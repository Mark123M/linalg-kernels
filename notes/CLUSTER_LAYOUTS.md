# Cluster / CTA Layout Algebra (fp16_gemm_3_1)

Config: `cluster_shape_mnk=(2,1,1)`, `use_2cta_instrs=True`, `mma_tiler_mnk=(256,256,64)`.
Example problem: M=N=1024, K=512.

---

## 1. `thr_id` — the unit of CTA cooperation

`make_tiled_mma(op)` with `CtaGroup.TWO` → `size(tiled_mma.thr_id) == 2`.
A single `tcgen05` MMA is executed by a **pair** of CTAs ("V" dimension).

```
        2-CTA MMA pair (one logical 256-row tile)
        +-----------------------------------------+
        |   CTA v=0 (leader)   |   CTA v=1 (peer)  |
        |   rows   0..127      |   rows 128..255   |
        +-----------------------------------------+
                  \_______________________/
                     issues ONE MMA inst
```

---

## 2. `cta_layout_mnk = make_layout((2,1,1))`

Maps a cluster coord `(m,n,k)` -> linear CTA-rank-in-cluster.

```
   cluster M axis (size 2)
   +--------+--------+
   | rank 0 | rank 1 |     N=1, K=1  (single column/depth)
   +--------+--------+
```

---

## 3. `tiled_divide(cta_layout_mnk, (thr_id,))` -> `cta_layout_vmnk`

Peel `V=thr_id` out of the **M** cluster mode:

```
   cluster M = 2
        |
        |  divide by thr_id = 2
        v
   V = 2 , M' = 1          (N, K untouched)

   modes:  ( V , M , N , K )
   sizes:  ( 2 , 1 , 1 , 1 )
            ^   ^   ^   ^
            |   |   |   +-- cluster tiling along K
            |   |   +------ cluster tiling along N
            |   +---------- leftover cluster tiling along M
            +-------------- which CTA in the MMA pair  (leader=0 / peer=1)
```

General case (cluster `(4,2,1)` + 2cta): `V=2, M=4/2=2, N=2, K=1` -> `(2,2,2,1)`.
V is **always** `thr_id`, always peeled from the M mode.

---

## 4. How `vmnk` is consumed

```
 cta_in_cluster_coord_vmnk = vmnk.get_flat_coord(rank)
        rank 0 -> (0,0,0,0)   v=0  => leader CTA
        rank 1 -> (1,0,0,0)   v=1  => peer   CTA

 num_mcast_participants = size(M) + size(N) - 1 = 1 + 1 - 1 = 1
        (no extra M/N multicast; only the V-pair shares data)

 vmnk.shape -> make_tiled_tma_atom_A/B   (TMA partition + multicast split)
        A multicasts using N-extent (mode 2)
        B multicasts using M-extent (mode 1)
```

---

## 5. Same V factor on the OUTPUT tile

`cta_tile_shape_mnk = (mma_tiler_mnk[0] // size(thr_id), N, K)`

```
   logical MMA tile         per-CTA tile (V split)
   M = 256             ->   M = 256/2 = 128
   N = 256                  N = 256
   K = 64                   K = 64
```

Grid = C(1024x1024) / per-CTA(128x256):

```
   M: 1024/128 = 8 CTAs   ┐
   N: 1024/256 = 4 CTAs   ┘ -> 32 CTAs = 16 clusters of 2
                              = 16 output tiles (256x256)
```

---

## TL;DR

```
 thr_id (=2) = CTAs per MMA  ──┬──▶ tiled_divide peels it from cluster M
                               │        => VMNK = (2,1,1,1)
                               │        => leader/peer id, mcast masks, TMA split
                               └──▶ halves the M output tile: 256 -> 128 per CTA
```
