# Epilogue Layout Setup in `host_function` (fp16_gemm_3_1)

Config: per-CTA tile `(128, 256)`, `io_dtype=fp16`, **no source C** (pure `A@B`),
C is **N-major** (row-major `(m,n)`), `epi_stages=2`.

The host builds 3 things (lines 746-766): `epi_tile`, the staged epilogue
SMEM layout, and the C TMA-store atom. They exist because the `128x256`
accumulator can't go TMEM->GMEM in one shot — it's streamed in **subtiles**.

---

## 1. `epi_tile = compute_epilogue_tile_shape(...)`  ->  `(128, 32)`

The subtile size for one epilogue iteration. Derivation for this config:

```
 warp grid     : cta_m=128 != 64        -> (warp_m, warp_n) = (4, 1)
 tile_m        : min(cta_m=128, 32*warp_m=128)        = 128
 n_perf        : no source C -> 4096 / tile_m = 4096/128 = 32   (256 % 32 == 0)
 n_min (align) : N-major, fp16 -> 128/16 * warp_n = 8
 tile_n        : min(cta_n=256, max(32, 8, 8))        = 32
                                                  => epi_tile = (128, 32)
```
- `tile_m=128`: 4 warps x 32 TMEM datapaths = the full 128 M rows at once.
- `tile_n=32`: 4096-element target / 128 rows; the SMEM-vs-pipeline sweet spot.

So the per-CTA `128x256` output divides into **8 subtiles** of `128x32` along N:

```
            per-CTA output tile  C[128 x 256]   (one CTA's M-half)
   N -> 0      32      64      96     128     160     192     224    256
        +-------+-------+-------+-------+-------+-------+-------+-------+
 M 128  | sub0  | sub1  | sub2  | sub3  | sub4  | sub5  | sub6  | sub7  |
        +-------+-------+-------+-------+-------+-------+-------+-------+
          128x32  ...                                     EPI_M=1, EPI_N=8
```

---

## 2. `epi_smem_layout_staged = make_smem_layout_epi(...)`

The SMEM staging buffer for the C subtile, **double-buffered** (`epi_stages=2`)
and swizzled for the TMA store:

```
   epi_smem_layout_staged : (128, 32, STAGE=2)  fp16, swizzled
                                       \____ ping-pong: store sub_i while
                                             writing sub_{i+1}

   epi_smem_layout = slice stage 0 -> (128, 32)   (template for TMA atom)
```

Only `128x32x2` fp16 of SMEM, vs the much larger 6-stage A/B pipeline — that
small footprint is exactly why `tile_n` was capped at 32.

---

## 3. `c_tma_atom = make_tiled_tma_atom(CopyBulkTensorTileS2GOp, c, epi_smem_layout, epi_tile)`

A **bulk tensor TMA store** (SMEM -> GMEM), tiled by `epi_tile=(128,32)`. One
`cute.copy(tma_atom_c, ...)` moves one `128x32` subtile from SMEM into the
right offset of global C. (S2G = store; the mainloop used G2S load atoms.)

---

## The data path these 3 objects wire up (kernel epilogue)

```
 TMEM acc          RMEM            RMEM         SMEM (staged)     GMEM
 128x256 fp32      fp32            fp16         128x32 fp16       C
   │                 │               │              │              │
   │  tcgen05        │  convert      │  r2s store   │  TMA bulk    │
   │  Ld32x32b       │  .to(fp16)    │  (swizzled)  │  store (S2G) │
   ▼  (t2r)          ▼               ▼              ▼              ▼
 [one 128x32 subtile per iteration] ───────────────────────────────►
                              loop subtile_idx = 0..7
                              double-buffered on epi_stages=2
```

Per output tile: **8 iterations**, each `128x32`. Stage `s = subtile_idx % 2`
ping-pongs so the TMA store of subtile *i* overlaps the TMEM->RMEM->SMEM of
subtile *i+1*.

---

## TL;DR

```
 compute_epilogue_tile_shape -> epi_tile (128,32)   = one epilogue subtile
 make_smem_layout_epi        -> SMEM (128,32,2)      = double-buffered stage
 make_tiled_tma_atom(S2G)    -> c_tma_atom           = SMEM->GMEM per subtile

 128x256 acc  =  8 subtiles x (128x32),  streamed TMEM->RMEM->SMEM->GMEM
```
