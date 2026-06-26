# MMA Tensor Layout Flow (fp16_gemm_3_1)

How a global matrix becomes the exact operand one `tcgen05` instruction reads.
Running example: M=N=1024, K=512, leader CTA `v=0`, `mma_tiler=(256,256,64)`.

> **Naming:** `tCxY` = **t**ensor, partitioned for the **C** computation (the MMA),
> in space **x** (`g`mem / `r`mem-fragment / `t`mem), operand **Y**.
> `tCgA` = A in gmem, MMA coords. `tCrA` = A as MMA fragment. `tCtAcc` = acc in TMEM.

---

## The funnel — each step narrows scope / changes coordinate system (no data moves)

```
 global matrix (M,N,K)
    │  local_tile        chop into CTA-tiles         coords: global, regrouped
    ▼
 gA/gB/gC  (one tile, grid-of-tiles)
    │  get_slice(v)      "I am CTA v in the pair"     pick a participant lens
    ▼
 thr_mma
    │  partition_*(g)    overlay the MMA atom         coords: MMA-atom, MY share
    ▼                    (256 M → my 128, 64 K → 4×16)
 tCgA/tCgB/tCgC  ───────►  GMEM side: drives TMA loads / epilogue stores
    │
    │  make_fragment_*(s)  bind to SMEM/TMEM          + STAGE = which buffer
    ▼
 tCrA/tCrB (smem descriptors), tCtAcc (tmem)  ──────►  compute side: cute.gemm
```

| stage | verb | does | coord system |
|---|---|---|---|
| `local_tile` | **chop** | pick work granularity (256×64 / 256×256); split into *(inside tile, which tile)*. No MMA yet. | global, regrouped |
| `get_slice(v)` | **whose** | bind participant id (CTA in the 2-SM pair; a lane for SIMT). Returns `thr_mma`. Computes nothing. | — |
| `partition_*` | **describe** | re-express the tile in the instruction's footprint for *my* share. M 256→128, K 64→4×16. | MMA-atom |
| `make_fragment_*` | **bind** | attach that layout to the real resource the instruction touches; add `STAGE`. | MMA-atom + buffer |

**One line:** *tile* = how big a chunk; *get_slice* = whose chunk; *partition* = that
chunk in the instruction's language; *make_fragment* = hand the instruction the
actual SMEM/TMEM it reads & writes.

---

## What you might miss

**1. The fork — same MMA coords, two sides.**
`partition_*(gA)` views **gmem** → tells **TMA** which tile to fetch + the epilogue
where to store. `make_fragment_*(sA)` views **smem/tmem** → what **MMA** consumes.
Identical MMA-coordinate structure ⇒ loaded data lands exactly where `cute.gemm`
expects. (`tma_partition` is the extra step that turns the gmem-side `tCg*` into
TMA-descriptor coordinates for `cute.copy`.)

**2. Reading the modes.** Leading **`MMA`** mode = *what one instruction touches*
(128×16 operand, 128×256 acc). Trailing modes = *how many times* (`MMA_K`=4 K-blocks)
or *which buffer* (`STAGE`).

**3. tcgen05-SS fragments are DESCRIPTORS, not registers.**
`tCrA = (1,1,4,6)` — value modes collapse to `1`: the 128×16 data isn't held in
registers, it's *referenced* by one SMEM descriptor. Only `MMA_K=4` and `STAGE=6`
iterate. The MMA reads SMEM (+ peer half) directly — the point of SS operands.

**4. The accumulator `_fake` pattern.** TMEM isn't allocated when layouts are built,
so `make_fragment_C` makes a **layout-only** template (`tCtAcc_fake`). The real
pointer arrives later and is rebound to the *same* layout:
`tCtAcc_base = make_tensor(tmem_ptr, tCtAcc_fake.layout)`.

---

## Verified layouts (`s@d` = stride `s` along original axis `d`)

```
thr_id       2:1                                       # V = 2 CTAs / MMA

gA           (256,64,4,8)     : (1@0,1@1,256@0,64@1)   # bM,bK,RestM,RestK
gC           (256,256,4,4)    : (1@0,1@1,256@0,256@1)  # bM,bN,RestM,RestN

tCgA   ((128,16),1,4, 4,8)    : ((1@0,1@1),0,16@1,256@0,64@1)
                  └ atom 128M×16K  └MMA_K=4   └Rest 4×8
tCgC   ((128,256),1,1, 4,4)   : ((1@0,1@1),0,0,256@0,256@1)
                  └ acc 128M×256N             └Rest 4×4

acc_shape    ((128,256),1,1)
tCtAcc_fake  ((128,256),1,1, 2): ((65536,1),0,0,256)
             # TMEM: N stride 1, M stride 2^16 (datapath); STAGE stride 256
             # 2 stages × 256 cols = 512 = num_tmem_cols

tCrA / tCrB  (1,1,4,6) : (0,0,2,1024)   # descriptor: MMA_K=4, STAGE=6
```

These line up 1:1 with the per-SM-per-instruction shape **(128, 256, 16)**:
`tCgA` atom = 128×16 A, `tCgB` = 128×16 B-half, `tCgC`/`tCtAcc` = 128×256 acc,
`MMA_K=4` = the four K=16 steps per 64-K tile.

---

## TL;DR

```
 local_tile     → how big a chunk of the matrix
 get_slice(v)   → which participant's share (CTA in the 2-SM pair)
 partition_*    → that chunk in MMA-atom coords   → GMEM side feeds TMA
 make_fragment_*→ bound to real SMEM/TMEM (+STAGE) → compute side feeds cute.gemm
 MMA mode = one instruction's footprint; trailing modes = count / buffer
```

---

# Object Reference — every object on the path to `cute.gemm`

What each represents and how it's built, in dataflow order.

## Roots (built on host, passed into the kernel)
| object | type | what it is | how created |
|---|---|---|---|
| `tiled_mma` | `cute.TiledMma` | the MMA **recipe**: atom shape + how a CTA-tile maps onto participants | `make_tiled_mma(op)` from `tcgen05.MmaF16BF16Op` |
| `mA_mkl/mB_nkl/mC_mnl` | `cute.Tensor` (gmem) | global A/B/C **coordinate handles** (the TMA tensors) | returned by `make_tiled_tma_atom_*` host-side |
| `a_smem_layout/b_smem_layout` | `cute.ComposedLayout` | SMEM **layout templates** for A/B (swizzled, staged) | `make_smem_layout_a/b(tiled_mma, mma_tiler, dtype, ab_stages)` |

## SMEM buffers (physical memory, allocated in-kernel)
| object | type | what it is | how created |
|---|---|---|---|
| `sA` | `cute.Tensor` (smem) | the **A SMEM buffer** — A k-tiles, 6 stages. `((128,16),1,4,6)` ≈ 128M×64K×6 | `smem.allocate_tensor(io_dtype, a_smem_layout.outer, swizzle=a_smem_layout.inner)` |
| `sB` | `cute.Tensor` (smem) | same for **B** (128N×64K×6) | `smem.allocate_tensor(..., b_smem_layout...)` |
| `sC` | `cute.Tensor` (smem) | epilogue **C staging buffer** (128×32 ×2) | `smem.allocate_tensor(..., epi_smem_layout_staged...)` |

## Global CTA-tiles (gmem, global coords)
| object | type | what it is | how created |
|---|---|---|---|
| `gA` | `cute.Tensor` (gmem) | A chopped into **pair-tiles**+grid: `(256,64, 4,8)` = (bM,bK,RestM,RestK) | `local_tile(mA_mkl, slice_(mma_tiler,(None,0,None)), (None,None))` |
| `gB` | `cute.Tensor` (gmem) | B: `(256,64, 4,8)` = (bN,bK,RestN,RestK) | `local_tile(mB_nkl, slice_(mma_tiler,(0,None,None)), …)` |
| `gC` | `cute.Tensor` (gmem) | C: `(256,256, 4,4)` = (bM,bN,RestM,RestN) | `local_tile(mC_mnl, slice_(mma_tiler,(None,None,0)), …)` |

`gA`'s tile is the **full 256-M pair tile**; the per-CTA 256→128 split happens in partition.

## MMA lens
| object | type | what it is | how created |
|---|---|---|---|
| `thr_mma` | `cute.ThrMma` | `tiled_mma` **projected to this CTA** (participant `v` in the pair) | `tiled_mma.get_slice(mma_tile_coord_v)` |

## MMA-coord global views (gmem, atom coords → feed TMA/epilogue)
| object | type | what it is | how created |
|---|---|---|---|
| `tCgA` | `cute.Tensor` (gmem) | global A, **my share**: `((128,16),1,4, 4,8)` — atom 128M×16K, MMA_K=4, grid 4×8 | `thr_mma.partition_A(gA)` |
| `tCgB` | `cute.Tensor` (gmem) | global B, my N-half: `((128,16),1,4, 4,8)` | `thr_mma.partition_B(gB)` |
| `tCgC` | `cute.Tensor` (gmem) | global C: `((128,256),1,1, 4,4)` — acc tile 128×256 | `thr_mma.partition_C(gC)` |

## MMA operand fragments (what `cute.gemm` reads — SMEM descriptors)
| object | type | what it is | how created |
|---|---|---|---|
| `tCrA` | `cute.Tensor` (smem-desc) | A operand = **SMEM descriptor**: `(1,1,4,6)` = (MMA, MMA_M, MMA_K=4, STAGE=6) | `tiled_mma.make_fragment_A(sA)` |
| `tCrB` | `cute.Tensor` (smem-desc) | B operand descriptor: `(1,1,4,6)` | `tiled_mma.make_fragment_B(sB)` |

Value modes are `1` — the 128×16 data lives in `sA`/`sB`; the descriptor points at it.

## Accumulator (what `cute.gemm` writes — TMEM)
| object | type | what it is | how created |
|---|---|---|---|
| `acc_shape` | `tuple` (Shape) | the **shape** of the per-CTA acc: `((128,256),1,1)` | `tiled_mma.partition_shape_C(mma_tiler[:2])` |
| `tCtAcc_fake` | `cute.Tensor` (layout-only) | **layout-only** acc template (no TMEM yet): `((128,256),1,1, 2)` | `tiled_mma.make_fragment_C(append(acc_shape, acc_stages))` |
| `tCtAcc_base` | `cute.Tensor` (tmem) | the **real** acc over TMEM (same layout, real ptr) | `make_tensor(tmem_ptr, tCtAcc_fake.layout)` after `tmem.retrieve_ptr` |
| `tCtAcc` | `cute.Tensor` (tmem) | one **stage slice** — the `cute.gemm` target | `tCtAcc_base[(None,None,None,acc_index)]` |

## TMA partitions (gmem↔smem load/store mapping)
| object | type | what it is | how created |
|---|---|---|---|
| `tAsA, tAgA` | `cute.Tensor` ×2 (smem, gmem) | A's **SMEM dest** + **GMEM src** in TMA coords | `tma_partition(tma_atom_a, …, group_modes(sA), group_modes(tCgA))` |
| `tBsB, tBgB` | `cute.Tensor` ×2 (smem, gmem) | same for B | `tma_partition(tma_atom_b, …, sB, tCgB)` |
| `tCsC, tCgC_tma` | `cute.Tensor` ×2 (smem, gmem) | epilogue C: SMEM src + GMEM dest | `tma_partition(tma_atom_c, …, sC, gC_epi)` |

## Object graph
```
 tiled_mma ──get_slice(v)──► thr_mma
                                │ partition_A/B/C
 mA/mB/mC ──local_tile──► gA/gB/gC ──────────────► tCgA/tCgB/tCgC ──┐
                                                                     │ tma_partition
 a/b_smem_layout ──allocate_tensor──► sA/sB ──┐                      ▼
                                              │ make_fragment_A/B   tAsA/tAgA …  (TMA loads)
                                              ▼                      │ fills
                                          tCrA/tCrB ◄────────────────┘
                                              │
 acc_shape ─make_fragment_C─► tCtAcc_fake ─(+tmem_ptr)─► tCtAcc_base ─[stage]─► tCtAcc
                                              │                                    │
                                              └────────► cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)
```

**Mental split:** `s*` = physical SMEM; `g*` = global tiles; `tCg*` = global-in-MMA-coords (→ TMA);
`tCr*` = SMEM operands (→ MMA); `tCtAcc*` = TMEM accumulator (← MMA); `tAsA/tAgA…` = TMA's view linking gmem↔smem.

---

# Q&A

## What is the `l` in `mA_mkl` / `mB_nkl` / `mC_mnl`?

The suffix just spells out CuTe's **M-N-K-L** mode-order convention — it documents each
tensor's logical axes, *not* extra data:

| name | axes |
|---|---|
| `mA_mkl` | (M, K, L) |
| `mB_nkl` | (N, K, L) |
| `mC_mnl` | (M, N, L) |

**`L` = the batch / group mode** — the number of independent GEMMs stacked together
(batched matmul). The persistent scheduler carries an `l` tile-coord and the grid is
`(*num_ctas_mn, L)`.

**In this tutorial `L = 1`** (a single, non-batched GEMM). Verified the tensors are
literally **2-D** — there is *no* materialized L axis:
```
a (input)     rank 2   (M, K)
a_tma_tensor  rank 2   (M, K)
gA            rank 4   (256, 64, RestM, RestK)   # no L mode
```
So here `l` is convention/placeholder: the scheduler's `l` coord is always 0 and never
indexes a tensor axis (`mma_tile_coord_mn` drops it). For a batched GEMM (L>1) the
tensors would be 3-D and `l` would select which matrix.

## What are "coordinate handles" (the TMA tensors `mA_mkl`/…)?

A **normal** `cute.Tensor` = `engine(pointer) + layout`; indexing `t[c]` computes an
address `ptr + layout(c)` and **dereferences memory**.

A **TMA tensor** is different: its engine is **not a pointer** — it's a **coordinate
iterator**. Indexing/slicing it yields a **logical `(M,K)` coordinate** into the global
matrix, *not* an address, and you never load/store through it directly. This is exactly
why the probe printed strides as `1@0, 256@0` (basis vectors `e₀, e₁`) instead of
integers — the values it produces are coordinates, not byte offsets.

Usage — slice the handle to one tile's coord, hand it to `cute.copy`:
```
mA_mkl  (coordinate handle)
  local_tile → gA → partition_A → tCgA → tma_partition → tAgA   (all coordinate slices)
  cute.copy(tma_atom_a, tAgA[k_tile], tAsA[stage], bar, mask)
                        ^descriptor    ^coordinate   → TMA unit issues the bulk copy
```
- **atom** = the TMA **descriptor** (real base addr + global shape/strides + swizzle,
  baked at host time) → *where the data is*.
- **handle** = the coordinate tensor → *which tile*.

`cp.async.bulk.tensor` addresses memory by **tensor coordinates against a descriptor**,
not flat pointers — so the kernel computes coordinates and the handle is what you index
to get them. That's why `make_tiled_tma_atom` returns the **pair** (atom + handle): same
memory, two roles. (Same as the raw `c` pointer-tensor vs `mC_mnl` coordinate-handle.)

## Partition vs fragment (and why "value modes are 1")

- **Partition** = a **view** re-labeling an *existing* tensor into MMA coords: "which
  slice is mine." Allocates nothing. `partition_A(gA) → tCgA`.
- **Fragment** = the **per-participant storage the MMA instruction operates on**.
  `make_fragment_A(sA) → tCrA`.

Classic SIMT MMA makes it concrete:
```
 gmem partition ──copy──► rmem FRAGMENT (holds values) ──► MMA reads registers
```
There a fragment **is a register array** — each thread holds e.g. 8 fp16 values. The
leading **value mode** = 8 = real data elements in registers.

**tcgen05 SS** (SMEM-source) twist: the tensor core reads A/B **directly from SMEM** —
operands never enter registers. So `tCrA` holds no values; it holds a **matrix
descriptor** (~64-bit: SMEM base offset, stride, swizzle). Hence `tCrA = (1,1,4,6)`:
```
 (  1  ,  1  ,  4  ,  6  )
  value  MMA_M MMA_K STAGE
   └─ ONE descriptor, not 128×16 values
```
- **"value modes are 1"** → the dimension that would carry data elements is collapsed to
  1 (no per-participant registers, just one descriptor).
- **"128×16 data lives in sA/sB"** → the actual elements sit in SMEM (TMA-loaded there).
- **"descriptor points at it"** → `tCrA`'s single entry tells the MMA the SMEM
  address/layout of the 128×16 tile.

| | SIMT MMA (Ampere HMMA) | **tcgen05 SS** (this kernel) |
|---|---|---|
| operand source | registers | **SMEM** |
| `make_fragment_A` | register tile, value mode = e.g. 8 | **descriptor**, value mode = **1** |
| data lives in | rmem (copied in) | **sA/sB** (stays in SMEM) |
| MMA reads from | registers | SMEM via descriptor |

So for tcgen05-SS a "fragment" is almost a misnomer — it's a **pointer-with-layout into
SMEM**, not a tile of values. That's the point of SS operands: keep A/B in SMEM, feed the
tensor core descriptors instead of burning registers streaming them through.

## Order of object use in one pipeline execution

Every object is defined **once, spanning all stages**; the mainloop just **indexes a
different slice each iteration**. Static object = the whole circular buffer; loop index =
the current slot.

**Index legend — which loop var selects which slice:**

| index | range | selects |
|---|---|---|
| `M_tile/N_tile` | per tile | `tAgA[…,M_tile,…]`,`tBgB[…,N_tile,…]` — this tile's gmem column |
| `acc` | per tile (0/1) | `tCtAcc_base[…,acc]` — which accumulator buffer |
| `k_tile` | 0..7 | `tAgA_slice[…,k_tile]` — which K-chunk of gmem |
| `s` (=`handle.index`) | 0..5 | `tAsA[s]`,`tBsB[s]`, `tCrA[…,s]`,`tCrB[…,s]` — AB pipeline buffer |
| `k_block` | 0..3 | `tCrA[…,k_block,s]` — which 16-wide K-block |
| `subtile` | 0..7 | `tTR_tAcc[…,subtile]`,`tCgC_grouped[…,subtile]` — which N-strip |
| `cb` (=`subtile%2`) | 0/1 | `tRS_sC[…,cb]`,`tCsC[…,cb]` — epilogue SMEM buffer |

**Timeline for one output tile:**
```
SETUP (once)
  tAgA_slice = tAgA[(None,M_tile,None)] ; tBgB_slice = tBgB[(None,N_tile,None)]
  tCtAcc     = tCtAcc_base[…, acc]

MAINLOOP  k_tile = 0..7   (s = k_tile % 6)          TMA ahead, MMA trails
  TMA: 1. ab_producer.acquire → buffer s free
       2. copy tAgA_slice[k_tile] ─► tAsA[s]        # gmem coord → SMEM A
       3. copy tBgB_slice[k_tile] ─► tBsB[s]        #            → SMEM B
  MMA: 4. ab_consumer.wait → buffer s full
       5. for k_block 0..3: gemm(tCtAcc, tCrA[k_block,s], tCrB[k_block,s])
                                       # tCrA[…,s] = descriptor INTO tAsA[s]
       6. handle.release() → buffer s free (TMA may refill for k_tile=s+6)

AFTER 8 k_tiles
  7. acc_empty.commit() (MMA) → tCtAcc complete (128×256)
  8. acc_consumer.wait() (epi)

EPILOGUE  subtile = 0..7   (cb = subtile % 2)
  9.  copy tTR_tAcc[subtile] ─► tTR_rAcc            # TMEM → RMEM (t2r)
  10. tRS_rC = tRS_rAcc.to(fp16)                    # convert
  11. copy tRS_rC ─► tRS_sC[cb]                     # RMEM → SMEM (r2s)
  12. fence + epilog barrier
  13. (warp0) copy tCsC[cb] ─► tCgC_grouped[subtile]  # SMEM → GMEM (TMA store)
  14. epilog barrier

DONE
  15. acc_full.release() → acc buffer free for next output tile
```

**Key linkages:**
- steps 2/3 ↔ 5: TMA fills `tAsA[s]`; `tCrA[…,s]` is the descriptor *pointing at* `tAsA[s]`.
  Same `s` ⇒ MMA reads exactly what TMA loaded.
- `s` cycles 0..5 while `k_tile` goes 0..7 — 6 buffers serve 8 chunks; step 6 frees `s`
  for `k_tile=s+6`.
- `acc` fixed for the whole tile (one accumulator reduced over 8 k_tiles); flips only
  between output tiles (double-buffer: epilogue drains tile N while MMA starts N+1).
- `cb` flips per subtile so the TMA store of `subtile` overlaps the t2r/r2s of `subtile+1`.

## What are `tAgA`/`tBgB` (and `tAsA`/`tBsB`)? How does `tma_partition` build them?

These are what `cute.copy` consumes in the TMA loop — the **TMA-copy** views of A/B.

```python
# tAsA: ((atom_v,rest_v), STAGE)        tAgA: ((atom_v,rest_v), RestM, RestK)
tAsA, tAgA = tma_partition(
    tma_atom_a,
    cta_in_cluster_coord_vmnk[2],                  # my coord in the multicast (N) dim
    make_layout(size(cta_layout_vmnk, mode=[2])),  # multicast group (N extent)
    group_modes(sA,   0, 3),                        # SMEM dest, tile modes fused
    group_modes(tCgA, 0, 3),                        # GMEM coords, tile modes fused
)
```

**Represent:**
- `tAgA` = global A coordinate handle **re-cut into TMA transfer units**: leading
  `(atom_v,rest_v)` = one TMA *box* (the whole tile one `cp.async.bulk.tensor` moves);
  trailing `(RestM,RestK)` = which tile. → gmem coords to fetch.
- `tAsA` = matching **SMEM dest**: same box, trailing `STAGE` = which buffer.
- A **matched (src,dst) pair** with identical box ⇒ `cute.copy(atom, tAgA[tile], tAsA[s])`
  aligns. You pass the box whole (`None`); only trailing modes are indexed.

**Construction:**
1. `group_modes(x,0,3)` fuses the inner `(MMA,MMA_M,MMA_K)` modes into one "whole tile"
   mode, leaving `RestM,RestK` (gmem) / `STAGE` (smem).
2. partition that tile into `(atom_v,rest_v)` per the atom's box/swizzle.
3. factor out multicast via `(coord, layout)` — removes the cluster-broadcast dim
   (A: N mode; B: M mode). N=M=1 here ⇒ trivial, full tile each.

```
 tCgA ((128,16),1,4, 4,8) ─group(0,3)→ (TILE,RestM,RestK) ─partition→ tAgA ((atom_v,rest_v),RestM,RestK)
 sA   ((128,16),1,4, 6)   ─group(0,3)→ (TILE,STAGE)       ─partition→ tAsA ((atom_v,rest_v),STAGE)
```

So `tma_partition` turns **MMA-coord** views (`tCgA`,`sA`) into **TMA-copy** views:
leading = one box to transfer, trailing = which-tile / which-buffer. `tBgB`/`tBsB` are
identical with **M** (mode 1) as the multicast dim instead of N.
