# Efficient Copying in CuTe DSL — TV-Layout Coalesced Loads

The TV-layout coalesced load used in memory-bound reduction kernels
(`quack/media/2025-07-10-membound-sol.md`, `quack/quack/cross_entropy.py`).

```python
sX = smem.allocate_tensor(gX.element_type, ...)            # one tile of smem

copy_atom = cute.make_copy_atom(                            # 1 thread, 128-bit cp.async
    cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=128)

thr_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn).get_slice(tidx)

tXgX = thr_copy.partition_S(blkX)        # this thread's gmem elements (view)
tXsX = thr_copy.partition_S(sX)          # matching smem dest (view)
tXrX = cute.make_rmem_tensor_like(tXgX)  # registers (real allocation)

cute.copy(copy_atom, tXgX, tXsX)         # async gmem -> smem
cute.arch.cp_async_commit_group(); cute.arch.cp_async_wait_group(0)
cute.autovec_copy(tXsX, tXrX); x = tXrX.load()   # smem -> registers
```

---

## 1. Hierarchy

Two parallel chains that meet at `partition`:

```
how-to-copy:  copy_atom  →  tiled_copy  →  thr_copy = get_slice(tidx)
what-to-copy: tensor     →  partition_S/D(tensor)   →  per-thread fragment
```

- **copy_atom** — the indivisible hardware op: one thread moves 128 bits via
  `cp.async`. Knows nothing about how many threads exist.
- **tiled_copy** — that atom stamped across a tile per a TV-layout so the threads
  collectively cover it. Compile-time, identical for all threads.
- **get_slice(tidx)** — a `ThrCopy`: the tiling viewed from one thread's seat.
  Allocates nothing; just exposes `partition_S/D`.
- **partition_S/D(tensor)** — applies the TV-layout to a real tensor, returning
  *this thread's* slice. Same `thr_copy` partitions src and dst, so they're
  element-for-element aligned (what `cute.copy` requires).

## 2. `tiler_mn` and `tv_layout`

Same brick, two viewpoints:

- **`tiler_mn`** — the brick's **shape in the global tensor `mX`**. It is what
  `cute.local_tile(mX, tiler_mn, coord)` cuts out: the chunk **one CTA** loads and
  reduces. `tiler_mn` sets the size; the tile/block coord picks which one.
- **`tv_layout`** — the **`(thread, value) → offset-in-brick`** map. Built so
  adjacent threads hit adjacent addresses (coalesced) and each thread's `vecsize`
  values are one contiguous 128-bit load.

`mX` is `(M, N) = (batch, reduction_len)` — one row per reduction. So:

```
tiler_mn = (cols_per_block,  vecsize * num_blocks_N * threads_per_row)
            \_ batch (M) _/   \_________ reduction (N) __________/
```

The CTA's threads form a 2-D grid `threads_per_row × cols_per_block`
(`num_threads = threads_per_row * cols_per_block`):

| Term | Architectural meaning |
|---|---|
| `threads_per_row` | threads sweeping the **reduction** dim N side by side (jointly reduce one row) |
| `cols_per_block` | `num_threads // threads_per_row` → how many **rows/reductions** the CTA does at once (named "cols" because the article draws batch horizontally) |
| `vecsize` | elements per 128-bit load (`128/dtype.width`; 4 fp32, 8 bf16) |
| `num_blocks_N` | **per-thread loop count**: separate 128-bit loads each thread issues to cover its slice of N, since one row > `threads_per_row · vecsize` |

So `tiler_mn[1]` is just one CTA's N decomposed: `threads_per_row` threads ×
`vecsize` per load × `num_blocks_N` passes. The brick is big because a CTA is big.

Sanity check — tile elements must equal threads × values/thread:

```
cols_per_block · (vecsize · num_blocks_N · threads_per_row)
  = num_threads · (vecsize · num_blocks_N)
  = num_threads · values_per_thread        ✓   (Value Layout = (vecsize, num_blocks_N))
```

**Concrete** (bf16, N=4096, threads_per_row=32, num_threads=128):

```
vecsize=8   cols_per_block=128/32=4   num_blocks_N=ceil(4096/8,32)=16
tiler_mn = (4, 8·16·32) = (4, 4096)
```
Threads 32×4=128. Along N: 32 thr × 8 elems × 16 passes = 4096 (full row). Along
batch: 4 rows at once. Each thread holds 8·16=128 values; tile 4·4096 = 128·128. ✓

## 3. The copy, architecturally

`cute.copy` with `CopyG2SOp` lowers to **`cp.async` (`LDGSTS`)**: each thread fires
`num_blocks_N` async 128-bit gmem→smem transfers that **bypass registers**;
`cp_async_wait_group(0)` blocks until they land. Coalesced because adjacent threads
hit adjacent addresses. Then `autovec_copy` pulls smem→registers for the reduction.

---

## Relation to the MMA path

`thr_copy.partition_S/D` is the same idea as `thr_mma.partition_A/B/C` in GEMM — a
thread-specialized **view, no storage**. `tXrX = make_rmem_tensor_like(tXgX)` is the
`make_fragment` half (the real allocation). See [[MMA_LAYOUTS]] for the MMA funnel.
