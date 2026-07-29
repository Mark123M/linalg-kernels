# `b2n4096` B200 Cholesky design

## Status

The tracked default is variant 11 (`torch_per_matrix_loop`). Variant 2 remains
the fastest previously submitted native path at 4.955 ms against an 11.177 ms
Torch/cuSOLVER batched baseline. Variant 11 is now confirmed by NSys to use
two copies of cuSOLVER's batch-one fused Xpotrf kernel; its profiling range is
3.630 ms, including the profiler/output adapter copies. It still requires an
official Popcorn timing gate before its default is considered final.

Variant 7 (`direct_adaptive_m128_m64_m32_fp32`) is the next-fastest variant
that also clears the exact-shape stress families, at 8.605 ms.

The shape file is self-contained and routes only contiguous CUDA FP32 tensors
with shape `(2, 4096, 4096)` to a selected specialized variant. Every other
input retains the Torch/cuSOLVER path.

## Stable registry

| ID | Implementation |
|---:|---|
| 0 | Torch/cuSOLVER baseline |
| 1 | left-looking NB512, micro64, fused split-consumer TF32 |
| 2 | left-looking NB512, micro64, inverse-GEMM TF32 |
| 3 | left-looking NB256, micro64, fused split-consumer TF32 |
| 4 | direct left-looking 128 to 64 at remaining 1024, TF32 |
| 5 | direct left-looking 128 to 64 to 32 at remaining 1024/256, TF32 |
| 6 | direct fixed micro64, TF32 |
| 7 | direct adaptive 128/64/32, FP32 GEMMs |
| 8 | MAGMA-style hybrid, direct ATen/LAPACK CPU POTRF64 |
| 9 | MAGMA-style hybrid, compiled fixed-shape CPU POTRF64 |
| 10 | MAGMA-style hybrid, direct ATen/LAPACK CPU POTRF128 |
| 11 | Torch/cuSOLVER, one single-matrix POTRF per batch entry |

## Batched-POTRF dispatch anomaly

Variant 0 hands the whole `(2, 4096, 4096)` tensor to
`torch.linalg.cholesky_ex`, which routes to cuSOLVER's batched POTRF. That
path is built for many small matrices and is pathological at this n. From a
single torch-only benchmark run
(`artifacts/tuning/b2n4096_20260728T143039Z/round_00_variant_00.api.json`):

| Entry | Shape | Torch |
|---:|---|---:|
| [10] | batch 1, n 4096 | 1.528 ms |
| [11] | batch 2, n 4096 | 11.177 ms |

Twice the work costs 7.32x the time. At batch 1 the dispatch takes the
single-matrix `Xpotrf` path instead, which is a fused tiled wavefront kernel.

The variant-0 NSys trace is retained at
`artifacts/nsys/b2_n4096_20260729T075049Z/v0_torch_cusolver/`. It confirms
PyTorch's `potrfBatched` dispatch:

| Kernel family | Count | Total |
|---|---:|---:|
| `potrf_cta_lower_batch<...,16>` | 256 | 1.466 ms |
| `potrfBatch_trsm_lower<...,16>` | 255 | 0.601 ms |
| `potrf_syrk_T16_nc_kernel` | 192 | 5.968 ms |
| `potrf_syrk_nc_kernel` | 63 | 2.571 ms |

Those 769 solver kernels total 10.607 ms. With setup, copying, cleanup, and
launch gaps, the captured range spans 11.550 ms.

Variant 11 factors each matrix separately so both calls take the batch-one
path. Its NSys trace is retained at
`artifacts/nsys/b2_n4096_20260729T075335Z/v11_torch_per_matrix_loop/`.
The two Xpotrf kernels take 1.394 and 1.382 ms, or 2.776 ms total. The full
captured range is 3.630 ms. It includes four approximately 73 us copy
kernels: two input clones plus two slice assignments into the profiler's
preallocated output. The normal return path still performs the two slice
assignments, so a native Xpotrf wrapper with direct column-major output
remains the preferred follow-up.

Entry [8] (`batch 2, n 2048`, 3.149 ms) shows the same signature: it costs
more than the 1.528 ms `batch 1, n 4096` factorization despite having a
quarter of the FLOPs. Entries [6] and [9] are worth checking for it too.

## Batch-parallel GPU algorithms

Variants 1–3 use an outer left-looking panel. Each large and inner history
update is one `cublasGemmStridedBatchedEx` call over both matrices. The two
factor producers and their consumer grids are launched per matrix so their
release/acquire flags and inverse storage remain independent.

The micro-factor kernel loads a 32-, 64-, or 128-wide lower tile into shared
memory, performs POTRF, and builds the inverse used by the below-panel solve.
Variant 1 splits each 64-column consumer job over two CTAs. Variant 2 uses one
strided-batched inverse GEMM followed by per-matrix copy-back kernels.

Variants 4–7 are direct left-looking recurrences. Every step performs one
strided-batched full-history GEMM and then launches a factor/solve grid for
each matrix. Variant 7 changes those GEMMs to FP32 computation while retaining
the same adaptive 128/64/32 sequence.

Matrix bases, strides, and products use 64-bit indexing. Each lower-copy
kernel zeros a complete upper triangle; staged kernels only write diagonal
and below-diagonal panels.

## Hybrid CPU–GPU recurrence

Variants 8–10 follow the lower recurrence in
`magma/src/zpotrf_gpu.cpp`:

1. Update both diagonal tiles with one strided-batched GPU GEMM.
2. Copy both tiles asynchronously into one preallocated pinned CPU tensor and
   record one readiness event.
3. Enqueue one strided-batched below-diagonal history GEMM.
4. Wait only for panel readiness and run batched CPU POTRF while the GPU
   history update executes.
5. Enqueue both factored tile copies after the update, then issue one GPU TRSM
   per matrix.

All CUDA work uses the existing default execution queue; no secondary queue
is created. Variant 9 exposes private phase helpers and warms a contiguous
fixed `(2,64,64)` `torch.compile(fullgraph=True, dynamic=False,
mode="max-autotune")` CPU specialization before timing.

`CHOLESKY_PROFILE_NVTX=1` enables panel-level ranges for diagonal update,
transfers, history update, panel wait, CPU POTRF, and GPU solve. Production
execution leaves them disabled.

## Compilation and resources

The extension targets `sm_100a` with `-O3`, fast math, extra device
vectorization, restricted pointers, line information, expensive ptxas
optimization, and spill reporting. Host code uses native ISA tuning, FMA,
fast/unsafe math, and C++20.

The local CUDA 13.1 compilation completed. `cuobjdump --dump-resource-usage`
reported `LOCAL:0` for every emitted production kernel. This proves the
compilation/resource property only; the local GPU cannot execute `sm_100a`.

## CPU alternatives

Direct ATen/LAPACK is the primary CPU backend. The compiled variant keeps a
fixed-shape LAPACK operation so compiler/Python overhead can be isolated. A
tensorized rank-one factorization was measured locally at about 0.67 ms for
`(2,64,64)` and rejected as unsuitable for panel overlap.

## Validation and promotion record

The exact-shape B200 sweep covered all 11 variants on dense, spectrum,
diagonal, low-rank, row-scaled, and tridiagonal inputs. The retained report is
`artifacts/validation/b2_n4096_20260728T142231Z/results.json`.

- Variants 0 and 7 passed every case.
- TF32 variants passed the target dense case but became non-finite on the
  low-rank stress input; inverse-GEMM variant 2 also failed spectrum.
- Hybrid variants passed dense but their CPU POTRF rejected the low-rank
  recurrence after the TF32 history update.
- The initial direct implementation left general-GEMM values above microtile
  diagonals. The final 128-wide wedge cleanup corrected both batch shapes.

The full screen is retained at
`artifacts/tuning/b2n4096_20260728T143039Z/summary.json`. All 11 variants
submitted cleanly in `--mode benchmark` (`check: pass`, 15/15 entries).
Variant 2 was fastest at 4.955 ms; variant 7 screened at 8.599 ms against an
11.177 ms baseline.

The automated promotion run is
`artifacts/tuning/b2n4096_20260728T143231Z/summary.json`:

| Variant | Median target mean |
|---:|---:|
| 0 | 11.093 ms |
| 7 | 8.605 ms |

That run swept only variants 0 and 7 because the stress-validation gate had
excluded the rest. The 0.995 threshold was 11.038 ms; variant 7 was 22.434%
faster, so the runner atomically changed `_DEFAULT_VARIANT` from 0 to 7.
The tracked default was subsequently set to variant 2 by hand.

Variant 2 was additionally confirmed on 2026-07-28 in `--mode test`
(17/17, submission 921947, residuals 0.00064 to 0.024 against a 20.0 limit)
and in `--mode benchmark`. Note that no official test shape routes to this
specialization — the largest is `(1, 2048, 2048)` and both `lowrank` cases
are n=256 and n=1024 — so the test-mode result exercises the Torch fallback
rather than the kernel.

Evidence is retained under:

- `artifacts/validation/` for dense, spectrum, diagonal, low-rank,
  row-scaled, and tridiagonal checks;
- `artifacts/tuning/` for screening and three alternating rounds;
- `artifacts/nsys/` for leading native and hybrid timelines;
- `artifacts/ncu/` for leading native kernel reports;
- `artifacts/helpers/ncu/` for analysis helpers.

Commands:

```bash
python cholesky/b2n4096/cholesky_b2n4096_runner.py autotune \
  --variants all --rounds 3
python cholesky/b2n4096/cholesky_b2n4096_runner.py ncu --variants 1,4
.venv/bin/modal run cholesky/b2n4096/cholesky_b2n4096_modal.py --variant 1
.venv/bin/modal run cholesky/b2n4096/cholesky_b2n4096_modal.py --validate
python cholesky/b2n4096/cholesky_b2n4096_runner.py submit --mode test
python cholesky/b2n4096/cholesky_b2n4096_runner.py submit --mode leaderboard
```

Nsight Systems now confirms both the 772-kernel batched baseline and the
two-kernel Xpotrf control. The profiler preflight was also corrected to emit
metadata for Python-orchestrated variant 11 instead of indexing beyond the
native metadata table.

If NCU shows GEMMs above 30% of target time and the kernel/SASS evidence does
not identify an SM100 Tensor Core path, the next candidate is a CUTLASS 3.x
SM100 TF32 TMA/TMEM GEMM. It is not added without that evidence.

## Integrated submission build reuse

The integrated `cholesky/cholesky.py` submission shares the native Xpotrf
extension with `(1,4096,4096)`. Its entry point accepts only batch counts 1
and 2 and retains the same per-matrix copy and Xpotrf loop used by variant
12. This is a build-latency change: it removes a duplicate CUDA compilation
but does not alter the factorization or timing path.

Popcorn submission 928863 passed public and secret test, benchmark, and
leaderboard evaluation. Its public geomean was 893.371 us and its secret
geomean was 893.050 us.
