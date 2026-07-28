# `b1n4096` B200 Cholesky design

## Status

The tracked default is variant 0 (`torch.linalg.cholesky_ex`). The B200 gate
completed on 2026-07-28 and retained it: the only fully passing native
candidate was substantially slower. The historical 1.53 ms baseline was
confirmed by the contemporaneous three-round median.

The shape file is self-contained and routes only contiguous CUDA FP32 tensors
with shape `(1, 4096, 4096)` to a selected specialized variant. Every other
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

## GPU-native algorithms

Variants 1–3 use an outer left-looking panel. One large history GEMM updates
each outer panel. Subsequent micro panels use inner history GEMMs. The
micro-factor kernel loads a 32-, 64-, or 128-wide lower tile into shared
memory, performs a positive-diagonal POTRF, and builds the inverse used by the
below-panel solve.

The fused variants launch one producer CTA and co-resident consumer CTAs.
Release/acquire flags publish the inverse after factorization. Consumers
preload the below-panel tile while the producer works, then apply the inverse.
Variant 1 splits each 64-column consumer job over two CTAs. Variant 2 retains
separate inverse GEMM and copy-back kernels as a launch/throughput control.

Variants 4–7 are direct left-looking recurrences. Every step performs one
full-history GEMM, then a shape-specialized POTRF/solve. The width policies
are explicit in the registry. Variant 7 is the numerical control: it changes
all history GEMMs from fast TF32 to FP32 computation while keeping the same
adaptive factor/solve sequence.

All matrix offsets and work counts that can exceed 32-bit products use
64-bit indexing. The output copy zeros the full upper triangle before staged
updates. Subsequent kernels only write the diagonal and below-diagonal
panels.

## Hybrid CPU–GPU recurrence

Variants 8–10 follow the lower-path recurrence in
`magma/src/zpotrf_gpu.cpp`:

1. Update the diagonal tile on the GPU.
2. Copy it asynchronously to a preallocated pinned CPU tensor and record a
   readiness event.
3. Enqueue the independent below-diagonal history GEMM.
4. Wait only for the panel event and run CPU POTRF while that GEMM executes.
5. Enqueue the factored tile copy after the history GEMM, then run GPU TRSM.

All CUDA work is ordered on the existing default execution queue. No
secondary queue is created. Variant 9 exposes private native copy, stage, and
finish helpers to Python. Its fixed `(1,64,64)` CPU specialization is warmed
with `torch.compile(fullgraph=True, dynamic=False, mode="max-autotune")`
before timing. The compiled result is copied into a pinned factor buffer
before the GPU upload.

`CHOLESKY_PROFILE_NVTX=1` enables panel-level ranges for diagonal update,
device-to-host transfer, history update, panel wait, CPU POTRF,
host-to-device transfer, and GPU solve. They are disabled in production.

## Compilation

The extension targets `sm_100a` and uses:

- host: `-O3`, native ISA tuning, FMA, fast/unsafe math, C++20;
- device: `-O3`, fast math, extra device vectorization, restricted pointers,
  line information, expensive ptxas optimization, spill diagnostics;
- cuBLAS: `cublasGemmEx` with explicit TF32 or FP32 compute modes.

The local CUDA 13.1 compilation completed. `cuobjdump --dump-resource-usage`
reported `LOCAL:0` for all emitted production kernels (copy, factor,
fused 128/64/32, inverse copy-back, and wedge cleanup). This is a compilation
check only; the local machine cannot execute `sm_100a`.

## CPU alternatives

Direct ATen/LAPACK is retained because local CPU measurements showed roughly
9.6 microseconds for one 64-wide panel. A tensorized, rank-one unrolled
implementation was rejected: its compiled `(2,64,64)` measurement was about
0.67 ms, far too slow for a pipelined panel. The compiled variant therefore
wraps the fixed-shape LAPACK operation rather than shipping the rejected
tensorized algorithm.

## Validation and promotion record

The exact-shape B200 sweep covered all 11 variants on dense, spectrum,
diagonal, low-rank, row-scaled, and tridiagonal inputs. The retained report is
`artifacts/validation/b1_n4096_20260728T142024Z/results.json`.

- Variants 0 and 7 passed all 66 case/variant checks.
- The TF32 paths passed the target dense case but became non-finite on the
  planted low-rank stress input; inverse-GEMM variant 2 also failed the
  planted-spectrum case.
- Hybrid variants 8–10 passed dense but their CPU panel POTRF rejected the
  low-rank stress recurrence after the TF32 history update.
- The first remote pass exposed upper-triangle values left by direct diagonal
  GEMMs. A final 128-wide wedge cleanup fixed that defect in both shapes.

The full-registry screen is in
`artifacts/tuning/b1n4096_20260728T142400Z/summary.json`. The fastest native
candidate was variant 1 at 2.683 ms; the library baseline was 1.530 ms.
Variant 1 is excluded on speed alone — it is 1.75x slower than the baseline,
so the stress-case failure never becomes the deciding factor. Every native
path lost to cuSOLVER at this shape: a single 4096-square matrix does not
fill the B200, so the per-panel launch recurrence pays overhead the library
does not.

The promotion run is in
`artifacts/tuning/b1n4096_20260728T142720Z/summary.json`. Across three
forward/reverse alternating rounds:

| Variant | Median target mean |
|---:|---:|
| 0 | 1.529 ms |
| 7 | 4.081 ms |

The required threshold was 1.521 ms, so the runner recorded
`retained_default_below_required_gain` and left variant 0 selected.

Artifacts are retained under:

- `artifacts/validation/` for dense, spectrum, diagonal, low-rank,
  row-scaled, and tridiagonal checks;
- `artifacts/tuning/` for the full screen and alternating Popcorn results;
- `artifacts/nsys/` for native and hybrid timelines;
- `artifacts/ncu/` for leading native kernel reports;
- `artifacts/helpers/ncu/` for report-analysis helpers.

The runner commands are:

```bash
python cholesky/b1n4096/cholesky_b1n4096_runner.py autotune \
  --variants all --rounds 3
python cholesky/b1n4096/cholesky_b1n4096_runner.py ncu --variants 1,4
.venv/bin/modal run cholesky/b1n4096/cholesky_b1n4096_modal.py --variant 1
.venv/bin/modal run cholesky/b1n4096/cholesky_b1n4096_modal.py --validate
python cholesky/b1n4096/cholesky_b1n4096_runner.py submit --mode test
python cholesky/b1n4096/cholesky_b1n4096_runner.py submit --mode leaderboard
```

Nsight Systems capture is pending because the authorized remote execution
request reached its tool usage limit. Nsight Compute is independently blocked
because neither `POPCORN_BREV_PROFILER_URL` nor `BREV_PROFILER_URL` is
configured. No metric substitution was made.

If NCU attributes more than 30% of target time to GEMMs and does not identify
an SM100 Tensor Core path in the kernel/SASS evidence, the next candidate is a
CUTLASS 3.x SM100 TF32 TMA/TMEM GEMM. It is not added speculatively.
