# `b2n2048` B200 Cholesky design

## Status

The tracked default remains variant 0, Torch/cuSOLVER. Variants 1--3 add the
experimental full-grid 64-square wavefront requested for this shape:

| ID | Update | Grid order |
|---:|---|---|
| 0 | Torch/cuSOLVER | library |
| 1 | scalar FP32 FMA | task-major, batch-interleaved |
| 2 | TF32 WMMA with FP32 accumulation | task-major, batch-interleaved |
| 3 | TF32 WMMA with FP32 accumulation | batch-major control |

CUDA 13.1 compilation for `sm_100a` succeeds. The FP32 specialization uses
128 registers per thread; both TF32 specializations use 158. All three report
zero stack, spill stores, and spill loads. These are compilation results, not
B200 correctness or timing results, so no native variant is promoted.

## Algorithm

`2048 / 64 = 32`, giving 528 lower-triangle tasks per matrix and 1,056 CTAs
for the batch. Task `(i,j)` has the column-major ID

```text
j * (2*T - j + 1) / 2 + i - j,  T = 32.
```

Every history dependency `(i,k)` and `(j,k)`, and diagonal dependency
`(j,j)`, has a lower ID. Each CTA:

1. loads its original 64-square target;
2. consumes all published history pairs in left-looking order;
3. factors a diagonal tile or solves against the published diagonal;
4. stores the finished tile, zeros its upper mirror, and release-publishes
   its flag.

Task-major order maps adjacent CTAs to independent matrices at the same DAG
position. Variant 3 is the arithmetic-identical batch-major control.

The three 64-by-68 shared buffers occupy 52,224 bytes. Scalar history tiles
use the measured XOR row/group swizzle from the corrected b1n4096 experiment.
The TF32 path uses 16-by-16-by-8 WMMA operations with explicit TF32
conversion and retains FP32 POTRF64, TRSM64, and accumulation.

The release/acquire flag protocol is also corroborated by
`references/qr-kernels/gaunernst.py`, a prior full-grid QR winner. Ordinary
CUDA blocks still have no documented scheduling-order guarantee. These
variants are B200-specific experiments and require repeated no-hang testing;
successful execution is not a portability claim.

## Build and validation

The extension targets `sm_100a` with host and device `-O3`, fast math,
device vectorization, restricted pointers, line information, expensive
ptxas optimization, and spill diagnostics. Only exact contiguous CUDA FP32
`(2,2048,2048)` inputs can use variants 1--3; all other inputs retain the
Torch path.

Required B200 sequence:

```bash
.venv/bin/modal run cholesky/b2n2048/cholesky_b2n2048_modal.py \
  --variant 1 --validate --repeats 100
.venv/bin/modal run cholesky/b2n2048/cholesky_b2n2048_modal.py \
  --variant 2 --validate --repeats 100
.venv/bin/modal run cholesky/b2n2048/cholesky_b2n2048_modal.py \
  --variant 3 --validate --repeats 100
python cholesky/b2n2048/cholesky_b2n2048_runner.py autotune \
  --variants 0,1,2,3 --rounds 3 --no-promote
python cholesky/b2n2048/cholesky_b2n2048_runner.py submit --mode test
```

Run 100 repeated exact-shape calls for each candidate before any promotion.
Promote only a candidate that passes property validation in every round and
has median target mean no greater than `0.995` times the contemporaneous
variant-0 median. Missing or renamed profiler metrics must be recorded before
interpreting any substitute, and useful NCU helpers belong under
`artifacts/helpers/ncu/`.
