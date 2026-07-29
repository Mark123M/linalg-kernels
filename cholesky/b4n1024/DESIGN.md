# `b4n1024` B200 Cholesky design

## Status

The tracked default is variant 1, the task-major batch-interleaved **FP32**
wavefront. Variant 2 was briefly the default and was reverted on 2026-07-29
after it was implicated in secret-seed validation failures; see
"Precision and the secret seed" below. Variants 1--3 add the experimental
full-grid 64-square wavefront requested for this shape:

| ID | Update | Grid order |
|---:|---|---|
| 0 | Torch/cuSOLVER | library |
| 1 | scalar FP32 FMA | task-major, batch-interleaved |
| 2 | TF32 WMMA with FP32 accumulation | task-major, batch-interleaved |
| 3 | TF32 WMMA with FP32 accumulation | batch-major control |

CUDA 13.1 compilation for `sm_100a` succeeds. The earlier offline compiler
inspection reported 128 registers for FP32 and 158 for TF32, with zero stack,
spill stores, and spill loads. Direct B200 runtime metadata instead reports
255 registers for FP32; the B200 measurement is authoritative and the earlier
number is retained only as a record of the mismatch. Direct B200 validation
results for FP32 and TF32 follow.

The first Modal B200 validation attempt for variant 1 did not reach CUDA
compilation or kernel execution: `nvcc` could not create its temporary output
under the configured `/cache/tmp` directory because the validation entry point
had not created it. The validation harness now creates that directory before
loading the extension, matching the existing profiling entry point. The
reported missing-NumPy warning was nonfatal and unrelated. This attempt
provides no correctness, liveness, resource, or timing evidence for variant 1.

The corrected Modal run then validated variant 1 on a B200 with CUDA runtime
13.0, CUDA compiler 13.3.73, PyTorch 2.12.0+cu130, and 148 SMs. All six
property cases passed, followed by 100 consecutive exact-shape launches with
no observed hang or CUDA failure:

| Case | Scaled residual | Reference scaled residual | Minimum diagonal |
|---|---:|---:|---:|
| dense | 0.00282003 | 0.00230063 | 2.01313 |
| spectrum | 0.00275447 | 0.00195653 | 2.00776 |
| diagonal | 0.000779399 | 0.000779399 | 1.0 |
| low-rank | 0.00326050 | 0.00289180 | 0.871066 |
| row-scaled | 0.00386076 | 0.00269626 | 0.960283 |
| tridiagonal | 0.000697545 | 0.000697545 | 1.54779 |

Every result was finite FP32 on the requested device, had an exactly zero
upper triangle, and was below the validation limit of 16. Runtime metadata
reported 544 tasks, 2,176 flag bytes, 52,224 bytes of dynamic shared memory,
255 registers, zero local memory, one active block per SM, 256 threads, and
two launches per factorization. This is direct correctness and no-hang
evidence, but not a timing comparison or a CUDA-wide scheduling guarantee.

Variants 2 and 3, the interleaved and batch-major TF32 orderings, both passed
the six property cases and completed 100 consecutive exact-shape launches
without an observed hang or CUDA failure. Their property outputs are
bit-for-bit identical, confirming that task ordering does not alter the
arithmetic. Direct B200 metadata for both reports 154 registers, zero local
memory, one active block per SM, and the same 52,224-byte dynamic shared-memory
allocation:

| Case | TF32 scaled residual | Reference scaled residual | TF32/reference |
|---|---:|---:|---:|
| dense | 0.665650 | 0.00230063 | 289.3 |
| spectrum | 0.329426 | 0.00195653 | 168.4 |
| diagonal | 0.000779399 | 0.000779399 | 1.0 |
| low-rank | 1.31049 | 0.00289180 | 453.2 |
| row-scaled | 0.720565 | 0.00269626 | 267.2 |
| tridiagonal | 0.0659180 | 0.000697545 | 94.5 |

These residuals pass the harness's standard scaled-residual floor of 16, but
are substantially worse than both cuSOLVER and variant 1. Variants 2 and 3
therefore advanced to the official performance comparison with a documented
accuracy tradeoff. Their identical arithmetic provides a clean timing control
for whether interleaving the four independent matrix DAGs hides
dependency-wait latency.

## Precision and the secret seed

Variant 2 was reverted as the default on 2026-07-29. The table above is not
evidence that TF32 is safe on the graded inputs, because
`cholesky_b4n1024_modal.py` generates `factors @ factors.T / LOW_RANK` with
`.diagonal().add_(4.0)`: a diagonally dominant Wishart matrix whose hardest
case spans only `logspace(-0.35, 0.35)`, so kappa is roughly 10--100. Every
benchmark row instead carries `cond=2`, a symmetric row/column dynamic-range
control worth about 1e4 on the matrix, so kappa(A) >= 1e4.

TF32's unit roundoff is `2^-11 = 4.9e-4`, which puts `u * kappa` near 5 --
at or past the Cholesky backward-stability boundary. Past that boundary a
trailing pivot can become non-positive on unlucky data, and the unguarded
`__fsqrt_rn` in `factor64` then returns NaN, failing the checker on
finiteness or positive-diagonal rather than on tolerance. FP32's `2^-24`
keeps `u * kappa` near 6e-4, four orders of margin.

`references/popcorn-eval/consts.py` documents that private/leaderboard
evaluation re-runs the same shapes "on a secret seed", and `eval.py` combines
that seed into every case's seed. Shapes and cases are unchanged, so only the
data differs -- exactly the signature of a conditionally stable
factorization: it passes on one seed and fails on another. Variant 1 costs
0.283% (527,187 ns versus 525,692 ns) and is already validated over six
property cases and 100 consecutive launches, so the tradeoff is not close.

Reinstating variant 2 would require measuring its residual on genuine
`cond=2` benchmark inputs, not on the harness's damped Wishart matrices.

## Direct Popcorn performance

The 2026-07-29 three-round Popcorn benchmark sweep passed every submission and
promoted variant 2:

| Variant | Median mean | Speedup vs variant 0 | Time reduction |
|---:|---:|---:|---:|
| 0, Torch/cuSOLVER | 1,325,182 ns | 1.000x | 0% |
| 1, FP32 interleaved | 527,187 ns | 2.514x | 60.22% |
| 2, TF32 interleaved | 525,692 ns | 2.521x | 60.33% |
| 3, TF32 batch-major | 815,919 ns | 1.624x | 38.43% |

Interleaving the four matrix DAGs reduced TF32 time by 35.57% relative to the
batch-major control, directly supporting the low-batch scheduling hypothesis.
TF32 was only 0.283% faster than the much more accurate FP32 interleaved
variant, although it won all three recorded rounds. This sweep used three
concurrent submit workers, so it did not enforce the intended serial
forward/reverse ordering. A serial head-to-head confirmation of variants 1
and 2 remains appropriate before treating their narrow ordering as settled.
The 60.33% gain over the contemporaneous default is far above the 0.5%
promotion threshold.

## Algorithm

`1024 / 64 = 16`, giving 136 lower-triangle tasks per matrix and 544 CTAs for
the batch. Task `(i,j)` has the column-major ID

```text
j * (2*T - j + 1) / 2 + i - j,  T = 16.
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
`(4,1024,1024)` inputs can use variants 1--3; all other inputs retain the
Torch path.

Required B200 sequence:

```bash
.venv/bin/modal run cholesky/b4n1024/cholesky_b4n1024_modal.py \
  --variant 1 --validate --repeats 100
.venv/bin/modal run cholesky/b4n1024/cholesky_b4n1024_modal.py \
  --variant 2 --validate --repeats 100
.venv/bin/modal run cholesky/b4n1024/cholesky_b4n1024_modal.py \
  --variant 3 --validate --repeats 100
python cholesky/b4n1024/cholesky_b4n1024_runner.py autotune \
  --variants 0,1,2,3 --rounds 3 --max-workers 1 \
  --wavefront-validated --no-promote
python cholesky/b4n1024/cholesky_b4n1024_runner.py submit --mode test
```

Run 100 repeated exact-shape calls for each candidate before any promotion.
The timing sweep is serial within each forward/reverse alternating round so
concurrent submissions cannot erase the intended order control.
Promote only a candidate that passes property validation in every round and
has median target mean no greater than `0.995` times the contemporaneous
variant-0 median. Missing or renamed profiler metrics must be recorded before
interpreting any substitute, and useful NCU helpers belong under
`artifacts/helpers/ncu/`.
