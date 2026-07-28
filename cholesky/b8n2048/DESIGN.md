# B200 batched Cholesky: `(8, 2048, 2048)`

## Status

Implemented 2026-07-27. The initial tracked default remains variant 0,
`torch.linalg.cholesky_ex`, until a native candidate:

1. passes every public checker row, including the native target benchmark;
2. passes all requested alternating timing rounds; and
3. improves the contemporaneous variant-0 median by at least 0.5%.

The 2026-07-27 screening round measured the target-row baseline at 4.908 ms.
All ten native variants passed every public benchmark row. Variant 5,
left-looking fixed-128 with the custom solve and TF32 updates, led at
3.316 ms, a 32.4% improvement over the contemporaneous baseline. This is a
one-round screening result, not yet the three-round promotion result.

CUDA 13.1 host-side compilation succeeds for `sm_100a`. Ptxas reports 32
registers/thread and no stack frame, spill stores, or spill loads for every
`POTRF128` specialization and for `TRSM128`. The official fallback test
submission passed all 17 rows, and native target-row checker validation
passed for variants 1--10. Three-round confirmation and NCU comparison
remain pending.

## Contract

The native path specializes one exact input:

- contiguous CUDA `torch.float32`;
- shape `(8, 2048, 2048)`;
- eight symmetric positive-definite matrices.

Every other input, and all inputs while variant 0 is selected, use:

```python
torch.linalg.cholesky_ex(data, check_errors=False).L
```

The extension allocates a distinct output. A first kernel copies only
`tril(A)` and writes zeros elsewhere, so the input is never aliased or
modified. A final kernel restores bitwise-zero strict-upper entries after
library operations have used those locations as column-major views.

The runner parses each rendered source and rejects it locally if it contains
the evaluator-prohibited token documented in the repository instructions.

## Block equations

For a lower Cholesky step,

```text
A = [ A11  *  ]       L = [ L11  0   ]
    [ A21 A22 ]           [ L21 L22 ]

L11 L11^T = A11
L21 = A21 L11^-T
L22 L22^T = A22 - L21 L21^T
```

Every public native variant ultimately performs sixteen 128-wide diagonal
leaf factors, fifteen triangular solves, and fifteen rank-k updates. The
outer panel width only changes how those operations are grouped and how much
of the matrix an update touches.

Recursive blocking is therefore an implementation detail inside a panel,
not a third algorithm family.

## `POTRF128` and custom `TRSM128`

The measured leaf and solve structure is ported from `b60n1024`.

`POTRF128` recursively partitions its shared-memory diagonal tile as
`128 -> 64 -> 32`. One warp performs each 32-wide unblocked factor, and
cooperating row groups perform the inter-block triangular solves and FP32
FMA updates. The tile has a padded leading dimension of 129 to reduce shared
bank conflicts. Diagonal square root and reciprocal use explicit
round-to-nearest intrinsics even though the translation unit uses aggressive
fast-math compilation.

One 256-thread `TRSM128` CTA owns a `64 x 128` row tile. It stages the right
hand side and the required diagonal rows in padded shared memory, keeps the
active 32-column blocks in scalarized registers, and advances in dependency
order across all 128 columns. Rows below a leaf are distributed over
independent batch/row-tile CTAs.

Compiled resources:

| kernel | threads | registers/thread | static shared | dynamic shared | local/spills |
|---|---:|---:|---:|---:|---:|
| each `POTRF128<begin>` | 256 | 32 | 1,024 B | 66,560 B | 0 B / none |
| custom `TRSM128` | 256 | 32 | 1,024 B | 50,304 B | 0 B / none |
| pointer-table setup | 256 | 16 | 1,024 B | 0 B | 0 B / none |

The 1,024-byte static figure is the cubin resource report. Dynamic shared
memory is opted in during `prepare`.

## Right-looking family

A right-looking outer stage at panel `[j:j+NB]` does:

```text
factor the NB-wide panel in 128-wide leaf steps
solve A[j+NB:n, j:j+NB] against the completed panel
update the complete (n-j-NB)-square trailing matrix
```

For `NB` 256 or 512, internal leaf steps factor the panel using the same
`POTRF128`, `TRSM128`, and GEMM primitives. The outer solve uses
`cublasStrsmBatched` for the wider completed panel. An `NB=128` hybrid stage
uses the custom solver. The outer update is one
`cublasGemmStridedBatchedEx` over all eight matrices.

This family maximizes GEMM shape regularity and exposes large operations
early, but every outer update writes a full trailing square even though only
the lower half is part of the answer.

## MAGMA-style left-looking family

At panel `[j:j+NB]`, the left-looking schedule first applies all history:

```text
A[j:n, j:j+NB] -=
    L[j:n, 0:j] @ L[j:j+NB, 0:j]^T
```

That is one strided-batched rectangular GEMM. The panel is then factored in
128-wide leaves. After each leaf, the implementation solves the entire
column below it and updates only the remaining columns of the current panel:

```text
A[k+128:n, k+128:j+NB] -=
    L[k+128:n, k:k+128] @
    L[k+128:j+NB, k:k+128]^T
```

This mirrors MAGMA's outer panel update in
`magma/src/zpotrf_batched.cpp:136-154` and its recursively split panel in
`magma/src/zpotrf_panel_batched.cpp:56-136`. MAGMA's right-looking
factor/solve/update recurrence is also retained as a structural reference in
`magma/src/zpotf2_batched.cpp:50-71`.

The implementation uses explicit 128-wide leaf steps rather than runtime
recursion. The distinctive comparison is outer left-looking rectangular
updates versus outer right-looking full-square updates.

## Schedules and variants

The compiled outer schedules are:

| schedule | panel widths |
|---|---|
| fixed-128 | sixteen 128-wide panels |
| fixed-512 | four 512-wide panels |
| adaptive | `512, 512, 512, 256, 128, 128` |

Variant IDs are stable:

| ID | name | family | schedule | solve policy | update math |
|---:|---|---|---|---|---|
| 0 | `torch_baseline` | PyTorch/cuSOLVER | library | library | library |
| 1 | `rl_fixed128_custom_tf32` | right | fixed-128 | custom | TF32 |
| 2 | `rl_fixed512_hybrid_tf32` | right | fixed-512 | wide cuBLAS, 128 custom | TF32 |
| 3 | `rl_adaptive_hybrid_tf32` | right | adaptive | wide cuBLAS, 128 custom | TF32 |
| 4 | `rl_adaptive_cublas_tf32` | right | adaptive | all cuBLAS | TF32 |
| 5 | `ll_fixed128_custom_tf32` | left | fixed-128 | custom | TF32 |
| 6 | `ll_fixed512_custom_tf32` | left | fixed-512 | custom | TF32 |
| 7 | `ll_adaptive_custom_tf32` | left | adaptive | custom | TF32 |
| 8 | `ll_adaptive_cublas_tf32` | left | adaptive | all cuBLAS | TF32 |
| 9 | `ll_adaptive_custom_fp32` | left | adaptive | custom | FP32 |
| 10 | `rl_adaptive_hybrid_fp32` | right | adaptive | hybrid | FP32 |

Variants 9 and 10 isolate update precision. They are correctness and
performance controls for cases where TF32 update error consumes too much of
the reconstruction tolerance.

## cuBLAS mapping and pointer preparation

The output is row-major, while cuBLAS interprets each matrix as column-major.
The underlying storage is therefore operated on through the transposed view.
For a rectangular update with destination rows `R`, columns `C`, and
rank `K`, the call computes the column-major `C x R` transpose:

```text
C^T <- C^T - B^T A
```

using `transa=T`, `transb=N`, leading dimension 2048, matrix stride
`2048 * 2048`, and batch count 8. TF32 variants select
`CUBLAS_COMPUTE_32F_FAST_TF32`; FP32 controls select
`CUBLAS_COMPUTE_32F`.

The right-side row-major triangular solve is mapped to a column-major
left-side solve with `side=LEFT`, `uplo=UPPER`, and `transa=T`.

All addresses needed by `cublasStrsmBatched` are materialized once per
invocation by a small GPU kernel into two device pointer tables. There are no
host-to-device pointer-array copies between stages.

## Native API and build

The internal extension exposes:

- `prepare(variant)`;
- `run(data, variant)`;
- `run_out(data, output, variant)`;
- `metadata()`.

`prepare` configures the factor and custom solve dynamic shared-memory
limits, requests maximum shared-memory carveout, and queries compiled
attributes. `metadata` returns one row per public variant with schedule,
solver, precision, resource, operation-count, launch-count, and occupancy
fields.

Compilation targets only `sm_100a` with:

```text
-O3 -DNDEBUG -std=c++20
--use_fast_math --extra-device-vectorization --restrict -lineinfo
-Xptxas=-O3,-v,-warn-spills,--allow-expensive-optimizations=true
```

The first pass deliberately does not introduce MathDx/cuSolverDx
device-linking. A cuSolverDx leaf should be added only if a complete B200
profile shows diagonal factorization is material in endpoint time.

## Correctness invariants

- The input is read-only and the returned tensor has independent storage.
- The lower triangle is copied exactly before arithmetic begins.
- Every factor leaf consumes a lower triangle whose preceding updates have
  completed in launch order.
- A solve is launched only after its diagonal leaf or outer panel is
  complete.
- Left-looking updates touch only the active rectangular panel; values above
  the desired triangle may be temporary workspace but are not later treated
  as valid factor entries.
- The final strict-upper write produces exact `+0.0f`.
- Checker acceptance requires correct shape, dtype, device, finiteness,
  positive diagonal, exact upper zeros, and reconstruction residual on the
  native `(8,2048)` benchmark row. Passing test mode alone is insufficient.

MAGMA is the algorithmic correctness reference. The official property-based
checker remains the acceptance authority because it validates reconstruction
against the original FP32 input.

## Runner and promotion

`cholesky_b8n2048_runner.py` provides:

- `submit --mode test|benchmark|leaderboard`;
- `autotune`, with alternating forward/reverse rounds and `--no-promote`;
- `ncu`, for hosted B200 reports at benchmark index 9.

Every benchmark result is fetched through the authenticated result API.
Promotion requires every public row to pass in every round. Candidates are
ranked by median target-row mean, then median target-row best time, then ID.
The fastest passing native candidate is atomically promoted only when its
median mean is no more than `0.995 * baseline_median`; otherwise variant 0 is
retained.

NCU collection retains the rendered source, command, raw response, report,
details, and logs. Any helper used for analysis must be copied under
`artifacts/helpers/ncu/`.

Before interpreting a report, record:

1. each expected metric name;
2. the actual metric name, or that it is absent;
3. units and normalization for both;
4. whether the conclusion is directly measured or inferred.

No renamed or “equivalent” metric may be substituted silently.

## Validation sequence

Environment-independent checks:

```bash
python3 -m py_compile \
  cholesky/b8n2048/cholesky_b8n2048.py \
  cholesky/b8n2048/cholesky_b8n2048_runner.py
rg -n 'stream' cholesky/b8n2048/cholesky_b8n2048.py
git diff --check -- cholesky/b8n2048
```

B200 sequence:

```bash
# 1. Official fallback correctness grid.
python3 cholesky/b8n2048/cholesky_b8n2048_runner.py \
  submit --mode test

# 2. One non-promoting target-row screen of variants 0-10.
python3 cholesky/b8n2048/cholesky_b8n2048_runner.py \
  autotune --variants all --rounds 1 --no-promote

# 3. Three alternating rounds: baseline plus the fastest passing candidates.
python3 cholesky/b8n2048/cholesky_b8n2048_runner.py \
  autotune --variants 0,1,5,6,7 --rounds 3

# 4. Profile the promoted winner and the fastest passing candidate from the
# other family. Set the hosted profiler URL first.
POPCORN_BREV_PROFILER_URL=<url> \
python3 cholesky/b8n2048/cholesky_b8n2048_runner.py \
  ncu --variants WINNER,1
```

The profile decision must account for factor, solve, and GEMM time shares;
tensor-core dispatch; achieved/theoretical occupancy; executed local-memory
traffic; shared-bank conflicts; scoreboard stalls; and launch gaps.

## Benchmark history

| Date | Variant | target mean | Status |
|---|---:|---:|---|
| prior measurement | 0, torch/cuSOLVER | ~4.88 ms | earlier measured baseline |
| 2026-07-27 | 0, torch/cuSOLVER | 4.908 ms | one-round screen; all public rows passed |
| 2026-07-27 | 5, left fixed-128 custom TF32 | 3.316 ms | one-round screen; fastest |
| 2026-07-27 | 7, left adaptive custom TF32 | 3.343 ms | one-round screen; second |
| 2026-07-27 | 6, left fixed-512 custom TF32 | 3.346 ms | one-round screen; third |
| 2026-07-27 | 1, right fixed-128 custom TF32 | 3.545 ms | fastest right-looking control |
| 2026-07-27 | 8, left adaptive cuBLAS TF32 | 3.594 ms | passing solve-policy control |
| 2026-07-27 | 9, left adaptive custom FP32 | 3.872 ms | passing precision control |
| 2026-07-27 | 2 | 4.093 ms | passing right fixed-512 hybrid |
| 2026-07-27 | 3 | 4.153 ms | passing right adaptive hybrid |
| 2026-07-27 | 4 | 4.254 ms | passing right adaptive all-cuBLAS |
| 2026-07-27 | 10 | 4.738 ms | passing right FP32 control |

The screen suggests that at this shape the lower arithmetic volume of
left-looking rectangular panels matters more than exposing larger
right-looking square GEMMs. Within the leading left-looking custom family,
fixed-128, fixed-512, and adaptive are separated by less than 1%, so all
three remain in the confirmation set. Variant 1 is retained to measure the
best competing algorithm family under the same rounds.

## Nsight Systems endpoint

The shape-local Modal launcher profiles one warmed factorization on B200:

```bash
.venv/bin/python -m modal run cholesky/b8n2048/cholesky_b8n2048_modal.py
# Optional comparison:
.venv/bin/python -m modal run cholesky/b8n2048/cholesky_b8n2048_modal.py --variant 5
```

`--variant -1` selects the tracked default, now native variant 11.
Metadata records 72 algorithm launches for its adaptive left-looking custom
schedule. Variant 0 is the torch baseline and is intentionally rejected by
this native out-parameter endpoint. Input generation, compilation,
preparation, warmup, and correctness validation are outside the capture; the
NVTX range contains exactly one factorization.

Artifacts are downloaded under `artifacts/nsys/`. `profile.nsys-rep` is the
forward-compatible UI/VeloQ input, `kernel-trace.csv` is the ordered GPU
timeline, `kernel-exec-trace.csv` separates API, queue, and execution time,
and `kernel-summary.csv` aggregates duration by kernel name. The SQLite
export, human-readable statistics, command, profiler version, environment,
preflight, stdout, and stderr are retained with the report.

## 2026-07-28 trailing-size adaptation

The warmed variant-5 timeline holds each POTRF128 near 145-148 us. Its TRSM
falls initially but then plateaus around 43 us after the grid drops below
one B200 wave. These are direct VeloQ timeline measurements; the report
contains no GPU metrics.

Variants 11 and 12 preserve the left-looking TF32 schedule while varying
the compile-time micro width:

| ID | Width schedule for remaining `R` | POTRF 128/64/32 | TRSM 128/64/32 |
|---:|---|---:|---:|
| 11 | 128 when `R > 1024`, otherwise 64 | 8 / 16 / 0 | 8 / 15 / 0 |
| 12 | 128 when `R > 1024`, 64 when `R > 256`, otherwise 32 | 8 / 12 / 8 | 8 / 12 / 7 |

Every step first applies the complete left-looking history GEMM to the
current width, then launches a matching direct factor and triangular solve.
The width-128 and width-64 factors are compile-time compositions of the
existing POTF2-32, local TRSM, and lower-only local update; width 32 ends
after the direct factor. Solve row tiles are 64 at widths 128 and 64, and
32 at width 32. Library GEMM dimensions therefore change with every width
band without introducing a new library dependency.

The history update contains every previously completed column, so changing
width cannot omit or repeat a rank contribution. Factors read only the
updated lower block, solves write the complete panel below it, and the
existing final upper-zero pass remains unchanged.

Both candidates passed the Modal B200 preflight with scaled reconstruction
residual `0.373713`. VeloQ confirmed `128 -> 64` ordering for ID 11 and
`128 -> 64 -> 32` for ID 12. In ID 12, factor medians were
146.144/54.176/22.656 us and solve medians were
68.384/23.392/10.656 us. The reductions are 63%, 58%, 66%, and 54%
across the two width transitions.

| ID | Kernel trace span | Delta from ID 5 | Autotune median target mean |
|---:|---:|---:|---:|
| 5 | 3.483 ms | baseline | 3.330 ms |
| 11 | 3.348 ms | -3.9% | 3.265 ms |
| 12 | 3.375 ms | -3.1% | 3.274 ms |

All public rows and the target row passed in all three alternating rounds.
ID 11 improved the authoritative target median by 1.9%, passed the 0.5%
gate, and is now the tracked default. Official test submission 920854
passed all 17 cases. The final capture measured factor medians
146.272/54.336 us, solve medians 67.584/23.392 us, and a 3.345 ms kernel
span, 3.9% below the old default. Static validation and the
case-insensitive rejected-token scan passed.
