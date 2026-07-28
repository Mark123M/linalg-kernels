# B200 `(16, 512, 512)` Cholesky

## Contract

`cholesky_b16n512.py` specializes contiguous CUDA FP32 input with exact
shape `(16, 512, 512)`. Every other input uses
`torch.linalg.cholesky_ex(data, check_errors=False).L`.

The native path copies the lower triangle, writes exact zeroes to the strict
upper triangle, and factors the lower triangle in place. It targets
`sm_100a` and has no runtime CUTLASS, MathDx, or cuSolver dependency.

## Current outer graph

The matrix is an `8 x 8` grid of `64 x 64` tiles. Panel `p` executes:

1. one `POTRF64(p,p)` CTA per matrix;
2. one `TRSM64(i,p)` CTA for every `i > p` and matrix;
3. one `UPDATE64(i,j,p)` CTA for every `i >= j > p` and matrix.

Kernel ordering provides all cross-node dependencies. The path contains one
copy, eight factors, seven solves, and seven updates: 23 launches.

This staged graph replaced the original eight-CTA cluster scheduler. The
cluster graph reserved 128 KiB per CTA, spent 74.18% of samples at barriers,
and measured 1.918 ms. The first staged implementation reduced the target
median to 884.489 microseconds.

## 2026-07-25 profile

The retained report is:

`artifacts/ncu/b16n512_20260725T054947Z/`
`v00_staged64_fp32_precise_scalar_u256/REPORT.md`.

The report contains copy plus the first three factor/solve/update stages.
The directly measured launch pattern is:

| Node | Duration | SM throughput | Eligible warps/scheduler | Dominant stall |
|---|---:|---:|---:|---|
| `POTRF64` | 107--108 us | about 0.67% | 0.055 | barrier, 65.4% |
| `TRSM64` | 55--56 us | 7.7--11.0% | 0.121 | barrier, 39.7% |
| `UPDATE64` | 50--82 us | 45.9--52.6% | 0.407--0.839 | long scoreboard |

The factor launches only 16 CTAs on 148 SMs. The important inefficiency is
inside each CTA: three warps wait while one warp performs the 32-row
triangular solve. PC sampling attributes 4,761 of 7,216 factor samples
(65.98%) to barriers.

The update is not bandwidth or coalescing limited:

- DRAM read throughput is 1.38% of peak;
- global loads use 3.936 sectors/request;
- stores use 31.51 bytes/sector;
- there are no local loads or stores.

It is shared-load/dependency limited. Shared loads reach 50.08% of peak and
long-scoreboard stalls cost 3.99 cycles per issued instruction.

## Redesigned `POTRF64`

The new diagonal factor is recursive:

```text
POTRF16(A00)
TRSM16(A10, L00)
A11 -= L10 L10^T
POTRF16(A11)
TRSM32(A10, L00)
A11 -= L10 L10^T
POTRF16(A1100)
TRSM16(A1110, L1100)
A1111 -= L1110 L1110^T
POTRF16(A1111)
```

The implementation uses:

- one warp-resident rank-update `POTF2-16` for each 16x16 diagonal;
- four-thread row groups for the 16x16 and 32x32 triangular solves;
- four warps for the intervening symmetric updates;
- a padded 64x65 shared tile;
- precise, Newton-refined, and raw reciprocal-root alternatives.

The rank-32 implementation remains as an isolation control, but its middle
solve and update also use the new all-warp routines. The goal is to shorten
the useful serial path that caused the barrier samples; merely deleting a
barrier would not remove the work that sibling warps were waiting for.

## Register-microtiled `UPDATE64`

The old update assigned one output at a time:

```text
C[r,c] = C[r,c] - sum_k A[r,k] * B[c,k]
```

That performs two shared loads for every FMA and loads `C` before the
dependent accumulation. The new 256-thread mapping gives each lane a 4x4
register microtile:

- eight warps cover `4 x 2` warp tiles of shape `16 x 32`;
- each K step loads four A and four B values;
- those eight loads feed sixteen FMAs;
- the product accumulates independently in sixteen registers;
- `C` is loaded only in the epilogue.

A 512-thread 2x4 microtile alternative trades less register reuse for more
resident warps. The original scalar update remains as an isolation control.

## Solve

The panel solve retains two measured strategies:

- scalar: one owner thread per row;
- sub4: four lanes cooperatively reduce each row dot product.

Both cache the 64 diagonal reciprocals once. The scalar path remains the
default because it won the previous controlled comparison.

## Variant registry

| ID | Factor | Update | Root | Solve | Update threads |
|---:|---|---|---|---|---:|
| 0 | recursive 16 | 4x4 microtile | precise | scalar | 256 |
| 1 | recursive 16 | 4x4 microtile | Newton | scalar | 256 |
| 2 | recursive 16 | 4x4 microtile | raw | scalar | 256 |
| 3 | rank 32 | 4x4 microtile | precise | scalar | 256 |
| 4 | recursive 16 | scalar control | precise | scalar | 256 |
| 5 | recursive 16 | 4x4 microtile | precise | sub4 | 256 |
| 6 | recursive 16 | 2x4 microtile | precise | scalar | 512 |

All former TCGen05 variants were deleted. They all passed public validation,
but their 1.097--1.144 ms medians were 212--259 microseconds slower than the
FP32 control. Repeated TMEM allocation, packing, and epilogue cost more than
the 64x64x64 product saved at this workload size.

Variant 0 is the unmeasured default. Autotuning, not expectation, promotes a
candidate.

## Resource and promotion checks

Native `prepare()` records registers, static shared memory, and local bytes
for every factor, solve, and update specialization. It rejects a kernel
above the accepted 8-byte compiler-local frame.

Promotion additionally requires:

- process success and every public row passing in every round;
- raw `benchmark.4.mean`, `best`, `worst`, and `err` nanoseconds;
- an unchanged tracked source hash;
- one exact default-marker replacement.

## Results history

| Date | Implementation | Target median | Result |
|---|---|---:|---|
| 2026-07-24 | cluster/DAG variant 12 | 1,917.825 us | removed |
| 2026-07-25 | staged scalar variant 0 | 884.489 us | profile baseline |
| 2026-07-25 | recursive/microtile variants 0--6 | not measured | current |

## User-run commands

These commands have not been run by the assistant:

```bash
python3 -m py_compile \
  cholesky/b16n512/cholesky_b16n512.py \
  cholesky/b16n512/cholesky_b16n512_runner.py
git diff --check
rg -n 'stream' cholesky/b16n512/cholesky_b16n512.py

python3 cholesky/b16n512/cholesky_b16n512_runner.py submit --mode test

python3 cholesky/b16n512/cholesky_b16n512_runner.py autotune \
  --variants all --rounds 3 --max-workers 4

POPCORN_BREV_PROFILER_URL=https://http--brev-profiler-proxy--dxfjds728w5v.code.run \
python3 cholesky/b16n512/cholesky_b16n512_runner.py ncu \
  --variants WINNER,3,4
```

## Nsight Systems endpoint

The shape-local Modal launcher profiles one warmed factorization on B200:

```bash
.venv/bin/python -m modal run cholesky/b16n512/cholesky_b16n512_modal.py
# Optional comparison:
.venv/bin/python -m modal run cholesky/b16n512/cholesky_b16n512_modal.py --variant 2
```

`--variant -1` selects the tracked default, currently variant 2. Its fused
factor/solve schedule has 16 launches per factorization. Input generation,
extension compilation, preparation, warmup, and correctness validation are
outside the capture; the NVTX range contains exactly one out-parameter
factorization.

Artifacts are downloaded under `artifacts/nsys/`. `profile.nsys-rep` is the
forward-compatible UI/VeloQ input, `kernel-trace.csv` is the ordered GPU
timeline, `kernel-exec-trace.csv` separates API, queue, and execution time,
and `kernel-summary.csv` aggregates duration by kernel name. The SQLite
export, human-readable statistics, command, profiler version, environment,
preflight, stdout, and stderr are retained with the report.

## 2026-07-28 trailing-size adaptation

VeloQ measured the variant-2 fused factor/solve launch near 49 us through
all non-final 64-wide panels; only the final launch fell to about 22 us.
The trace exposes kernels, runtime, synchronization, and NVTX records but
no GPU metrics, so this plateau is a direct timeline measurement.

Variants 7 and 8 retain the raw-root 64-wide fused kernel for their prefix
and append a 32-wide direct factor/solve plus a lower-only rank-32 update:

| ID | Width schedule for remaining `R` | POTRF 64/32 | TRSM 64/32 | Launches |
|---:|---|---:|---:|---:|
| 7 | 64 when `R > 256`, otherwise 32 | 4 / 8 | 4 / 7 | 24 |
| 8 | 64 when `R > 128`, otherwise 32 | 6 / 4 | 6 / 3 | 20 |

The 32-wide fused kernel uses 128 threads and the existing warp-direct
POTF2-32 arithmetic. Each non-factor CTA redundantly factors the same
diagonal block, then solves one 32-row tile. The accompanying rank-32
update partitions only the lower trailing matrix into 32x32 tiles, so no
strict-upper value is written or later consumed. Width cutovers are
64-aligned, and the established width-64 update completes before the
first width-32 factor.

Both candidates passed the Modal B200 preflight; scaled reconstruction
residuals were `0.003330` for ID 7 and `0.003154` for ID 8. VeloQ confirmed
that the width-64 factor/solve launches precede the width-32 launches.
Across the two traces, fused factor/solve medians fell from 48.8-49.1 us
to 27.2-27.5 us (about 44%), while lower-update medians fell from
14.8-15.3 us to 7.6-8.0 us (about 48-49%).

| ID | Kernel trace span | Delta from default | Autotune median target mean |
|---:|---:|---:|---:|
| 2 | 598.462 us | baseline | 498.724 us |
| 7 | 591.456 us | -1.2% | 563.407 us |
| 8 | 558.046 us | -6.8% | 541.606 us |

Every public row and the target row passed in all three alternating
autotune rounds. The authoritative benchmark rejected the favorable
single-capture result: the best adaptive ID regressed 8.6% versus ID 2.
The 0.5% promotion gate therefore retained variant 2. Static validation
and the case-insensitive rejected-token scan passed.

## Cutlass-name clone experiment

A cutlass-named clone has been added for the current tracked default variant.
The public base variant is `2` and the cloned public variant is `9`.
The clone compiles identical CUDA algorithm source after renaming every custom
`__global__` kernel entry point and matching launch/configuration reference to
use a `cutlass_` prefix.

The 2026-07-28 runner autotune promoted variant 9. Median mean time was
494.656 us for variant 9 versus 498.027 us for variant 2, clearing the 0.5%
promotion gate.
