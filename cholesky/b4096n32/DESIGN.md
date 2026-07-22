# Batched Cholesky: B=4096, N=32

## Status

- Raw CUDA Cholesky--Crout extension implemented.
- Thirty-four B200 launch/math/prefetch/shuffle/algorithm variants implemented.
- Modal correctness, timing, resource, Nsight Systems, and targeted Nsight
  Compute tooling implemented.
- Popcorn leaderboard sweep implemented but not run by the assistant.
- Production variant 3 was selected by the original full Popcorn leaderboard
  sweep and remains the default while variants 10--33 are evaluated.

## Correctness reference

LAPACK `SRC/spotf2.f` is the scalar correctness reference. For the lower factor,
its recurrence is

```text
L[j,j] = sqrt(A[j,j] - sum(k<j, L[j,k]^2))
L[i,j] = (A[i,j] - sum(k<j, L[i,k]*L[j,k])) / L[j,j], i>j
```

MAGMA's `magmablas/zpotf2_kernels.cu` and
`magmablas/zpotf2_devicesfunc.cuh` demonstrate two useful GPU ideas: fully
unroll the small fixed-width factorization and pack multiple independent
matrices into one CTA. The implementation here does not reproduce MAGMA's
shared-memory panel layout or its block barriers.

## B200 mapping

One warp factors one matrix. Lane `i` owns row `i` of `L` and retains its 32
entries in registers. At iteration `j`, lane `i` computes `L[i,j]`; lane `j`
broadcasts its previous row entries using warp shuffles. There are no block
barriers and no shared-memory allocations.

The input is symmetric, so the kernel reads `A[j,i]` instead of the strided
lower entry `A[i,j]`. A warp therefore reads one contiguous input row per
Crout step. Each lane finally writes its owned output row through eight aligned
`float4` stores, inserting exact zeros above the diagonal.

The B200 compute-capability 10.0 resource envelope used for the launch design
is 64 resident warps, 2,048 resident threads, 32 resident CTAs, 64K 32-bit
registers, and 228 KiB shared memory per SM. The fixed workload has about 28
matrices per SM on a 148-SM B200, so every candidate uses launch bounds that
target at least 32 resident warps. Runtime CUDA occupancy queries, rather than
these paper limits alone, determine each compiled variant's actual capacity.

Reference: NVIDIA Blackwell Tuning Guide,
https://docs.nvidia.com/cuda/blackwell-tuning-guide/

## Variant space

Variant IDs are stable because temporary Popcorn submissions replace only the
production ID constant.

| ID | Warps/CTA | Configuration |
|---:|----------:|---------------|
| 0 | 1 | round-to-nearest sqrt/divide |
| 1 | 1 | approximate reciprocal sqrt + one Newton step |
| 2 | 2 | round-to-nearest sqrt/divide |
| 3 | 2 | approximate reciprocal sqrt + one Newton step |
| 4 | 4 | round-to-nearest sqrt/divide |
| 5 | 4 | approximate reciprocal sqrt + one Newton step |
| 6 | 8 | round-to-nearest sqrt/divide |
| 7 | 8 | approximate reciprocal sqrt + one Newton step |
| 8 | 16 | round-to-nearest sqrt/divide |
| 9 | 16 | approximate reciprocal sqrt + one Newton step |
| 10 | 2 | Newton-refined reciprocal sqrt, relaxed launch bound control |
| 11 | 2 | Newton-refined reciprocal sqrt, prefetch final 2 columns |
| 12 | 2 | Newton-refined reciprocal sqrt, prefetch final 4 columns |
| 13 | 2 | Newton-refined reciprocal sqrt, prefetch final 6 columns |
| 14 | 2 | Newton-refined reciprocal sqrt, prefetch final 8 columns |
| 15 | 2 | unrefined reciprocal sqrt |
| 16 | 2 | unrefined reciprocal sqrt, prefetch final 2 columns |
| 17 | 2 | unrefined reciprocal sqrt, prefetch final 4 columns |
| 18 | 2 | unrefined reciprocal sqrt, prefetch final 6 columns |
| 19 | 2 | unrefined reciprocal sqrt, prefetch final 8 columns |
| 20 | 2 | unrefined reciprocal sqrt, inline-PTX columns 26--31 one iteration ahead |
| 21 | 2 | unrefined reciprocal sqrt, inline-PTX columns 26--31 two iterations ahead |
| 22 | 2 | unrefined reciprocal sqrt, two dot-product accumulators |
| 23 | 2 | unrefined reciprocal sqrt, four dot-product accumulators |
| 24 | 2 | unrefined reciprocal sqrt, one accumulator, two-shuffle lookahead |
| 25 | 2 | unrefined reciprocal sqrt, two accumulators, two-shuffle lookahead |
| 26 | 2 | unrefined reciprocal sqrt, four accumulators, two-shuffle lookahead |
| 27 | 2 | unrefined reciprocal sqrt, one-matrix right-looking outer product |
| 28 | 2 | variant 27 with next-diagonal root lookahead |
| 29 | 2 | left-looking, two matrices per warp in width-16 subdivisions |
| 30 | 2 | right-looking, two matrices per warp in width-16 subdivisions |
| 31 | 2 | variant 30 with next-diagonal root lookahead |
| 32 | 2 | left-looking, four matrices per warp in width-8 subdivisions |
| 33 | 2 | right-looking, four matrices per warp in width-8 subdivisions |

Variant 3 remains the production choice until the expanded sweep completes.
The compiler uses `-O3`, fast global math, explicit vectorization, FMA
contraction, and `sm_100a`; precise variants call explicit round-to-nearest
intrinsics so they remain the numerical control group.

The NCU capture of variant 3 localized 80% of sampled not-issued
long-scoreboard stalls to the first dependent operations for columns 27--30.
Variants 11--14 keep selected final-column inputs live from kernel entry. They
use explicit scalar registers plus a zero-instruction dependency point intended
to preserve scalar form through CUDA front-end lowering. Final scheduling is a
ptxas decision, so acceptance requires SASS to show that the selected LDGs
actually moved ahead of factorization rather than back beside their consumers.

All new variants are specialized to the winning two-warp CTA mapping. Their
launch bound requires 14 CTAs/SM instead of 16. At 148 SMs, 14 resident CTAs
per SM provide 2,072 slots for the 2,048-block grid, so allocations through
roughly 72 registers/thread can still admit the entire workload in one wave.
This deliberately trades unused theoretical occupancy for input-load
lookahead. Runtime metadata must show zero local-memory allocation before a
prefetch variant is accepted.

Variant 10 is the no-prefetch control for the relaxed launch bound. Variants
15--19 remove the Newton correction after `rsqrtf`. This deletes the correction
dependency chain at every pivot, but its numerical accuracy is not assumed:
Modal validation and the complete Popcorn checker remain mandatory.

Variants 20 and 21 retain the raw-root control while replacing the ineffective
entry-prefetch marker with explicit `ld.volatile.global.f32` inline PTX. Only
columns 26--31 are loaded, either one or two full Cholesky iterations before
their consumers. The compiler memory clobber prevents deletion and ordinary
load folding, while the bounded lead limits register lifetimes. Acceptance
still requires SASS to show the six LDGs at the intended lead distance; inline
PTX makes the loads observable but ptxas remains responsible for scheduling.

Variants 22--26 target the fixed 527-shuffle communication path without
changing matrix ownership. Two- and four-accumulator variants split the serial
dot-product FMA dependency chain, then combine the partial accumulators with a
fixed reduction. Lookahead variants issue two independent pivot broadcasts
before consuming either value. This keeps the shuffle count unchanged but gives
ptxas independent FMA work with which to overlap shuffle-result latency. The
changed FP32 summation order requires the full correctness gate, and register
metadata plus local-memory allocation determine whether the additional live
values are viable.

Variants 27--33 change the factorization schedule or the physical warp mapping.
The right-looking kernel keeps all 32 input residuals in the same register
array that eventually holds the factor. After normalizing column `k`, it
applies that column's outer product to all trailing residual columns. Each
matrix still executes 496 FFMAs and 527 shuffles, and each individual residual
receives its terms in increasing `k` order, but consecutive FFMAs target
different registers instead of one serial dot-product accumulator. Variant 28
updates column `k+1` first and issues its reciprocal square root before the
remaining column-`k` updates; the eventual SASS, rather than source order, must
confirm whether the root remains ahead of its consumer.

The subdivision kernels exploit the `width` operand of warp shuffle. Width 16
splits one physical warp into two independent matrices with two rows per
thread; width 8 splits it into four matrices with four rows per thread. One
hardware shuffle therefore carries two or four unrelated pivots, while each
thread issues two or four row FFMAs. Relative to the raw one-matrix control,
the expected full-workload shuffle totals are 2,158,592 at width 32, 1,079,296
at width 16, and 539,648 at width 8. The tradeoff is fewer physical scheduler
contexts and roughly two or four times as many persistent row registers per
thread. Width-16 launches require seven CTAs/SM for one finite-grid wave;
width-8 launches require four. Zero local-memory allocation is mandatory for
accepting either mapping.

The algorithm sweep deliberately excludes a row-wise Banachiewicz kernel. With
row ownership it would remove the 31 inverse broadcasts, but would replace 31
warp-wide column-normalization instructions with 496 mostly single-lane
normalizations. Warp-reduction dot products would require roughly five
shuffles per scalar output. Both mappings are structurally worse for the
measured MIO/eligibility bottleneck than the right-looking loop interchange.

The native extension explicitly uses C++20, matching the PyTorch 2.12 extension
ABI on both Modal and the Popcorn evaluator. An earlier C++17 override compiled
on Modal but failed while parsing an ATen header in Popcorn; the extension cache
key was advanced when this was corrected.

The evaluated path uses CUDA's ordinary default-queue launch syntax. This is
semantically identical to the former explicit raw handle value of zero and
does not alter the selected device kernel or its launch dimensions. The raw
handle plumbing and dynamically assembled CUDA type identifier were removed in
anticipation of stricter source validation. Arbitrary nondefault-queue launches
are intentionally outside this fixed-shape submission path.

## Autotuning policy

The exact `(4096,32,32)` shape uses the custom extension. Every other shape
uses `torch.linalg.cholesky_ex(..., check_errors=False).L`. Consequently all
non-target leaderboard timings are held constant across temporary submissions.

The Popcorn action launches the variants selected by `--variants` concurrently
so Popcorn can queue them across its workers; omitting the option selects all
thirty-four. It records each process's raw command output, parses the public B200
geomean, and emits a ranked JSON summary. It does not rewrite the tracked
specialization. After the results
are reviewed, update the production ID above and record the score here.

Nsight Systems profiling retains the forward-compatible `.nsys-rep` plus small
text/JSON summaries. `nsys stats` receives a temporary SQLite export under
`/tmp`; that database is deleted before the Modal volume is committed and is
never downloaded as an artifact.

Nsight Compute profiling is deliberately opt-in per variant. The worker replays
one input/output pair in a plain loop and lets the profiler choose the launch:
`--kernel-name` matches the function name base shared by every variant and
`--launch-skip` counts only matching launches, so the warmup is discarded by
NCU rather than partitioned by hand. The timing worker's eight-tensor ring is
pointless under kernel replay, which flushes the caches between passes anyway.
The selected sections cover compute and memory throughput, the overview and
hierarchical single-precision rooflines, occupancy, scheduler behavior, warp
stalls, instruction mix, and source/SASS counters without paying for the
indiscriminate `full` section set. The hierarchical roofline distinguishes the
L1, L2, and device-memory ceilings; the standard Speed-of-Light percentages
remain available for comparison with earlier captures. Each run
downloads Popcorn-compatible `ncu-details.txt`, `ncu-details.csv`, and
`profile.ncu-rep` artifacts, plus preflight and profiler diagnostics. NCU uses
kernel replay with cache flushing so every replay pass begins from a consistent
cache state. Its durations are collected at the locked base SM clock, roughly
1.7x the boost-clock kernel time the tuning and Nsight Systems runs report, so
every environment record carries both the current and the maximum SM clock and
NCU durations are never compared directly against those runs. The profiler
subprocess runs with `TMPDIR` redirected to container
storage because its tree launcher creates FIFOs there and the Modal volume at
`/cache` does not support them. Artifact naming follows
`popcorn-cli/docs/profiling.md`.
`artifacts/helpers/ncu/roofline_summary.py` extracts the overview and
hierarchical FP32 roofline values from a run directory or individual report and
can emit either a compact table or JSON.

### Toolchain image pin — 2026-07-22

The Modal image now starts from NVIDIA's newest published development container,
`nvidia/cuda:13.3.0-devel-ubuntu24.04`, then installs the CUDA 13.3 Update 1
compiler, command-line, and development-library meta-packages (`13.3.1-1`) from
NVIDIA's signed Ubuntu repository. NVIDIA has not published a `13.3.1` container
tag. Because NVIDIA's 13.3.0 runtime and development images hold cuBLAS at
13.5.1.27, the build explicitly removes both cuBLAS holds, installs runtime and
development packages 13.6.0.2, and restores the holds. The profiling layers are
pinned independently: Nsight Compute
`2026.2.1` comes from `cuda-nsight-compute-13-3=13.3.1-1`, while Nsight Systems
CLI `2026.3.1.157` comes from NVIDIA's standalone package and is checked against
its recorded SHA-256 before installation. Each Systems capture now records
`nsys-version.txt`, matching the existing `ncu-version.txt` provenance.

PyTorch remains `2.12.0` with the `cu130` wheel to avoid changing the benchmark
runtime at the same time as the system compiler and profilers. A CUDA 13.3
compiler versus CUDA 13.0 PyTorch build warning is therefore expected and is
not suppressed. Existing NCU 2025.4.x reports remain valid historical artifacts,
but their metric names, section availability, and helper column layouts must not
be assumed to match 2026.2.1. The previously observed Modal driver 580.95.05
meets CUDA 13.x's `>=580` minor-compatibility floor, but it is older than the
610.43.02 driver associated with CUDA 13.3; new-driver-only features are not
assumed available until a fresh preflight confirms them.

| Date | Selected ID | Public B200 geomean | Notes |
|------|------------:|---------------------:|-------|
| 2026-07-20 | 3 (`w2_refined`) | 0.0018265698079868835 s | Best of ten full leaderboard submissions; IDs 7 and 9 were close runners-up |

## Verification commands

These commands are intentionally documented but have not been run by the
assistant.

```bash
python3 -m py_compile \
  cholesky/b4096n32/cholesky_b4096n32.py \
  cholesky/b4096n32/cholesky_b4096n32_modal.py

git diff --check

# The command must print no matches.
rg -n 'stream' cholesky/b4096n32/cholesky_b4096n32.py

# Modal correctness/resource sweep and kernel-only timing.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action tune

# One Nsight Systems report set per variant; all artifacts are downloaded.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action profile

# One targeted Nsight Compute capture for the production variant.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action ncu --variants 3

# Compare the precise-root control and production variant in separate captures.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action ncu --variants 2,3

# Profile the relaxed-bound control and both new candidate families.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action ncu --variants 10,11,12,13,14,15,16,17,18,19

# Confirm placement and stalls for the raw-root control and bounded PTX loads.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action ncu --variants 15,20,21

# Compare accumulator counts and explicit two-shuffle lookahead.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action ncu --variants 15,22,23,24,25,26

# Check the provisional production route against the official tests.
popcorn submit --no-tui --leaderboard cholesky --gpu B200 --mode test \
  cholesky/b4096n32/cholesky_b4096n32.py

# Concurrently submit all thirty-four variants, parse their public geomeans, and rank them.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action popcorn

# Or submit only an explicit subset.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action popcorn --variants 3,15,20,21,22,23,24,25,26

# Isolate the raw left-looking control and the algorithm/subdivision sweep.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action popcorn --variants 15,27,28,29,30,31,32,33

# Capture the high-signal occupancy, shuffle, and scoreboard metrics for the sweep.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action ncu --variants 15,27,28,29,30,31,32,33
```

The requested feedback after `--action tune` is its JSON output plus all ptxas
spill warnings. After `--action profile`, inspect or provide the downloaded
statistics files. After `--action ncu`, provide `ncu-details.txt` for each
selected variant; the `.ncu-rep` is only needed for GUI inspection. After
`--action popcorn`, provide `summary.json` so the winning ID can replace the
provisional default.
