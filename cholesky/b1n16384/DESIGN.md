# b1n16384: single-matrix n=16384 Cholesky specialization for B200

Benchmark entry 13: `{"batch": 1, "n": 16384, "cond": 2, "seed": 48284}`.
1 GiB FP32 input; the second-largest single-matrix entry.

## Provenance

Direct port of the b1n32768 phase-2h winner `ll_nb1024_invgemm_tf32`
(42.86 ms = 5.15x there; see `cholesky/b1n32768/DESIGN.md` for the full
derivation, NCU evidence, and dead ends). Every kernel is micro=128-tile
local, so only the `kN` constant changes. The dead ends measured at
n=32768 (cublasStrsm, BF16x9 emulation, Ssyrk math mode, flat/rank-8
factors, light apply, fused per-micro kernel) are not re-ported; the
b1n32768 registry documents why.

## Algorithm

Left-looking outer panels (NB variant axis), micro = 128:

```
copy_lower (exact strict-upper zeros)
for panel in 0..kN step NB:
    if panel > 0:  history GEMM  C[panel:, panel:panel+NB] -= L L^T slab   (TF32)
    for micro in panel..panel+NB step 128:
        if micro > panel:  inner GEMM (k = micro - panel <= NB-128, TF32)
        factor128_kernel: one CTA, 512 threads, wide redundant-corner
            factor + T = inv(L11) build (skipped on the final micro)
        if not last micro:
            apply GEMM  scratch = X T^T   (dense TF32; T upper is exact zeros)
            copy_back_kernel (float4, scratch -> panel sub-column)
zero_wedges (restore exact zeros in NB-block wedges)
```

Wedge-correctness argument is inherited unchanged from b1n32768 (GEMM
garbage confined to strict-upper wedges of NB diagonal blocks; no
operand ever reads a wedge; one final zeroing pass).

Accuracy: checker gate at n=16384 allows ~16*eps*n ~= 3.1e-2 relative
residual; TF32 quantization on history/inner/apply GEMMs contributes
~1e-3. Same TF32 usage passed the authoritative checker at n=32768
(2x looser gate, same error structure); confirmed here by the test runs
inside the first autotune.

## Variant registry

IDs are stable and append-only.

| ID | name | NB | apply | role |
|---:|---|---:|---|---|
| 0 | `ll_nb1024_invgemm_tf32` | 1024 | cuBLAS GEMM + copy-back | ported winner, default |
| 1 | `ll_nb2048_invgemm_tf32` | 2048 | cuBLAS GEMM + copy-back | NB axis (8 panels) |
| 2 | `ll_nb512_invgemm_tf32` | 512 | cuBLAS GEMM + copy-back | NB axis (32 panels; history GEMMs cheaper per call at half n) |
| 3 | `ll_nb1024_inv_tf32` | 1024 | in-kernel 256-thread apply | control (b1n32768 v12 apply path) |

## Resources

Identical to b1n32768 (all kernels micro-local): factor 512 threads /
153600 B dynamic SMEM (1 CTA/SM), apply 256 threads / 132096 B,
copy/wedges/copy-back 256 threads static. Workspaces per call: `t_inv`
{128,128} fp32; `scratch` {kN-128, 128} fp32 (8.3 MB, invgemm variants
only). Output allocated `empty_like`; input never written.

## Expectations (pre-measurement estimates)

Boost-clock budget for v0, scaled from the measured b1n32768 splits
(factor ~87 us/micro is shape-independent): factor chain 128 x 87 us
~= 11.1 ms, apply GEMM + copy-back ~1.5 ms, inner GEMMs ~1 ms, history
GEMMs ~2-3 ms, copy ~0.3 ms, ~600 launch gaps ~2.4 ms -> ~18-19 ms.
cuSOLVER baseline unmeasured here (pure n^3 scaling from the 220.7 ms
n=32768 baseline says ~28 ms; overheads usually make it worse). The
factor chain is a larger fraction than at n=32768, so factor-side
levers (and launch-gap reduction) matter more at this shape.

## Risks

1. NB=1024 optimality does not transfer - that is what variants 1/2
   probe.
2. Launch-gap share doubles relative to n=32768 (same ~600 launches,
   half the work); if measured large, the fused-micro idea is NOT the
   answer (proven neutral at 32768) - CUDA-graph-free batching of the
   copy-back into neighboring kernels would be.
3. TF32 residual at the 2x tighter gate - covered by the first
   `--mode test` / autotune pass; FP32-apply fallback would be a new
   append-only variant if ever needed.

## User-run verification

```bash
# 1. Correctness + timing sweep (validates TF32 at n=16384 on the
#    authoritative checker and picks NB):
python3 cholesky/b1n16384/cholesky_b1n16384_runner.py autotune --variants 0,1,2,3 --rounds 3

# 2. Winner split:
python3 cholesky/b1n16384/cholesky_b1n16384_runner.py ncu --variants <winner>

# 3. Board refresh once promoted:
python3 cholesky/b1n16384/cholesky_b1n16384_runner.py submit --mode leaderboard

# Sandbox-safe checks:
python3 -m py_compile cholesky/b1n16384/cholesky_b1n16384.py
rg -n 'stream' cholesky/b1n16384/cholesky_b1n16384.py   # must print nothing
git diff --check
```

## Score history

| Date | Selected ID | Public B200 geomean | Notes |
|------|------------:|--------------------:|-------|
| - | - | - | no leaderboard submission yet |

## Benchmark history

| Date | Variant | benchmark.13 mean |
|------|--------:|------------------:|
| - | - | not yet measured |

## Nsight Systems endpoint

The shape-local Modal launcher profiles one warmed factorization on B200:

```bash
.venv/bin/python -m modal run cholesky/b1n16384/cholesky_b1n16384_modal.py
# Optional comparison:
.venv/bin/python -m modal run cholesky/b1n16384/cholesky_b1n16384_modal.py --variant 0
```

`--variant -1` selects the tracked default, currently variant 0. Its
NB=1024 inverse-GEMM schedule has 511 algorithm launches per factorization.
Input generation, extension compilation, preparation, warmup, and correctness
validation are outside the capture; the NVTX range contains exactly one
out-parameter factorization.

Artifacts are downloaded under `artifacts/nsys/`. `profile.nsys-rep` is the
forward-compatible UI/VeloQ input, `kernel-trace.csv` is the ordered GPU
timeline, `kernel-exec-trace.csv` separates API, queue, and execution time,
and `kernel-summary.csv` aggregates duration by kernel name. The SQLite
export, human-readable statistics, command, profiler version, environment,
preflight, stdout, and stderr are retained with the report.

## 2026-07-28 trailing-size adaptation

VeloQ measured essentially every inverse-building POTRF128 launch near
83 us in the warmed default timeline. The report exposes kernels, runtime,
synchronization, and NVTX data but no GPU metrics, so this fixed cost is a
direct timeline result rather than a counter-based diagnosis.

NB remains 1024 in all new variants:

| ID | Width schedule for remaining `R` | POTRF 128/64/32 | TRSM 128/64/32 |
|---:|---|---:|---:|
| 4 | 128 to `R=2048`, then 64 | 112 / 32 / 0 | 112 / 31 / 0 |
| 5 | 128 to `R=4096`, then 64 | 96 / 64 / 0 | 96 / 63 / 0 |
| 6 | 128/64/32 at `R=4096` and `R=1024` | 96 / 48 / 32 | 96 / 48 / 31 |

The appended 64- and 32-wide factors use precise POTF2-32 arithmetic.
Width 64 performs a compile-time 32-row solve/update before its second
direct factor. Both widths build an exact-zero-upper dense inverse of the
lower factor. Inner history GEMMs, inverse-apply GEMMs, and float4
copy-backs are specialized by width; the established width-128 path is
unchanged.

Every cutover is on an NB=1024 boundary. Therefore an outer history GEMM
always updates one whole 1024-column panel, and all micro steps within that
panel share one width. The inverse identity preserves the triangular solve,
and the final NB-wedge cleanup is unchanged.

All three variants passed the Modal B200 preflight. Their scaled
reconstruction residual was `0.128921`, below the preflight limit of 16.
VeloQ confirmed that the factor specializations occur in `128 -> 64` order
for IDs 4/5 and `128 -> 64 -> 32` order for ID 6. Median factor durations
were 82.752 us, 66.848 us, and 30.560 us in ID 6, reductions of 19.2% and
54.3% at the width transitions. Matching width-specialized copy-back
medians fell from 3.104 us at width 128 to 1.728 us at width 64 and
1.344 us at width 32. The history/apply calls also selected multiple
dimension-specific cuBLAS kernel families.

| ID | W128/W64/W32 factor p50 (us) | Kernel trace span (ms) | Delta from default |
|---:|---:|---:|---:|
| 0 | 82.8 / - / - | 15.405 | baseline |
| 4 | 82.8 / 66.9 / - | 16.937 | +9.9% |
| 5 | 83.0 / 67.2 / - | 17.081 | +10.9% |
| 6 | 82.8 / 66.8 / 30.6 | 17.124 | +11.2% |

The smaller solver kernels satisfy the stepwise duration goal, but doubling
the number of micro steps costs more than it saves at this shape. Per the
timeline screen, none advanced to the authoritative autotune. Variant 0
remains the default and no final-default capture was needed. Static
validation and the case-insensitive rejected-token scan passed.

## Cutlass-name clone experiment

A cutlass-named clone has been added for the current tracked default variant.
The public base variant is `0` and the cloned public variant is `7`.
The clone compiles identical CUDA algorithm source after renaming every custom
`__global__` kernel entry point and matching launch/configuration reference to
use a `cutlass_` prefix.

The 2026-07-28 runner autotune retained variant 0. Median mean time was
15.128629 ms for variant 0 versus 15.161088 ms for variant 7, so the
cutlass-named clone was rejected.
