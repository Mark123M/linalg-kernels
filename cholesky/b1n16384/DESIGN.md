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
