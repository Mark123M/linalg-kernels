# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Build efficient batched dense Cholesky factorization for the NVIDIA B200 GPU, one
specialization per benchmark shape, submitted through the Popcorn evaluation
infrastructure. `AGENTS.md` is the authoritative project contract — read it before
making changes.

## Non-Negotiable: No Environment Workarounds

The assistant sandbox has no GPU, no CUDA compiler, and no network runners. **Never
add hacky code, fallbacks, stubs, tolerance changes, or environment branches to make
something runnable in the sandbox.** The user runs local RTX 4050 (WSL2) and Modal
B200 commands on request. When blocked: make no workaround edit, state exactly what
cannot be run and why, give the user the exact command, state which output is needed,
and wait. Read-only checks (`py_compile`, source searches, `git diff --check`) are
fine but must be clearly distinguished from real CUDA compilation and GPU validation.
Do not launch Modal or any remote runner unless the user explicitly asks. See
`AGENTS.md` for the full policy.

## Commands

```bash
# Correctness test against the official checker (user runs these)
popcorn submit --leaderboard cholesky --gpu B200 --mode test --no-tui cholesky/bBnN/cholesky_bBnN.py

# Leaderboard submission (true performance)
popcorn submit --leaderboard cholesky --gpu B200 --mode leaderboard --no-tui cholesky/bBnN/cholesky_bBnN.py

# Per-shape Modal B200 harness (pattern established by b4096n32; user runs these)
.venv/bin/modal run cholesky/bBnN/cholesky_bBnN_modal.py --action tune      # correctness/resource sweep + kernel timing
.venv/bin/modal run cholesky/bBnN/cholesky_bBnN_modal.py --action profile   # Nsight Systems per variant
.venv/bin/modal run cholesky/bBnN/cholesky_bBnN_modal.py --action ncu --variants <ids>   # targeted Nsight Compute
.venv/bin/modal run cholesky/bBnN/cholesky_bBnN_modal.py --action popcorn --variants <ids>  # concurrent leaderboard sweep, ranked summary.json

# Sandbox-safe checks
python3 -m py_compile cholesky/bBnN/cholesky_bBnN.py
git diff --check
rg -n 'stream' cholesky/bBnN/cholesky_bBnN.py   # must print NO matches before handoff
```

Use `.venv/bin/python` / `.venv/bin/modal` (repo virtualenv); the `popcorn` CLI is on
the user's PATH.

## Architecture

- `cholesky/bBnN/cholesky_bBnN.py` — single-file Popcorn submission for one shape:
  `custom_kernel(data: input_t) -> output_t`, importing from `task` (provided by the
  Popcorn runner, not the repo — local harnesses install a stub module before import).
  A specialization handles only its exact `(batch, n)` shape with a custom kernel
  (raw CUDA via `torch.utils.cpp_extension.load_inline`); every other shape falls
  back to `torch.linalg.cholesky_ex(..., check_errors=False).L` so non-target
  leaderboard timings stay constant across submissions.
- `cholesky/bBnN/DESIGN.md` — **required** per shape: algorithm design, variant
  table, measurement policy, and progress/score log. Keep it current.
- `cholesky/bBnN/cholesky_bBnN_modal.py` — Modal B200 harness with `--action
  tune/profile/ncu/popcorn`. It generates checker-matching inputs, validates the
  factor, times kernels, and downloads artifacts. The popcorn sweep submits
  temporary copies where only the `# POPCORN_VARIANT` constant line is replaced —
  variant IDs stay stable and the tracked file is never rewritten by tooling.
- `cholesky/bBnN/artifacts/` — git-ignored tuning JSON, popcorn sweep results
  (`summary.json`), `nsys/` and `ncu/` captures, and `helpers/` analysis scripts.
- `cholesky/cholesky.py`, `submission.py` — torch-fallback stubs (eventual combined
  submission lives in `cholesky.py`).
- `probe_submission.py` — eval-box environment probe; **do not modify** (same for
  vendored/reference trees and untracked files unless the user puts them in scope).
- `eigh/` — completed prior project (batched symmetric eigendecomposition) with its
  own `AGENTS.md`/`CLAUDE.md`; useful for Popcorn/Modal harness patterns and
  eval-box findings, but not active work.

## Build and Submission Constraints

- The evaluation server rejects any submission file containing the substring formed
  by the letters `s-t-r-e-a-m` — anywhere, even inside a longer word or comment.
  They also reject any workarounds like _QUEUE_TYPE = "cuda" + "Str" + "eam_t"
- Native extensions must use C++20 (PyTorch 2.12 extension ABI on both Modal and the
  Popcorn evaluator; C++17 compiled on Modal but broke on the evaluator).
- MathDx 26.06/cuSolverDx 0.5 must provide `cusolverdx.hpp`, bundled CUTLASS
  headers, and `libcusolverdx.fatbin`, discovered from `MATHDX_ROOT` or the
  installed package; fail clearly if incomplete. cuSolverDx extensions genuinely
  require relocatable device code, device LTO, and device-linking the packaged
  fatbin — that is product-required integration, not a workaround. Warm
  specializations and configure block-kernel dynamic shared memory before CUDA
  graph capture.

## References

- `lapack/` — linalg algorithm correctness references (e.g. `SRC/spotf2.f` is the
  scalar Cholesky reference)
- `magma/` — efficient linalg algorithm examples (e.g. `magmablas/zpotf2_kernels.cu`
  for batched small-matrix patterns)
- `CUDALibrarySamples/` — GPU-accelerated library examples; device extensions under
  `CUDALibrarySamples/MathDx` are of particular interest
- `cccl/cub/` — efficient parallel computing primitives
- `cuda-samples/` — general kernel examples
- `cutlass/` — efficient kernel examples
- `popcorn-cli/` — full evaluation infrastructure, **DO NOT CHEAT**
  (`popcorn-cli/docs/profiling.md` defines profiling artifact naming)

## Problem Description

Implement batched dense Cholesky factorization.

Input is `A`, a `batch x n x n` CUDA tensor in `torch.float32`. Every matrix
is symmetric positive definite up to FP32 roundoff. Return a lower-triangular
FP32 tensor `L` with positive diagonal such that `A = L @ L.T`.

The checker validates shape, dtype, device, finiteness, lower-triangular
structure, positive diagonal, and the reconstruction residual against the
original FP32 input. Correctness is property-based rather than elementwise
comparison with one library implementation.

Inputs cover dense covariance-like matrices, planted spectra, diagonal
matrices, damped low-rank matrices, scaled rows and columns, and
tridiagonal SPD matrices. The `cond` field is a deterministic dynamic-range
control; it is not an exact requested condition number for every case.

The benchmark grid emphasizes batched factorization: from thousands of small
matrices through a single 32768 x 32768 matrix. The low-batch entries from
n=32 through n=1024 each contain 2^22 FP32 input elements, so matrix scaling
is compared at a consistent 16 MiB input footprint. Paired high-batch entries
at n=512, 1024, 2048, and 4096 separately exercise throughput and batch
parallelism. Among passing submissions, ranking is by the geometric mean of
runtime across all benchmark entries.

## Benchmark Shapes

- {"batch": 4096, "n": 32, "cond": 2, "seed": 41032}
- {"batch": 1024, "n": 64, "cond": 2, "seed": 41064}
- {"batch": 256, "n": 128, "cond": 2, "seed": 41128}
- {"batch": 64, "n": 256, "cond": 2, "seed": 41256}
- {"batch": 16, "n": 512, "cond": 2, "seed": 41512}
- {"batch": 640, "n": 512, "cond": 2, "seed": 510512}
- {"batch": 4, "n": 1024, "cond": 2, "seed": 42024}
- {"batch": 60, "n": 1024, "cond": 2, "seed": 511024}
- {"batch": 2, "n": 2048, "cond": 2, "seed": 44048}
- {"batch": 8, "n": 2048, "cond": 2, "seed": 512048}
- {"batch": 1, "n": 4096, "cond": 2, "seed": 48096}
- {"batch": 2, "n": 4096, "cond": 2, "seed": 514096}
- {"batch": 1, "n": 8192, "cond": 2, "seed": 48192}
- {"batch": 1, "n": 16384, "cond": 2, "seed": 48284}
- {"batch": 1, "n": 32768, "cond": 2, "seed": 48368}

## Test Shapes

- {"batch": 16, "n": 32, "cond": 2, "seed": 53124}
- {"batch": 16, "n": 64, "cond": 2, "seed": 53125}
- {"batch": 16, "n": 128, "cond": 2, "seed": 3321}
- {"batch": 8, "n": 256, "cond": 2, "seed": 94010}
- {"batch": 4, "n": 512, "cond": 2, "seed": 32523}
- {"batch": 2, "n": 1024, "cond": 2, "seed": 4327}
- {"batch": 1, "n": 2048, "cond": 2, "seed": 224466}
- {"batch": 8, "n": 128, "cond": 5, "seed": 1200, "case": "spectrum"}
- {"batch": 8, "n": 128, "cond": 5, "seed": 1201, "case": "diagonal"}
- {"batch": 32, "n": 32, "cond": 5, "seed": 1202, "case": "spectrum"}
- {"batch": 32, "n": 32, "cond": 5, "seed": 1203, "case": "diagonal"}
- {"batch": 16, "n": 64, "cond": 5, "seed": 1204, "case": "spectrum"}
- {"batch": 16, "n": 64, "cond": 5, "seed": 1205, "case": "diagonal"}
- {"batch": 4, "n": 256, "cond": 4, "seed": 32524, "case": "lowrank"}
- {"batch": 4, "n": 512, "cond": 4, "seed": 32525, "case": "rowscale"}
- {"batch": 4, "n": 512, "cond": 1, "seed": 32526, "case": "tridiagonal"}
- {"batch": 2, "n": 1024, "cond": 4, "seed": 4330, "case": "lowrank"}

## Current State

- `b4096n32` is the only shape with a custom kernel (see its `DESIGN.md` for the
  variant table, selected production variant, and leaderboard score history). All
  other `cholesky/bBnN/` files are torch fallbacks awaiting specialization.
