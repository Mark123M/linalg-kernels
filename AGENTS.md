## Goal
Build an efficient batched dense Cholesky factorization algorithm for the B200 GPU in cholesky.py for specific shapes

## Non-Negotiable: No Environment Workarounds
- All code for a specific batch B dimension N case must go under `cholesky/bBnN/cholesky_bBnN.py`
- For each batch B dimension N case, you must maintain a `cholesky/bBnN/DESIGN.md` for documenting algo design and progress.
- **Do not add hacky code because the assistant cannot run, compile, or validate something in its environment.** If a required GPU, dependency, compiler, network resource, permission, or external service is unavailable, stop and ask the user to run the relevant command or provide the missing output.
- The user can run local RTX 4050 and Modal B200 commands for the assistant. Give the user the exact command and state what output is needed instead of changing the project to accommodate the assistant sandbox.
- Do not add synthetic/random fallback modes, fake success paths, dependency stubs, warning suppression, alternate algorithms, silent CPU/torch fallbacks, environment-specific branches, or build-system shims solely to make an assistant-side test run.
- Do not change toolchain versions, compiler selection, benchmark inputs, validation tolerances, or execution paths merely to silence a warning or bypass an unavailable environment. Explain the issue and ask the user before making a real project-level toolchain change.
- Do not upload or execute the repository on an external service unless the user explicitly asks. A request to implement or inspect code is not authorization to launch Modal or another remote runner.
- Read-only inspection and environment-independent checks such as parsing, `py_compile`, source searches, and `git diff --check` are fine. Clearly distinguish these from actual CUDA compilation and GPU validation.
- Product-required integration code is not an environment workaround. For example, cuSolverDx genuinely requires relocatable device code, device LTO, and device-linking its packaged fatbin. Implement such requirements through supported, maintainable build mechanisms and fail clearly when dependencies are absent.
- **Required behavior when blocked:** make no workaround edit; tell the user exactly what cannot be run and why; provide the exact command for the user to run; state which output is needed; then wait for the user's response.

## Current Implementation State

## Build and Submission Constraints
- MathDx 26.06/cuSolverDx 0.5 must provide `cusolverdx.hpp`, its bundled CUTLASS headers, and `libcusolverdx.fatbin`. Discover a complete installation from `MATHDX_ROOT` or the installed MathDx package and fail with a clear error if any component is missing.
- The cuSolverDx native extension requires relocatable device code and device LTO, followed by a device-link step that includes the packaged fatbin. Warm the specialization and configure block-kernel dynamic shared memory before CUDA graph capture.
- The evaluation server rejects `cholesky.py` if it contains the substring formed by the letters `s-t-r-e-a-m`, even inside a longer word or comment. Preserve the raw `queue_handle` convention and check the submission file before handoff.
- Keep unrelated user changes intact. In particular, do not modify `probe_submission.py`, vendored/reference trees, or untracked files unless the user places them in scope.

## References
- lapack/ directory contains the golden correctness references
- quack/ directory has the most efficient CuTeDSL kernel examples
- cutlass/ directory also has efficient kernel examples
- magma/ directory has linalg algorithm examples
- popcorn-cli contains the full evaluation infrastructure, **DO NOT CHEAT**.

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
