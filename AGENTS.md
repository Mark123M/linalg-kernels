## Goal
Build an efficient eigendecomposition algorithm for the B200 GPU in eigh.py for specific shapes

## Non-Negotiable: No Environment Workarounds
- **Do not add hacky code because the assistant cannot run, compile, or validate something in its environment.** If a required GPU, dependency, compiler, network resource, permission, or external service is unavailable, stop and ask the user to run the relevant command or provide the missing output.
- The user can run local RTX 4050 and Modal B200 commands for the assistant. Give the user the exact command and state what output is needed instead of changing the project to accommodate the assistant sandbox.
- Do not add synthetic/random fallback modes, fake success paths, dependency stubs, warning suppression, alternate algorithms, silent CPU/torch fallbacks, environment-specific branches, or build-system shims solely to make an assistant-side test run.
- Do not change toolchain versions, compiler selection, benchmark inputs, validation tolerances, or execution paths merely to silence a warning or bypass an unavailable environment. Explain the issue and ask the user before making a real project-level toolchain change.
- Do not upload or execute the repository on an external service unless the user explicitly asks. A request to implement or inspect code is not authorization to launch Modal or another remote runner.
- Read-only inspection and environment-independent checks such as parsing, `py_compile`, source searches, and `git diff --check` are fine. Clearly distinguish these from actual CUDA compilation and GPU validation.
- Product-required integration code is not an environment workaround. For example, cuSolverDx genuinely requires relocatable device code, device LTO, and device-linking its packaged fatbin. Implement such requirements through supported, maintainable build mechanisms and fail clearly when dependencies are absent.
- **Required behavior when blocked:** make no workaround edit; tell the user exactly what cannot be run and why; provide the exact command for the user to run; state which output is needed; then wait for the user's response.

## Solution Steps:
1. Blocked Householder tridiagonalization kernel (**implemented**)
2. Tridiagonal eigensolver (**current stage:** batched cuSolverDx HTEV with all eigenvectors; a custom divide-and-conquer solver may replace the large-N intermediate path)
3. Backtransform for eigenvectors

## Current Implementation State
- `eigh.py` contains the blocked Householder tridiagonalization and its driver. It mutates a work copy of `A` and writes LAPACK/cuSOLVER-style FP32 tensors `D(batch, N)` and `E(batch, N-1)`.
- The HTEV stage uses cuSolverDx 0.5 `job::all_vectors` with `Arrangement<row_major>`. The contiguous PyTorch output `V(batch, N, N)` therefore contains logical eigenvectors in its columns, matching the later backtransform convention.
- HTEV is destructive: on success `D` becomes ascending eigenvalues and `E` is zeroed. Preserve pre-HTEV `D/E` copies only when validation needs the original tridiagonal matrix.
- HTEV execution is specialized lazily by `(N, architecture, mode)`. Use block execution when the aligned `D + E + V` footprint fits shared memory: `N <= 240` on SM100/B200 and `N <= 158` on SM89. Use one-matrix-per-thread execution above those limits.
- HTEV is asynchronous and uses the existing raw queue-handle convention. Inspect `info` only after synchronization and report every nonzero entry; never silently fall back to another solver.
- The large-N HTEV thread path is a functional intermediate path that must be benchmarked before it is accepted as the final performance solution.
- Backtransform is not implemented yet. Leave `custom_kernel`'s final output path unchanged until it exists. Until then, validate HTEV eigenvectors against the tridiagonal matrix `T`, while comparing HTEV eigenvalues with the original symmetric matrix's spectrum.

## Build and Submission Constraints
- MathDx 26.06/cuSolverDx 0.5 must provide `cusolverdx.hpp`, its bundled CUTLASS headers, and `libcusolverdx.fatbin`. Discover a complete installation from `MATHDX_ROOT` or the installed MathDx package and fail with a clear error if any component is missing.
- The cuSolverDx native extension requires relocatable device code and device LTO, followed by a device-link step that includes the packaged fatbin. Warm the specialization and configure block-kernel dynamic shared memory before CUDA graph capture.
- The evaluation server rejects `eigh.py` if it contains the substring formed by the letters `s-t-r-e-a-m`, even inside a longer word or comment. Preserve the raw `queue_handle` convention and check the submission file before handoff.
- `eigh_bench_local_rtx4050.py` is the local validation/benchmark harness. `eigh_bench_b200_modal.py` is the B200 launcher. Harness-only helpers belong in those files, not in submitted product code.
- Keep unrelated user changes intact. In particular, do not modify `probe_submission.py`, vendored/reference trees, or untracked files unless the user places them in scope.

## References
- lapack/ directory contains the golden correctness references
- quack/ directory has the most efficient CuTeDSL kernel examples
- cutlass/ directory also has efficient kernel examples
- magma/ directory has linalg algorithm examples
- popcorn-cli contains the full evaluation infrastructure, **DO NOT CHEAT**.

## Kernel Writing Tips
- Don't add synchronization unless it's needed for correctness
- Ensure static code are optimized for compiler (ex. using cutlass.Constexpr, unrolling etc.)
- Use type annotations as much as possible

## Problem Description
Implement batched real symmetric eigendecomposition.

Input is `A`, a `batch x n x n` CUDA tensor in `torch.float32`. Every input
matrix is symmetric up to FP32 roundoff.

Return `(Q, L)` in the same eigenvector convention as `torch.linalg.eigh(A)`:
`Q` is a `batch x n x n` FP32 tensor whose columns are orthonormal
eigenvectors, and `L` is a `batch x n` FP32 tensor of eigenvalues sorted in
ascending order. The checker validates the invariant `A @ Q = Q @ diag(L)`,
reconstructs `A = Q @ diag(L) @ Q.T`, and checks orthogonality of `Q`.

Eigenvectors are not unique. Individual signs may flip, and repeated or
tightly clustered eigenvalues may rotate within their eigenspaces. Correctness
is therefore based on matrix identities, not elementwise comparison against a
reference eigensolver.

This shape set mirrors the optimizer-statistics motivation in `qr_v2`: square
matrices produced from gradient views and second-moment style statistics.
Batched `512 x 512` is the central target, while `1024`, `2048`, and `4096`
cover larger square factors.

Test and benchmark specs include a `cond` field. In this task `cond` is a
deterministic dynamic-range knob, not an exact requested condition number.
Some cases create spectra spanning `10^-cond` to `1`; others apply row/column
scaling or generate covariance-like positive semidefinite matrices. Stress
cases include rank-deficient, near-rank-deficient, repeated-eigenvalue,
clustered-eigenvalue, diagonal, banded, row-scaled, and mixed inputs.
Correctness tests also include LAPACK DDRVST-inspired matrix types from
`TESTING/EIG/ddrvst.f`: zero, identity, diagonal spectra, dense
planted-spectrum matrices, random symmetric matrices, and high- and
low-magnitude banded variants.

The `mixed` case builds a heterogeneous batch: each matrix is independently
assigned a conditioning profile at a random position in the batch. This
mirrors real optimizer-statistics batches, where per-layer or per-block
factors do not share one numerical structure. The benchmark set includes
mixed batches and homogeneous ill-conditioned batches, so robustness is
ranked, not only gated.

Correctness is residual-gated against the original FP32 input. Low-bit FP16,
FP8, or NVFP4 work is allowed as an internal implementation strategy:
returned factors must still be FP32 and must represent a numerically
meaningful eigendecomposition. Residuals are measured in FP64 to reduce
checker noise, and the gates are intentionally dimension-scaled and
invariant-based rather than elementwise reference comparisons. This leaves
room for approximate low-bit solutions while still rejecting non-orthogonal
factors, unsorted eigenvalues, and outputs that do not reconstruct the input.
The hard gates are the eigen-equation residual `A @ Q - Q @ diag(L)`,
reconstruction residual `Q @ diag(L) @ Q.T - A`, and orthogonality residual
`Q.T @ Q - I`, each applied relative to the corresponding matrix L1 norm.

Among passing submissions, ranking is by runtime using the geometric mean of
benchmark cases.

## Benchmark Shapes
- {"batch": 20, "n": 32, "cond": 1, "seed": 43214}
- {"batch": 40, "n": 176, "cond": 1, "seed": 423011}
- {"batch": 40, "n": 352, "cond": 1, "seed": 123456}
- {"batch": 640, "n": 512, "cond": 2, "seed": 1029}
- {"batch": 60, "n": 1024, "cond": 2, "seed": 75342}
- {"batch": 8, "n": 2048, "cond": 1, "seed": 224466}
- {"batch": 640, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"}
- {"batch": 60, "n": 1024, "cond": 2, "seed": 770002, "case": "mixed"}
- {"batch": 640, "n": 512, "cond": 0, "seed": 770003, "case": "rankdef"}
- {"batch": 640, "n": 512, "cond": 0, "seed": 770004, "case": "clustered"}
- {"batch": 60, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"}
- {"batch": 640, "n": 512, "cond": 0, "seed": 780001, "case": "lapack_dense_even_spectrum"}
- {"batch": 60, "n": 1024, "cond": 0, "seed": 780007, "case": "lapack_dense_geometric_spectrum"}
