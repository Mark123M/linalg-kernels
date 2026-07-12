## Goal
Build an efficient eigendecomposition algorithm for the B200 GPU in eigh.py for specific shapes

## Solution Steps:
1. Blocked householder tridiagonalization kernel
2. Divide and conquer eigensolver
3. Backtransform for eigenvectors

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