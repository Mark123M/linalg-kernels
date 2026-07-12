# CLAUDE.md

## Goal

Build an efficient **batched real symmetric eigendecomposition** (`eigh`) for the **B200 GPU** in `eigh.py`, submitted through the popcorn evaluation infrastructure. Among passing submissions, ranking is by runtime (geometric mean over benchmark cases). Batched `512 x 512` is the central target — the first kernel is optimized for N ≈ 512; `1024`, `2048`, `4096` cover larger square factors.

Solution plan:
1. Blocked Householder tridiagonalization kernel (in progress, `eigh.py`)
2. Divide and conquer eigensolver
3. Backtransform for eigenvectors

## Problem spec (condensed from AGENTS.md)

- Input: `A`, `batch x n x n` CUDA tensor, `torch.float32`, symmetric up to FP32 roundoff.
- Output `(Q, L)` in the `torch.linalg.eigh` convention: `Q` orthonormal eigenvector columns (FP32), `L` ascending eigenvalues (FP32).
- Correctness is **residual-gated in FP64**, not elementwise: eigen-equation `A@Q - Q@diag(L)`, reconstruction `Q@diag(L)@Q.T - A`, orthogonality `Q.T@Q - I`, each relative to the matrix L1 norm. Signs/rotations within eigenspaces are free. Internal low-precision (FP16/FP8/NVFP4) is allowed; returned factors must be FP32 and meaningful.
- Stress cases: rank-deficient, near-rank-deficient, repeated/clustered eigenvalues, diagonal, banded, row-scaled, mixed-conditioning batches, LAPACK DDRVST-inspired types. Robustness is ranked, not only gated.
- Benchmark shapes (batch × n): 20×32, 40×176, 40×352, **640×512**, 60×1024, 8×2048, plus mixed/rankdef/clustered/nearrank/lapack variants at 512 and 1024.

## Repo layout

- `eigh.py` — the kernel under development (CuTe DSL). Also contains the jit/disk-cache plumbing (`EIGH_ENABLE_DISK_CACHE`, `EIGH_CUTE_CACHE_DIR` env vars) and `custom_kernel` entry point.
- `eigh_bench_local_rtx4050.py` — local runner/benchmark (CUDA-graph replay + L2-rotated tensor sets via quack's bench utils). Includes a torch reflector+GEMV baseline and a `torch.allclose` check of the kernel's GEMV output.
- `lapack/` — golden correctness references (e.g. `dsytd2`/`dsytrd` for tridiagonalization, `TESTING/EIG/ddrvst.f` for test matrix types).
- `quack/` — the most efficient CuTe DSL kernel examples; the primary pattern source (`quack/quack/softmax.py` SoftmaxBackward, `reduce.py`, `copy_utils.py`, `bench/bench_utils.py`).
- `cutlass/` — more kernel examples; `magma/` — linalg algorithm examples; `notes/` — layout notes; `references/` — misc reference kernels.
- `popcorn-cli/` — full evaluation infrastructure. **DO NOT CHEAT.**
- `submission.py` — submission stub.

## Environment

- Run Python via `.venv/bin/python` (cutlass DSL 4.5.x installed there).
- The `task` module is provided by the popcorn runner, not the repo. Local scripts must stub it before importing `eigh`:
  ```python
  task = types.ModuleType("task"); task.input_t = torch.Tensor; task.output_t = object
  sys.modules.setdefault("task", task)
  ```
- Local GPU is an **RTX 4050 Laptop (~190 GB/s DRAM, 6 GB)** under WSL2 — fine for correctness and memory-bound roofline checks, but the deployment target is B200; don't over-tune to local ratios.

## Current kernel state (`eigh.py`)

Panel tridiagonalization kernel, one CTA per batch matrix, `num_threads` (128/256) split into row groups of `threads_per_row` (N-dependent, 8→256). Per panel column `i` (loop `cutlass.range(panel_size, unroll_full=True)`):

1. **Column load**: 1D async tiled copy of the panel subcolumn into smem `sCol`, predicated `col_idx >= i and col_idx < panel_n` (cp.async zero-fills masked lanes, so `v` is zero below `i` for free).
2. **Reflector (every row group redundantly)**: `sColBcast` = broadcast view of `sCol` with layout `(rows_per_tile, tiler_n) stride (0,1)`, partitioned by the 2D matvec tiled copy — so each row group's register copy of the column lines up elementwise with its A-tile rows. `norm_sq` via `row_sum` (warp reduction, `block_sum` through `reduction_buffer` when `warps_per_row > 1`). Semantics: `x` includes alpha at index `i`; `beta = -sign(alpha)*||x||`; `tau = (beta-alpha)/beta`; `v = x/(alpha-beta)` (so `v[i] = alpha/denom`, *not* the conventional 1).
3. **Pipelined GEMV** `b = A' @ v`: double-buffered smem tile `sA (rows_per_tile, tiler_n, 2)`; prefetch tile `t+1` (one `cp_async_commit_group` per iteration, `cp_async_wait_group(1)`), reduce tile `t` with `row_sum(a * v)`; lane 0 of each row group stores into smem `sB`. Column bound via static `predicate_k(limit=panel_n)`; row bound via scalar `if` (quack convention). No barrier around `sA` — each thread reads back exactly what it copied.
4. Barriers: inside the GEMV loop only when `warps_per_row > 1` (protects `reduction_buffer` reuse); unconditional at end of each `i` iteration (`sCol`/`sB` reuse).

Under `debug_printf=True`, `b` is stored to `mTri[:, panel_row_base+j, col]` for host-side verification (throwaway hook).

Next step: the rank-2 update consuming `sB` (LAPACK `dsytd2`: `w = tau*b - (tau^2/2)(b.v) v`, then `A -= v w^T + w v^T`).

Status: GEMV verified vs float64 torch for N ∈ {8, 33, 128, 1000, 4096} and panel_size up to 4 (rel err ≤ 2.5e-7); benchmark parity with cuBLAS `bmm` (1.03x at N=1024, 0.97x at N=4096 on local GPU, DRAM-bound both).

## CuTe DSL performance rules (measured in this project)

- The DSL is a **tracing JIT**: Python runs at compile time; layouts/partitions/`local_tile`/`domain_offset` with `const_expr` args cost **nothing at runtime**. Dynamic *values* (predicates, limits, pointer offsets) compile to a few integer ops — cheap, like hand-written CUDA indexing.
- **Never respecialize per loop iteration** (`cutlass.range_constexpr` + per-i `const_expr` tile sizes): it duplicates the kernel body per iteration — measured 22x compile time and up to 1.85x runtime regression (I-cache) at panel_size=32, while ceil-rounding made the "smaller" tiles identical anyway. Use `cutlass.range(..., unroll_full=True)` with one traced body + dynamic predicates instead.
- **Keep predicate limits static when possible**: with a static limit, most of `predicate_k`'s per-thread booleans constant-fold away. Making the bound depend on runtime `i` turned them into live registers and measured 3–11% *slower* — for <2% traffic saved. Columns `< i` are already nullified by `v`'s zeros.
- `cp.async` needs ≥4-byte transactions: `num_copy_elems=1` copies only work for fp32.
- Smem budget: `sCol + sB + 2*rows_per_tile*tiler_n` elements (~96 KB at N=4096 fp32) — revisit before pushing N higher.
- Don't add synchronization unless needed for correctness; document why each barrier exists.
- Use `cutlass.Constexpr` / `const_expr` type annotations aggressively so the compiler sees static structure.
- The DSL warns at ≥64 iterations of a static loop; treat that as a hard smell.

## Verify & bench

```bash
# correctness + reflector debug output
.venv/bin/python eigh_bench_local_rtx4050.py --batch 2 --n 512

# benchmark (CUDA graphs + L2 rotation) with torch GEMV baseline + allclose gate
.venv/bin/python eigh_bench_local_rtx4050.py --batch 640 --n 512 --bench --skip-expected --bench-repeats 3
```

When verifying kernel changes, always cover: an odd N (predicates), N=4096 (`warps_per_row=2` → `block_sum` path + loop barrier), and `panel_size > 1` (smem-reuse barriers across `i`). Compare against float64 torch references mirroring the kernel's exact reflector semantics (see `torch_reflector_gemv` in the runner).
