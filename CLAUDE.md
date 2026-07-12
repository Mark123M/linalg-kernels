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
- `eigh_bench_local_rtx4050.py` — local runner/benchmark (CUDA-graph replay + L2-rotated tensor sets via quack's bench utils). `--panel-size` selects ps for both sides; the baseline (`torch_dlatrd_panel`) runs the full DLATRD panel op sequence through cuBLAS bmm (refresh, reflector, matvec, inner/outer corrections, w, V/W appends) so the comparison is fair at any ps, and the allclose gate checks every panel column's w against the kernel's tri output.
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

Panel tridiagonalization kernel, one CTA per batch matrix, `num_threads` (128/256) split into row groups of `threads_per_row` (N-dependent, 8→256; constructor arg `threads_per_row=` overrides, used to exercise `warps_per_row>1` paths at small N locally). Implements the full DLATRD column flow steps 1-5 (only the trailing DSYR2K remains). Per panel column `i` (loop `cutlass.range(panel_size, unroll_full=True)`):

1. **Column load + refresh**: 1D async tiled copy of the panel subcolumn into smem `sCol`, predicated `col_idx >= i and col_idx < panel_n` (cp.async zero-fills masked lanes, so `v` is zero below `i` for free). While the cp.async is in flight, the column refresh (dlatrd.f:297) accumulates `corr[p] = Σ_{j<i} V[p,j]·W[i,j] + W[p,j]·V[i,j]` in registers: **thread-per-row, serial-j** (thread owns `p = tidx + m·num_threads`, the w-append striding, `tiler_n/num_threads` accumulators) — unlike the outer pass's warp-per-row, register accumulation bridges the cp.async wait with no smem staging, and at fixed `j` consecutive threads read consecutive `p` (coalesced; the per-`j` coefficients `W[i,j]`/`V[i,j]` are warp-broadcast reads). No reductions inside, `j`-trip CTA-uniform. After the wait+barrier, a masked subtract `sCol[p] -= acc[m]` (`i <= p < panel_n`: preserves v's zero prefix and the zero pad tail) + **one new barrier** publishes the refreshed column; `alpha` and everything downstream read it unchanged. Uniform at i=0 (zero-trip j-loop, subtract of 0). Rejected alternative (register-only refresh, no store-back): the norm needs the `sColBcast` fragment layout and w-formation re-reads `sCol` after the matvec clobbered `sB`; keeping corr alive costs +tiler_n smem (`sCorr` — breaks N=4096 local launch) for a ~4-iteration smem-pass saving. Revisit as a B200 micro-opt.
2. **Reflector (every row group redundantly)**: `sColBcast` = broadcast view of `sCol` with layout `(rows_per_tile, tiler_n) stride (0,1)`, partitioned by the 2D matvec tiled copy — so each row group's register copy of the column lines up elementwise with its A-tile rows. `norm_sq` via `row_sum` (warp reduction, `block_sum` through `reduction_buffer` when `warps_per_row > 1`). Semantics (LAPACK `dlarfg` convention): `x` includes alpha at index `i`; `beta = -sign(alpha)*||x||`; `tau = (beta-alpha)/beta`; `v` tail `= x/(alpha-beta)` with `v[i] = 1` forced via a dynamic-`if` element loop. Zero-tail columns take the `tau=2` sign-flip reflector (valid under residual gating) instead of LAPACK's `xnorm==0 => H=I` shortcut — deliberate, per input restrictions.
3. **Inner correction GEMVs** (dlatrd.f:316-318, 322-324): `s_w = Wᵀv`, `s_v = Vᵀv` into smem arrays `sSw/sSv`, placed between the tile-0 prefetch and the matvec loop to hide the first gmem fetch. Row group `r` owns panel column `j = jt*rows_per_tile + r` (a row of the gmem workspace), so fragments align with `tVrV` like `sColBcast`. `j >= i` is masked (never branched — a group-divergent branch around `row_sum` would break `block_sum`'s CTA barrier); dynamic trip count `ceil(i/rows_per_tile)` is CTA-uniform.
4. **Pipelined GEMV** `b = A' @ v`: double-buffered smem tile `sA (rows_per_tile, tiler_n, 2)`; prefetch tile `t+1` (one `cp_async_commit_group` per iteration, `cp_async_wait_group(1)`), reduce tile `t` with `row_sum(a * v)`; lane 0 of each row group stores into smem `sB`. Column bound via static `predicate_k(limit=panel_n)`; row bound via scalar `if` (quack convention). No barrier around `sA` — each thread reads back exactly what it copied.
5. **Outer correction GEMVs** (dlatrd.f:319-321, 325-327): `b -= V·s_w + W·s_v`, one warp per row — lane `l` owns panel column `l (+32m)` with its coefficient preloaded to a register; warp-shuffle reduction, lane 0 rewrites `sB[p]`. Skipped via a warp-uniform `if i > 0` (no CTA barrier inside).
6. **w formation** (dlatrd.f:328-331): `w = τ·b + (-τ/2)(wᵀv)·v` using `wᵀv = τ·(bᵀv)` (one `row_sum` over a broadcast `sB` view); a thread-strided loop appends `v`/`w` as rows of the gmem workspace and (under `debug_printf`) stores `w` to `mTri[:, panel_row_base+p, col]` for host verification.

**V/W panels live in gmem workspace**, not smem (removes the smem ceiling — every benchmark N/ps launches; a smem-resident variant is a possible later optimization for N≤1024). Per-batch shape `(rows, cols)` from `Eigh.workspace_shape()`: rows = panel_size padded to `rows_per_tile`, cols = `tiler_n`; row-major so inner-GEMV reads and w-appends are coalesced (outer reads are strided but L2-hot). **Callers must `torch.zeros`-allocate**: unwritten rows/pad columns are read under masks before first write and must be finite (`0 * NaN` poisons reductions); the kernel writes only valid data, so the invariant survives graph replays. Kernel/`__call__` signature: `(data, tri, V, W)`.

Barriers per column: inside reductions only when `warps_per_row > 1` (`reduction_buffer` WAR); unconditional after the refresh subtract (thread-strided `sCol` rewrite, read cross-thread by alpha + the reflector fragments), after the matvec (publishes `sB`+`sSw`/`sSv` to the outer pass), after the outer pass (lane 0s rewrote `sB`), and at column end (`sCol` WAR vs next cp.async; also orders this CTA's gmem V/W appends — all its threads share one SM's L1).

Remaining DLATRD step: after the panel, block rank-2 update `A -= V Wᵀ + W Vᵀ` (DSYR2K, called from `dsytrd.f`). `panel_size=1` collapses the flow to `dsytd2`.

Status: full panel pipeline (refresh → reflector → inner GEMVs → matvec → outer correction → w) verified vs a float64 mirror of the **true DLATRD recurrence** (stale-column caveat gone) at N ∈ {65, 128, 512, 1000, 2048} and panel_size up to 16, plus `threads_per_row=64` at N=512 for the `block_sum` paths (rel err ≤ 1.3e-6). All benchmark N/ps combos launch (workspace in gmem). Perf (full clocks, 640×512, vs the fair full-panel cuBLAS baseline): ps=1 kernel 3.884 ms vs torch 4.129 (1.06x); ps=8 31.08 vs 40.14 (1.29x); ps=16 63.79 vs 80.51 (1.26x). Kernel per-column is flat (3.88→3.89→3.99 ms: refresh free at ps=8, +2.6% at ps=16; ps=1 floor 3.89 vs 3.81 pre-refresh, the always-on subtract pass + barrier) while torch per-column jumps 4.13→5.02 at ps>1 (per-column bmm launches for refresh + 4 correction GEMVs re-read V/W each time). Absolute ms vary hugely with laptop power state (power-saving mode ≈ 500 MHz SM runs ~1.5x slower); compare within one session only. **Occupancy/L2 note (local)**: the old smem-resident panels capped occupancy at 1 CTA/SM, so ~20 resident 1 MB matrices fit the 24 MB L2 and re-read columns hit cache (1.78 ms/col at ps=8); with panels in gmem, smem drops to ~20 KB, occupancy rises, the aggregate working set blows L2, and every column pays DRAM roofline (~3.8 ms/col at any ps). Deliberately capping occupancy (dummy smem carveout / launch attr) is a possible knob; on B200 the aggregate exceeds L2 either way, so this is mostly a local-GPU artifact.

## CuTe DSL performance rules (measured in this project)

- The DSL is a **tracing JIT**: Python runs at compile time; layouts/partitions/`local_tile`/`domain_offset` with `const_expr` args cost **nothing at runtime**. Dynamic *values* (predicates, limits, pointer offsets) compile to a few integer ops — cheap, like hand-written CUDA indexing.
- **Never respecialize per loop iteration** (`cutlass.range_constexpr` + per-i `const_expr` tile sizes): it duplicates the kernel body per iteration — measured 22x compile time and up to 1.85x runtime regression (I-cache) at panel_size=32, while ceil-rounding made the "smaller" tiles identical anyway. Use `cutlass.range(..., unroll_full=True)` with one traced body + dynamic predicates instead.
- **Keep predicate limits static when possible**: with a static limit, most of `predicate_k`'s per-thread booleans constant-fold away. Making the bound depend on runtime `i` turned them into live registers and measured 3–11% *slower* — for <2% traffic saved. Columns `< i` are already nullified by `v`'s zeros.
- `cp.async` needs ≥4-byte transactions: `num_copy_elems=1` copies only work for fp32.
- Smem budget: `sCol + sB + 2*rows_per_tile*tiler_n` elements (V/W are gmem) — ~20 KB at N=512, ~98 KB at N=4096 (right at the local 99 KB / sm_89 limit). Exceeding the arch limit fails at **launch**, not compile ("launch shared memory exceeds current GPU arch"), with the allocated byte count in the message.
- Don't add synchronization unless needed for correctness; document why each barrier exists.
- Use `cutlass.Constexpr` / `const_expr` type annotations aggressively so the compiler sees static structure.
- The DSL warns at ≥64 iterations of a static loop; treat that as a hard smell.
- **Never use a ternary expression with a dynamic condition** (`x if dyn_cond else y`): the condition is resolved at trace time and one branch is silently baked in (trace-time-arbitrary — can even differ between compiles). Use a dynamic `if` statement, which the preprocessor lowers correctly. Cost of a masked scalar update this way is negligible.

## Verify & bench

```bash
# correctness + reflector debug output
.venv/bin/python eigh_bench_local_rtx4050.py --batch 2 --n 512

# benchmark (CUDA graphs + L2 rotation) with torch w-column baseline + allclose gate
.venv/bin/python eigh_bench_local_rtx4050.py --batch 640 --n 512 --bench --skip-expected --bench-repeats 3
```

When verifying kernel changes, always cover: an odd N (predicates), a `warps_per_row=2` config (`block_sum` path + reduction barriers — use the `threads_per_row=64` override at N=512, since N=4096 no longer fits local smem with V/W panels), and `panel_size > 1` (correction GEMVs + smem-reuse barriers across `i`). Compare against float64 torch references mirroring the kernel's exact recurrence — reflector, corrections, and w formation (see `torch_reflector_w` in the runner and the scratchpad `test_panel.py` mirror).
