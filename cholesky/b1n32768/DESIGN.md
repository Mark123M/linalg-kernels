# Single-Matrix Cholesky: B=1, N=32768

## Status

Scaffolded 2026-07-26; phases 1-2 complete the same day. All six
implemented variants pass the authoritative checker at n=32768 (TF32,
BF16x9, and inverse-apply TRSM all confirmed legal). Tracked default:
**variant 2 at 74.48 ms** vs the 220.7 ms torch baseline (2.96x).
Measured dead ends: SYRK math-mode TF32 (v5) and BF16x9 speed (v4).
Variants 6-7 (fused GMEM-flag panel) are registered but reject at
`prepare`/`launch` with a clear error; phase-3 work gated on the NCU
evidence for the serial-micro-chain hypothesis.

## Contract

- `custom_kernel` handles exactly `(1, 32768, 32768)` contiguous FP32 CUDA
  input; every other shape falls back to
  `torch.linalg.cholesky_ex(..., check_errors=False).L`.
- The input tensor is never written: the evaluator reuses it across timed
  repeats and rechecks it against a pristine copy. `copy_lower_kernel`
  reads it once into a fresh `at::empty_like` output; all factorization is
  in the output.
- The strict upper triangle of the returned factor holds exact zeros:
  `copy_lower_kernel` writes them, library GEMMs are allowed to dirty only
  the strict-upper wedges of NB-diagonal blocks (proof below), and
  `zero_wedges_kernel` restores those in one final pass.
- The submission file contains no banned token; kernel launches use
  `cudaLaunchKernelEx` on a zero-initialized `cudaLaunchConfig_t`, and
  cuBLAS work lands on the caller's execution queue through
  `at::cuda::getCurrentCUDABlasHandle()` with an RAII state guard
  (math/atomics/pointer modes, emulation strategy saved and restored).
- Native extension: C++20, `sm_100a` only, `-lcublas`, lazy `load_inline`
  keyed by a SHA-256 source tag so non-target shapes never compile it.

## Why this shape is different

- Baseline row 14 is `torch`/cuSOLVER at **220.7 ms** (197 public runs,
  spread < 1 ms), which is ~53 TFLOP/s = ~71% of the B200's ~74 TFLOP/s
  SIMT FP32 peak. A CUDA-core rewrite has no headroom; tensor cores
  (TF32 ~1.1 PF dense, BF16x9 emulation ~250 TF effective) are the entire
  opportunity.
- The 4 GiB input exceeds the evaluator's 256 MiB batching target, so the
  timing window holds exactly one call and stops after ~3 repeats; the L2
  is flushed before each repeat, so every phase reads straight from HBM.
- The checker's scaled residual divides by `eps_fp32 * n * norm(A)`; at
  n=32768 the base gate of 16 admits a relative residual of roughly
  16 * 1.19e-7 * 32768 = 6.2e-2. TF32 trailing GEMMs already pass the
  production checker at n=512 (`b640n512` v20) and n=1024 (`b60n1024` v6)
  where the same formula is 32x tighter. This is still an inference about
  the server checker, not a measurement; the phase-1 `--mode test` probe
  on variant 1 resolves it before deeper optimization.

## Algorithm: left-looking outer panels, 128-wide micro steps

MAGMA's GPU potrf applies history left-looking at the panel level
(`magma/src/shpotrf_gpu.cpp:206-272`, `zpotrf_gpu.cpp:294-343`); this
specialization uses the same shape because at batch=1 it dominates
right-looking staging on every axis that matters here:

| metric | right-looking (NB=1024) | left-looking (NB=1024) |
|---|---|---|
| big-GEMM flops | 2n^3/3 full-square (or SYRK, unverified TF32) | n^3/3 via plain `cublasGemmEx` |
| trailing-matrix HBM traffic | ~91 GB re-read/rewrite | ~23 GB history reads |
| TF32 dependency | `cublasSsyrk` math mode | `cublasGemmEx` compute type (proven) |
| big library calls | per-panel | 31 total |

Loop structure (`launch_staged<Id>`), `kN=32768`, `kMicro=128`,
variant `nb` in {1024, 2048}:

```
copy_lower                                    # exact strict-upper zeros
for panel in range(0, kN, nb):
    if left-looking and panel > 0:
        gemm_history: C[p:n, p:p+nb] -= A[p:n, 0:p] @ A[p:p+nb, 0:p]^T
    for micro in range(panel, panel + nb, 128):
        if micro > panel:
            gemm_inner: C[m:n, m:m+128] -= X[m:n, p:m] @ X[m:m+128, p:m]^T
        factor128[_inv]                       # 1 CTA, SMEM tile, potf2 chain
        trsm step on X[m+128:n, m:m+128]      # inverse apply or cublasStrsm
    if right-looking:
        ssyrk_trailing (TF32 math mode)       # control variant 5 only
zero_wedges                                   # restore exact upper zeros
```

### Wedge-correctness proof

Library GEMMs write the full rectangle `C[m:n, m:m+128]` (and the panel
rectangle for the history GEMM), so garbage lands only where row < column
inside those rectangles - i.e. the strict upper wedge of an NB-diagonal
block. No later operand reads a wedge:

- history GEMM operands: rows >= p, columns < p; every earlier panel's
  wedge lives at rows < p (disjoint);
- inner GEMM operands: rows >= m, columns < m (strictly lower);
- `factor128_kernel` loads only `column <= row`;
- `trsm_apply_kernel` / `cublasStrsm` read rows below the diagonal block
  and the lower triangle of the block itself;
- variant 5's SYRK reads rows >= p+nb, columns < p+nb (strictly lower).

Hence one final `zero_wedges_kernel` pass (~67 MB at NB=1024) restores
exact zeros everywhere.

### Node strategies

- **copy_lower** (`copy_lower_kernel`, grid 2048 x 256): float4
  grid-stride over 2^30 elements; quads fully below the diagonal copy,
  fully above store zeros, the one straddling quad per row splits
  elementwise. ~8 GiB traffic, ~1.2-1.5 ms expected.
- **factor128** (`factor128_kernel<BuildInverse>`, grid 1 x 256): ports
  the `b60n1024` panel chain - `potf2_32` (single-warp right-looking 32
  strip), `local_trsm` (4-lane row groups), `local_update` - over a
  128x129 padded SMEM tile. `BuildInverse` additionally builds
  T = inv(L11) in SMEM: four warp-parallel 32x32 forward-substitution
  inverses (one lane per column), then 32->64->128 combines through
  `inv([[A,0],[B,C]]) = [[inv A, 0], [-inv(C) B inv(A), inv C]]`, dense
  inner loops made exact by storing the strict upper of T as zeros.
  Written to the 64 KB `t_inv` workspace. The inverse costs ~1.6 MFLOP -
  noise against the launch itself.
- **trsm step**, winner path (`trsm_apply_kernel`, grid = row tiles,
  256 threads, 132 KB SMEM): X := X @ T^T per 128x128 tile, in place
  (each output tile depends only on its own input tile and T). 8x8
  register microtiles; the k loop is clipped to `warp_column*64 + 64`
  because T[c][k] = 0 for k > c, and the remaining zero products are
  exact no-ops. Baseline path instead calls `cublasStrsm` (row-major
  right-solve mapped to a col-major left-solve on the transposed upper
  view) - correct but expected slow (native FP32 TRSM).
- **big GEMMs**: `cublasGemmEx` with per-call compute type
  (`CUBLAS_COMPUTE_32F`, `32F_FAST_TF32`, or
  `32F_EMULATED_16BFX9` under a `CUBLAS_VER_MAJOR >= 13` guard).
  Row-major tensors are presented as their column-major transposes with
  `OP_T`/`OP_N`; the symmetric product makes the transposed result land
  exactly on the intended row-major destination.
- **variant 5 SYRK**: `cublasSsyrk` under `CUBLAS_TF32_TENSOR_OP_MATH`
  set by the RAII guard. Empirical control for whether SYRK engages TF32
  tensor cores; writes only the stored triangle.

### Phase-3 plan (variants 6-7, not yet implemented)

Fused panel kernel replacing the per-micro launch chain: persistent CTAs
with an occupancy-clamped grid, CTA 0 owning diagonal factor+invert work,
row CTAs spinning on per-micro GMEM flags
(`st.release.gpu` publish after `__syncthreads()`, consumer
`ld.relaxed.gpu` + `fence.acquire.gpu`, `__nanosleep` backoff - the
protocol proven in `references/qr-kernels/gaunernst.py:749-763,
1129-1294`). Gated on phase-2 NCU evidence that the serial factor chain
plus launch gaps cost >= ~5 ms.

## Variant registry

IDs are stable and append-only.

| ID | name | schedule | NB | trsm | big math | inner math | role |
|---:|---|---|---:|---|---|---|---|
| 0 | `ll_nb1024_strsm_fp32_all` | left | 1024 | cublasStrsm | FP32 | FP32 | correctness baseline |
| 1 | `ll_nb1024_strsm_tf32_big` | left | 1024 | cublasStrsm | TF32 | FP32 | accuracy probe: TF32 isolated on the n^3/3 flops |
| 2 | `ll_nb1024_inv_tf32_all` | left | 1024 | inverse apply | TF32 | TF32 | expected winner |
| 3 | `ll_nb2048_inv_tf32_all` | left | 2048 | inverse apply | TF32 | TF32 | NB axis |
| 4 | `ll_nb1024_inv_bf16x9_big` | left | 1024 | inverse apply | BF16x9 | FP32 | accuracy-safe tensor fallback |
| 5 | `rl_nb1024_ssyrk_tf32` | right | 1024 | inverse apply | Ssyrk+TF32 mode | TF32 | SYRK math-mode control |
| 6 | `ll_nb1024_fused_tf32` | left | 1024 | fused | TF32 | TF32 | phase 3 |
| 7 | `ll_nb2048_fused_tf32` | left | 2048 | fused | TF32 | TF32 | phase 3 |
| 8 | `ll_nb1024_flatf_tf32` | left | 1024 | inverse apply | TF32 | TF32 | regressed (95.4 ms; alias-chain lesson) |
| 9 | `ll_nb1024_flatf_lightapply_tf32` | left | 1024 | light apply | TF32 | TF32 | regressed (100.6 ms; carveout/L2 lesson) |
| 10 | `ll_nb1024_rank8f_tf32` | left | 1024 | inverse apply | TF32 | TF32 | rank-8 register-blocked factor |
| 11 | `ll_nb2048_rank8f_tf32` | left | 2048 | inverse apply | TF32 | TF32 | v10 x NB axis |
| 12 | `ll_nb1024_rank8w_tf32` | left | 1024 | inverse apply | TF32 | TF32 | wide 512-thread redundant-corner factor |
| 13 | `ll_nb1024_microfused_tf32` | left | 1024 | fused micro | TF32 | TF32 | one launch per micro: CTA 0 factors while consumer CTAs prefetch X tiles, release-flag handoff |
| 14 | `ll_nb1024_invgemm_tf32` | left | 1024 | inverse GEMM | TF32 | TF32 | apply as cuBLAS TF32 GEMM (dense T, exact upper zeros) into scratch + float4 copy-back |

## Native API and resource policy

Pybind entry points `prepare` / `run` / `run_out` / `metadata`, matching
the established shape modules. `prepare(variant)` opts the factor and
apply kernels into their dynamic SMEM sizes
(`cudaFuncAttributeMaxDynamicSharedMemorySize`, carveout 100) and rejects
unimplemented variants. All linear and pointer arithmetic on the matrix
is `int64_t` by decree: n^2 = 2^30 elements leaves zero headroom for
int32 `row * kN` products.

| kernel | threads | dynamic SMEM |
|---|---:|---:|
| `factor128_kernel<false>` | 256 | 66,560 B |
| `factor128_kernel<true>` | 256 | 148,992 B |
| `trsm_apply_kernel` | 256 | 132,096 B |
| `copy_lower_kernel` | 256 | 0 |
| `zero_wedges_kernel` | 256 | 0 |

Workspace: one `{128,128}` FP32 `t_inv` tensor per call (64 KB,
caching-allocator noise after the untimed warm call). Launch count for
implemented variants is 768 at both NB values
(1 copy + big GEMMs + inner GEMMs + 256 factors + 255 trsm steps +
1 wedge pass); `metadata` reports it per variant along with registers,
SMEM, local bytes, and occupancy.

A ptxas local frame is diagnostic metadata here, not a rejection
criterion (the `b60n1024` policy); acceptance is by measured endpoint
time and checker verdict.

## Popcorn rounds

### 2026-07-26 - phase 1: plumbing + TF32 legality (direct measurements)

- `submit --mode test` (v0 tracked): submission 913404, all 17 checker
  cases passed. Note the test grid tops out at n=2048, so this run
  validated compile + fallback only; kernel-path validation came from
  the benchmark runs below (benchmark row 14 is the target shape and the
  evaluator checks every row before timing, then rechecks each repeat).
- `autotune --variants 0,1 --rounds 1`
  (`artifacts/autotune/b1n32768_20260726T094015Z/`), all 15 public rows
  passing for both variants:

| variant | benchmark.14 mean | vs 220.7 ms baseline |
|---:|---:|---:|
| 0 `ll_nb1024_strsm_fp32_all` | 271.64 ms | 0.81x (slower, expected) |
| 1 `ll_nb1024_strsm_tf32_big` | 86.48 ms | **2.55x** |

- Promotion: v1 became the tracked default.
- Inference from the v0/v1 split: switching only the big history GEMMs
  from FP32 to TF32 removed ~185 ms, consistent with the FP32 GEMM
  estimate; the remaining ~86 ms budget should be roughly TF32 GEMMs
  ~20 ms + FP32 `cublasStrsm` ~20-25 ms + FP32 inner GEMMs ~10-15 ms +
  serial factor chain and ~768 launches ~15-20 ms + copy ~1.5 ms. The
  strsm and inner-GEMM terms are exactly what variants 2-5 remove;
  NCU/nsys evidence is still needed before treating this split as fact.

### 2026-07-26 - phase 2: full sweep (direct measurements)

`autotune --variants 0,1,2,3,4,5 --rounds 3`
(`artifacts/autotune/b1n32768_20260726T094501Z/`); every variant passed
all 15 public rows in all 3 rounds; per-round spreads were < 0.6%:

| variant | median benchmark.14 mean | vs baseline |
|---:|---:|---:|
| 2 `ll_nb1024_inv_tf32_all` | **74.48 ms** | **2.96x** (promoted) |
| 3 `ll_nb2048_inv_tf32_all` | 75.81 ms | 2.91x |
| 1 `ll_nb1024_strsm_tf32_big` | 86.77 ms | 2.54x |
| 4 `ll_nb1024_inv_bf16x9_big` | 152.15 ms | 1.45x |
| 5 `rl_nb1024_ssyrk_tf32` | 270.47 ms | 0.82x |
| 0 `ll_nb1024_strsm_fp32_all` | 272.93 ms | 0.81x |

Conclusions (measured unless noted):

- Inverse-apply TRSM + TF32 inner GEMMs beat `cublasStrsm` + FP32 inner
  by 12.3 ms - smaller than the pre-measurement model, so the earlier
  ~20-25 ms strsm estimate was too pessimistic.
- NB=2048 loses ~1.3 ms to NB=1024: deeper-k history GEMMs buy nothing
  (already compute-bound), while the longer per-panel micro chain and
  larger wedge pass cost a little.
- `cublasSsyrk` under `CUBLAS_TF32_TENSOR_OP_MATH` runs at FP32 speed
  (v5 = v0 within noise): SYRK does not engage TF32 tensor cores on this
  cuBLAS build. Right-looking via library SYRK is a dead end here.
- BF16x9 emulated big GEMMs are ~5x slower than TF32 (still pass the
  checker); v4 stays as an accuracy fallback only.
- Inference pending NCU: with strsm/inner terms now bounded at ~12 ms,
  the remaining ~74 ms budget is likely dominated by the serial micro
  chain (256 single-CTA factors with 147 SMs idle, 224 skinny m=128
  inner GEMMs, 255 applies, ~768 launch gaps) - potentially 35-45 ms.
  If NCU confirms, the fused-panel variants (6/7) and/or a lighter
  factor+apply single-launch fusion are the next lever, targeting
  ~40-50 ms.

### 2026-07-26 - phase 2b: NCU on the winner (direct measurements)

`ncu --variants 2,3` (`artifacts/ncu/b1n32768_20260726T095215Z/`),
NCU 2026.2.0, 10 launches captured per report (panel-0 prefix: copy,
then factor/apply/inner-GEMM rounds). NCU locks clocks to base, so
durations are ~1.7x boost-clock wall time; shares are the evidence.

Per-launch `gpu__time_duration.sum`, v2 report:

| kernel | NCU duration | x count | inferred share of 74.5 ms |
|---|---:|---:|---:|
| `factor128_kernel<true>` | 320.8 us | 256 | ~48 ms (dominant) |
| `trsm_apply_kernel` | 92 us | 255 | ~14 ms |
| inner GEMM (`cutlass3x_sm100_tensorop_s128x256x8tf32gemm...`) | 31 us | 224 | ~4-8 ms |
| `copy_lower_kernel` | 1.39 ms | 1 | ~1 ms |

The inner-GEMM kernel name confirms cuBLAS dispatched an SM100 TF32
tensor-op CUTLASS kernel (TF32 genuinely engaged).

Warp stalls for `factor128_kernel` (launch:1, by reason): barrier
47.6%, wait 19.0%, short_scoreboard 13.9% - warps idle behind the
warp-serial blocked chain (`potf2_32` rank-1 updates, `local_trsm`'s
sequential 64-column loop, `local_update`, the sqrt/div chain all
appear in the by-line histogram). The b60n1024-derived blocked factor
was latency-tolerant only because 60 CTAs ran concurrently; at batch=1
it serializes on a single SM and is ~65% of the runtime.

Actions taken (variants 8/9, IDs appended):

- **v8 `ll_nb1024_flatf_tf32`** - replaces the blocked chain with a
  flat right-looking spotf2-128: per column, one root, CTA-parallel
  scale, row-parallel rank-1 trailing update (each thread owns rows;
  the update is independent FMAs, not a latency chain). Estimated
  factor ~40-90 us NCU vs 320.8, chain ~10-15 ms vs ~48.
- **v9 `ll_nb1024_flatf_lightapply_tf32`** - v8 plus a light
  `trsm_apply` that skips the shared copy of T and reads it through
  the read-only cache (`__ldg`), halving dynamic shared memory
  (132,096 -> 66,048 B) so two CTAs share an SM.

### 2026-07-26 - phase 2c: flat-factor regression and rank-8 redesign

`autotune --variants 2,8,9 --rounds 3`
(`artifacts/autotune/b1n32768_20260726T101123Z/`), all passing:
v2 74.26 ms (promotion: already_default), **v8 95.4 ms, v9 100.6 ms -
both regressions.**

Root causes (analysis, consistent with the deltas):

- **v8 (+82 us per factor)**: the naive flat rank-1 update stores into
  the shared tile every iteration while the next iteration loads from
  the same tile through the same base pointer; the compiler cannot
  prove non-aliasing, so each store serializes against the following
  loads (~30-cycle chain per element). The blocked chain, despite its
  warp serialization, accumulates in registers and stores once - which
  is exactly why it wins. Lesson recorded: SMEM-destination inner
  loops must either accumulate in registers or read operands from a
  separately-qualified buffer.
- **v9 (+20 us per apply on top of v8)**: `configure_dynamic` requests
  shared-memory carveout 100, leaving `__ldg` reads of T almost no L1;
  the k-loop pulls ~1 MB per CTA from L2 instead of a one-time 64 KB
  shared copy. Light-apply direction dropped.

Action - **v10/v11 `ll_nb1024/2048_rank8f_tf32`**: rank-8
register-blocked flat factor (MAGMA `POTF2_NB = 8` precedent): per
8-column group, a one-warp 8x8 corner potf2, per-row 8-entry solves
held entirely in registers, publication of the solved columns into a
separate `__restrict__` panel buffer (removes the alias chain;
`kPanelLd = 9` keeps it bank-conflict-free), and a rank-8 trailing
update with four concurrent accumulators (one store per 8 FMAs).
NB=2048 re-enters as v11 because the earlier NB tie was measured under
a factor-dominated budget.

### 2026-07-26 - phase 2d: rank-8 factor round (direct measurements)

`autotune --variants 2,10,11 --rounds 3`
(`artifacts/autotune/b1n32768_20260726T102400Z/`), all passing:

| variant | median benchmark.14 mean | vs baseline |
|---:|---:|---:|
| 10 `ll_nb1024_rank8f_tf32` | **60.89 ms** | **3.62x** (promoted) |
| 11 `ll_nb2048_rank8f_tf32` | 62.13 ms | 3.55x |
| 2 `ll_nb1024_inv_tf32_all` | 74.65 ms | 2.96x |

- The rank-8 register-blocked factor bought 13.6 ms over the blocked
  chain - real but well short of the ~38 ms the NCU-derate inference
  predicted. Recorded correction: the 320.8 us NCU factor duration
  divided by the 1.7x clock derate over-attributed the wall budget to
  the factor; the apply / GEMM / launch-gap terms must be larger than
  the subtraction suggested. Per-kernel truth for the new winner needs
  its own capture before the next design bet (fused panel vs apply
  optimization vs launch-count reduction).
- NB=2048 remains ~1.2 ms behind NB=1024 even with the cheap factor.

### 2026-07-26 - phase 2e: NCU on v10 and the wide-factor redesign

`ncu --variants 10` (`artifacts/ncu/b1n32768_20260726T102939Z/`),
direct measurements from the report + `ncu-details.txt`:

- `factor128_kernel<1,2>` (rank-8): **230.7 us** NCU (vs 320.8 blocked;
  the 90 us NCU delta / 1.7 = 53 us boost x 256 exactly reproduces the
  measured 13.6 ms sweep gain, validating the clock-derate model).
  `trsm_apply` 91.6 us, inner GEMM ~31 us, copy 1.40 ms - unchanged.
- Factor kernel internals: **0 spills, 40 regs/thread, achieved
  occupancy 12.5% (7.99 warps on one SM), issued IPC 0.72, 11.1 warp
  cycles per issued instruction, 188K issued instructions.** Stalls:
  barrier 54.7%, wait 17.2%, short_scoreboard 10.3%; hot lines are the
  corner-potf2 loop (25.2%) and the solve phase (22.5%); the rank-8
  update itself is now 2-4%-per-line.
- Reading: the kernel is pure latency exposure - 8 warps cannot hide
  sqrt/SMEM/FMA latency, the corner/solve phases idle most warps
  behind 48 CTA barriers, and the inverse-build combines contribute
  roughly a third of the instruction stream (dense loops over known
  zeros).

Actions:

- **v12 `ll_nb1024_rank8w_tf32`** (kFactorWide, 512 threads): every
  thread factors the 8x8 corner redundantly in registers (~150-cycle
  chain run by all warps in parallel; removes the warp-serial corner,
  its shuffles, and one barrier per group), four threads share each
  sub-panel row (redundant register solve, quarter-split rank-8
  update, stride-16 4-blocks), two CTA barriers per group, sixteen
  warps for latency hiding.
- **Shared pure win**: triangular loop bounds in the inverse-build
  combines (`k >= column` / `k <= row`), halving their instruction
  count for every inverse variant (v2/v10/v11 improve too; noted for
  cross-round comparability).

### 2026-07-26 - phase 2f: wide factor round (direct measurements)

`autotune --variants 10,12 --rounds 3`
(`artifacts/autotune/b1n32768_20260726T104145Z/`), all passing:

| variant | median benchmark.14 mean | vs baseline |
|---:|---:|---:|
| 12 `ll_nb1024_rank8w_tf32` | **48.32 ms** | **4.57x** (promoted) |
| 10 `ll_nb1024_rank8f_tf32` | 61.33 ms | 3.60x |

- The 512-thread redundant-corner factor bought 13.0 ms - the
  occupancy/latency diagnosis held. Inferred factor kernel now
  ~144 us NCU / ~85 us boost, chain ~21.5 ms.
- v10 re-measured 61.33 vs 60.89 previous round (+0.7%, noise edge):
  the shared triangular-combine bounds did not help at 8 warps
  (variable-trip loops unroll worse); v12's 16 warps absorb them.
- Budget model at 48.3 ms (inference, capture pending): factor chain
  ~21.5 ms, applies ~14 ms, inner GEMMs ~4 ms, history GEMMs + launch
  gaps ~8 ms, copy ~1 ms. Factor chain and row work are now comparable
  sizes - the regime where fused-panel overlap (factor k+1 concurrent
  with apply k) converts their sum into their max. Next lever: fused
  per-micro factor+apply kernel (single launch, GMEM flag, apply CTAs
  prefetch X tiles while the factor CTA works), then full panel fusion.

### 2026-07-26 - phase 2g: v12 NCU capture and the fused per-micro kernel

Leaderboard locked in first: submission 913708, public geomean
**1.8695 ms**, secret 1.8921 ms, all runs passing (v12 tracked).

`ncu --variants 12` (`artifacts/ncu/b1n32768_20260726T105623Z/`),
direct measurements via veloq, first-panel launches:

| kernel | NCU duration | boost est. (/1.7) | x count | total |
|---|---:|---:|---:|---:|
| `factor128_kernel<true, kFactorWide>` | 147.6 us | ~86.8 us | 256 | ~22.2 ms |
| `trsm_apply_kernel` (255 tiles, largest) | 91.8 us | ~54 us | 255 (shrinking) | ~8 ms |
| inner GEMM (cutlass TF32) | 31.3 us | ~18 us | 224 | ~4 ms |
| `copy_lower_kernel` | 1.39 ms | ~0.8 ms | 1 | ~0.8 ms |

Remainder of the 48.3 ms wall (~13 ms) = history GEMMs + ~768 launch
gaps. Factor kernel internals: 78 registers, no spills, 154624 B SMEM,
16 warps resident; stalls 38.4% barrier / 17.2% wait / 11.5%
short_scoreboard - the serial dependency chain is the floor, not a
scheduling defect. Two consequences: (a) in-kernel factor tuning is
near exhausted; (b) the apply (54 us max) fits entirely inside the
factor's 87 us shadow, so fusion converts factor+apply+2 gaps into
~factor+tail per micro.

Variant 13 `ll_nb1024_microfused_tf32` implements the fused per-micro
kernel (`fused_micro_kernel`, 512 threads, 153600 B dynamic SMEM =
max(factor 153.6 KB, apply 132 KB), one CTA per SM):

- Grid = min(1 + tiles, co-resident capacity from
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` x SM count), latched
  once; `prepare()` TORCH_CHECKs capacity >= 2, so the spin cannot
  deadlock (148 SMs x 1 CTA on B200; tiles <= 255 means consumers own
  at most 2 tiles each).
- CTA 0 = exact v12 factor path (`factor_wide` + `build_inverse_128`),
  stores `t_inv`, then `__syncthreads()` +
  `st.release.gpu.global.u32` on this micro's flag slot. Consumers
  prefetch their first X tile into SMEM before spinning
  (`ld.global.relaxed.gpu.L1::no_allocate.u32` + `__nanosleep(64)`,
  then `fence.acquire.gpu`) - the tile load rides under the factor's
  critical path. Protocol per `references/qr-kernels/gaunernst.py`
  749-763, 1173-1214.
- Apply adapted to 512 threads: 4x4 warp grid, 32x32 per warp, 8x4
  register microtile per thread, k clipped per 32-column stripe
  (finer than v12's 64-column clip; ~17% fewer lane-FMAs).
- Flag workspace: 256 int32, `at::zeros` per call (write-once slots,
  no per-micro reset; one extra tiny fill launch counted in metadata).
- Last micro (tiles = 0): grid 1, CTA 0 skips inverse build and
  publish.
- Correctness argument unchanged from v2-v12: identical operations in
  identical order, only launch packaging differs; all consumer inputs
  except `t_inv` were written by earlier kernels on the same queue,
  and `t_inv` is ordered by the release/acquire flag.

Expected: deletes ~8 ms of exposed apply time plus ~256 launch gaps
(~1-2 ms) -> ~38-40 ms. Not projected as promised - measured next by
autotune. Beyond v13: full panel fusion (v6/v7 slots) would also pull
the 128x128 diagonal slice of the inner update in-kernel so factor k+1
stops waiting on the full-column cuBLAS GEMM; that targets the ~22 ms
factor-chain floor itself.

### 2026-07-26 - phase 2h: fused-micro results and the inverse-GEMM apply

`autotune --variants 12,13 --rounds 3`
(`artifacts/autotune/b1n32768_20260726T111616Z/`), all passing:

| variant | median benchmark.14 mean | vs baseline |
|---:|---:|---:|
| 13 `ll_nb1024_microfused_tf32` | **48.53 ms** | **4.55x** (promoted) |
| 12 `ll_nb1024_rank8w_tf32` | 48.85 ms | 4.52x |

v13 won every round but only by ~0.35 ms - far under the ~9 ms
projection. `ncu --variants 13`
(`artifacts/ncu/b1n32768_20260726T111923Z/`, direct measurements):
`fused_micro_kernel` = 234.8 us vs 147.6 + 91.8 = 239.4 us for the
separate pair; 65.6% barrier stalls, 61.6% of samples on the
consumers' post-spin barrier (cuda.cu:997 = first instruction after
it). Post-mortem: the prefetch hid only the X-tile *load*; the apply
*compute* (~40 us/tile at 16 warps = 25% occupancy, 2 serial tiles on
most consumers because the grid is capped at 148 CTAs) cannot start
before the flag and runs serially after the factor. The apply was
latency-bound all along, not load-bound - same occupancy disease NCU
already diagnosed in the factor at phase 2e.

Variant 14 `ll_nb1024_invgemm_tf32` draws the sharper conclusion:
stop hand-rolling the thin memory-bound product. Since `t_inv` has
exact strict-upper zeros, X := X T^T is a plain dense GEMM, so per
micro: v12 factor kernel (unchanged), then
`cublasGemmEx` (OP_T/OP_N, m=128, n=rows, k=128, TF32,
A=t_inv ld 128, B=X ld kN, beta=0) into a `(kN-128) x 128` scratch
column, then a float4 `copy_back_kernel` (256x256, ~33 MB traffic,
est. ~9 us). cuBLAS runs this shape with tensor cores and deep
pipelines - est. ~10-15 us vs the 87 us in-kernel tail. Accuracy:
TF32 quantization of X and T adds ~1e-3 relative error to panel
entries against the ~6e-2 gate at n=32768 (the history GEMMs already
put TF32-scale error into the same entries; checker verdict rules).
Est. ~40-42 ms if the apply GEMM lands in the cutlass fast path -
measured next, not promised. If it holds, the follow-up folds the
copy-back into the fused kernel's consumer CTAs (k-split GemmEx
operands read scratch directly) to take the copy-back off the
critical path too.

## Roofline and expectations (pre-measurement estimates)

- Big GEMM flops: n^3/3 = 1.17e13. At an effective 600-900 TF (TF32,
  k up to 31744): ~15-20 ms. BF16x9: ~45-55 ms. FP32: ~150-190 ms.
- History HBM reads: ~n^3/(6*NB)*4 B = 23 GB at NB=1024, 11 GB at
  NB=2048; arithmetic intensity ~NB/2 flop/B is 4x above the B200 ridge,
  so NB=1024 should already be compute-bound (variant 3 tests whether
  deeper k wins anyway).
- Serial micro chain: 256 x (factor ~30-100 us + inner GEMM ~10-20 us +
  trsm apply ~10-15 us) = ~10-20 ms, plus ~768 launches ~2-4 ms. This is
  the phase-3 target.
- Copy + wedge pass: ~1.5 ms. Expected total for variant 2: ~30-40 ms
  (5.5-7x the 220.7 ms baseline). `cublasStrsm` variants add an
  estimated ~15-25 ms (native FP32 TRSM at ~30-50 TF over ~1.1e12 flops
  of 128-wide solves' library overhead) - acceptable for probe duty.

## Risks

1. **TF32 legality at n=32768 on the authoritative checker** - inferred
   only; variant 1 exists to answer it with a real `--mode test` before
   the winner path is tuned. Fallback axes: BF16x9 (v4), FP32 (v0).
2. **BF16x9 / emulation API availability** - compile-guarded on
   `CUBLAS_VER_MAJOR >= 13`; `metadata` column `emulation_available`
   reports the build's answer; v4 fails loudly, never silently.
3. **SYRK TF32 math-mode engagement** - unknown; v5 is a measurement,
   not a dependency.
4. **Inverse-apply numerics** - multiplying by inv(L11) instead of
   solving is safe at cond~2 (MAGMA `ztrtri_diag` precedent, gaunernst
   `build_t128_diag` precedent); the FP32 potf2 chain and inverse are
   full precision, and the checker probe covers the rowscale /
   tridiagonal secondary cases.
5. **180 s test timeout** - first `--mode test` pays the JIT build
   (~1-2 min class) plus 17 checker reconstructions; the compile is
   cached for later runs. If the first run times out, resubmit once (the
   extension cache persists on the evaluator image between processes of
   one submission; a second submission rebuilds).
6. **int32 overflow** - all matrix indexing goes through
   `matrix_index(row, column)` returning `int64_t`; review checklist
   item for every new kernel.

## Autotuning and profiling

`cholesky_b1n32768_runner.py` (copy of the `b60n1024` runner):

- `submit --mode test|benchmark|leaderboard` - tracked source, official
  checker.
- `autotune --variants 0,1,2,3,4,5 --rounds 3` - concurrent public
  benchmark sweeps, authenticated result fetch, all 15 public rows must
  pass, ranks by median of per-round `benchmark.14.mean`, atomically
  promotes the winner into the tracked `# POPCORN_VARIANT` line.
- `ncu --variants <ids>` - hosted Brev B200 Nsight Compute capture at
  benchmark index 14 (needs `POPCORN_BREV_PROFILER_URL`).

Measurement policy: leaderboard-visible numbers come only from Popcorn
runs; NCU numbers are clock-locked and never compared against benchmark
means; every DESIGN entry labels direct measurements vs inferences.

## User-run verification

```bash
# DONE 2026-07-26: submit --mode test (17/17 pass, submission 913404)
# DONE 2026-07-26: autotune --variants 0,1 --rounds 1 (v1 promoted, 86.48 ms)

# DONE 2026-07-26: autotune --variants 0,1,2,3,4,5 --rounds 3 (v2 promoted, 74.48 ms)
# DONE 2026-07-26: ncu --variants 2,3 (factor chain = dominant cost)

# DONE 2026-07-26: autotune --variants 2,8,9 (v8/v9 regressed, v2 stays)

# DONE 2026-07-26: autotune --variants 2,10,11 (v10 promoted, 60.89 ms)

# DONE 2026-07-26: ncu --variants 10 (occupancy 12.5%, latency-bound)

# DONE 2026-07-26: autotune --variants 10,12 (v12 promoted, 48.32 ms)

# DONE 2026-07-26: ncu --variants 12 (factor 147.6 us / apply 91.8 us / inner 31.3 us)
# DONE 2026-07-26: submit --mode leaderboard (public geomean 1.8695 ms, submission 913708)

# DONE 2026-07-26: autotune --variants 12,13 (v13 promoted, 48.53 ms, +0.35 ms only)
# DONE 2026-07-26: ncu --variants 13 (fused 234.8 us ~= separate sum; apply compute serializes)

# DONE 2026-07-26: autotune --variants 13,14 (v14 promoted, 42.86 ms, 5.15x)

# 19. Capture the v14 split to size the next lever (factor chain expected dominant):
python3 cholesky/b1n32768/cholesky_b1n32768_runner.py ncu --variants 14

# 20. Refresh the board with v14 (~3% geomean over the 913708 submission):
python3 cholesky/b1n32768/cholesky_b1n32768_runner.py submit --mode leaderboard

# Sandbox-safe checks:
python3 -m py_compile cholesky/b1n32768/cholesky_b1n32768.py
rg -n 'stream' cholesky/b1n32768/cholesky_b1n32768.py   # must print nothing
git diff --check
```

## Score history

| Date | Selected ID | Public B200 geomean | Notes |
|------|------------:|--------------------:|-------|
| 2026-07-26 | 12 | 1.8695 ms | secret 1.8921 ms, submission 913708, all runs passing |

Benchmark-mode history for the target row (not leaderboard geomeans):

| Date | Variant | benchmark.14 mean |
|------|--------:|------------------:|
| 2026-07-26 | 0 | 271.64 ms |
| 2026-07-26 | 1 | 86.48 ms (promoted) |
| 2026-07-26 | 2 | 74.48 ms (promoted, 3-round median) |
| 2026-07-26 | 3 | 75.81 ms |
| 2026-07-26 | 4 | 152.15 ms |
| 2026-07-26 | 5 | 270.47 ms |
| 2026-07-26 | 8 | 95.39 ms (regression, recorded) |
| 2026-07-26 | 9 | 100.59 ms (regression, recorded) |
| 2026-07-26 | 10 | 60.89 ms (promoted, 3-round median) |
| 2026-07-26 | 11 | 62.13 ms |
| 2026-07-26 | 12 | 48.32 ms (promoted, 3-round median) |
| 2026-07-26 | 13 | 48.53 ms (promoted, 3-round median; v12 48.85 ms same sweep) |
| 2026-07-26 | 14 | 42.86 ms (promoted, 3-round median; v13 48.59 ms same sweep - the inverse-GEMM apply bought 5.7 ms) |

## Nsight Systems endpoint

The shape-local Modal launcher profiles one warmed factorization on B200:

```bash
.venv/bin/python -m modal run cholesky/b1n32768/cholesky_b1n32768_modal.py
# Optional comparison:
.venv/bin/python -m modal run cholesky/b1n32768/cholesky_b1n32768_modal.py --variant 14
```

`--variant -1` selects the tracked default, currently variant 14. Its
NB=1024 inverse-GEMM schedule has 1,023 algorithm launches per
factorization. Input generation, extension compilation, preparation, warmup,
and correctness validation are outside the capture; the NVTX range contains
exactly one out-parameter factorization.

Artifacts are downloaded under `artifacts/nsys/`. `profile.nsys-rep` is the
forward-compatible UI/VeloQ input, `kernel-trace.csv` is the ordered GPU
timeline, `kernel-exec-trace.csv` separates API, queue, and execution time,
and `kernel-summary.csv` aggregates duration by kernel name. The SQLite
export, human-readable statistics, command, profiler version, environment,
preflight, stdout, and stderr are retained with the report.

## 2026-07-28 trailing-size adaptation

VeloQ measured the default inverse-building POTRF128 launches at roughly
83 us throughout the warmed timeline. The trace includes kernels, runtime,
synchronization, and NVTX records but no GPU metrics, so this is direct
timing evidence only.

All appended schedules keep NB=1024:

| ID | Width schedule for remaining `R` | POTRF 128/64/32 | TRSM 128/64/32 |
|---:|---|---:|---:|
| 15 | 128 to `R=4096`, then 64 | 224 / 64 / 0 | 224 / 63 / 0 |
| 16 | 128 to `R=8192`, then 64 | 192 / 128 / 0 | 192 / 127 / 0 |
| 17 | 128/64/32 at `R=8192` and `R=1024` | 192 / 112 / 32 | 192 / 112 / 31 |

The precise 64- and 32-wide direct factors and inverse builders mirror the
16384 implementation. The width-64 factor composes two POTF2-32 blocks
with a local solve/lower update; width 32 is direct. Inner history GEMMs,
inverse-apply GEMMs, and float4 copy-backs use the selected compile-time
width. The default 128-wide implementation is not refactored.

The two cut points are NB-aligned, so each 1024-wide outer panel has one
micro width. No rank contribution crosses a width boundary, the dense
inverse retains exact strict-upper zeros, and the existing NB-wedge cleanup
restores the output contract.

All three variants passed the Modal B200 preflight with scaled
reconstruction residual `0.061490` against a limit of 16. VeloQ showed
the requested `128 -> 64` order for IDs 15/16 and `128 -> 64 -> 32` for
ID 17. In ID 17 the factor medians were 83.136 us, 67.136 us, and
30.720 us, reductions of 19.2% and 54.2%. Width-specialized copy-back
medians fell from 4.896 us to 2.048 us and 1.632 us. The history/apply
dimensions selected distinct cuBLAS kernel families as the width changed.

| ID | W128/W64/W32 factor p50 (us) | Kernel trace span (ms) | Delta from default |
|---:|---:|---:|---:|
| 14 | 82.9 / - / - | 44.583 | baseline |
| 15 | 82.9 / 67.0 / - | 44.615 | +0.1% |
| 16 | 82.8 / 66.9 / - | 48.634 | +9.1% |
| 17 | 83.1 / 67.1 / 30.7 | 49.225 | +10.4% |

Although every smaller factor/copy specialization is materially faster per
launch, the additional launches erase the gain; IDs 16/17 regress clearly
and ID 15 is indistinguishable from the default capture. None passed the
faster-than-default timeline screen, so no authoritative autotune was run.
Variant 14 remains the default. Static validation and the case-insensitive
rejected-token scan passed.
