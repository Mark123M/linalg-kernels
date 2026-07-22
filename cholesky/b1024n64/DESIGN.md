# b1024n64 — Batched Cholesky, 1024 × 64 × 64 FP32 on B200

## Status

- Implemented: raw CUDA extension with 21 active custom variants across four
  kernel families, plus a cuSolverDx POTRF baseline as variant 19. Modal
  tune/nsys/ncu/popcorn tooling adapted from `b4096n32`. IDs 20 and 21 are
  reserved for the measured-and-retired rendezvous experiments; the two new
  instruction-footprint hybrids are active as variants 22 and 23.
- Measured (2026-07-20/21): Modal tune (×2), full 20-variant Popcorn
  leaderboard sweep, and two NCU rounds (crout family; smem + library family).
  See **Profiling findings** below.
- **Fastest kernel: variant 13** (`w4_int_f2_raw_right_rootlook`), by the Modal
  tune median (24.69 µs; min≈median, <1% jitter — the same warm-ring/full-clock
  methodology the real eval uses). This is the authoritative speed ranking. The
  leaderboard geomean (winner nominally v6, 0.00188028 s) dilutes the (1024,64)
  signal across 15 shapes and is noise-dominated within the top cluster; NCU
  locked-clock+cache-flush durations are diagnostic only and can invert the
  real ranking (they did for v18 — see findings).
- Production default: variant 6 in the current file. Recommend flipping
  to **variant 13**, after a focused leaderboard re-submit of {13, 6, 10}
  confirms it (the first sweep's noise put v6 nominally on top). v13 is a
  right-looking rank-1 kernel that fits in 255 registers with **zero spills**;
  its no-lookahead siblings v12/v14 spill 512 B/thread and are ~8× slower, so
  the win is real but sits at the edge of the register cap.
- Rejected after B200 sweep: experimental variants 20 and 21 added one late
  CTA rendezvous to the v6 and v13 algorithms, respectively. Both were
  spill-free and correct, but v20 tied/slightly lost to v6 and v21 was 1.98%
  slower than v13. Their IDs remain reserved but are omitted from all active
  sweeps; baselines 6 and 13 remain unchanged.
- Awaiting B200 measurement: v22 keeps v6's register Crout columns 0--47 and
  factors only the 16x16 tail with compact, non-unrolled shared-memory loops.
  v23 keeps v13's root-lookahead right-looking prefix through k=31, then
  factors the resulting 32x32 Schur complement with compact shared-memory
  rank-1 loops. These test whether deleting the late unrolled instruction
  footprint works where a one-time CTA rendezvous did not. A local CUDA 13.1
  SM100a build passes: v22 is 4,344 static instructions / 69,504 bytes, 164
  registers, 16,640 B shared, zero stack/spills; v23 is 5,512 instructions /
  88,192 bytes, 255 registers, 16,896 B shared, zero stack/spills.

## Correctness reference

LAPACK `SRC/spotf2.f` (lower, left-looking):

```
L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2)
L[i][j] = (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j]      (i > j)
```

All kernels read `A[i][j]` as `a[j*64 + i]` — valid for the symmetric input
and coalesced for row-major storage. MAGMA context: at n=64 FP32, MAGMA's
batched potrf stays entirely in the fused single-kernel `lpout` path
(crossover to Level-3 blocked code is n>192), factoring each matrix inside one
threadblock with one thread per row — the model for the shared-memory family.
The checker validates shape/dtype/device, finiteness, exact zeros in the
strict upper triangle, positive diagonal, and a scaled reconstruction
residual; it is property-based, so refined/raw reciprocal-root variants are
admissible if the residual gate holds.

## B200 mapping

- **Warp families (A, C)**: one warp factors one 64×64 matrix; each lane owns
  two factor rows in registers (`float row[2][64]`, 128 persistent floats).
  n=64 > warp width forces the two-rows-per-lane split, and that is also the
  structural fix for the b4096n32 bottleneck: every `__shfl_sync` pivot
  broadcast now feeds **two** FMA chains, so the shuffle:FMA ratio halves
  (~2,080 shuffles vs ~4,032 FFMAs per matrix; the 32×32 kernel ran 527:496).
  - Row ownership layouts: *blocked* (rows `lane`, `lane+32`; two scalar
    coalesced loads per column) vs *interleaved* (rows `2*lane`, `2*lane+1`;
    one coalesced `float2` load per column — half the load instructions).
    Pivot owner lane/slot are compile-time constants under the fully unrolled
    column loop.
  - Occupancy is a non-issue: 1,024 warps total ≈ 7 warps/SM on 148 SMs in
    one wave, so launch bounds are loose (Warps=2/4/8 → 512/256/128 CTAs →
    MinBlocks 4/2/1, register budget up to 255/thread). Latency hiding must
    come from ILP, not warp count. **Zero spills is a hard acceptance gate**
    (metadata `local_bytes_per_thread`, plus `-Xptxas -warn-spills` output).
  - **Confirmed bottleneck** (NCU): the ~6,950-instruction fully unrolled body
    (~110 KB SASS) is instruction-fetch bound — `no_instruction` is the #1
    warp stall at ~3.1 cycles/issued-inst, and NCU's rule text names
    instruction-cache misses explicitly. The 110 KB body vastly exceeds the
    SM's small L1 instruction cache; barrier-free warps drift to scattered PCs
    across it, so the instantaneous set of needed I-cache lines overflows and
    thrashes. Raising warp count alone does **not** fix it (v6→v8, 8→16
    warps/SM, barely moved it: 3.09→2.88). Only a barrier-synchronized
    block-per-matrix layout, whose warps re-converge to a shared PC every
    column and reuse I-cache lines, collapses it (~0.11) — see findings.
- **Shared-memory family (B)**: one CTA factors one matrix (or two at
  `MatricesPerCta=2`); 64 threads per matrix, thread `tx` owns row `tx`;
  factor kept in `float[64][65]` shared tiles (padded leading dimension 65,
  bank-conflict-free; 16.25 KiB per matrix). Per column: coalesced symmetric
  load, dot loop with dynamic indexing (`#pragma unroll 8`, tiny code),
  inverse broadcast through a shared scalar, two barriers → 128 barriers per
  matrix. 1,024 CTAs (MinBlocks 7) ≈ one wave with classic occupancy-based
  latency hiding and a loop-based body measured in hundreds of instructions.
  `smem_reg` keeps the own row in registers (full unroll) so each dot FMA
  reads one register + one broadcast shared operand — half the shared-memory
  traffic of the pure version.
- **Prefix/tail hybrids (E)**: retain one warp per matrix and the measured
  register algorithm only through the instruction-fetch inflection, then
  publish the remaining rows into a private shared tile. v22 uses a 16x65
  tile per warp (16.25 KiB/CTA at Warps=4); v23 uses a 32x33 tile per warp
  (16.5 KiB/CTA). Their tail loops use `#pragma unroll 1`, so the goal is fewer
  unique instructions rather than more warp alignment. Only warp rendezvous
  are used: once at handoff and where a completed shared column must become
  visible to the next iteration.
- **cuSolverDx baseline (variant 19)**: `Size<64,64> + Precision<float> +
  Type<real> + Function<potrf> + FillMode<lower> + Arrangement<row_major> +
  SM<1000> + Block() + BlockDim<128> + BatchesPerBlock<1>`, one CTA per
  matrix, staged through dynamic shared memory, writeback masks the strict
  upper triangle to exact zeros. Built as a separate setuptools CUDAExtension:
  `-rdc=true`, `-gencode=arch=compute_100a,code=lto_100a`, device link
  carries `libcusolverdx.fatbin` (`nvcc_dlink`), and the TU omits
  `--use_fast_math` because the packaged fatbin is FTZ-off/precise-division
  and nvlink rejects mismatched RDC objects. MathDx discovery:
  `MATHDX_ROOT` → pip `nvidia/mathdx` under site-packages → CPATH →
  `/usr/local`, `/usr`; clear failure when incomplete. The dynamic
  shared-memory attribute is configured at module load (`prepare()`), before
  any capture or timing.

## Variant table

| ID | Name | Kernel | Config |
|---:|---|---|---|
| 0 | `w4_blk_precise` | crout_64 | Warps=4, blocked, scalar loads, precise root, MinBlocks=2 |
| 1 | `w4_blk_refined` | crout_64 | as 0, rsqrt + Newton |
| 2 | `w4_blk_raw` | crout_64 | as 0, raw rsqrt |
| 3 | `w4_int_f2_precise` | crout_64 | Warps=4, interleaved, float2, precise |
| 4 | `w4_int_f2_refined` | crout_64 | as 3, refined — **initial default** |
| 5 | `w4_int_f2_raw` | crout_64 | as 3, raw |
| 6 | `w4_int_scalar_raw` | crout_64 | interleaved, two scalar loads (load-width ablation) |
| 7 | `w2_int_f2_raw` | crout_64 | Warps=2, MinBlocks=4 |
| 8 | `w8_int_f2_raw` | crout_64 | Warps=8, MinBlocks=1 |
| 9 | `w4_int_f2_raw_acc2` | crout_64 | DotAccumulators=2 (4 chains total) |
| 10 | `w4_int_f2_raw_acc4` | crout_64 | DotAccumulators=4 (8 chains total) |
| 11 | `w4_int_f2_raw_look2` | crout_64 | ShuffleLookahead=2 (paired pivot shuffles) |
| 12 | `w4_int_f2_raw_right` | right_looking_64 | Warps=4, rank-1 updates |
| 13 | `w4_int_f2_raw_right_rootlook` | right_looking_64 | + early-rsqrt schedule |
| 14 | `w2_int_f2_raw_right` | right_looking_64 | Warps=2, MinBlocks=4 |
| 15 | `smem64_m1_precise` | smem_64 | 64 threads, 1 matrix/CTA, MinBlocks=7 |
| 16 | `smem64_m1_raw` | smem_64 | as 15, raw rsqrt |
| 17 | `smem64_m2_raw` | smem_64 | 128 threads, 2 matrices/CTA, MinBlocks=4 |
| 18 | `smem64_reg_raw` | smem_reg_64 | register-row hybrid, MinBlocks=7 |
| 19 | `cusolverdx_potrf` | potrf_dx | library baseline, BlockDim=128 |
| 20 | `retired_sync47` | reserved | rejected v6 + CTA rendezvous experiment |
| 21 | `retired_sync27` | reserved | rejected v13 + CTA rendezvous experiment |
| 22 | `w4_int_scalar_raw_shared_tail16` | crout_hybrid_64 | v6 through column 47, compact shared 16x16 tail |
| 23 | `w4_int_f2_raw_right_rootlook_shared_tail32` | right_hybrid_64 | v13 through k=31, compact shared 32x32 tail |

Family rationale:

- **A (crout, 0–11)**: minimum communication per FLOP, fully register
  resident, zero barriers. Variants sweep layout (0–2 vs 3–5), load width
  (6), warps per CTA (7, 8), and the ILP knobs ported from b4096n32 that gave
  ptxas independent work to overlap shuffle latency (9–11).
- **C (right-looking, 12–14)**: same register budget, but each shuffle feeds
  independent trailing-column registers instead of a serial dot chain —
  attacks short-scoreboard stalls differently. Most spill-prone family (all
  128 floats live from entry).
- **B (smem, 15–18)**: instruction-cache and occupancy hedge. NCU verdict:
  the *pure* smem kernels (15–17) do **not** lose on barriers (only 0.16
  cyc/inst) but on instruction count — 3× the register kernel from smem
  address math and two LDS per FMA. The register-row hybrid (18) keeps
  instruction count at register-kernel levels and its barriers do collapse the
  I-cache stall — but on warm full-clock timing (tune) it is only mid-pack
  (37 µs, 11th), well behind the register kernels, because its dominant stall
  becomes the global-load path. The whole family is slower than crout/right on
  the real metric.
- **D (19)**: state-of-the-art library reference measured in the identical
  harness, and a real submission variant per project decision (MathDx assumed
  available on the eval box, as in eigh).
- **E (22–23)**: instruction-footprint hybrids derived from the two measured
  champions. The existing kernels are deliberately unchanged so the focused
  sweep compares each hybrid to an exact baseline.

Variant IDs are stable and append-only. IDs 20 and 21 are reserved and rejected
by the CLI; an empty variant selection expands to active IDs 0--19, 22, and 23.
The popcorn sweep rewrites only the `_DEFAULT_VARIANT = N  # POPCORN_VARIANT`
line in temporary copies.

## Measurement policy

- Exact `(1024, 64, 64)` uses the extension; every other shape falls back to
  `torch.linalg.cholesky_ex(..., check_errors=False).L`, so non-target
  leaderboard timings stay constant across submissions and the popcorn
  geomean ranking isolates this shape.
- Modal tune: 6 checker-matching input cases (dense/spectrum/diagonal/
  lowrank/rowscale/tridiagonal, seeds `53125+i`), validation gate = finite,
  strict-upper exactly zero, positive diagonal, scaled residual ≤
  `max(16, 8×reference)` with the n=64 epsilon scale; timing via 8-tensor
  input ring, 16 warmups, 200 calls/sample, 5 repeats with order reversal,
  seed base `41064`.
- NCU runs at locked base SM clock (durations ≈ 1.7× boost-clock tuning
  numbers — never compare directly); every capture records current and max SM
  clocks. Kernel replay with cache flush; `TMPDIR` redirected off the Modal
  volume because it cannot create FIFOs.
- Acceptance gates before any variant becomes default: zero
  `local_bytes_per_thread`, Modal validation pass on all 6 cases, Popcorn
  test mode pass, then best public leaderboard geomean.

## Profiling findings

Three measurement lenses, in decreasing order of trust for *ranking speed*:

1. **Modal tune median** — full clock (1965 MHz), warm 8-input ring, 200 calls
   × 5 repeats, order-reversed. min≈median (<1% jitter) and it matches the real
   eval's methodology, so this is the authoritative speed ranking.
2. **Popcorn leaderboard geomean** — the actual score, but diluted across 15
   shapes (only (1024,64) varies), so the per-shape signal is swamped by noise
   within the top cluster.
3. **NCU** — locked base clock, kernel-replay with **cache flush**. Diagnostic
   only (stalls, roofline); its cache-flush durations can *invert* the real
   ranking and must not be read as speed. Rounds:
   `artifacts/ncu/b1024_n64_20260720T103641Z` (crout v4–v8, v10, v11) and
   `.../b1024_n64_20260721T071810Z` (v16, v18, v19).

**Authoritative speed ranking (tune, 2026-07-21, µs median; all pass):**

| var | name | µs | var | name | µs |
|---|---|---:|---|---|---:|
| **13** | int f2 right **rootlook** | **24.69** | 8 | int f2 w8 | 30.72 |
| 6 | int scalar raw | 26.70 | 1 | blk refined | 32.53 |
| 10 | int f2 acc4 | 26.74 | 18 | smem_reg | 37.15 |
| 9 | int f2 acc2 | 28.38 | 0 | blk precise | 46.53 |
| 4 | int f2 refined | 28.75 | 3 | int f2 precise | 50.73 |
| 11 | int f2 look2 | 28.79 | 16 | smem pure | 61.55 |
| 7 | int f2 w2 | 29.39 | 15 | smem precise | 67.58 |
| 5 | int f2 raw | 30.16 | 17 | smem m2 | 68.43 |
| 2 | blk raw | 30.75 | 19 | cuSolverDx | 80.67 |
|  |  |  | 12/14 | right, **no** rootlook | ~208 |

- **The right-looking rank-1 kernel with root-lookahead (v13) is fastest** —
  its independent trailing-column updates plus the early-`rsqrt` schedule beat
  the left-looking crout. It fits in **255 registers with zero spills**.
- **Spill cliff**: v12/v14 (right-looking, *no* lookahead) spill **512 B/thread**
  (the full `state[2][64]`) at a ptxas-chosen 185 regs → **~208 µs, ~8× slower**.
  The zero-`local_bytes_per_thread` gate exists exactly to catch this.
- **v18 is mid-pack (37 µs, 11th)** on this metric — see the NCU caveat below;
  its NCU "win" was a cache-flush artifact.

**Regime: latency-bound, not compute or bandwidth.** Every variant sits at
~10% of FP32 peak and ~4% of DRAM. v6 runs at 10.5% occupancy, 0.86 waves, "no
eligible warp" 79% of cycles, 8.2 warp-cycles per issued instruction. Root
cause is structural: warp-per-matrix × 1,024 matrices = 1,024 warps total → a
single 0.86 wave → ~7 warps/SM regardless of any per-thread knob.

**Crout family (run 1), stall cycles per issued instruction:**

| var | config | NCU µs | occ% | no_inst | long_sb | short_sb | wait | board s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v6 | int **scalar** raw | **57.98** | 10.5 | 3.09 | **1.43** | 0.24 | 1.11 | 0.0018803 (1st) |
| v11 | int f2 look2 | 59.17 | 10.4 | 2.85 | 2.00 | 0.20 | 1.00 | 0.0018939 |
| v8 | int f2 w8 | 59.62 | 12.3 | 2.88 | 2.19 | 0.20 | 0.99 | 0.0018864 |
| v5 | int **f2** raw | 60.06 | 10.6 | 2.96 | 2.14 | 0.19 | 0.97 | 0.0018884 |
| v7 | int f2 w2 | 60.35 | 10.5 | 3.02 | 2.14 | 0.19 | 0.97 | 0.0018850 |
| v4 | int f2 refined | 61.31 | 10.6 | 3.26 | 1.74 | 0.20 | 1.00 | 0.0018930 |
| v10 | int f2 acc4 | 61.47 | 10.5 | 3.68 | 1.19 | 0.27 | 0.84 | 0.0018818 (2nd) |

- **The b4096n32 shuffle bottleneck is solved.** `short_scoreboard` is ~0.2
  everywhere and SHFL carries zero long-scoreboard samples — two-rows-per-lane
  did its job.
- **New #1 stall is instruction fetch** (`no_instruction` ≈ 3.1 cyc/inst; the
  fully unrolled ~110 KB body at ~7 warps/SM).
- **v6 (scalar loads) beats v5 (float2) via load MLP, not bandwidth.** Two
  independent LDG.32 (127/matrix) vs one LDG.64 (64/matrix) cut long-scoreboard
  2.14 → 1.43; SASS charges the stall to the first FFMA of each column's dot
  (412 → 288 samples). At this occupancy the per-warp memory-level parallelism
  from splitting the load is the whole ~3–5%. Verified three ways (WarpState
  ratio, SASS opcode attribution, LDG counts).
- **Within-cluster leaderboard order is noise.** NCU ranks
  v6 < v11 < v8 < v5 < v7 < v4 < v10; the board ranks
  v6 < v10 < v7 < v8 < v5 < v4 < v11. Only "v6 first" survives both. Rank by
  NCU; treat board deltas < ~1% as noise. Caveat: NCU flushes caches, so its
  `long_scoreboard` is a cold-L2 worst case — the real eval runs warm (the
  16 MiB input is L2-resident), so in production the I-cache term dominates
  even more relative to loads.

**Smem + library (run 2):**

| var | kernel | NCU µs | occ% | inst/mat | no_inst | barrier | short_sb | long_sb | board s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v18 | smem_reg hybrid | **54.98** | 19.9 | 7030 | 0.12 | 0.64 | 1.42 | 8.04 | 0.0031639 |
| v6 | crout scalar (ref) | 57.98 | 10.5 | 6951 | 3.09 | 0.00 | 0.24 | 1.43 | 0.0018803 |
| v16 | smem pure | 100.77 | 20.2 | 21742 | 0.11 | 0.16 | 1.49 | 3.19 | 0.0019934 |
| v19 | cuSolverDx potrf | 134.94 | 32.2 | 12696 | 0.15 | 4.43 | 12.64 | 1.97 | 0.0020278 |

- **What removes the I-cache stall is barrier synchronization, not occupancy.**
  Both smem variants drop `no_instruction` from ~3.1 to ~0.11 — but the cause
  is not simply more warps. Raising warps in the *barrier-free* crout kernel
  barely helps (v6→v8, 8→16 warps/SM: 3.09→2.88). The smem kernels'
  `__syncthreads()` every column re-converges a CTA's warps to a **shared PC**,
  so they demand and reuse the same I-cache lines instead of each warp
  streaming its own path through the 110 KB body. More resident warps also
  help hide the residual fetch latency, but synchronization is the driver. v18
  keeps the same 7,030-instruction body as v6 yet has `no_instruction` 0.12 vs
  3.09 — pure occupancy cannot explain that gap; the barriers can.
- **Pure smem kernels lose on instruction count, not barriers.** v16 executes
  3× the instructions (21,742 vs 6,951: smem address math + two LDS per FMA +
  loop control); its barrier stall is only 0.16 cyc/inst. This corrects the
  original "loses on barriers" hypothesis.
- **v18's NCU duration (54.98 µs) is a cache-flush artifact, not real speed.**
  NCU flushes caches every replay, so v18's dominant stall — long-scoreboard on
  global loads (8.04) — pays full cold-DRAM latency, while the register kernels
  pay it too; under NCU v18 edges ahead. But on the **warm** tune (loads hit
  L2), the crout kernels speed up far more and v18 falls to 37 µs / 11th. Same
  reason its leaderboard score is bad (though the exact 0.00316 magnitude is
  additionally a bad-sample outlier — a 1/15-shape geomean cannot move that far
  from kernel speed alone). **Lesson: rank by tune, diagnose by NCU; never rank
  by NCU cache-flush duration on a load-latency-bound kernel.**
- **cuSolverDx (v19) is not competitive at n=64**: 80.67 µs on tune (134.94 µs
  NCU), short-scoreboard 12.6 + MIO 8.4 + barrier 4.4 — generic blocked
  machinery drowning a tiny matrix. Keep as the baseline; it confirms the
  custom kernels are the right call.

**Levers, ranked by evidence:**

1. **Late CTA rendezvous in the register kernels — tested, rejected.** A deeper SASS pass over
   `b1024_n64_20260721T074424Z` found that v6's 6,960-static-instruction,
   111,360-byte body is fetch-stall-free for almost all of the first 48 Crout
   columns: only 19 of 768 not-issued `no_instruction` samples occur through
   column 47. Columns 48--63 contribute 702 and output contributes 47. The
   same late-body cliff appears in v10 (951 samples, 117,376-byte body) and v13
   (909 samples, 111,616-byte body). Variants 20 and 21 tested the minimal
   experiment: keep the register algorithms but realign independent CTA warps
   once, after v6 column 47 and v13 k=27. On the same focused warm run, v20 was
   26.744 us versus v6 at 26.730 us (+0.05%, effectively tied/slightly worse),
   while v21 was 25.540 us versus v13 at 25.044 us (+1.98%). Both retained
   identical register counts, zero local bytes, and passed all six cases.
   Therefore the sampled fetch-stall localization does not translate into a
   warm-runtime win: barrier wait and/or constrained scheduling consumes the
   benefit. Do not add a barrier cadence sweep.
2. **Reduce unique late-body instructions with a register/shared hybrid — now
   implemented as v22/v23, awaiting B200.** v22 cuts over exactly where v6's
   sampled fetch cliff begins: columns 0--47 remain the scalar-load register
   Crout algorithm, while a non-unrolled shared loop handles columns 48--63.
   v23 cuts the root-lookahead right-looking algorithm after k=31, near its
   observed instruction-cache inflection, and handles the 32x32 remainder in
   a compact shared rank-1 loop. Unlike v20/v21, this actually removes static
   instructions; it pays shared-memory address/load cost only in the tail.
   Local SM100a SASS confirms a 37.6% reduction for v22 versus v6 (4,344
   versus 6,960 instructions) and a 21.0% reduction for v23 versus v13 (5,512
   versus 6,976).
   Accept only if all six cases pass, local bytes remain zero, and the warm
   timing beats its unchanged parent.
3. **Root-lookahead + right-looking remains the current measured best (v13).**
   Its warm-tune edge comes from avoiding the spill cliff and hiding diagonal
   `rsqrt` latency. The new cold-cache NCU capture does not rank it: v13 is
   59.65 us versus v6 at 58.50 us and has similar long-scoreboard (1.49 versus
   1.43 cycles/issued instruction) plus worse no-instruction (3.79 versus
   3.13). This is another direct example of NCU cache-flush timing inverting
   the authoritative warm result (24.69 versus 26.70 us).
4. **Do not start with naive explicit prefetch or generic coalescing advice.**
   nvcc already hoists v6's 127 scalar LDGs far ahead through the fully
   unrolled body, including 48 static LDGs in the first 512 instructions.
   NCU's 50% excessive-sector warning is real but splits evenly: LDG accounts
   for 520,192 excessive sectors and scattered row STG for 524,288. The
   coalesced float2 v5 was already slower than scalar v6 on warm tune, while
   coalescing output would require a register/shared-memory transpose. These
   are lower priority than the localized fetch cliff. Revisit explicit
   prefetch only after the hybrid sweep, and verify that it changes SASS
   load/use distance rather than merely source order.
5. **Avoid the precise root** — now **confirmed** by the low-jitter tune, not
   just hypothesized: `fsqrt.rn` + `fdiv.rn` on the serial column critical path
   costs interleaved f2 precise (v3) 50.73 µs vs refined (v4) 28.75 (+76%), and
   blocked precise (v0) 46.53 vs refined (v1) 32.53 (+43%). Refined ≈ raw
   (rsqrt-based) is essentially free. The earlier "unproven / v0 didn't show it"
   note was an artifact of the noisy leaderboard geomean.

**NCU 2025.4.1 metric compatibility note:** the inherited condensed helper
expected several names that this capture does not emit:
`sm__inst_executed.sum`, `sm__inst_executed_pipe_fp32...`,
`dram__bytes_{read,write}.sum`, the
`smsp__warp_issue_stalled_*_per_warp_active.pct` family, and active-normalized
MIO PQ counters. The actual direct metrics used here are
`smsp__inst_executed.sum`, `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`,
`dram__bytes.sum.per_second`, and elapsed-normalized MIO PQ counters. No
missing metric was silently replaced. Spatial hotspot counts are PC-sampling
measurements; the explanation that late warp-PC drift causes the concentration
is an architectural inference supported by the synchronized-smem comparison,
not a direct cache-miss counter.

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

## Verification commands (user-run; the sandbox has no GPU)

```bash
python3 -m py_compile cholesky/b1024n64/cholesky_b1024n64.py \
    cholesky/b1024n64/cholesky_b1024n64_modal.py
rg -n 'stream' cholesky/b1024n64/cholesky_b1024n64.py   # must print nothing
git diff --check

# Focused correctness + resources + warm kernel timing for each hybrid and its
# unchanged parent (report JSON + ptxas spill warnings).
.venv/bin/modal run cholesky/b1024n64/cholesky_b1024n64_modal.py \
    --action tune --variants 6,13,22,23

# Official checker on the default variant
popcorn submit --leaderboard cholesky --gpu B200 --mode test --no-tui \
    cholesky/b1024n64/cholesky_b1024n64.py

# Profile only after the correctness/resource/timing gate passes
.venv/bin/modal run cholesky/b1024n64/cholesky_b1024n64_modal.py \
    --action profile --variants 6,13,22,23
.venv/bin/modal run cholesky/b1024n64/cholesky_b1024n64_modal.py --action ncu \
    --variants 6,13,22,23

# Full 22-active-variant leaderboard sweep, ranked summary.json
.venv/bin/modal run cholesky/b1024n64/cholesky_b1024n64_modal.py --action popcorn
```

## Progress / score log

| date | change | result |
|---|---|---|
| 2026-07-20 | initial implementation: 19 custom variants + cuSolverDx baseline, Modal harness, DESIGN.md | built; awaiting sweep |
| 2026-07-20 | Modal tune + full 20-variant Popcorn leaderboard sweep | all 20 pass; winner v6 `w4_int_scalar_raw` geomean 0.00188028 s; top ~12 within ~1% |
| 2026-07-20 | NCU round 1 (crout v4–v8, v10, v11) | latency/occupancy-bound (~10% FP32, ~4% DRAM); shuffle solved; I-cache (`no_instruction` ~3.1 cyc) is #1 stall; v6 wins via load MLP |
| 2026-07-21 | NCU round 2 (v16 smem, v18 smem_reg, v19 cuSolverDx) | barrier synchronization (not occupancy) kills the I-cache stall (`no_instruction` 3.1→0.11); pure smem loses on 3× instruction count not barriers; cuSolverDx uncompetitive |
| 2026-07-21 | Modal tune #2 (all 20, low-jitter, eval methodology) | **v13 fastest at 24.69 µs** (right-looking + rootlook), v6 26.70, v10 26.74; v18 mid-pack 37.15 (NCU duration was a cache-flush artifact); v12/v14 spill 512 B → ~208 µs; precise root confirmed +43–76%. Recommend default → v13 after a focused {13,6,10} leaderboard re-submit |
| 2026-07-21 | Deep SASS pass on same-round v6/v10/v13 NCU captures | fetch stalls localize to the late ~40% of all three 111--117 KB register bodies; first experiment is a minimal late CTA rendezvous, not more occupancy or naive prefetch; exact LDG/STG excessive-sector attribution recorded above |
| 2026-07-21 | Added append-only CTA-rendezvous variants 20/21 and focused tune selection | v20 = v6 + sync after j=47; v21 = v13 + sync after k=27; awaiting B200 correctness/resource/warm-timing sweep of 6,13,20,21 |
| 2026-07-21 | Focused Modal tune of 6,13,20,21 | all pass and remain spill-free; v20 26.744 us vs v6 26.730 (+0.05%, no win); v21 25.540 us vs v13 25.044 (+1.98%, regression). The baseline v6 stayed within +0.13% of both prior runs, while v13 was +1.45% with wider jitter, so this is not a general runtime regression |
| 2026-07-21 | Retired failed rendezvous variants | restored the active kernel/harness to variants 0--19 and the original 19-column metadata ABI; retained focused `--variants` tune selection and added an explicit metadata-shape check |
| 2026-07-21 | Added v22/v23 instruction-footprint hybrids | v22 = v6 register columns 0--47 + compact shared 16x16 tail; v23 = v13 root-lookahead through k=31 + compact shared 32x32 tail; original v6/v13 untouched; awaiting focused B200 sweep of 6,13,22,23 |
| 2026-07-21 | Local CUDA 13.1 SM100a build + SASS count | build/load passes; v22: 164 regs, 16,640 B smem, zero stack/spills, 4,344 instructions (−37.6% vs v6); v23: 255 regs, 16,896 B smem, zero stack/spills, 5,512 instructions (−21.0% vs v13) |
