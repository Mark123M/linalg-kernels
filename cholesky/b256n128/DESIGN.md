# b256n128 — Batched Cholesky, 256 × 128 × 128 FP32 on B200

## Status

- Implemented: self-contained raw CUDA extension with twenty-six blocked 64+64
  variants and two unblocked controls, plus a Modal B200 tuning/profiling and
  Popcorn sweep harness. IDs 13--19 are append-only spill-remediation variants,
  IDs 20--25 are barrier-focused descendants of v9, and IDs 26--27 are
  cooperative-L11 (shared-reloaded) descendants of v22; the meanings of IDs
  0--25 are unchanged.
- Tracked default: variant 1 (`phase_v6_refined_u8`). It is correct, but v9 is
  the current target-shape winner and should be the leading Popcorn candidate.
- First B200 tune (2026-07-21): all 13 variants passed all six numerical
  cases, but the dynamically indexed 128-float row array put every kernel in
  local memory (512--720 B/thread), so none passed the resource gate. The
  target medians ranged from 0.103 ms (v10) to 1.221 ms (v12). See **Results**.
- Spill-remediation variants 13--19
  keep the panel/trailing rows in the padded shared tile and reserve register
  arrays only for the fully unrolled v6/v13 diagonal factor scope.
- Second B200 tune (2026-07-21): all 20 variants passed every numerical case.
  v9 is fastest at 0.0834 ms despite local-memory allocation; resource usage is
  now diagnostic only and does not affect selection.
- Awaiting Popcorn test, full leaderboard sweep, and focused confirmation.
- No verification or GPU command listed in this document was run by the
  assistant, per request.

## Algorithm and dependency graph

For each matrix,

```text
A00 = L00 L00ᵀ
A10 = L10 L00ᵀ
S11 = A11 - L10 L10ᵀ
S11 = L11 L11ᵀ
```

Every custom route launches one 128-thread CTA per matrix. The CTA uses a
128×129 FP32 dynamic shared tile (66,048 bytes); the padding gives row/column
shared accesses a non-power-of-two leading dimension. Warp 0 handles the
64×64 diagonal factors, while threads 32–95 own the 64 rows of A10 and A11.
The fourth warp participates in tiled updates and the cooperative epilogue.

During the A00 factor, the two row warps load A10 and A11 through their
symmetric upper entries. For a fixed logical column, the 64 row owners read
contiguous values. Variants 13--19 stage these rows directly into the
padded shared tile and perform the solve in place; this is the spill-free
candidate backend added after the first B200 resource report. Variants 0--7,
9, and 10 retain register rows. Variant 8 instead uses 16-byte `cp.async` copies into an
aligned flat scratch region at the end of the shared allocation; the copy is
issued while warp 0 factors A00, then row registers are filled with a rotated
shared-memory access that avoids same-bank column loads.

The A10 solve is a right-looking row solve:

```text
for k = 0..63:
    L10[row,k] = residual[row,k] / L00[k,k]
    residual[row,k+1:] -= L10[row,k] * L00[k+1:,k]
```

L00 reciprocal diagonals are retained in the padding column, so every row
reuses one value and no division occurs in the TRSM loop. The full and compact
TRSM variants compare complete unrolling against unroll factor 8.

### Trailing-update families

- **Phase-separated SYRK:** finish all L10 rows, then apply FP32 `fmaf` rank
  updates from the shared L10 tile. Variants 13--17 use in-place shared row
  state; variants 0--5 retain one A11 row per thread in registers.
- **Interleaved rank-1:** after each L10 column is solved and published, update
  every register-resident A11 row immediately. This replaces a separate SYRK
  phase with one CTA rendezvous per L10 column.
- **Tiled SIMT:** distribute the ten lower-triangular 16×16 output tiles over
  four warps and accumulate with FP32 FMA before the L11 factor.
- **TF32 WMMA experiment:** use four warps and 16×16×8 WMMA operations with
  manually rounded TF32 inputs and FP32 accumulators. This variant is never
  eligible unless it passes every property-based residual check.

### 64×64 diagonal factors

- The **v6-derived Crout factor** assigns two interleaved rows to each lane,
  loads the symmetric input with scalar coalesced reads, broadcasts each pivot
  with warp shuffles, and accumulates with FMA.
- The **v13-derived right-looking factor** keeps the 64×64 lower state in two
  register rows per lane. Its reciprocal-root lookahead starts the next root
  before applying the remaining independent rank-1 updates.
- The **CTA-64 L11 factor** keeps the updated A11 rows in the existing row
  registers and publishes each completed factor column through shared memory.
- The **warp-tail L11 factor** uses the CTA route for columns 0–31, after which
  the second row warp owns the remaining 32×32 factor and finishes it with
  shuffle broadcasts.

The final lower factor occupies the lower part of the padded shared tile. The
baseline epilogue has all 128 threads cooperatively write aligned `float4`
vectors to global memory; the overlap variants distribute the same writes
across the factor phases. Values above the diagonal are generated as exact
zero rather than read from uninitialized shared storage.

The v6/v13 register state is now declared inside each diagonal-factor helper,
so it dies immediately after publishing the factor. The TF32 route temporarily
uses the top half of the tile as aligned WMMA scratch and then refactors A00;
this extra work avoids carrying or spilling the 128-float L00 state across
TRSM and SYRK.

## Variant table

| ID | Name | Configuration |
|---:|---|---|
| 0 | `phase_v6_precise_u8` | v6 L00/L11, register-row phase update, precise root/divide, TRSM unroll 8 |
| 1 | `phase_v6_refined_u8` | as 0, reciprocal root plus Newton refinement; provisional default |
| 2 | `phase_v6_raw_u8` | as 0, raw reciprocal root |
| 3 | `phase_v13_raw_full` | v13 L00/L11, register-row phase update, fully unrolled TRSM |
| 4 | `phase_v13_raw_u8` | as 3, TRSM unroll 8 |
| 5 | `phase_rows_raw` | v6 L00, phase update retained in registers, CTA-64 L11 |
| 6 | `rank1_rows_raw` | v6 L00, interleaved TRSM/rank-1 update, CTA-64 L11 |
| 7 | `rank1_warptail_raw` | as 6, CTA columns 0–31 then one-warp 32×32 tail |
| 8 | `async_phase_v13_raw` | asynchronous A10/A11 staging, phase update, v13 factors |
| 9 | `simt_tile_v13_raw` | register-row TRSM, four-warp 16×16 FP32-FMA update, v13 factors |
| 10 | `tf32_wmma_v13_raw` | register-row TRSM, four-warp 16×16×8 TF32 WMMA update, v13 factors; refactor A00 after scratch use |
| 11 | `full128_refined` | unblocked 128-thread right-looking control, refined root |
| 12 | `full128_raw` | unblocked control, raw root |
| 13 | `phase_shared_v6_precise_u8` | shared-row counterpart of 0 |
| 14 | `phase_shared_v6_refined_u8` | shared-row counterpart of 1; primary spill-free candidate |
| 15 | `phase_shared_v6_raw_u8` | shared-row counterpart of 2 |
| 16 | `phase_shared_v13_raw_full` | shared-row counterpart of 3 |
| 17 | `phase_shared_v13_raw_u8` | shared-row counterpart of 4 |
| 18 | `simt_shared_v13_raw` | shared-row counterpart of 9 |
| 19 | `tf32_shared_v13_raw` | shared-row counterpart of 10; experimental TF32 |
| 20 | `simt_tile_v13_raw_overlap` | v9 arithmetic; warp 3 zeros the upper-right output during L00, warps 1--3 write the completed left half during L11, then the CTA writes only L11 |
| 21 | `simt_tile_v6_raw` | v9 tiled update with lower-register v6 Crout L00/L11 factors; no output overlap |
| 22 | `simt_balanced_v13_raw` | v9 factors with balanced SYRK: two full tiles on warps 0--1 and one full plus two packed diagonal tiles on warps 2--3 |
| 23 | `simt_balanced_v13_raw_overlap` | combines 20's output schedule with 22's balanced SYRK |
| 24 | `simt_tile_v6_raw_overlap` | combines output overlap with v6 factors and the original v9 SYRK |
| 25 | `simt_balanced_v6_raw_overlap` | combines output overlap, v6 factors, and balanced packed-diagonal SYRK |
| 26 | `simt_balanced_v13_warptail` | v22 balanced SYRK; cooperative L11 (CTA cols 0--31 + warp-2 32x32 tail) reloaded from shared; no overlap |
| 27 | `simt_balanced_v13_rows` | v22 balanced SYRK; full CTA-cooperative rows L11 reloaded from shared; no overlap |

Variant IDs are stable and append-only. The Popcorn action changes only the exact
`_DEFAULT_VARIANT` marker in temporary files and never edits the tracked
solution.

## Extension and compiler policy

The native module exposes allocation and allocation-free entry points plus a
CPU metadata tensor. Metadata records registers per thread, local bytes per
thread, static and dynamic shared bytes, active CTAs and warps per SM, factor
and update families, root mode, preload mode, TRSM unrolling, and whether the
route uses tensor arithmetic.

All kernels are compiled specifically for SM100a with C++20, `-O3`, fast math,
extra device vectorization, line information, and ptxas optimization/resource
and spill reporting. Dynamic shared-memory opt-in is configured once when the
extension is loaded, before any warmup, capture, or timing. A source hash is
part of the extension name to prevent stale compiled modules after edits.

Production selection requires:

1. All six Modal input families pass validation.
2. Popcorn test mode passes.
3. The candidate has the best confirmed performance.

Registers, local memory, shared memory, and occupancy remain recorded for
profiling but are not correctness or eligibility gates.

The exact `(256,128,128)` shape uses the native extension. Every other shape
uses `torch.linalg.cholesky_ex(..., check_errors=False).L`, holding all other
leaderboard entries constant across variant submissions.

## Measurement policy

Modal tuning uses dense, planted-spectrum, diagonal, damped-low-rank,
row-scaled, and tridiagonal `(256,128,128)` inputs. Validation checks shape,
dtype, device, finiteness, exact strict-upper zeros, positive diagonal, and a
scaled reconstruction residual bounded by `max(16, 8×reference)`. The scale is
`eps(float32) × 128 × ||A||∞`.

Timing uses an eight-input/output ring, 16 warmups per variant, 200 calls per
sample, five repeats, and reversed variant order on alternating repeats. Modal
target-shape timing is the tie-breaker, not the final selection metric.

The Popcorn action submits every selected variant in every requested round,
stores raw output and result files, and ranks the median public B200 geomean.
The first run should cover all variants. The three fastest correct candidates
are then submitted for three focused rounds. The tracked default is updated
only after reviewing those results.

Nsight Systems retains the `.nsys-rep`, command, preflight resource/validation
record, stdout/stderr, and text statistics.

`--action ncu` profiles through the hosted GPU Mode Brev B200 Nsight Compute
service (popcorn-cli `--profile-brev`), because Modal does not reliably expose
hardware performance counters. It runs entirely on the local machine through the
`popcorn` CLI (no Modal GPU): each selected variant is baked into a temporary
submission via the `# POPCORN_VARIANT` marker and submitted for the b256n128
benchmark (`--benchmark-index 2`). `custom_kernel` only runs the tuned kernel at
the `(256, 128, 128)` shape, so a wrong index silently profiles the torch
fallback instead — confirm the index against the live leaderboard. The CLI
downloads `ncu-details.txt`, `ncu-details.csv`, and the `.ncu-rep` per variant
into `artifacts/ncu/<run>/v<id>_<name>/`, alongside the submitted `.py`, the
`popcorn-command.json`, the streamed `popcorn.log.txt`, and a run-level
`summary.json`. The Brev service controls the NCU version, section list, and
launch window server-side, so its metric names, units, and available sections
are not guaranteed to match the Modal pin; flag any such difference before
interpreting counters (AGENTS.md profiling policy).

`--action ncu-modal` keeps the older self-hosted Modal path: it captures one
warmed kernel with cache-controlled replay and downloads the `.ncu-rep`,
command, version, preflight data, and text/CSV detail exports into
`artifacts/ncu_modal/`. Its requested NCU sections cover compute, memory, launch,
occupancy, scheduler, warp-state, instruction, source, and hierarchical FP32
roofline data (`--sections fast` collects the launch/occupancy/SpeedOfLight
smoke set). If a profiler version lacks a requested section or metric, that
endpoint fails and records the mismatch; no equivalent metric is silently
substituted.

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

## Variant 9 NCU analysis — 2026-07-22

Capture:
`artifacts/ncu/b256_n128_20260722T073501Z/v9_simt_tile_v13_raw`.
NCU 2025.4 kernel replay with cache control reports 189.0 us; this duration is
diagnostic and is not compared directly with the warm tune median of 83.4 us.

The kernel is scheduler-starved rather than compute- or DRAM-bound:

- 12,756,736 warp instructions (49,831/matrix), 255 registers/thread, 944
  local bytes/thread, 0.86 waves/SM, and 10.83% achieved occupancy.
- Only 0.115 eligible warps/scheduler-cycle and 10.39% issue-active. FP32 is
  3.27% of the profiler-derived roof and DRAM is 1.03% of peak.
- Warp cycles per issued instruction are dominated by barrier 9.75, then
  no-instruction 2.35, wait 1.30, long scoreboard 0.88, and short scoreboard
  0.42. NCU attributes 58.7% of the average issue interval to CTA barriers.
- Executed-SASS filtering is required: the source table contains 49,640 rows
  but only 16,866 execute for v9. The two v13 factors occupy 14,914 of those
  static rows and receive 902/1,121 no-instruction samples. The TRSM/SYRK
  middle executes 8.34M/12.76M dynamic instructions, including 3.74M LDS and
  2.34M FFMA.
- True barrier samples localize to the synchronization after L11 (2,052),
  after L00 (1,449), after TRSM (635), after SYRK (115), and a minor internal
  site (31): 4,282 total. The two warp-only factors therefore cause most CTA
  idle time even though A10/A11 loading already overlaps L00.
- NCU reports 81,664 compiler-spill requests, but replacing v9's local row
  state with shared state (v18) is not a win: the same capture makes v18 20.3%
  slower, with 11.5% more instructions, 8.7% less issue activity, and much
  higher short-scoreboard pressure. Warm tuning agrees (100.8 vs 83.4 us).

Profiler-version caveat: the old `selected_metrics.py` expects several absent
`sm__...` and `...per_warp_active` names. More importantly, the old
`sass_opcode_stalls.py` hardcodes column numbers whose meanings changed in
NCU 2025.4 (its claimed long/short columns are barrier/MIO here). No substitute
metrics or those mislabeled results are used. Capture-native helpers live in
`artifacts/helpers/ncu/`.

Prioritized experiments:

1. **Overlap the epilogue with L11.** While warp 0 factors L11, warps 1--3 can
   write completed L00 and L10 regions; after the existing rendezvous, all
   warps write only the bottom-right L11 region. Also test upper-zero stores on
   otherwise-idle warp 3 during L00. This directly fills the two largest
   barrier-wait windows without changing arithmetic.
2. **Share one no-inline raw v13 factor body between L00 and L11.** The two
   inlined bodies duplicate roughly 119 KiB each and dominate instruction-fetch
   samples. A runtime input/base mode can retain one update body; measure call
   overhead, stack growth, spills, and warm timing rather than assuming a win.
3. **Add a v9 update with v6 Crout diagonal factors.** At n=64 the v6 factor is
   only about 8% slower than v13 but uses materially fewer registers. This is a
   low-complexity control for whether lower factor pressure beats v13's local
   optimum inside the larger fused kernel.
4. **Specialize diagonal SYRK tiles.** Pack the 136 lower elements of each
   diagonal 16x16 tile into five accumulators/lane and rebalance six full plus
   four diagonal tasks across warps. This removes predicated diagonal work and
   reduces shared loads, but targets a smaller stall component than 1--3.
5. **Sweep TRSM unroll 4/16/full only after the above.** Long-scoreboard delay
   is visible in TRSM, but moving the row wholesale to shared memory already
   regressed. Preserve the local-row route and use warm timing as authority.

### Barrier-focused implementation

IDs 20--25 implement experiments 1, 3, and 4 while leaving v9 unchanged as
the direct control. The overlapped epilogue divides output into three disjoint
regions: warp 3 writes exact zeros to the upper-right quadrant while L00 runs,
warps 1--3 write the complete left half while warp 0 factors L11, and all four
warps write only the lower-right quadrant after the final CTA rendezvous. This
does not remove the L11 dependency, but converts part of both large waiting
windows into useful global stores and reduces post-factor output from 4,096 to
1,024 `float4` vectors per CTA.

The balanced update assigns the six full 16x16 lower tiles as two each to
warps 0--1 and one each to warps 2--3. Warps 2--3 also receive two diagonal
tiles each; their 136 useful lower elements are packed into five accumulators
per lane instead of executing eight predicated accumulator slots. Useful
element counts are therefore 512, 512, 528, and 528 across the warps before
the post-SYRK rendezvous. IDs 21, 24, and 25 separately test whether shortening
the warp-0-only factor phases with lower register pressure outweighs v6's
slower standalone arithmetic.

## Variant 23 NCU analysis — 2026-07-22

Capture:
`artifacts/ncu/b256_n128_20260722T105901Z/v23_simt_balanced_v13_raw_overlap`
(`simt_balanced_v13_raw_overlap`, benchmark index 2). The report was collected
by the hosted Brev service with NCU **2026.2.0**; kernel-replay duration is
168.6 us / 194,392 elapsed cycles and is diagnostic only (not compared with the
warm tune median).

Tooling note: the VeloQ `ncu` path could not build its sidecar here — the
report is NCU 2026.2.0 but the only local `ncu_report` module is 2025.4.1, which
lacks the `timed_warp_samples` API VeloQ calls. No workaround was applied; the
figures below come from the local `ncu` 2025.4.1 binary + Python module
(`source_info`/`correlation_ids`, which read the newer report cleanly) and the
server-exported `ncu-details.txt`.

The kernel is **barrier-latency bound**, not compute- or DRAM-bound, and the
overlap schedule did not remove the dominant stall:

- Compute (SM) throughput 16.5%, Memory 26.6%, **DRAM 1.15%**, FP32 4% of peak.
  Issue Slots Busy 9.64% (one instruction issued every ~10 cycles); eligible
  warps/scheduler 0.11 of 16; No Eligible 90.0%.
- `barrier` is **55.4%** of the 120,181 warp-sampling stall samples (next:
  no_instructions 13.4%). Warp State attributes **61.8%** of the 17.4-cycle
  average issue interval (10.7 cycles) to CTA barriers.
- Source-correlated `pcsamp` barrier samples land on exactly three CTA
  `__syncthreads()` sites (cuda.cu line -> this repo's `.py` line): **post-L11
  factor** 819/`.py:851` = 2002 (**47.8%**), **post-L00 factor** 869/`.py:901`
  = 1390 (**33.2%**), **post-TRSM** 920/`.py:952` = 682 (**16.3%**). Top three =
  97.3%. The two warp-0-only diagonal factors alone are 81% of barrier stalls.
  This matches the v9 executed-SASS localization (after-L11 2052, after-L00
  1449, after-TRSM 635) almost exactly.
- The output overlap barely moved the largest barrier (v9 2052 -> v23 2002): the
  overlappable left-half output (2,048 `float4` across 96 threads, ~21 each)
  drains long before warp 0 finishes the 64x64 L11 factor, so warps 1--3 idle
  at the post-L11 barrier for the rest of that phase.

Two deliberate non-leads:

- **Occupancy is a false lead here.** NCU's scheduler/occupancy rules quote
  "Est. Speedup ~73%", but **Waves/SM = 0.86**: with only 256 matrices over 148
  SMs the grid cannot fill even one wave at the 2-block/SM limit (needs 296).
  Cutting registers (255) or shared memory (66 KB) to lift the per-SM block
  limit *lowers* waves further; resident-warp count is capped by the fixed
  256-block grid, not by resources. The only wall-clock lever is per-matrix
  critical-path latency.
- **Local memory is secondary.** The `float local[128]` state spills (255
  regs) and is 90.6% of L1TEX sectors, uncoalesced (1 of 32 bytes/sector; rule
  Est. ~25% each on local ld/st), but it is well-cached (L1 hit 87%) and only
  ~7% of stall cycles (long+short scoreboard). It cannot touch the 55--62%
  barrier, and moving rows to shared already regressed 20% (v18).

Prioritized experiments:

1. **Fuse the L11 factorization into the trailing update (left-looking).** As
   each balanced-SYRK column-block completes, factor the ready leftmost S11
   columns instead of hitting a barrier and idling through a separate warp-0
   factor. This is the natural extension of the interleaved TRSM/rank-1 pattern
   (v6) across the SYRK->L11 boundary and directly attacks the 47.8% post-L11
   barrier. Larger change; deferred.
2. **Cooperative L11 factor after the balanced SYRK (IDs 26--27, below).** Cheap
   test of spreading the L11 factor across warps 1--2 instead of warp 0 alone.
3. **Two matrices per CTA with decoupled per-matrix barriers.** Cross-matrix ILP
   to keep the scheduler fed during warp-0 factors; 2x shared drops to 1
   block/SM, so it must be measured. Bigger swing; deferred.

### Cooperative-L11 experiment (IDs 26--27)

IDs 26--27 implement experiment 2. Both keep v22's balanced packed-diagonal
SYRK but replace the sync-free warp-0-only `kL11Right` factor with a
CTA-cooperative rows factor over the 64 trailing rows (threads 32--95): ID 26
uses the warp-tail split (`kL11WarpTail`: CTA columns 0--31, then warp 2 owns
the 32x32 shuffle tail), ID 27 uses the full CTA rows factor (`kL11Rows`).
Because the balanced SYRK leaves S11 in **shared** memory while these factors
expect the trailing block in **registers**, a guarded `reload_trailing_from_shared`
seeds `local[kHalf..]` from the shared tile before the factor; the reload is
protected by the factor's first internal barrier and only compiles for the
tiled-update modes. Output overlap is intentionally off: a cooperative factor
keeps warps 1--2 busy, so no warp is free to overlap output — the two strategies
are mutually exclusive, which is also why `OverlapOutput` `static_assert`s
against these factors. Against v22 (`kL11Right`, no overlap) this isolates the
L11-parallelism axis. Expectation is guarded: the cooperative factor replaces
one large post-L11 barrier with per-column barriers (kL11Rows adds ~192, the
warp-tail ~96), which prior shared-route results suggest may regress; the value
is a measured point on that axis, and warm tuning is the authority.

## Results

| Date | Stage | Result |
|---|---|---|
| 2026-07-21 | Implementation | 13 variants and Modal harness added; no commands run |
| 2026-07-21 | First Modal tune | CUDA 13.1 / Torch 2.12 / B200. All numerical checks passed. All IDs rejected only by local-memory gate: v0--v10 512--720 B/thread at 255 registers; v11--v12 512 B/thread at 142 registers. Fastest medians: v10 0.1031 ms, v9 0.1049 ms, v2 0.2222 ms, v3 0.2322 ms, v4 0.2348 ms, v1 0.2384 ms. Artifact: `artifacts/tuning_20260721T112113Z.json`. TF32 remains provisional despite passing the Modal residual gate. |
| 2026-07-21 | Spill remediation | Appended shared-row IDs 13--19; factor register arrays scoped; TF32 restores L00 by recomputation. Awaiting resource/timing rerun. |
| 2026-07-21 | Second Modal tune | All 20 variants pass all numerical cases. Raw ranking: v9 0.0834 ms, v18 0.1008, v10 0.1026, v19 0.1193, v17 0.1726. v15 is the fastest zero-local route at 0.1743 ms, but resource usage no longer gates selection. Artifact: `artifacts/tuning_20260721T114143Z.json`. |
| 2026-07-21 | Selection-policy correction | Eligibility now means numerical correctness only; performance chooses among correct variants. Local-memory and register metrics are diagnostic. |
| 2026-07-22 | Barrier-focused variants | Appended IDs 20--25 for overlapped output, v6 factors, balanced packed-diagonal SYRK, and their combinations. Awaiting B200 validation/timing. |
| 2026-07-22 | v23 NCU analysis | Brev NCU 2026.2.0. v23 is barrier-bound: `barrier` 55.4% of stall samples; post-L11/L00/TRSM syncs = 97.3% of barrier samples; overlap barely moved the post-L11 wait (2052->2002). Occupancy is a false lead (0.86 waves, 256-block grid). See **Variant 23 NCU analysis**. |
| 2026-07-22 | Cooperative-L11 variants | Appended IDs 26--27 (balanced SYRK + shared-reloaded CTA-cooperative L11, warp-tail and full-rows). Read-only checks only; awaiting B200 validation/timing. |
| pending | Popcorn full sweep | Awaiting all-variant public geomeans |
| pending | Focused top-three sweep | Awaiting three-round median and production ID |

## Verification commands

These commands are intentionally documented but have not been run by the
assistant.

```bash
python3 -m py_compile \
  cholesky/b256n128/cholesky_b256n128.py \
  cholesky/b256n128/cholesky_b256n128_modal.py

git diff --check

# This must print no matches.
rg -n 'stream' cholesky/b256n128/cholesky_b256n128.py

# Compile, validate all input families, collect resources, and time all IDs.
.venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
  --action tune

# Faster focused comparison against the v9 control.
.venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
  --action tune --variants 9,20,21,22,23,24,25

# Capture timelines for representative factor/update families.
.venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
  --action nsys --variants 9,20,21,22,23,24,25

# Capture detailed counters on the hosted Brev B200 NCU service (b256n128 is
# benchmark index 2). Needs the popcorn CLI; export POPCORN_BREV_PROFILER_URL or
# rely on the built-in default. Runs locally, not on a Modal GPU.
.venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
  --action ncu --variants 9,20,21,22,23,24,25

# Fallback: the older self-hosted Modal NCU path.
.venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
  --action ncu-modal --variants 9,20,21

# Verify the provisional tracked submission through the official checker.
popcorn submit --leaderboard cholesky --gpu B200 --mode test --no-tui \
  cholesky/b256n128/cholesky_b256n128.py

# Submit all twenty-six variants once and rank their public geomeans.
.venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
  --action popcorn

# Replace IDs with the fastest correct top three from the full sweep.
.venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
  --action popcorn --variants ID1,ID2,ID3 --rounds 3

# After the confirmed winner is made the tracked default.
popcorn submit --leaderboard cholesky --gpu B200 --mode leaderboard --no-tui \
  cholesky/b256n128/cholesky_b256n128.py
```

Please provide the tuning JSON after `--action tune`, `ncu-details.txt` for
each captured variant, and the Popcorn `summary.json`. The `.nsys-rep` and
`.ncu-rep` files are only needed if GUI or source-correlated follow-up analysis
is required.
