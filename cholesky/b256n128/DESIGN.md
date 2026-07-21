# b256n128 — Batched Cholesky, 256 × 128 × 128 FP32 on B200

## Status

- Implemented: self-contained raw CUDA extension with eighteen blocked 64+64
  variants and two unblocked controls, plus a Modal B200 tuning/profiling and
  Popcorn sweep harness. IDs 13--19 are append-only spill-remediation variants
  added after the first B200 result; the meanings of IDs 0--12 are unchanged.
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

The final lower factor occupies the lower part of the padded shared tile. All
128 threads cooperatively write aligned `float4` vectors to global memory;
values above the diagonal are generated as exact zero rather than read from
uninitialized shared storage.

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
record, stdout/stderr, and text statistics. Nsight Compute captures one warmed
kernel with cache-controlled replay and downloads the `.ncu-rep`, command,
version, preflight data, and text/CSV detail exports. Requested NCU sections
cover compute, memory, launch, occupancy, scheduler, warp-state, instruction,
source, and hierarchical FP32 roofline data. If a profiler version lacks a
requested section or metric, the endpoint fails and records the mismatch; no
equivalent metric is silently substituted.

## Results

| Date | Stage | Result |
|---|---|---|
| 2026-07-21 | Implementation | 13 variants and Modal harness added; no commands run |
| 2026-07-21 | First Modal tune | CUDA 13.1 / Torch 2.12 / B200. All numerical checks passed. All IDs rejected only by local-memory gate: v0--v10 512--720 B/thread at 255 registers; v11--v12 512 B/thread at 142 registers. Fastest medians: v10 0.1031 ms, v9 0.1049 ms, v2 0.2222 ms, v3 0.2322 ms, v4 0.2348 ms, v1 0.2384 ms. Artifact: `artifacts/tuning_20260721T112113Z.json`. TF32 remains provisional despite passing the Modal residual gate. |
| 2026-07-21 | Spill remediation | Appended shared-row IDs 13--19; factor register arrays scoped; TF32 restores L00 by recomputation. Awaiting resource/timing rerun. |
| 2026-07-21 | Second Modal tune | All 20 variants pass all numerical cases. Raw ranking: v9 0.0834 ms, v18 0.1008, v10 0.1026, v19 0.1193, v17 0.1726. v15 is the fastest zero-local route at 0.1743 ms, but resource usage no longer gates selection. Artifact: `artifacts/tuning_20260721T114143Z.json`. |
| 2026-07-21 | Selection-policy correction | Eligibility now means numerical correctness only; performance chooses among correct variants. Local-memory and register metrics are diagnostic. |
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

# Capture timelines for representative factor/update families.
.venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
  --action nsys --variants 6,8,11,14,17,18,19

# Capture detailed counters for the same representative set.
.venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
  --action ncu --variants 6,8,11,14,17,18,19

# Verify the provisional tracked submission through the official checker.
popcorn submit --leaderboard cholesky --gpu B200 --mode test --no-tui \
  cholesky/b256n128/cholesky_b256n128.py

# Submit all twenty variants once and rank their public geomeans.
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
