# B200 `(640, 512, 512)` Cholesky

## Contract

`cholesky_b640n512.py` specializes contiguous CUDA FP32 input with exact
shape `(640, 512, 512)`. Every other shape continues through
`torch.linalg.cholesky_ex(data, check_errors=False).L`.

The custom path returns a separate output tensor, reads only the input lower
triangle, writes exact zeroes to the strict upper triangle, and factors every
matrix independently. It targets `sm_100a` without a runtime CUTLASS,
MathDx, or cuSolver dependency.

## Why this differs from B16

The B16 implementation uses a 23-launch staged graph to expose enough
matrix-tile parallelism. B640 already supplies 640 independent matrices.
That is enough CTA-level work to hide the serial diagonal path without
splitting each matrix across launches.

The B640 path therefore launches exactly one CTA per matrix and fuses the
complete static Cholesky graph into that CTA. This is a software execution
strategy rather than a special hardware scheduler.

## Fused static DAG

For outer tile size `NB`, matrix `m`, and panel `p`, CTA `m` executes:

```text
COPY_LOWER_AND_ZERO_UPPER(m)

for p:
    LOAD_DIAGONAL(m, p)
    POTRF(m, p)
    STORE_DIAGONAL(m, p)

    for i > p:
        LOAD_PANEL(m, i, p)
        TRSM(m, i, p)
        STORE_PANEL(m, i, p)

    for i > p:
        LOAD_AND_RETAIN_LEFT_PANEL(m, i, p)
        for p < j <= i:
            LOAD_RIGHT_PANEL_UNLESS_DIAGONAL(m, j, p)
            UPDATE(m, i, j, p)
```

All dependencies are CTA-local. CTA barriers separate genuine shared-tile
producer and consumer phases. The barrier between the completed panel-store
loop and the first update load was removed after profiling: both operations
assign every panel element to the same thread, so per-thread global ordering
plus the following shared producer/consumer barrier is sufficient. No
inter-CTA communication, atomics, clusters, or device-side work queue are
required.

## Tile strategies

### `NB=64`

- 128 or 256 threads.
- Two padded `64 x 65` FP32 shared tiles plus 64 reciprocals: 33,536 bytes.
- Recursive `16 -> 32 -> 64` diagonal factorization.
- Four-lane row groups for factor-internal `TRSM16/32`.
- Outer panel solve uses either one owner thread per row or a four-lane row
  group.
- The trailing update uses a 4x4 register microtile per lane.
- A solved left panel remains in shared memory while the CTA updates every
  destination in that tile row.
- The TCGen05 variant reuses the same two shared allocations as K-major TF32
  operand buffers after the panel solve. It allocates 64 TMEM columns once,
  issues eight `M64N64K8` products per right-looking update, accumulates in
  FP32 TMEM, and uses four warps for the TMEM/global epilogue.

### `NB=32`

- 128 threads, four warps.
- Two padded `32 x 33` FP32 shared tiles plus 32 reciprocals: 8,576 bytes.
- Recursive `POTRF16`, `TRSM16`, rank-16 update, `POTRF16`.
- Scalar outer panel solve.
- The trailing update uses a 2x4 register microtile per lane.

This control follows the scale of NVIDIA's
`CUDALibrarySamples/MathDx/cuSolverDx/10_Advanced/blocked_potrf.cu`,
which demonstrates one CTA per `N=512` matrix with `NB=32`, 128 threads,
and roughly 400 batches. The implementation here remains custom and
right-looking.

## Root modes

- `raw`: hardware reciprocal square root and `diagonal = value * inverse`.
- `Newton`: one reciprocal-root refinement before forming the diagonal.
- `precise`: round-to-nearest square root and division.

Only factor and solve arithmetic select a root mode. Variants 0--6 use FP32
FMA trailing updates. Variant 7 converts update operands to TF32 and retains
FP32 accumulation; its diagonal factor and all triangular solves remain
FP32. Variant 0 is the measured default.

## Nsight Compute round `20260725T081712Z`

The report set contains variants 0, 1, and 3. All direct numbers below come
from Nsight Compute 2026.2.0 reports under
`artifacts/ncu/b640n512_20260725T081712Z`; the 7 ms capture duration is
instrumented and is used only for comparisons. The corresponding three-round
Popcorn benchmark ranked variant 0 first at 4,352,872.7 ns.

### Direct measurements

| Metric | v0 raw/scalar | v1 Newton/scalar | v3 raw/sub4 |
|---|---:|---:|---:|
| NCU duration | 7,057,920 ns | 7,102,048 ns | 7,575,232 ns |
| Registers/thread | 64 | 60 | 64 |
| Static + driver shared/block | 34,560 B | 34,560 B | 34,560 B |
| Waves/SM | 1.081 | 1.081 | 1.081 |
| Achieved occupancy | 41.96% | 41.97% | 42.18% |
| Eligible warps/scheduler | 0.893 | 0.840 | 1.236 |
| Issue instructions/cycle/scheduler | 0.435 | 0.435 | 0.503 |
| FMA-heavy pipe, elapsed peak | 19.18% | 17.66% | 19.81% |
| Tensor pipe, elapsed peak | 0% | 0% | 0% |
| L1TEX throughput, elapsed peak | 41.76% | 41.41% | 58.88% |
| Shared loads | 393,011,200 | 393,011,200 | 423,976,960 |
| Excess shared wavefronts | 9,912,320 | 9,912,320 | 204,308,480 |
| Register spills | 0 | 0 | 0 |

Variant 0 is neither DRAM-bandwidth nor FP32-pipe limited. DRAM read and
write rates are 209.9 and 223.6 GB/s, only 2.74% and 2.91% of the respective
reported peaks. L1TEX is busier because the scalar update executes
393 million shared loads, but even its elapsed throughput is only 41.76%.
The tensor pipe is completely unused. Global coalescing is already close to
ideal at 31.49/32 load bytes and 31.05/32 store bytes per sector; NCU assigns
less than 1.3% combined speedup to those access-pattern rules. Branch
efficiency is 98.53%, L2 hit rate is 48.40%, and there are no local or shared
spills.

VeloQ's timed warp-state data contains 3,514,313 samples for variant 0:
barrier 45.84%, long scoreboard 20.46%, wait 9.98%, short scoreboard 5.40%,
and selected 6.92%. The hottest single PC is the post-panel barrier with
980,916 barrier samples. The update destination read contributes the largest
long-scoreboard group, while the scalar FMA update contributes the largest
short-scoreboard group.

Variant 3 is the useful negative control. Four-lane TRSM reduces barrier
samples from 45.84% to 18.19%, but short-scoreboard rises to 18.61%,
MIO-throttle to 4.49%, shared loads rise 7.9%, and excessive shared
wavefronts rise 20.6x. Its NCU duration regresses 7.3%. The subgroup solve is
therefore not the next base strategy.

The 640-block launch is register-limited to four blocks per SM. On 148 SMs
that gives a 592-block full wave and a 48-block remainder; NCU reports 1.081
waves/SM and estimates a 50% uniform-block tail. The measured per-SM active
cycles range from 17.06% below to 25.42% above the mean. Without a PM
time-series, attributing all of that imbalance to the final wave would be an
inference, but the launch geometry alone makes residency a first-order target
rather than a cosmetic occupancy change.

### Profiler evidence gaps

The profiling policy expected time-series metrics such as
`pmsampling:sm__throughput.avg.pct_of_peak_sustained_elapsed` and
`pmsampling:smsp__warps_issue_stalled_barrier.avg`. The actual reports expose
aggregate `warpsampling:smsp__pcsamp_*` values and PM configuration
(2 us maximum interval), but no `pmsampling:*` metric instances. No timeline
shape or per-interval tail claim is made.

The aggregate counters
`derived__memory_l1_wavefronts_shared_excessive` (wavefronts) and
`derived__memory_l2_theoretical_sectors_global_excessive` (sectors) are
present, but VeloQ reports their instances as `unattributed_sass`; only their
aggregate values are interpreted. Timed warp samples and source/SASS
correlation are available normally, so barrier and scoreboard hotspot
locations are direct measurements.

## Variant registry

| ID | Outer tile | Root | Outer solve | Update | Threads |
|---:|---:|---|---|---|---:|
| 0 | 64 | raw | scalar | 4x4 | 256 |
| 1 | 64 | Newton | scalar | 4x4 | 256 |
| 2 | 64 | precise | scalar | 4x4 | 256 |
| 3 | 64 | raw | sub4 | 4x4 | 256 |
| 4 | 32 | raw | scalar | 2x4 | 128 |
| 5 | 32 | Newton | scalar | 2x4 | 128 |
| 6 | 64 | raw | scalar | 4x4, FP32 | 256, min 5 blocks/SM |
| 7 | 64 | raw | scalar | TCGen05 TF32/FP32 | 128, min 6 blocks/SM |

Native preparation reports registers, static shared memory, and local
memory plus update mode, requested minimum residency, and TMEM columns. A
candidate is rejected if the compiler creates a local frame above 8 bytes.

## Implemented optimization hypotheses

1. All variants remove the redundant post-panel barrier identified by the
   980,916-sample hotspot.
2. Variant 6 adds a five-block launch bound. If ptxas keeps it spill-free,
   640 CTAs fit within the 740-CTA resident capacity instead of entering a
   48-CTA second wave.
3. Variant 7 uses 128 threads and 33.5 KiB shared memory, targeting six
   blocks per SM and an 888-CTA resident capacity. A 64-column TMEM
   allocation does not reduce that target.
4. Variant 7 replaces the 4x4 scalar update and its 393 million shared-load
   instruction footprint with TCGen05. The left operand is packed once per
   trailing tile row and retained while right operands change.
5. Tensor completion is observed by one lane in each epilogue warp followed
   by a warp barrier. This avoids a CTA-wide completion barrier before every
   TMEM epilogue; a CTA barrier remains before operand-buffer reuse.

Variant 7 deliberately changes only the right-looking products. Keeping
POTRF and TRSM in FP32 isolates tensor-update accuracy and performance from
the serial diagonal path.

## User-run verification

These commands have not been run by the assistant:

```bash
python3 -m py_compile \
  cholesky/b640n512/cholesky_b640n512.py \
  cholesky/b640n512/cholesky_b640n512_runner.py
git diff --check
rg -n 'stream' cholesky/b640n512/cholesky_b640n512.py

python3 cholesky/b640n512/cholesky_b640n512_runner.py autotune \
  --variants all --rounds 3 --max-workers 4

python3 cholesky/b640n512/cholesky_b640n512_runner.py autotune \
  --variants 6,7 --rounds 3 --max-workers 2

POPCORN_BREV_PROFILER_URL=URL \
python3 cholesky/b640n512/cholesky_b640n512_runner.py ncu \
  --variants 0,6,7

popcorn submit \
  --leaderboard cholesky \
  --gpu B200 \
  --mode test \
  --no-tui \
  cholesky/b640n512/cholesky_b640n512.py

popcorn submit \
  --leaderboard cholesky \
  --gpu B200 \
  --mode benchmark \
  --no-tui \
  --output /tmp/cholesky-b640n512-result.json \
  cholesky/b640n512/cholesky_b640n512.py
```
