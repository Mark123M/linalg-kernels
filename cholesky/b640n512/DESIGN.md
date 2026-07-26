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

Variants 0--8 use the original right-looking DAG. For outer tile size `NB`,
matrix `m`, and panel `p`, CTA `m` executes:

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

Variants 9--17 use a block-left-looking current-column DAG:

```text
for p:
    C_pp = A_pp
    for q < p:
        C_pp -= L_pq L_pq^T
    POTRF(C_pp)
    STORE_FINAL(L_pp)

    for i > p:
        C_ip = A_ip
        for q < p:
            C_ip -= L_iq L_pq^T
        TRSM(C_ip, L_pp)
        STORE_FINAL(L_ip)

ZERO_BLOCKS_STRICTLY_ABOVE_THE_BLOCK_DIAGONAL()
```

In variants 9--12, each `C_ip` is a distributed 4x4-per-thread register
tile. It is initialized directly from immutable input, retains every
previous-column contribution, and is published to shared memory only once.
The factor or triangular solve consumes that shared tile immediately. No
intermediate Schur complement is written to global memory.

Variants 9--12 retain two padded shared tiles rather than reserving a third
tile for `L_pp`. Their left-looking update uses both operand buffers, then
reloads the small diagonal factor into the released operand buffer while the
register accumulator is published to the other buffer.

Variants 13--16 reserve a third padded tile for the current result. The two
operand tiles remain static shared memory; the result tile uses the supported
opt-in dynamic allocation so the kernel does not exceed the static shared
limit. This allows each 64-wide product to be accumulated independently and
subtracted once, matching the stable blocked update order. The extra tile
changes the residency target from five to four CTAs per SM.

Variant 17 keeps the same three-tile allocation but holds the current
accumulator in the result tile between products. That permits the original
4x4 lane ownership with only one 4x4 product in registers. Each previous
operand pair is loaded once instead of once per 2x4 row phase.

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
- The left-looking schedule uses the same 4x4 ownership, but initializes the
  accumulator from `A_ip` and performs negative FMAs for every `q < p`
  before a single shared publication.
- The product-stable left-looking schedule splits that ownership into two
  2x4 row phases. It keeps both a current accumulator and a 64-wide product
  in registers, then subtracts the completed product once.
- The warp2 factor maps rows `2*lane` and `2*lane+1` to each lane. One warp
  performs a direct 64-column right-looking factor using shuffle broadcasts
  and one warp barrier per factor column, replacing the recursive CTA
  barriers.
- The block16 TRSM keeps 16 adjacent right-hand-side values in registers.
  Earlier solved values update all 16 independent accumulators, then the
  local triangular block is solved right-looking. This preserves two row
  warps while reducing repeated shared reads and exposing FMA ILP.
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

Only factor and solve arithmetic select a root mode. Variants other than 7
use FP32 FMA updates. Variant 7 converts update operands to TF32 and retains
FP32 accumulation; its diagonal factor and all triangular solves remain
FP32. Variant 8 is the measured default.

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

## Nsight Compute round `20260725T192553Z`

Variant 6 was captured after the five-block launch bound removed the
four-block remainder wave. Direct measurements from Nsight Compute 2026.2.0
show:

| Metric | Variant 6 |
|---|---:|
| NCU-instrumented duration | 5,121,248 ns |
| Registers/thread | 48 |
| Shared allocation/block | 34,560 B |
| Achieved occupancy | 50.59% |
| Eligible warps/scheduler | 1.155 |
| Cycles with no eligible warp | 49.71% |
| Barrier warp samples | 46.31% |
| Long-scoreboard samples | 20.30% |
| Wait samples | 9.45% |
| L1TEX throughput | 57.69% |
| FP32 FMA-heavy pipe | 20.94% |
| DRAM read/write rate | 312.3 / 318.6 GB/s |
| Shared loads | 393,011,200 |
| Excess shared wavefronts | 2% |
| Local loads/stores | 0 / 0 |

The launch has 0.865 waves per SM because 640 CTAs are fewer than the
five-block capacity of 740 CTAs. Its grid-limited occupancy ceiling is
`640*8/(148*64) = 54.05%`, so the measured 50.59% is already close to the
attainable ceiling. Lowering register count alone cannot add useful resident
warps without assigning multiple CTAs to each matrix.

Source-correlated timed samples identify three targets:

1. The diagonal-load/factor entry region contributes 851,221 barrier
   samples, 60.6% of all barrier samples. The triangular global-load
   assignment is imbalanced before a CTA barrier.
2. The trailing-update destination read contributes 364,693
   long-scoreboard samples, 59.2% of that stall class. The old update loads
   the destination only after completing its 64-step product.
3. The recursive diagonal factor and right-looking panel/update regions
   account for most remaining barrier samples. The scalar update is not
   limited by FP32 throughput or HBM bandwidth.

The profiler exposes aggregate warp sampling but no `pmsampling:*`
time-series metrics. No temporal tail or sawtooth conclusion is made.
Per-line instances of the excessive shared/global counters were also
reported as `unattributed_sass`; their small aggregate values are used
without assigning them to source lines.

## Popcorn round `20260725T205247Z`

Three benchmark rounds produced:

| Variant | Result |
|---:|---|
| 8 | Passed every row; median target mean 3,331,130.7 ns |
| 9 | Target launched but failed with NaN/Inf |
| 10 | Target launched but failed with NaN/Inf |
| 11 | No target row was emitted after the fallback-only rows |
| 12 | No target row was emitted after the fallback-only rows |

Variant 8 was promoted and is the new default. Its preload moved the largest
measured dependent global read ahead of the update FMA loop and improved the
previous approximately 4.35 ms result to 3.33 ms.

The subsequent `test` submission passed, but none of the public test shapes
has batch 640. Those rows exercise the PyTorch fallback and therefore do not
validate the specialization. Variant 8's passing benchmark row is the
target-shape correctness evidence.

Variants 9 and 10 shared both the direct rank-1 subtraction order and the
original unordered output-panel handoff, isolating the failure from the
warp2 factor. Variants 13--16 use per-block products, a separate shared
result tile, and explicit publication barriers. The missing target records
for 11 and 12 are consistent with target-entry preparation or resource
rejection; the public result did not include an exception string, so this is
an inference rather than a direct compiler diagnostic. The block8 variant
reduces that pressure without weakening the local-frame gate.

## Popcorn round `20260725T211522Z`

The one-round correctness gate produced:

| Variant | Result |
|---:|---|
| 8 | Passed; target mean 3,347,978.6 ns |
| 13 | Failed with NaN/Inf |
| 14 | Passed; target mean 5,396,915.2 ns |
| 15 | Failed with NaN/Inf |
| 16 | Failed with NaN/Inf |

Variant 14 proves that the synchronized product-stable left-looking DAG is
valid for the target input. It is nevertheless 61.2% slower than variant 8,
so it is a correctness control rather than the new optimization base.
Variants 13, 15, and 16 each replace one component of 14 and fail: recursive
factorization, block8 solve, and precise-root warp2 factorization,
respectively. This target is numerically sensitive to those arithmetic
sequences; they are not combined with further performance changes.

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
| 8 | 64 | raw | scalar | right-looking 4x4, destination preload | 256, min 5 |
| 9 | 64 | raw | scalar | left-looking current 4x4, recursive factor | 256, min 5 |
| 10 | 64 | raw | scalar | left-looking current 4x4, warp2 factor | 256, min 5 |
| 11 | 64 | raw | block16 | left-looking current 4x4, warp2 factor | 256, min 4 |
| 12 | 64 | raw | block16 | left-looking current 4x4, warp2 factor | 256, min 5 |
| 13 | 64 | raw | scalar | product-stable left-looking 2x4, recursive factor | 256, min 4 |
| 14 | 64 | raw | scalar | product-stable left-looking 2x4, warp2 factor | 256, min 4 |
| 15 | 64 | raw | block8 | product-stable left-looking 2x4, warp2 factor | 256, min 4 |
| 16 | 64 | precise | scalar | product-stable left-looking 2x4, warp2 factor | 256, min 4 |
| 17 | 64 | raw | scalar | shared-accumulator left-looking 4x4, warp2 factor | 256, min 4 |
| 18 | 64 | raw | scalar | right-looking 4x4 preload, warp2 factor | 256, min 5 |

Native preparation reports registers, static shared memory, and local
memory plus update mode, requested minimum residency, TMEM columns, schedule,
and factor mode. A candidate is rejected if the compiler creates a local
frame above 8 bytes.

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

6. Variant 8 is the isolated latency control. It loads each destination into
   its existing register accumulator before the 64-step product and applies
   negative FMAs, moving the dependent global load off the epilogue.
7. Variant 9 replaces the mutable right-looking Schur complement with a
   register-resident left-looking current tile. It reads `A_ip` once and
   stores only final `L_ip`; the original recursive factor and scalar solve
   remain as controls.
8. Variant 10 replaces the recursive `16 -> 32 -> 64` diagonal factor with
   the warp2 factor. This isolates whether removing intermediate CTA barriers
   outweighs giving the complete 64x64 diagonal factor to one warp.
9. Variants 11 and 12 add block16 TRSM. Variant 11 permits four-block
   residency; variant 12 asks ptxas to retain the five-block geometry. The
   pair measures whether extra registers improve dependency-level
   parallelism enough to offset lower residency.
10. Variants 13--16 form each 64-wide product in registers and subtract it
    once from the current tile, matching blocked POTRF update order rather
    than applying 64 sequential negative FMAs directly to `A`. A third
    shared tile retains the current result while the two source tiles are
    loaded, and explicit publication barriers establish the documented
    block-wide happens-before edge before later panel loads.
11. The product-stable mapping uses two 2x4 row phases per thread, keeping
    the accumulator and product register sets small enough for four-block
    residency. Variants 13 and 14 isolate recursive versus warp2
    factorization; variant 16 is the precise-root safety control.
12. Variant 15 reduces the register-blocked solve from 16 to 8 columns. This
    tests the same shared-read/ILP hypothesis while avoiding the local-frame
    pressure that prevented variants 11 and 12 from emitting the target
    benchmark row.
13. Variant 17 retains the current tile in shared memory between completed
    64-wide products. It loads every left/right operand pair once, halving
    the operand traffic of variant 14 while retaining its passing root,
    factor, solve, and blocked subtraction order.
14. Variant 18 changes only variant 8's diagonal factor. The passing warp2
    factor from variant 14 replaces the recursive factor, directly testing
    the barrier hotspot measured in the right-looking profile without
    changing the winning update schedule.

## User-run verification

The assistant ran Python syntax checking, `git diff --check`, and the rejected
token search. It did not compile CUDA or run any GPU, Popcorn, or Brev
command. The complete user verification sequence is:

```bash
python3 -m py_compile \
  cholesky/b640n512/cholesky_b640n512.py \
  cholesky/b640n512/cholesky_b640n512_runner.py
git diff --check
rg -n 'stream' cholesky/b640n512/cholesky_b640n512.py

python3 cholesky/b640n512/cholesky_b640n512_runner.py autotune \
  --variants all --rounds 3 --max-workers 4

python3 cholesky/b640n512/cholesky_b640n512_runner.py autotune \
  --variants 8,14,17,18 --rounds 3 --max-workers 4

POPCORN_BREV_PROFILER_URL=URL \
python3 cholesky/b640n512/cholesky_b640n512_runner.py ncu \
  --variants 8,14,17,18

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
