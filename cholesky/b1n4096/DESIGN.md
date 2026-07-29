# `b1n4096` B200 Cholesky design

## Status

The tracked default is variant 16 (`native_xpotrf_lower_fused_copy`). Its
three-round B200 median was 1.413225 ms, versus 1.532546 ms for the
contemporaneous Torch/cuSOLVER baseline. It passed the official Popcorn
property checks and cleared the required 0.995 promotion ratio.

Variants 11--15 are a second-generation redesign based directly on Algorithm
3 of ICL-UTK-987-2017. They are implemented and compile for `sm_100a`. Their
isolated leaves reconstruct correctly, but the final variant-11 NSys duration
is 6.095 ms, including 4.112 ms in 32 serial M128 leaf calls. This is
decisively slower than the approximately 1.53 ms library baseline, so none is
promoted and no version is ported to `b2n4096`.

Variant 17 began as the static 148-worker persistent experiment. Its first
Popcorn benchmark passed correctness but measured 4.765932 ms, 3.38 times
variant 16. History-load remapping reduced it to 2.314259 ms and task-wave
scheduling reached 2.245847 ms. It is now retained as the two-CTA-per-SM
296-worker occupancy experiment.

Variant 18 is the one-CTA-per-SM successor. Each CTA multiplexes eight
partially accumulated tiles in shared memory and skips blocked tiles instead
of holding the worker at the first unavailable dependency. B200 compilation,
correctness, and timing are pending; variant 16 remains the default.

The shape file is self-contained and routes only contiguous CUDA FP32 tensors
with shape `(1, 4096, 4096)` to a selected specialized variant. Every other
input retains the Torch/cuSOLVER path.

## Stable registry

| ID | Implementation |
|---:|---|
| 0 | Torch/cuSOLVER baseline |
| 1 | left-looking NB512, micro64, fused split-consumer TF32 |
| 2 | left-looking NB512, micro64, inverse-GEMM TF32 |
| 3 | left-looking NB256, micro64, fused split-consumer TF32 |
| 4 | direct left-looking 128 to 64 at remaining 1024, TF32 |
| 5 | direct left-looking 128 to 64 to 32 at remaining 1024/256, TF32 |
| 6 | direct fixed micro64, TF32 |
| 7 | direct adaptive 128/64/32, FP32 GEMMs |
| 8 | MAGMA-style hybrid, direct ATen/LAPACK CPU POTRF64 |
| 9 | MAGMA-style hybrid, compiled fixed-shape CPU POTRF64 |
| 10 | MAGMA-style hybrid, direct ATen/LAPACK CPU POTRF128 |
| 11 | NB512, recursive leaf128, fused LL IB32/LB32, cuBLAS TRSM |
| 12 | NB512, recursive leaf256, fused LL IB32/LB8, cuBLAS TRSM |
| 13 | NB256, direct leaf256, fused LL IB32/LB8, cuBLAS TRSM |
| 14 | NB512, recursive leaf128, fused LL IB32/LB32, inverse GEMM |
| 15 | variant 11 schedule with FP32 history/update GEMMs |
| 16 | native `cusolverDnXpotrf`, fused physical-triangle copy |
| 17 | static 296-worker, two-CTA-per-SM persistent FP32 wavefront |
| 18 | static 148-worker, eight-slot shared-accumulator persistent FP32 wavefront |

## Paper-faithful fused left-looking leaf

`fused_ll_potf2<M,IB,LB,Threads>` is a one-CTA, loop-inclusive leaf for
`M={64,128,256}`. The complete `i` loop remains inside one launch. At each
iteration, threads load the current lower `(M-i) x IB` panel into registers
and accumulate

```text
A[i:M, i:i+IB] -=
    A[i:M, 0:i] @ A[i:i+IB, 0:i].T
```

in `LB`-wide chunks. The current and next history fragments are
double-buffered in registers. The corresponding pivot fragment is transposed
into one of two padded shared-memory buffers. The update loop is ordered
history element first and output column second, exposing the `IB` independent
register accumulators instead of creating one `LB`-deep dependency chain per
column.

After all history chunks are accumulated, the complete updated `(M-i) x IB`
panel is staged column-major in shared memory. The POTF2 and below-diagonal
solve are fused column-by-column across the full panel: each factored column
is scaled for every active row, then those row owners update their remaining
panel entries. The finished panel is stored only after this fused solve. The
kernel never updates a future panel, which is the paper's left-looking
distinction from the old `factor_wide` rank-8 right-looking substeps.

The isolated registry contains all 72 combinations of:

- `M={64,128,256}`;
- `IB={8,16,24,32}`;
- `LB={8,16,32}`;
- 128 or 256 threads.

`IB=16`, `LB=16` was only the initial integrated setting. The Modal leaf sweep
checks every configuration independently and times an initialized leaf
launch. This is an isolated implementation diagnostic, not an official
submission acceptance gate. The Modal NCU endpoint captures exactly one
`fused_ll_potf2` launch after preflight, using the full profiler set for
instruction count, dependency stalls around FP32 square-root/divide/FMA,
registers, shared traffic, spills, and latency.

The first local CUDA 13.1 `sm_100a` build emitted all 72 configurations.
`artifacts/helpers/ncu/leaf_resource_summary.py` parsed the resulting binary:
65 configurations in the first implementation had zero stack/local storage.
Spill data is reported as a tuning signal only; it never rejects a leaf,
variant, or submission. The final integrated leaves have:

| Leaf | Threads | Registers | Shared bytes | Local bytes |
|---:|---:|---:|---:|---:|
| M128, IB32, LB32 | 128 | 146 | 24,976 | 0 |
| M256, IB32, LB8 | 256 | 75 | 35,024 | 0 |

The high M128 register count is not a rejection condition; its measured
latency won the complete sweep.

## MAGMA source cross-check

The paper notation was resolved against the implementation it describes:

- `magma/magmablas/zpotf2_kernels.cu` keeps the inner-panel loop inside one
  CTA and calls the fused leaf for each `POTF2_NB` panel.
- `magma/magmablas/zpotf2_devicesfunc.cuh` gives every row owner `rC`, `rA`,
  and `rp` register arrays. It copies `rp` to `rA`, prefetches the next block,
  then accumulates with the history loop outside the output-column loop.
- The same file writes the complete updated panel to shared memory and calls
  `zpotf2_sminout_fixsize_device`, whose column-by-column factorization also
  solves every row below the diagonal block.
- `magma/src/zpotrf_panel_native.cpp` implements the recursive
  `POTRF(A11)`, `TRSM(A21)`, `HERK(A22)`, `POTRF(A22)` schedule used here.

MAGMA is column-major. Its row owners can populate the pivot transpose from
their already coalesced register loads. A literal translation is wrong for a
contiguous Torch row-major tensor: those row-owner loads stride by `lda`.
The attempted register-derived pivot transpose regressed M128 from roughly
137 us to 148--156 us and M256 from 453 us to 514--809 us. It was reverted.
The retained row-major implementation enumerates contiguous history elements
within each pivot row for that load while keeping MAGMA's register
accumulation and whole-panel factor/solve structure.

## Recursive NB512 factorization

Variants 11, 12, 14, and 15 preserve the outer NB512 left-looking history
GEMM. The updated 512-square diagonal tile is factored recursively in
128- or 256-wide leaves:

```text
factor A00
A10 = A10 @ inv(A00.T)
A11 -= A10 @ A10.T
factor A11
```

The internal update is limited to rows inside the diagonal 512 tile. Each
completed leaf uses one cuBLAS triangular solve for the remaining rows of
that tile. Once all leaves are complete, variants 11, 12, and 15 issue one
width-512 cuBLAS solve for the entire below-panel region. This replaces the
eight width-64 below-panel solves in variant 1. Variant 13 applies the same
structure directly at NB256. Variant 14 constructs the completed width-512
inverse with cuBLAS TRSM, applies it with one TF32 GEMM, and copies the
contiguous result back.

Variant 15 is the numerical control: both the outer history GEMM and recursive
leaf-to-leaf updates use FP32 compute. POTF2 and cuBLAS TRSM remain FP32 in
all five variants.

## GPU-native algorithms

Variants 1–3 use an outer left-looking panel. One large history GEMM updates
each outer panel. Subsequent micro panels use inner history GEMMs. The
micro-factor kernel loads a 32-, 64-, or 128-wide lower tile into shared
memory, performs a positive-diagonal POTRF, and builds the inverse used by the
below-panel solve.

The fused variants launch one producer CTA and co-resident consumer CTAs.
Release/acquire flags publish the inverse after factorization. Consumers
preload the below-panel tile while the producer works, then apply the inverse.
Variant 1 splits each 64-column consumer job over two CTAs. Variant 2 retains
separate inverse GEMM and copy-back kernels as a launch/throughput control.

Variants 4–7 are direct left-looking recurrences. Every step performs one
full-history GEMM, then a shape-specialized POTRF/solve. The width policies
are explicit in the registry. Variant 7 is the numerical control: it changes
all history GEMMs from fast TF32 to FP32 computation while keeping the same
adaptive factor/solve sequence.

All matrix offsets and work counts that can exceed 32-bit products use
64-bit indexing. The output copy zeros the full upper triangle before staged
updates. Subsequent kernels only write the diagonal and below-diagonal
panels.

## Torch/cuSOLVER reverse engineering

The B200 variant-0 timeline is retained at
`artifacts/nsys/b1_n4096_20260729T074025Z/v0_torch_cusolver/`. PyTorch
2.12.0+cu130 dispatches batch-one Cholesky to `cusolverDnXpotrf`; the public
PyTorch implementation calls the looped `Xpotrf` path when the batch count is
one and `potrfBatched` when it is greater than one.

The actual factorization is one proprietary CUDA kernel:

```text
kernel<getrf_wo_pivot_params_<
    float,0,256,1,64,64,68,8,1,1>>
```

Its NSys duration is 1.411 ms. It launches 2,080 CTAs of 256 threads.
`4096/64 = 64` and `64*65/2 = 2080`, so the grid contains exactly one CTA for
every tile in a 64-by-64 lower-triangular tile matrix. It uses 200 registers
per thread and 52,224 dynamic shared bytes. The latter is exactly
`3*64*68*sizeof(float)`, identifying three padded 64-by-68 FP32 tile buffers.

The targeted full NCU capture is retained at
`artifacts/ncu/b1_n4096_torch_ncu_20260729T074311Z/v0_torch_cusolver/`.
The exact FP32 kernel has 6,377 static SASS instructions, including 1,354
`FFMA`, 1,112 `LDS`, 381 `STS`, and 34 asynchronous
global-to-shared `LDGSTS` instructions. Tensor-pipe utilization and all
reported TF32 tensor-operation counts are zero; this algorithm uses scalar
FP32 CUDA-core arithmetic, not Tensor Cores.

The SASS contains a global acquire loop:

```text
LDG.E.STRONG.SYS
YIELD
ISETP.GE
BRA
```

and completion publication sequences with `MEMBAR.SC.GPU`, cache
invalidation, and a global flag store. Combined with the triangular CTA grid,
three shared tiles, and long unrolled FFMA regions, this supports the
following algorithmic reconstruction:

1. Map each CTA to one lower-triangular 64-square output tile.
2. Wait until the tile's diagonal/history predecessors have been published.
3. Load the target and two predecessor tiles, accumulate the complete
   left-looking history for that output tile, and keep the tile CTA-local.
4. Run the diagonal POTF2 or off-diagonal triangular solve in the same CTA.
5. Store the completed tile and publish its dependency flag.

This is an inference from launch geometry and proprietary SASS, not published
cuSOLVER source. It is nevertheless qualitatively different from variants
11--15: cuSOLVER exposes the complete tiled dependency graph in one launch,
whereas the custom schedule serializes 32 leaf solvers and surrounds them
with separate library operations.

NCU measures 1.418 ms for the replayed factor kernel, 12.46% achieved
occupancy, 0.419 issued instructions per scheduler cycle, 0.658 eligible
warps per scheduler cycle, 35.61% SM throughput, and 44.16% L1/TEX
throughput. Register allocation limits residency to one CTA per SM, but the
grid still supplies 14.05 waves per SM. No local-memory or spill requests are
measured. These resource facts describe the retained solver; none is used as
an acceptance or rejection gate.

The expected direct executed-FP32-FMA metric
`smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` is absent in Nsight
Compute 2026.2. The report exposes only derived FFMA aliases and per-cycle
rates, so no derived value is substituted as a direct dynamic instruction
count. Direct total executed instructions are available as
`smsp__inst_executed.sum = 578,603,350`. Source correlation is unavailable
because the proprietary cubin has no line information; SASS disassembly is
available.

The most promising next candidate is therefore not another serial panel
variant. It is a thin native `cusolverDnXpotrf` wrapper:

- copy the immutable input into a solver-owned output while zeroing the
  physically unused triangle in the same kernel;
- invoke the same batch-one Xpotrf kernel directly;
- return a transposed-stride view so the physical column-major factor is the
  required logical lower factor;
- for `b2n4096`, compare two looped Xpotrf calls against PyTorch's much slower
  batch-greater-than-one `potrfBatched` dispatch.

For batch one this can remove Torch's separate approximately 58 us triangle
cleanup while retaining the 1.411 ms factor kernel. For batch two, the
PyTorch dispatch distinction creates much larger potential headroom and must
be verified with its own NSys trace before implementation.

## Native Xpotrf variant

Variant 16 is the promoted thin wrapper around `cusolverDnXpotrf`. The input
is symmetric, so the copy kernel writes its physical upper triangle into a
column-major-strided output and zeros the physical lower triangle. The solver
then factors that allocation with `CUBLAS_FILL_MODE_LOWER`. Returning the
same allocation with strides `(4096*4096, 1, 4096)` exposes the physical
column-major factor as the required logical row-major lower triangle.

`prepare` creates the Xpotrf handle and parameters, queries both workspaces,
and preallocates device info and workspace storage before measurement. The
measured path is one vectorized 512-CTA copy followed by one Xpotrf call; it
does not run Torch's later triangle-cleanup operation.

The promotion report is
`artifacts/tuning/b1n4096_20260729T083054Z/summary.json`:

| Variant | Three-round median target mean |
|---:|---:|
| 0 | 1.532546 ms |
| 16 | 1.413225 ms |

Variant 16 passed the official property checker and improved the baseline by
7.79%, so it became the tracked default.

## Static persistent 64-square wavefront

Variant 17 assigns the complete 2,080-tile lower triangle to 148 persistent
workers. The column-major task number is

```text
task(i,j) = j*(129-j)/2 + (i-j),  0 <= j <= i < 64.
```

Every history tile `(i,k)`, pivot tile `(j,k)`, and diagonal tile `(j,j)`
needed by `(i,j)` has a smaller task number. The initial table pinned the
first 148 tasks one per worker, processed remaining tasks in decreasing task
order, assigned weight `j+1` to the least-loaded worker, and finally sorted
each worker list. Consequently a worker could not wait on one of its own
later tasks and the inter-worker wait graph followed strictly decreasing task
numbers. Its estimated worker weights were 308--311.

The kernel uses one cooperative grid of 148 CTAs, 256 threads per CTA, and
120 KiB dynamic shared memory. Runtime checks require CC 10.0, exactly 148
SMs, cooperative-launch support, and one measured resident CTA per SM. Each
worker loads its target tile, immediately consumes every published history
pair, factors a diagonal tile or solves an off-diagonal tile, stores it, and
release-publishes its completion. Tile owners also zero the corresponding
upper output, so no cleanup launch follows. The only preceding operation is
an 8,320-byte flag reset.

The first Popcorn round passed and measured:

| Variant | Target mean |
|---:|---:|
| 0 | 1.533019 ms |
| 16 | 1.411205 ms |
| 17 | 4.765932 ms |

NSys at
`artifacts/nsys/b1_n4096_20260729T100829Z/` confirms one 0.928 us flag reset
and one 4.757400 ms persistent factorization kernel, with no cuBLAS or
cuSOLVER work in the factorization. The reset and host synchronization are
negligible; the kernel itself is the regression.

The full NCU report is
`artifacts/ncu/b1_n4096_variant_ncu_20260729T101047Z/`. The reproducible
comparison helper is
`artifacts/helpers/ncu/compare_persistent_v17.py`. Key measurements against
the retained cuSOLVER tile-wavefront report are:

| Metric | Initial v17 | cuSOLVER |
|---|---:|---:|
| NCU duration | 4.764896 ms | 1.417856 ms |
| theoretical occupancy | 12.50% | 12.50% |
| issued instructions / scheduler cycle | 0.1263 | 0.4193 |
| eligible warps / scheduler cycle | 0.1746 | 0.6578 |
| elapsed FP32 FMA-pipe utilization | 6.80% | 22.86% |
| shared-load bank conflicts | 628,914,688 | 521,216 |
| local loads / stores | 682,752 / 2,368 | 0 / 0 |
| maximum SM active-cycle excess | 9.16% | 14.11% |

The schedule is therefore not the principal regression: its measured SM
imbalance is smaller than cuSOLVER's. The initial history mapping causes an
average 4.3-way conflict over 190,450,240 shared-load requests; 76.23% of
shared-load wavefronts are excessive. DRAM read utilization is only 0.094%,
while L1/TEX reaches 60.95%, so this is on-chip addressing/issue pressure,
not HBM bandwidth. Warp sampling attributes 2,471,750 of 4,911,341 samples
(50.3%) to LG throttle, concentrated on the two history loads. Only 6.3% of
samples are selected/issuing.

SASS explains both effects. The fully unrolled history update contains 512
generic `LD.E` instructions rather than statically addressed `LDS`
instructions. Their `0x110` row offsets equal `68*sizeof(float)`. Separately,
the dynamically indexed `float* history[2][2]` table creates the 32-byte
local frame and its measured local traffic. The old 16-by-16 thread-block
mapping also makes consecutive four-column register blocks alias banks when
the shared leading dimension is 68. This is a measured arithmetic/load
mapping defect, not a reason to reject the kernel for spilling.

The revised source keeps the schedule unchanged and makes one targeted
microkernel correction:

- eliminate the dynamic pointer table and derive each of the four history
  addresses directly from one shared base;
- map each warp to four row blocks by eight column blocks, covering the
  complete 16-by-16 grid of 4-by-4 thread tiles across eight warps;
- stage each 16-byte history chunk at
  `physical_group = logical_group XOR (row >> 2)`;
- invert that swizzle at use sites and issue explicit `ld.shared.f32`
  instructions from precomputed 32-bit shared addresses.

For every history `k`, this mapping gives one distinct shared bank per
distinct address for both operands; repeated addresses within a warp are
broadcasts. The diagonal tile remains in the unswizzled layout expected by
the POTRF64/TRSM control.

The revised B200 NSys capture is
`artifacts/nsys/b1_n4096_20260729T104741Z/`. It passes the dense preflight
with the same 0.001912 scaled reconstruction residual. The persistent kernel
now takes 2.314259 ms, a 51.35% improvement over the initial 4.757400 ms
capture. Compiled resources fell from 246 to 168 registers per thread and
from 32 to zero reported local bytes. The flag reset is 1.344 us, so the
remaining gap is still entirely inside the factorization kernel. At
approximately 1.64 times the 1.411 ms variant-16 result, variant 17 remains
well outside the promotion gate. A new NCU capture is required to determine
whether the remaining cost is arithmetic issue rate, shared traffic, or
dependency waiting; NSys alone cannot distinguish them. Register and
local-memory results remain diagnostics and never serve as acceptance gates.

The corresponding revised NCU report is
`artifacts/ncu/b1_n4096_variant_ncu_20260729T105405Z/`. It confirms that the
load-mapping repair addressed the original kernel defect:

| Metric | Revised v17 | cuSOLVER |
|---|---:|---:|
| NCU duration | 2.316384 ms | 1.417856 ms |
| registers per thread | 168 | 200 |
| local loads / stores | 0 / 0 | 0 / 0 |
| issued instructions / scheduler cycle | 0.2293 | 0.4193 |
| eligible warps / scheduler cycle | 0.3510 | 0.6578 |
| elapsed FP32 FMA-pipe utilization | 14.12% | 22.86% |
| shared-load bank conflicts | 2,718,208 | 521,216 |
| shared-store bank conflicts | 3,051,037 | 260,300 |

The revised SASS contains 1,322 static `FFMA.FTZ` and 525 explicit `LDS`
instructions, with no local-memory instructions. The remaining gap is
primarily a scheduling defect. VeloQ reports 1,165,873 barrier-state samples,
52.45% of all 2,222,692 samples. Source correlation attributes 806,980
samples to the history-dependency wait before staging the next published
tile pair and another 178,795 to the diagonal-dependency wait before TRSM.
Together these actual publication waits account for 84.55% of barrier-state
samples. The nonblocking readiness-check rendezvous accounts for only 94,216
samples. Thus ordinary CTA synchronization is not the dominant observation:
workers are arriving at unavailable DAG nodes and holding their fixed worker
slot. The measured maximum SM active-cycle excess also rose to 14.95%, nearly
the cuSOLVER result, despite the table's nearly equal arithmetic weights.

`artifacts/helpers/ncu/simulate_persistent_schedule.py` reproduces this
head-of-line effect with a phase-level model that lets each target consume a
history pair as soon as that pair is published. Across history-heavy,
balanced, and leaf-heavy cost assumptions, the prescribed weight-balanced
table has a modeled makespan of 422.6--883.6 units. Plain task-ID
round-robin takes 363.6--852.6 units and reduces accumulated dependency wait
by 1.10--7.13 times. A dependency-aware greedy assignment does not improve
the modeled critical-path makespan beyond round-robin and is much more
sensitive to assumed leaf costs.

The next v17 table therefore uses `owner(task) = task mod 148`. This preserves
the first wave, strict increasing task order within each worker, and the
acyclic proof while keeping consecutive global task waves aligned across
workers. It deliberately accepts wider estimated arithmetic weights,
281--345, to reduce publication head-of-line blocking. This is a profiler-led
schedule correction; it does not alter the repaired history microkernel.

NSys at `artifacts/nsys/b1_n4096_20260729T110614Z/` measures this task-wave
table:

| Schedule | Persistent kernel |
|---|---:|
| weight-balanced static table | 2.314259 ms |
| task-ID round-robin | 2.245847 ms |

Round-robin improves the kernel by 68.412 us, or 2.96%, while retaining the
same 168 registers, zero local bytes, 120 KiB dynamic shared memory, and
148-by-256 cooperative launch. The dense preflight again passes with a
0.001912 scaled reconstruction residual. The trace contains exactly one
8,320-byte, 1.216 us flag reset and one factorization kernel; there are no
library factorization or update calls. The schedule correction is therefore
real but far smaller than the simple dependency model predicted. At 1.637
times the 1.371975 ms Xpotrf solver kernel in the retained variant-16 NSys
capture, v17 is still not promotable. A round-robin NCU capture is needed to
measure how much publication waiting actually changed and to separate the
remaining dependency floor from shared-tile and POTRF/TRSM work.

The round-robin NCU capture is
`artifacts/ncu/b1_n4096_variant_ncu_20260729T111249Z/`:

| Metric | Weight-balanced | Round-robin | Change |
|---|---:|---:|---:|
| NCU duration | 2.316384 ms | 2.250720 ms | -2.84% |
| issued instructions / scheduler cycle | 0.2293 | 0.2377 | +3.67% |
| eligible warps / scheduler cycle | 0.3510 | 0.3641 | +3.72% |
| elapsed FP32 FMA-pipe utilization | 14.12% | 14.51% | +2.77% |
| all barrier-state samples | 1,165,873 | 1,089,008 | -6.59% |
| history-publication barrier samples | 806,980 | 604,405 | -25.10% |
| diagonal-publication barrier samples | 178,795 | 303,279 | +69.62% |

The two publication waits total 907,684 samples under round-robin versus
985,775 under the weight-balanced table, a 7.92% reduction. Round-robin
therefore does align history waves better, but it delays some diagonal owners
and moves much of the saved wait to the TRSM diagonal dependency. The
nonblocking readiness-check rendezvous is unchanged at 94,518 samples.
Publication waiting still represents 83.35% of all barrier-state samples and
42.47% of all timed warp samples. SM active-cycle imbalance also remains
material at +15.64%/-10.16%. The static persistent schedule is approaching a
dependency-wait floor rather than an arithmetic-load-balance optimum.
A dedicated diagonal worker was rejected in the schedule model: diagonal
tiles still perform all `j` history updates, so that worker carries weight
2,080 versus 271--345 elsewhere and raises the modeled makespan by
2.6--5.8 times rather than accelerating the diagonal chain.

The actual tile work still has an independent measured defect. NCU reports
only 6.562 useful bytes per 32-byte sector for global loads and 26,075,136
excessive sectors, 33% of the total. This comes from loading each thread's
aligned 4-by-4 register tile with sixteen scalar operations while a warp is
split across four matrix rows. An experiment loaded each local row as one
aligned `float4`, reducing sixteen scalar target loads per thread to four
vector loads without changing ownership or arithmetic.

The experiment's NCU report is
`artifacts/ncu/b1_n4096_variant_ncu_20260729T112837Z/`. SASS confirms the
intended change from sixteen `LDG.E` instructions to four `LDG.E.128`
instructions. Dynamic global-load instructions fell from 2,657,218 to
2,432,752, global-load sectors fell from 56,360,309 to 53,172,909, and useful
bytes per sector improved from 6.562 to 12.691. This traffic reduction had no
performance value: NCU duration changed only from 2.250720 to 2.249600 ms,
while eligible warps fell 1.92%, publication-wait samples rose 1.80%, and
short-scoreboard samples rose 7.22%. The user's NSys capture measured
approximately 2.37 ms versus the preceding 2.245847 ms. Because the intended
memory effect is verified but produces no repeatable latency gain, the
`float4` target-load experiment is rejected and the scalar loads are
restored. This also demonstrates that the NCU source rule's theoretical
8.25% sector opportunity was not on the critical path.

The aggregate shared-store counter remains 3,051,025 conflicts over
1,411,200 requests, with a 3.2-way average conflict. VeloQ correctly rejects
the aggregate bank-conflict metrics as `not-a-source-counter`. The derived
global and shared excessive-transaction metrics exist, but all 40 global and
955 shared SASS instances are unattributed to source lines in this NCU 2026.2
report despite line information. No source-line substitution is made; the
target-load attribution above is an inference from its exact warp mapping and
the otherwise coalesced output stores. Shared-layout changes are deferred
until they can address a measured critical path rather than aggregate
transaction counts alone.

Nsight Compute 2026.2 again does not expose the requested direct metric
`smsp__sass_thread_inst_executed_op_ffma_pred_on.sum`. It exposes derived
`*_x2` aliases and per-cycle FFMA rates; none is substituted as a direct
executed-instruction count. All other metrics used in the comparison helper
were present under their expected names.

### Two-resident-CTA experiment

The next variant-17 experiment increases the cooperative grid from 148 to
296 workers, exactly two 256-thread CTAs per each of the target B200's 148
SMs. CUDA does not define an SM assignment order for ordinary blocks, so the
kernel still uses a cooperative launch and requires the occupancy API to
report exactly two active blocks per SM before it may run. This doubles the
resident warps from 8 to 16 per SM and raises theoretical warp occupancy from
12.5% to 25%. The goal is to let a second worker execute when the first is
polling an unpublished tile.

Dynamic shared memory falls from 120 KiB to 96 KiB per CTA. The kernel's
actual shared layout occupies 88,836 bytes, so this does not change its
algorithm or staging buffers; two blocks consume approximately 194 KiB
including static allocations, below the CC 10.0 limit. Registers are the
binding constraint. The previous 168 registers per thread cannot admit two
256-thread blocks in a 65,536-register SM, so the kernel now declares
`__launch_bounds__(256, 2)`. This asks ptxas to cap allocation at 128
registers per thread. Any resulting local-memory traffic is recorded as a
performance diagnostic and does not reject the variant.

With 296 workers, the original reverse-greedy static algorithm assigns the
first 296 task IDs one per worker, assigns the remaining tasks in descending
task order to the least-loaded worker using weight `j+1`, and then sorts each
worker's task list. Every dependency still has a smaller task ID, so no CTA
can wait on its own later work. The resulting estimated weights are 154--159
and each CTA owns 7--8 tiles. The variant remains ID 17 and variant 16
remains the default until official correctness and the promotion timing gate
pass.

### Eight-slot shared-accumulator scheduler

Variant 18 returns to one cooperative 256-thread CTA per each of 148 B200
SMs, but exposes eight logical tile workers inside every CTA. The static
148-worker table uses the original reverse-greedy weighted assignment, has
14--15 tasks per CTA, and retains estimated arithmetic weights of 308--311.
Every per-worker list is sorted by the existing column-major task ID, which
is exactly lexicographic `(j,i)` order.

The dynamic shared-memory request is 208 KiB. Its fixed layout is:

| Storage | Bytes |
|---|---:|
| Two pairs of swizzled 64-by-68 history tiles | 69,632 |
| Eight padded 64-by-65 FP32 accumulators | 133,120 |
| Inverse diagonal and POTRF panel | 2,560 |
| Slot state and scheduler control | 180 |
| Total used | 205,492 |

The request selects the SM100 228 KiB carveout while leaving explicit
alignment headroom. Runtime preparation requires CC 10.0, 148 SMs,
cooperative-launch support, sufficient per-block opt-in shared memory, and
exactly one active block per SM from the occupancy API.

Each slot stores a task ID, `(i,j)`, `next_k`, a fresh/parked state, and its
partial accumulator. `next_k` is the first history column not yet applied:

```text
T(i,j) = A(i,j) - sum over k < next_k of L(i,k) @ L(j,k).T
```

Thread zero scans all eight slots and selects the smallest column-major task
whose next dependency pair is published. Blocked slots are skipped. The CTA
loads the selected accumulator into the existing 4-by-4-per-thread register
mapping, applies every consecutively ready `k` update using the existing
double-buffered history microkernel, and keeps the accumulator in registers
through that burst. It writes the tile back to its padded shared slot only
when the next dependency is absent. Completed history on a diagonal tile
runs POTRF64 immediately; an off-diagonal tile becomes selectable when
`L(j,j)` is published, then runs TRSM64. Only final `L(i,j)` is
release-published.

The eight-slot prefix cannot deadlock despite each CTA owning more than
eight tasks. Completed slots immediately admit the next task in that
worker's sorted list. Consider the globally smallest unfinished task: all
earlier same-worker tasks have completed, so it must be in the active prefix;
all of its dependencies have smaller task IDs, so they are complete.
Therefore at least one active task is ready until all 2,080 tasks finish.
Static ownership also guarantees one writer per accumulator, so no tile
claim atomic or global accumulator is required.

Column-major priority is the initial policy requested for variant 18.
Row-frontier `(i,j)` priority remains a separate future comparison rather
than being mixed into this experiment. Register spills and local-memory
traffic remain profiler diagnostics, never correctness rejection criteria.

## Hybrid CPU–GPU recurrence

Variants 8–10 follow the lower-path recurrence in
`magma/src/zpotrf_gpu.cpp`:

1. Update the diagonal tile on the GPU.
2. Copy it asynchronously to a preallocated pinned CPU tensor and record a
   readiness event.
3. Enqueue the independent below-diagonal history GEMM.
4. Wait only for the panel event and run CPU POTRF while that GEMM executes.
5. Enqueue the factored tile copy after the history GEMM, then run GPU TRSM.

All CUDA work is ordered on the existing default execution queue. No
secondary queue is created. Variant 9 exposes private native copy, stage, and
finish helpers to Python. Its fixed `(1,64,64)` CPU specialization is warmed
with `torch.compile(fullgraph=True, dynamic=False, mode="max-autotune")`
before timing. The compiled result is copied into a pinned factor buffer
before the GPU upload.

`CHOLESKY_PROFILE_NVTX=1` enables panel-level ranges for diagonal update,
device-to-host transfer, history update, panel wait, CPU POTRF,
host-to-device transfer, and GPU solve. They are disabled in production.

## Compilation

The extension targets `sm_100a` and uses:

- host: `-O3`, native ISA tuning, FMA, fast/unsafe math, C++20;
- device: `-O3`, fast math, extra device vectorization, restricted pointers,
  line information, expensive ptxas optimization, spill diagnostics;
- cuBLAS: `cublasGemmEx` with explicit TF32 or FP32 compute modes.

The local CUDA 13.1 compilation completed. `cuobjdump --dump-resource-usage`
reported `LOCAL:0` for all emitted production kernels (copy, factor,
fused 128/64/32, inverse copy-back, and wedge cleanup). This is a compilation
check only; the local machine cannot execute `sm_100a`. The initial
persistent kernel was compiled later by CUDA 13.3 on B200 and reported 246
registers, 120 KiB dynamic shared memory, and 32 local bytes. Its spills are
recorded as diagnostic evidence only. The revised swizzled history mapping
compiled on the same B200 with 168 registers, 120 KiB dynamic shared memory,
and zero reported local bytes.

## CPU alternatives

Direct ATen/LAPACK is retained because local CPU measurements showed roughly
9.6 microseconds for one 64-wide panel. A tensorized, rank-one unrolled
implementation was rejected: its compiled `(2,64,64)` measurement was about
0.67 ms, far too slow for a pipelined panel. The compiled variant therefore
wraps the fixed-shape LAPACK operation rather than shipping the rejected
tensorized algorithm.

## Validation and promotion record

The historical exact-shape B200 sweep covered variants 0--10 on dense, spectrum,
diagonal, low-rank, row-scaled, and tridiagonal inputs. The retained report is
`artifacts/validation/b1_n4096_20260728T142024Z/results.json`.

- Variants 0 and 7 passed all 66 case/variant checks.
- The TF32 paths passed the target dense case but became non-finite on the
  planted low-rank stress input; inverse-GEMM variant 2 also failed the
  planted-spectrum case.
- Hybrid variants 8–10 passed dense but their CPU panel POTRF rejected the
  low-rank stress recurrence after the TF32 history update.
- The first remote pass exposed upper-triangle values left by direct diagonal
  GEMMs. A final 128-wide wedge cleanup fixed that defect in both shapes.

The full-registry screen is in
`artifacts/tuning/b1n4096_20260728T142400Z/summary.json`. The fastest native
candidate was variant 1 at 2.683 ms; the library baseline was 1.530 ms.
Variant 1 is excluded on speed alone — it is 1.75x slower than the baseline,
so the stress-case failure never becomes the deciding factor. Every native
path lost to cuSOLVER at this shape: a single 4096-square matrix does not
fill the B200, so the per-panel launch recurrence pays overhead the library
does not.

The promotion run is in
`artifacts/tuning/b1n4096_20260728T142720Z/summary.json`. Across three
forward/reverse alternating rounds:

| Variant | Median target mean |
|---:|---:|
| 0 | 1.529 ms |
| 7 | 4.081 ms |

The required threshold was 1.521 ms, so the runner recorded
`retained_default_below_required_gain` and left variant 0 selected.

Artifacts are retained under:

- `artifacts/validation/` for dense, spectrum, diagonal, low-rank,
  row-scaled, and tridiagonal checks;
- `artifacts/tuning/` for the full screen and alternating Popcorn results;
- `artifacts/nsys/` for native and hybrid timelines;
- `artifacts/ncu/` for leading native kernel reports;
- `artifacts/helpers/ncu/` for report-analysis helpers.

The runner commands are:

```bash
python cholesky/b1n4096/cholesky_b1n4096_runner.py autotune \
  --variants all --rounds 3
python cholesky/b1n4096/cholesky_b1n4096_runner.py ncu --variants 1,4
.venv/bin/modal run cholesky/b1n4096/cholesky_b1n4096_modal.py --variant 1
.venv/bin/modal run cholesky/b1n4096/cholesky_b1n4096_modal.py --validate
python cholesky/b1n4096/cholesky_b1n4096_runner.py submit --mode test
python cholesky/b1n4096/cholesky_b1n4096_runner.py submit --mode leaderboard
```

The isolated NCU captures are under
`artifacts/ncu/b1_n4096_leaf_ncu_20260729T061054Z/`. Direct measurements for
the selected configurations are:

| Leaf | Duration | Registers | Active occupancy | Executed instructions |
|---:|---:|---:|---:|---:|
| M64, IB8, LB16, t128 | 56.416 us | 80 | 6.228% | 27,412 |
| M128, IB8, LB16, t128 | 167.456 us | 79 | 6.248% | 115,518 |
| M256, IB16, LB16, t256 | 591.296 us | 94 | 12.502% | 611,088 |

The canonical B200 stall metrics were available as
`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`. The requested
raw executed FP32 FMA metric
`smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` was absent in this
Nsight Compute version. It exposed only derived aliases such as
`derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2`; those aliases
are not reported as direct executed-instruction counts. Direct total
instruction count instead uses `smsp__inst_executed.sum`.

The target SASS contains one reciprocal and one reciprocal-square-root MUFU
site plus two slow-path calls. Its static instruction image is 1,714
instructions for the M128 winner and 3,207 for M256. The dominant static
arithmetic/data-movement groups are 298/647 FP32 FMA instructions and
298/597 shared loads, respectively. This evidence points to the serialized
panel arithmetic and data movement; CTA barrier management is not treated as
the optimization objective.

If NCU attributes more than 30% of target time to GEMMs and does not identify
an SM100 Tensor Core path in the kernel/SASS evidence, the next candidate is a
CUTLASS 3.x SM100 TF32 TMA/TMEM GEMM. It is not added speculatively.

The paper-redesign commands are:

```bash
.venv/bin/modal run cholesky/b1n4096/cholesky_b1n4096_modal.py \
  --panel-configs all
.venv/bin/modal run cholesky/b1n4096/cholesky_b1n4096_modal.py \
  --panel-ncu-configs 32,57
.venv/bin/modal run cholesky/b1n4096/cholesky_b1n4096_modal.py \
  --ncu-variant 0
.venv/bin/modal run cholesky/b1n4096/cholesky_b1n4096_modal.py \
  --validate
python cholesky/b1n4096/cholesky_b1n4096_runner.py autotune \
  --variants 0,1,11,12,13,14,15 --rounds 3
```

The B200 leaf sweep at
`artifacts/tuning/b1_n4096_leaf_20260729T061015Z/leaf-sweep.json` checked
all 72 configurations against isolated FP32 reconstruction. The median
winners were M64 IB8/LB16/128 threads at
58.304 us, M128 IB8/LB16/128 threads at 168.928 us, and M256
IB16/LB16/256 threads at 625.600 us. The integrated M128 variants were
updated to the measured IB8/LB16 winner; the M256 variants already used the
measured winner.

After correcting the pivot-history mapping to use contiguous global row
loads followed by the intended shared-memory transpose, the selected
remeasure is retained at
`artifacts/tuning/b1_n4096_leaf_20260729T061839Z/leaf-sweep.json`. All three
leaves still reconstruct correctly. Medians were 58.592 us for M64,
169.184 us for M128, and 598.240 us for M256. The change materially helped
M256 relative to the first sweep but did not explain the M128 panel cost.

The six-family Modal full-shape report at
`artifacts/validation/b1_n4096_20260729T061923Z/results.json` is diagnostic
only. It labels variants 11--13 non-finite on its locally constructed
low-rank stress case and variant 14 non-finite on its spectrum and low-rank
cases; variant 15 passes all six. These labels never reject a submission.
Only an official Popcorn check decides submission correctness, including
when it disagrees with the local diagnostic.

Variant 16 is now the promotion reference. Variant 17 or 18 may replace it
only after official correctness and a three-round median at or below
`0.995 x` its contemporaneous variant-16 median. No persistent
implementation is ported to `b2n4096` before that gate is met.
