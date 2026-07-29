# B200 batched Cholesky: `(60, 1024, 1024)`

## Contract

`cholesky_b60n1024.py` specializes exactly one input:

- CUDA
- contiguous
- `torch.float32`
- shape `(60, 1024, 1024)`

Every other input is dispatched to:

```python
torch.linalg.cholesky_ex(data, check_errors=False).L
```

The extension never aliases or modifies the input. It returns a separate
lower-triangular FP32 allocation.

The submitted Python source must not contain the evaluator-rejected token
documented in the repository instructions. The runner checks every rendered
candidate before submitting it.

## Algorithm

The production candidates are staged right-looking factorizations. A matrix
is partitioned into eight `128 x 128` outer blocks, with `64 x 64` microtiles:

```text
copy
for p = 0..7:
    POTRF128(p,p)
    TRSM64x128(rows below p)
    update(remaining lower microtiles)
optional final upper-zero
```

Kernel completion publishes each stage to the next stage. A custom-update
candidate therefore has 23 launches:

```text
1 copy + 8 factor + 7 solve + 7 update
```

The two cuBLAS controls have 24 operations because they copy both input
triangles and zero the strict upper triangle after the last factor.

This follows the structure used by MAGMA's batched POTRF panel code:
factor a diagonal block recursively, solve the panel below it, then update
the trailing matrix. The block size is deliberately B200-specific rather
than copied from a general-GPU MAGMA tuning table.

## Node strategies

### Copy

Sixteen CTAs cooperate on each matrix. Custom-update candidates copy only the
lower triangle and write zero above it. cuBLAS candidates copy the complete
matrix because each library update writes a full trailing square.

### `POTRF128`

The factor is recursively split `128 -> 64 -> 32`.

- A single warp performs each right-looking `POTF2-32`.
- Four- or eight-lane row groups perform the triangular solves between
  recursive pieces.
- The `32 x 32` links use FP32 FMA updates.
- The `64 x 64` root link uses either FP32 FMA or TCGen05 TF32 into FP32
  TMEM.
- Precise mode uses round-to-nearest square root and divide.
- Refined mode uses reciprocal square root, one Newton correction, a
  corrected diagonal, and a final precise reciprocal.

The diagonal tile is padded to 129 floats per row in dynamic shared memory.

### `TRSM64x128`

One CTA owns one `64 x 128` row tile. The solve proceeds in dependency order
over the 128 columns. Four-lane groups use 256 threads and eight-lane groups
use 512 threads.

This is equivalent to two 64-column halves, each containing two 32-column
recursive pieces. Keeping one CTA responsible for the complete 128 columns
avoids an additional publication launch between halves.

The profile-driven implementation treats those halves as four explicit
32-column register blocks:

- the complete `64 x 128` right-hand side is loaded once into shared memory;
- the panel leading dimension is `128 + subgroup_width`, so the four- or
  eight-lane row groups map onto distinct banks;
- only the 32 diagonal rows needed by the active recursive block are staged;
- each lane retains its four or eight values from the active block in
  scalarized registers;
- solved values are subgroup-broadcast and written back to the shared panel
  only after the 32-column block is complete.

The sub4 solve therefore uses 50,304 bytes of dynamic shared memory and the
sub8 solve uses 51,328 bytes, down from the original 102,400-byte launch
budget. The active-block register array has compile-time indices so ptxas can
scalarize it. A local frame is recorded but is not a static rejection
criterion: the autotuner measures the compiled kernel, and NCU spill traffic
determines whether a frame is costly in practice.

### FP32 trailing update

One 256-thread CTA owns a lower-triangular `64 x 64` destination microtile.
It:

1. loads two padded `64 x 128` panels into shared memory;
2. preloads sixteen destination values per thread into a `4 x 4` register
   microtile;
3. accumulates the rank-128 product with FP32 FMAs;
4. stores only valid lower-triangular values on a diagonal tile.

The preload overlaps destination latency with panel consumption and removes
a destination read from the epilogue.

### TCGen05 trailing update

The rank-128 product is issued as two independent rank-64 TF32 stages.
Each stage is packed into K-major shared memory and consumed by ordered
`M64N64K8` instructions. The path uses:

- asynchronous-proxy shared-memory fences;
- ordered TCGen05 MMA issue;
- `tcgen05.commit` to an mbarrier;
- parity waits;
- `tcgen05.fence::after_thread_sync` before TMEM loads;
- a four-warp FP32 TMEM epilogue;
- explicit TMEM allocation, deallocation, and permit relinquish.

`outer` mode uses this path only for global trailing updates. `all` mode also
uses it for the `64 x 64` link inside `POTRF128`.

### cuBLAS controls

Variants 5 and 6 replace each custom update launch with one
`cublasGemmStridedBatchedEx` over all 60 matrices:

```text
C_tail[b] = C_tail[b] - L_panel[b] * L_panel[b]^T
```

The row-major allocation is treated as its column-major transpose:

- `transa = T`
- `transb = N`
- `lda = ldb = ldc = 1024`
- matrix stride `1024 * 1024`
- batch count `60`

Variant 5 selects `CUBLAS_COMPUTE_32F`; variant 6 selects
`CUBLAS_COMPUTE_32F_FAST_TF32`. The PyTorch-owned current cuBLAS handle is
used, and the extension links `-lcublas`.

These controls perform approximately twice the necessary update arithmetic
because they write full trailing squares. They test whether seven large
library calls outperform the lower-triangular custom tile grids.

## Cluster-DAG control

Variant 7 is the only persistent cluster candidate:

- 8 CTAs per cluster;
- 16 clusters;
- matrix indices `cluster_id + 16 * local_index`;
- at most four matrices per cluster;
- 372 compute nodes plus cooperative copy, or 373 total nodes per matrix.

The 372 compute nodes are:

```text
8 POTRF + 56 TRSM + 308 update
```

Rank-zero DSM contains four independent state arrays. Cluster-scope acquire
loads and release stores publish dependencies. A CAS claims a ready node,
and the completed-node counter uses a cluster-scope atomic add. CTAs scan
from staggered cursors and use bounded nanosleep backoff when no ready node
is immediately available.

The control intentionally has no phase-barrier alternative. It exists to
measure whether fine-grained update/solve overlap can recover the scheduling
overhead that performed poorly for the earlier `(16, 512, 512)` experiment.

## Focused variants

| ID | scheduler | root / row group | update |
|---:|---|---|---|
| 0 | staged | precise / sub4, 256 | custom FP32 |
| 1 | staged | refined / sub8, 512 | custom FP32 |
| 2 | staged | precise / sub4, 256 | TCGen05 outer |
| 3 | staged | precise / sub4, 256 | TCGen05 all |
| 4 | staged | refined / sub8, 512 | TCGen05 all |
| 5 | staged | precise / sub4, 256 | cuBLAS FP32 full tail |
| 6 | staged | precise / sub4, 256 | cuBLAS fast-TF32 full tail |
| 7 | cluster DAG | refined / sub8, 512 | TCGen05 all |

Variant 6 is the tracked default selected by the completed baseline
autotuning run. The autotuner remains the only mechanism that changes it
after a new sweep.

## Native API and resource policy

The extension exposes:

- `prepare(variant)`
- `run(data, variant)`
- `run_out(data, out, variant)`
- `metadata()`

Preparation opts each kernel into the smallest role-specific dynamic
shared-memory budget used by the implementation (about 49--108 KiB), requests the
maximum shared-memory carveout, and queries compiled resources. A variant is
rejected when it cannot be configured or launched, not merely because ptxas
allocated a local frame. There is no substitution for a failing variant.

Metadata records threads, registers, local memory, static and dynamic shared
memory, per-kernel active-block estimates, scheduler, root mode, row-group
width, arithmetic mode, inner tensor use, tile sizes, launch count, cluster
size, node count, and TMEM columns.

The build retains ptxas spill warnings. Static local bytes are diagnostic
metadata rather than a performance verdict: acceptance is based on
correctness and measured endpoint time, while follow-up NCU inspection uses
executed local load/store traffic and its stall contribution when a winning
candidate spills.

Compilation targets only `sm_100a` and uses:

```text
-O3 -DNDEBUG -std=c++20
--use_fast_math --extra-device-vectorization --restrict -lineinfo
-Xptxas=-O3,-v,-warn-spills
```

## Autotuning and profiling

`cholesky_b60n1024_runner.py` provides three actions.

### `autotune`

- renders each selected variant as a standalone submission;
- performs three alternating forward/reverse rounds by default;
- uses Popcorn `--mode benchmark` and the supported `--output` path;
- fetches the authenticated result object;
- requires all public rows to pass;
- parses row 7 `mean`, `err`, `best`, and `worst` as raw-nanosecond
  `Decimal` values;
- ranks by median `mean`, median `best`, then variant ID;
- atomically promotes the winner only if the tracked source and SHA-256 are
  unchanged.

### `ncu`

The profiling action uses only Popcorn's hosted Brev profiler at benchmark
index 7. It retains candidate source, command JSON, stdout/stderr, raw JSON,
downloaded CSV/text details, archives, and `.ncu-rep` files.

If B200 profiler versions rename or omit metrics, analysis must record the
expected and actual metric names, units, normalization, and whether any
conclusion is inferred. Metrics must never be silently substituted.

### `submit`

The submit action accepts `test`, `benchmark`, and `leaderboard`.

## Profile-driven optimization: 2026-07-25

The completed baseline sweep selected variant 6 at a median mean of
`3,107,934.1383541333 ns`. Its hosted B200 NCU report captured the first ten
of 24 operations: copy, and factor/solve/GEMM for panels zero through two.
Within that captured window:

| role | time | share |
|---|---:|---:|
| solve | 1,572,800 ns | 56.7% |
| factor | 786,048 ns | 28.3% |
| cuBLAS TF32 GEMM | 302,464 ns | 10.9% |
| copy | 114,656 ns | 4.1% |

The first solve had 53.4% short-scoreboard samples, 54.7% aggregate
shared-load bank conflicts, 84.3% L1/shared throughput, less than 1% DRAM
utilization, and a 102,400-byte dynamic shared-memory allocation. This is the
evidence for the register-block and width-matched padding changes above.

The expected B200 aggregate shared-bank metric names were present, but the
NCU 2026.2/VeloQ source query reported them as not being source counters.
Consequently the aggregate conflict ratio and line-level short-scoreboard
sampling are direct measurements; attribution to the old 129-float panel
stride is an address-mapping inference corroborated by SASS.

The factor kernels remain the second optimization target. Their 60-CTA grid
underfills 148 SMs, and 57.2% of sampled stalls were at barriers while a
single warp performed each `POTF2-32`. A multi-CTA recursive factor is not
folded into this iteration because it changes publication and correctness
semantics. It should be a separately tuned candidate after the solve change
has passed.

The focused eight-variant contract is intentionally retained. A
sub8-plus-cuBLAS-TF32 candidate would require either removing one requested
control or expanding the sweep; it is not silently substituted for an
existing variant.

## Validation status

The previous variant-6 implementation passed the baseline autotune and
produced the NCU evidence above. The register-block solve is a new,
unvalidated source revision. Required evidence, in order:

1. CUDA compilation reports resources and spill warnings for every variant;
2. the public test submission passes through the library fallback;
3. target benchmark row 7 passes for all eight variants in every tuning
   round;
4. the promoted winner is reprofiled, checking solve allocation, occupancy,
   shared-load conflicts, short-scoreboard stalls, executed local-memory
   traffic, and elapsed time.

## Nsight Systems endpoint

The shape-local Modal launcher profiles one warmed factorization on B200:

```bash
.venv/bin/python -m modal run cholesky/b60n1024/cholesky_b60n1024_modal.py
# Optional comparison:
.venv/bin/python -m modal run cholesky/b60n1024/cholesky_b60n1024_modal.py --variant 6
```

`--variant -1` selects the tracked default, now variant 9. The former
variant-6 staged cuBLAS-TF32 schedule has 24 algorithm launches per
factorization. Input
generation, extension compilation, preparation, warmup, and correctness
validation are outside the capture; the NVTX range contains exactly one
out-parameter factorization.

Artifacts are downloaded under `artifacts/nsys/`. `profile.nsys-rep` is the
forward-compatible UI/VeloQ input, `kernel-trace.csv` is the ordered GPU
timeline, `kernel-exec-trace.csv` separates API, queue, and execution time,
and `kernel-summary.csv` aggregates duration by kernel name. The SQLite
export, human-readable statistics, command, profiler version, environment,
preflight, stdout, and stderr are retained with the report.

## 2026-07-28 trailing-size adaptation

VeloQ measured each variant-6 128-wide POTRF at approximately 146-150 us
while the corresponding solves fell from roughly 136 us toward 42 us.
The report contains timeline/runtime, synchronization, kernel, and NVTX
records but no GPU metrics, so no kernel-internal cause is inferred.

Variants 8 and 9 append precise width-specialized stages:

| ID | Width schedule for remaining `R` | POTRF 128/64/32 | TRSM 128/64/32 |
|---:|---|---:|---:|
| 8 | 128 when `R > 512`, otherwise 64 | 4 / 8 / 0 | 4 / 7 / 0 |
| 9 | 128 when `R > 512`, 64 when `R > 128`, otherwise 32 | 4 / 6 / 4 | 4 / 6 / 3 |

The 64- and 32-wide factors reuse POTF2-32, with the 64-wide path adding
one local solve/update and a second direct factor. Matching compile-time
solve kernels use 64- or 32-row tiles. Large trailing matrices continue
through handle-based TF32 strided-batched GEMM. Once the post-panel
remainder is at most 128, rank-specialized custom kernels update only the
lower 64x64 or 32x32 tiles.

Each cutover is width-aligned. Panel solves complete before their trailing
update, diagonal update tasks suppress strict-upper stores, and the final
upper-zero kernel remains authoritative. Thus the adaptive schedule changes
only the factorization partition, not the block Cholesky dependencies.

Both variants passed the Modal B200 preflight; their scaled reconstruction
residuals were `0.658807` and `0.658804`. VeloQ confirmed the requested
width ordering. In ID 9, factor medians fell from 149.440 us to 54.400 us
and 22.463 us, solve medians from 118.271 us to 25.440 us and 10.624 us,
and custom lower-update medians from 22.016 us to 6.816 us. These are
64%, 59%, and 69% reductions at the relevant transitions.

| ID | Kernel trace span | Delta from ID 6 | Autotune median target mean |
|---:|---:|---:|---:|
| 6 | 2.417 ms | baseline | 2.245 ms |
| 8 | 2.210 ms | -8.6% | 2.201 ms |
| 9 | 2.203 ms | -8.9% | 2.200 ms |

All public rows and the target row passed in all three alternating rounds.
ID 9 improved the authoritative target median by 2.0%, passed the 0.5%
gate, and is now the tracked default. Official test submission 920853
passed all 17 cases. The final capture measured factor medians
149.503/54.367/22.464 us, solve medians
119.200/25.472/10.624 us, custom lower-update medians 21.984/6.816 us,
and a 2.206 ms kernel span, 8.7% below the old default. Static validation
and the case-insensitive rejected-token scan passed.

## Cutlass-name clone experiment

A cutlass-named clone has been added for the current tracked default variant.
The public base variant is `9` and the cloned public variant is `10`.
The clone compiles identical CUDA algorithm source after renaming every custom
`__global__` kernel entry point and matching launch/configuration reference to
use a `cutlass_` prefix.

The 2026-07-28 runner autotune retained variant 9. Median mean time was
2.200835 ms for variant 10 versus 2.201686 ms for variant 9, which was faster
but below the 0.5% promotion gate.

## Persistent CPU--GPU lookahead experiment

Variants 11, 12, and 13 are append-only hybrid candidates; variant 9 remains
the tracked default. The 2026-07-29 variant-9 Systems trace has a 2.207 ms
kernel span with no kernel overlap. Factor work totals 1.017 ms, including
four approximately 150 us width-128 factors. Those launches contain only 60
blocks for 148 SMs, so a partial CPU split cannot remove their single GPU
wave. All three hybrid candidates therefore send all 60 width-128 diagonal
tiles to the CPU and retain GPU-only width-64 and width-32 tails.

One 256-thread cooperative kernel owns the complete factorization. Its
296-block grid is constrained to two resident blocks on each B200 SM. It
reuses the precise shape-local factor and solve routines, the TF32 Tensor
Core rank-128 update, and lower-only FP32 tail updates. The hybrid update
uses warp-level `mma.sync.m16n8k8` with TF32 inputs and FP32 accumulators;
it deliberately avoids TCGen05 because that instruction class reserves an
SM-wide virtual resource and limits a kernel entry to one resident CTA.
After each panel
solve, the three 64x64 tiles forming the next 128x128 diagonal are updated
first and packed directly into mapped pinned storage. The CPU begins POTRF
after that generation is published while the resident grid processes all
other lower trailing tiles. The CPU factor is consumed only after the
trailing work and CPU generation have both completed.

Ready, completed, abort, and final generations occupy separate cache lines.
The GPU uses system-scope PTX acquire/release loads and stores; the CPU uses
matching C++ acquire/release operations. No atomic read-modify-write is used.
The wait binding releases the GIL, has a 30-second failure bound, and the
abort generation lets every cooperative block leave the scheduler before
exit. Panel and factor buffers are cacheable pinned tensors of shape
`(60,128,128)`.

Variant 11 compiles fixed-shape `cholesky_ex`; variant 12 compiles a
four-step blocked-32 recurrence containing batched POTRF, triangular solve,
and matrix-product updates. Both use full-graph static Inductor
`max-autotune`, warm before timing, and use at most 60 CPUs from the process
affinity. Profiling emits separate CPU wait, POTRF, and publish NVTX ranges
for every outer panel.

Variant 13 keeps exactly the same persistent GPU work and mapped-memory
protocol, but replaces the opaque batched PyTorch POTRF with a native outer
batch loop. The extension is built with OpenMP enabled, so
`at::parallel_for(0, 60, 1)` assigns independent matrices to the PyTorch CPU
team. Each worker sets its MKL-local thread count to one, copies one packed
lower panel directly into the mapped factor buffer, and invokes the exported
single-matrix FP32 LAPACK Cholesky routine. The input is row-major and only
its lower triangle is populated, so it is reinterpreted as column-major and
factored with `uplo=U`; the resulting row-major lower factor is precisely the
triangle consumed by the GPU. The binding releases the GIL and returns the
first failing one-based batch index. Its NVTX role is `mkl_outer`.

This third backend is deliberately a native comparison rather than another
Inductor graph. Inspection of the generated variant-11 graph showed that
`torch.compile` delegates the whole operation to the opaque
`aten.linalg_cholesky_ex` CPU operator. That operator iterates over the batch
internally, so compilation does not create matrix-level parallelism. Variants
11 and 12 remain the warmed `torch.compile` controls required to distinguish
the scheduling change from the GPU pipeline.

The first variant-11 Modal preflight on 2026-07-29 rejected the kernel
before launch: the 124-register binary admitted only one resident block per
SM instead of the required two. A kernel-local `__maxnreg__(120)` ceiling
reduced the compiled entry to 119 registers per thread with zero stack and
zero local memory, but the repeated preflight still reported one. Cubin
metadata then identified the actual limiter as
`EIATTR_TCGEN05_1CTA_USED`, not registers. The persistent update was changed
to warp-level TF32 MMA, removing that entry attribute. The resulting binary
uses 96 registers per thread with zero stack and zero local memory. Register
allocation admits two blocks, while 76 KiB dynamic shared memory plus the
per-block reservation admits two but not three, so the static residency is
exactly two. Both native and cutlass-renamed extensions compile with this
revision.

The repeated variant-11 B200 run succeeded and confirmed two resident blocks
per SM, but regressed to a 28.319 ms persistent-kernel span. Its four CPU
POTRF ranges measured 6.197, 6.022, 5.837, and 5.461 ms, totaling 23.516 ms;
CPU wait ranges added 2.624 ms. This identifies the compiled CPU solver, not
the mapped-memory handshake, as the dominant limiter. The GPU and CPU ranges
did overlap, correctness passed with a scaled residual of `0.6594 < 16`, and
the trace contained one persistent launch rather than a chain of small
launches.

The native backend passed a local fixed-shape numerical check with zero
failure status, zero upper triangle for the packed input convention, and a
maximum relative reconstruction residual below `2.0e-7`. Local compilation
also confirmed the OpenMP outer path and one-thread MKL control entry. No
B200 timing claim is made for variant 13 yet. Its required next evidence is a
variant-13 Modal preflight and Systems report showing all four `mkl_outer`
ranges and their overlap with the persistent kernel. Promotion still
requires every public row to pass and at least a 0.5% median-mean gain over
variant 9.
