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
scalarize it; the normal resource check still rejects the specialization if
it creates a local frame larger than eight bytes.

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
maximum shared-memory carveout, queries compiled resources, and rejects a
kernel with a local frame larger than eight bytes. There is no substitution
if a variant spills or cannot launch.

Metadata records threads, registers, local memory, static and dynamic shared
memory, per-kernel active-block estimates, scheduler, root mode, row-group
width, arithmetic mode, inner tensor use, tile sizes, launch count, cluster
size, node count, and TMEM columns.

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

1. CUDA compilation reports zero disallowed local-memory frames;
2. the public test submission passes through the library fallback;
3. target benchmark row 7 passes for all eight variants in every tuning
   round;
4. the promoted winner is reprofiled, checking solve allocation, occupancy,
   shared-load conflicts, short-scoreboard stalls, and elapsed time.
