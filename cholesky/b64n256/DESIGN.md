# B200 batched Cholesky: `B=64`, `N=256`

## Scope and dispatch

`cholesky_b64n256.py` owns one specialization: a contiguous CUDA FP32 tensor
with shape `(64, 256, 256)`. `custom_kernel` calls the raw extension only for
that exact shape. Every other shape returns
`torch.linalg.cholesky_ex(data, check_errors=False).L`.

The production ID is variant 18. Variant IDs are append-only. The Popcorn
sweep may change only the integer on the tracked `# POPCORN_VARIANT` line
after its promotion gates pass.

Variants 18--21 are isolated descendants of the profiled 512-thread variant
11. Variants 22--28 are append-only experiments derived from the fused
left-looking panel design in `references/icl-utk-987-2017.pdf`. Variant 18
remains unchanged as their timing and correctness control.

Variants 29--38 are append-only descendants of the B200-winning variant 25.
They test the synchronization, register residency, thread-block geometry, and
mixed outer/inner blocking ideas in `references/imple_tuning_batch.pdf` and
`references/fast-batch-2014.pdf`. Variant 25 is their unchanged control.

## Matrix layout

The single-CTA implementation retains only the lower 2-by-2 block layout:

```text
             128 columns       128 columns
          +-----------------+-----------------+
128 rows  | A00, ld=129     | not retained    |
          +-----------------+-----------------+
128 rows  | A10, ld=128     | A11, ld=129     |
          +-----------------+-----------------+
```

`A00` and `A11` include one padding column. The three tiles occupy 49,408
floats, or 197,632 bytes. Tensor variants add a 64-by-64 conversion area and
barrier/allocation words for 214,032 dynamic bytes. Both are below B200's
227-KiB per-CTA opt-in limit.

Variant 18 changes only `A10` to `ld=129` and moves `A11` by 128 floats. Its
tensor specialization therefore uses 214,544 dynamic bytes. All `A10`
staging, solve, output, and indirect factor/update accesses use the selected
compile-time layout; variants 0--17 retain `ld=128`.

The cluster implementation splits the rows of each 128-row tile between its
two CTAs. Each rank owns 64 rows of `A00`, `A10`, and `A11`. Its base storage
is 98,816 bytes. Tensor variants reserve a 96-KiB panel area plus
barrier/allocation words, for 197,136 bytes per CTA. The larger area holds
one 64-by-128 A panel and up to four independently packed 32-by-128 B panels.
It is deliberately greater than half of B200's per-SM shared capacity, so the
two cluster ranks cannot reside on the same SM.

Every output element is first set to zero. Only the computed lower triangle is
subsequently written, making strict-upper zeros exact rather than a side
effect of an input copy.

Variant 29 and its descendants defer all three factor-block writes until the
factorization is complete. They then traverse the complete lower triangle
once and write zero for every upper-triangular position. This replaces the
baseline's 65,536-element zero pass plus 32,896 factor stores with one
65,536-element final pass, while removing the two intermediate output loops
and their CTA barriers. The first warp relinquishes its no-longer-used TMEM
permit before final writeback, so the deferred path also needs no terminal
CTA barrier.

## Factorization

The outer factorization is:

```text
L00 = POTRF128(A00)
L10 = TRSM-right(A10, L00)
A11 = A11 - L10 L10^T
L11 = POTRF128(A11)
```

The default `POTRF128` recursively performs two 64-row factors. Each 64-row
factor recursively performs two cooperative `POTF2-32` factors separated by
a 32-row TRSM and lower-triangular update. Variant 4 stops at cooperative
`POTF2-64`; variant 5 changes its rank-one work mapping. The diagonal
factorization remains FP32 in every variant.

The scalar TRSM assigns complete rows to threads. A row is solved from left to
right, so no CTA-wide synchronization is required within that row. The
8-lane TRSV alternative assigns one subwarp to a row and reduces each dot
product with width-8 shuffles.

Variant 19 keeps scalar TRSM for every recursive 32- and 64-row panel, but
uses the existing 8-lane mapping for only the outer 128-by-128 `A10` solve.
Its 64 subgroups give all 512 threads useful outer-solve work while avoiding
subgroup overhead in the smaller recursive panels.

Variant 20 keeps all v11 arithmetic and layouts, but limits each `POTF2-32`
column phase to threads 0--127. Those four warps use named CTA barrier 1 with
an explicit 128-thread participant count. Threads 128--511 wait only at the
single full-CTA barrier at factor exit. The original full-CTA POTF2 path is
unchanged for variants 0--19 and 21.

Variants 25--27 replace each recursive right-looking `POTRF128` with a direct
left-looking factorization using panel widths 8, 16, and 32. At each step all
threads update only the current rectangular panel from already-computed
factor columns. Warp zero factors the small diagonal block with warp-local
synchronization, after which complete rows independently solve the panel
below the diagonal. The panel update, diagonal factor, and row solve use three
CTA barriers per panel instead of three CTA barriers per column in each of
eight `POTF2-32` calls. Panel updates and diagonal arithmetic remain FP32.

Variants 23 and 24 block the outer `A10` solve at 32 columns. A left-looking
cross-panel update first removes contributions from previously solved
columns, then every row performs a 32-column FP32 triangular solve. Variant
23 computes the cross-panel update with FP32 FMA. Variant 24 packs
128-by-8 and 32-by-8 operands and issues rectangular 128-by-32 TF32
operations, while retaining the panel solve and stored factor in FP32.

Variant 28 combines the 16-column left-looking `POTRF128`, the tensor
32-column outer solve, padded `A10`, and batched K slices. It intentionally
keeps only the outer 128-by-128 Schur update on the original tensor path;
variants 25--28 do not issue the recursive 64-by-64 tensor updates.

Variants 30 and 35 replace the LL8 below-panel row solve's repeated shared
loads and stores with an eight-value register array per participating thread.
Each complete row is loaded once, solved left to right in registers, and
written once. Variant 31 instead assigns a four-lane subgroup to each row.
Its 128 subgroups make up to 120 below-panel rows active concurrently, testing
whether shorter reductions outweigh shuffle and subgroup-synchronization
costs.

Variants 32 and 33 retain the v25 algorithm while reducing the CTA from 512
threads to 256 and 128. The 214,544-byte shared allocation still limits the
kernel to one CTA per SM, so this is a direct test of useful-warps versus
latency-hiding rather than an occupancy-capacity experiment.

Variants 34 and 36--38 combine the winning inner width of eight with a
32-column left-looking outer solve. This is the two-level blocking reported
by Dong et al.: an outer right-looking update, a recursively left-looking
panel, an outer block size of 32, and an inner block size of eight. The
cross-panel outer updates use TF32 tensor operations while diagonal factors
and triangular solves remain FP32. Variants 35--38 test the useful
combinations of register rows, deferred output, and outer blocking.

Cluster tensor variants block the 128-column outer TRSM at 64 columns. They
solve the first half in FP32, convert copies of that panel and the off-diagonal
factor block in scratch, use a group-2 128-by-64 TF32 MMA for the cross-panel
GEMM, subtract into the FP32 second half, and finish its triangular solve in
FP32. Conversion therefore never damages the panel later written to output.

The SIMT update walks 16-by-16 lower tiles and accumulates each result in FP32
with fused multiply-add. The single-CTA path uses all block threads for each
factor, solve, and update phase; no warp is permanently assigned to only one
algorithmic stage.

Root modes are:

- precise: round-to-nearest square root and division;
- refined: reciprocal square root plus one Newton step;
- raw: hardware reciprocal square root without refinement.

## TCGen05 path

The source has no CUTLASS dependency. It includes the CUDA 13.3 `<cuda/ptx>`
interface and implements the small amount of descriptor, TMEM-allocation,
completion-barrier, and TMEM-load machinery needed by this kernel.

Input panels are explicitly converted with `cvt.rna.tf32.f32` into conversion
scratch. Completed FP32 factor panels are copied to global output before this
conversion. `tcgen05.mma` accumulates products as FP32 in TMEM. After the
completion barrier, cooperative TMEM loads subtract the products from the
still-FP32 residual.

TF32 operands use the PTX canonical K-major byte layout. The descriptor
leading dimension is the operand row count, its stride dimension is eight,
and neither matrix-major bit is set. Reusing a conversion tile waits for the
preceding MMA to finish before overwriting its bytes.

Variant 21 packs several K=8 operand slices into the existing 4,096-float
conversion area before issuing tensor operations. A 64-by-64 update packs all
eight slices and uses one commit/wait group. A 128-by-128 update packs four
slices at a time and uses four groups. Across both recursive 64 updates and
the outer 128 update this reduces 32 commit/wait groups to six, without
changing shared-memory size. Variants 0--20 retain one commit/wait per K=8
slice.

Variant 22 combines variant 18's padded `A10` with the existing batched-slice
mechanism. It is the low-risk control for whether the two independently
successful layout and commit-count changes compose. Variant 28 also batches
the rectangular outer-solve update in groups of up to three K=8 slices, the
largest group that fits both operands in the existing conversion area.

One-SM update shapes are 64-by-64 and 128-by-128. The 64-row TMEM epilogue
uses the four 16-row subpartition groups; every lane executes each synchronous
TMEM load and the active 16 lanes consume the values. Every tensor update
allocates 128 physical TMEM columns, including a 64-row update, so sequential
allocations never increase their column count as prohibited by the PTX ISA.

The two-SM path uses `tcgen05.mma.cta_group::2.kind::tf32` with a global
128-by-64 instruction shape. Layout-B maps each CTA's directly addressable
half to 32 output columns. Rather than depending on the peer-B half when
feeding descriptors from generic shared-memory stores, the implementation
packs each desired 32-column block as the directly addressable half of an
independent 128-by-64 operation. The outer 128-column update therefore issues
four such blocks; the 64-column cross-panel GEMM and recursive update issue
two. Their accumulators occupy distinct 32-column TMEM regions. Each rank
provides its local 64 A rows and consumes its local 64-row accumulator
partition.

The early design proposed one-SM TCGen05 for eligible 64-row recursive updates
inside the same cluster kernel. The PTX ISA forbids that construction: every
`tcgen05` instruction in a kernel must use the same CTA-group qualifier.
Consequently the `all` cluster variants keep `cta_group::2` throughout. A
64-by-64 recursive panel is embedded in rank 0's half of zero-padded
128-by-64 operands; two group-2 MMAs compute its 32-column blocks and rank 0
subtracts the result into the rank-1-owned residual through DSMEM. This costs
unused tensor work, but it is legal and keeps the factorization fused.

TCGen05 is intentionally experimental. Descriptor interpretation, TMEM
addressing, the transition between one-SM and two-SM allocation permits, and
the numerical effect of TF32 must all pass the same validation gates as the
FP32 variants. No relaxed tensor-specific tolerance exists.

## Two-CTA cluster path

One launch contains 128 CTAs arranged as 64 portable two-CTA clusters.
`cooperative_groups::this_cluster()` supplies rank and synchronization.
`map_shared_rank` maps the peer's identically laid-out dynamic shared
allocation, so a factor value is addressed by logical matrix row regardless
of its owner.

Cluster synchronization occurs at recursive 64-row boundaries, after the
outer panel solve, and around the outer update. Cooperative `POTF2` uses only
CTA-local barriers because each 64-row diagonal factor has one owner. This
keeps fine-grained column barriers off the DSMEM synchronization path.

The host uses `cudaLaunchKernelEx` with a cluster dimension of two. Dynamic
shared-memory opt-in and the maximum shared-memory carveout are configured for
every specialization in `prepare()`, before the tuning warmup.

During bring-up, a temporary raw-product dump showed that a single
128-by-128 group-2 instruction fed by the original descriptors reproduced
the first B half in the second half. PTX Layout-B documentation and the
official two-CTA GEMM tutorial established the accumulator mapping; isolated
32-column blocks then matched all four expected quadrants to roughly
`1.7e-4` through `3.0e-4` relative Frobenius error. The probe writes were
removed after the block decomposition passed normal validation.

## Variants

| ID | Threads/CTA | Placement | Factor | TRSM | Update | Root |
|---:|---:|---|---|---|---|---|
| 0 | 256 | 1 CTA | recursive 32 | scalar row | FP32 SIMT | precise |
| 1 | 256 | 1 CTA | recursive 32 | scalar row | FP32 SIMT | refined |
| 2 | 256 | 1 CTA | recursive 32 | scalar row | FP32 SIMT | raw |
| 3 | 256 | 1 CTA | recursive 32 | 8-lane | FP32 SIMT | refined |
| 4 | 256 | 1 CTA | POTF2-64 control | scalar row | FP32 SIMT | refined |
| 5 | 256 | 1 CTA | rank-1 POTF2-64 | scalar row | FP32 SIMT | refined |
| 6 | 512 | 1 CTA | recursive 32 | scalar row | FP32 SIMT | refined |
| 7 | 256 | 1 CTA | recursive 32 | scalar row | TCGen outer | refined |
| 8 | 256 | 1 CTA | recursive 32 | scalar row | TCGen all eligible | refined |
| 9 | 256 | 1 CTA | recursive 32 | 8-lane | TCGen all eligible | refined |
| 10 | 256 | 1 CTA | recursive 32 | scalar row | TCGen all eligible | raw |
| 11 | 512 | 1 CTA | recursive 32 | scalar row | TCGen all eligible | refined |
| 12 | 128 | 2-CTA cluster | recursive 32 | scalar row | FP32 SIMT | refined |
| 13 | 256 | 2-CTA cluster | recursive 32 | scalar row | FP32 SIMT | refined |
| 14 | 128 | 2-CTA cluster | recursive 32 | tensor block-panel | TCGen outer | refined |
| 15 | 128 | 2-CTA cluster | recursive 32 | tensor block-panel | TCGen all eligible | refined |
| 16 | 256 | 2-CTA cluster | recursive 32 | tensor block-panel | TCGen all eligible | refined |
| 17 | 256 | 2-CTA cluster | recursive 32 | tensor block-panel | TCGen all eligible | raw |
| 18 | 512 | 1 CTA | recursive 32 | scalar row | TCGen all eligible, padded A10 | refined |
| 19 | 512 | 1 CTA | recursive 32 | scalar recursive, 8-lane outer | TCGen all eligible | refined |
| 20 | 512 | 1 CTA | recursive 32, 128 participants | scalar row | TCGen all eligible | refined |
| 21 | 512 | 1 CTA | recursive 32 | scalar row | TCGen all eligible, batched K slices | refined |
| 22 | 512 | 1 CTA | recursive 32 | scalar row | TCGen all, padded A10, batched K | refined |
| 23 | 512 | 1 CTA | recursive 32 | LL32 outer, FP32 update | TCGen all, padded A10 | refined |
| 24 | 512 | 1 CTA | recursive 32 | LL32 outer, TF32 update | TCGen all, padded A10 | refined |
| 25 | 512 | 1 CTA | LL8 POTRF128 | scalar outer | TCGen outer, padded A10 | refined |
| 26 | 512 | 1 CTA | LL16 POTRF128 | scalar outer | TCGen outer, padded A10 | refined |
| 27 | 512 | 1 CTA | LL32 POTRF128 | scalar outer | TCGen outer, padded A10 | refined |
| 28 | 512 | 1 CTA | LL16 POTRF128 | LL32 outer, TF32 update | TCGen outer, padded A10, batched K | refined |

`_variant_metadata()` returns one row per ID with registers, local bytes,
static and dynamic shared bytes, occupancy, cluster size, tensor use,
recursive base, factor/TRSM/update modes, root mode, and launch-bound threads.
It also records the A10 leading dimension, outer TRSM mode, POTF2 participant
count, whether TC slices are batched, both panel widths, and whether the
blocked outer solve uses tensor operations.

## Autotuning and promotion

`cholesky_b64n256_modal.py` exposes only three actions.

### Modal tune

The Modal B200 route uses CUDA 13.3 Update 1 and PyTorch 2.12/cu130. It
validates six target-sized input families: dense, planted spectrum, diagonal,
damped low rank, row-scaled, and tridiagonal.

At remote function startup it creates the configured `/cache/tmp` directory.
The `/cache` volume mount hides image-build-time contents, while NVCC requires
the configured temporary directory to exist before extension compilation.

Each result must have the requested shape, CUDA FP32 dtype/device, finite
entries, exact strict-upper zeros, and positive diagonals. Its reconstruction
residual is scaled by `eps * n * ||A||inf`. The limit is
`max(16, 8 * reference_scaled_residual)`, where the reference uses PyTorch on
the identical FP32 input.

Timing uses eight input/output slots, 16 warmups, 200 calls per sample, and
five samples whose variant order alternates forward and reverse. The returned
payload, including resource rows and every case result, is written locally.

### Brev NCU

The NCU action runs locally and calls Popcorn's hosted Brev profiler at
benchmark index 3. Each selected ID is rendered into a retained submission.
The command, standard output/error, raw result, CSV/text details, `.ncu-rep`
files, and a combined summary remain under `artifacts/ncu/`.

There is no Modal NCU route and no local RTX route.

### Popcorn sweep

The leaderboard action renders and submits every selected ID in every round
with bounded local concurrency. It retains all sources, commands, logs, raw
result files, scores, and median-geomean ranking.

A variant is promotion-eligible only when:

1. the supplied tuning JSON contains passing results for all six exact-shape
   Modal cases; and
2. the variant returned a score with a successful process status in every
   requested Popcorn round.

Before promotion, the script re-reads the tracked source and compares both its
content and SHA-256 hash with the pre-sweep snapshot. A mismatch refuses the
write. Otherwise, it atomically replaces the file after changing exactly the
one production marker. The winning rendered source is retained next to the
sweep summary.

## Compiler policy

The extension targets only `sm_100a` and uses C++20, `-O3`, fast math, extra
device vectorization, restricted kernel pointers, line information, ptxas
optimization, resource reporting, and spill warnings. There is no external
device-library dependency.

## B200 results: 2026-07-23

A consolidated Modal run validated every stable ID on all six `(64,256,256)`
input families. The retained payload is
`artifacts/tune/tuning_20260723T091738Z.json`.

| Rank | ID | Median ms | Placement/update |
|---:|---:|---:|---|
| 1 | 11 | 0.436361 | 1 CTA, 512 threads, TCGen all |
| 2 | 8 | 0.453568 | 1 CTA, 256 threads, TCGen all |
| 3 | 9 | 0.461529 | 1 CTA, 8-lane TRSV, TCGen all |
| 4 | 10 | 0.463688 | 1 CTA, raw root, TCGen all |
| 5 | 7 | 0.466301 | 1 CTA, TCGen outer |
| 6 | 17 | 0.680507 | 2 CTA, 256 threads, TCGen all/raw |
| 7 | 16 | 0.688399 | 2 CTA, 256 threads, TCGen all |
| 8 | 6 | 0.715085 | 1 CTA, 512 threads, FP32 SIMT |
| 9 | 2 | 0.735159 | 1 CTA, raw root, FP32 SIMT |
| 10 | 1 | 0.736595 | 1 CTA, refined root, FP32 SIMT |
| 11 | 5 | 0.755193 | 1 CTA, rank-1 POTF2-64 |
| 12 | 3 | 0.761914 | 1 CTA, 8-lane TRSV, FP32 SIMT |
| 13 | 0 | 0.764548 | 1 CTA, precise root, FP32 SIMT |
| 14 | 15 | 0.766805 | 2 CTA, 128 threads, TCGen all |
| 15 | 14 | 0.800571 | 2 CTA, 128 threads, TCGen outer |
| 16 | 4 | 1.063629 | 1 CTA, POTF2-64 control |
| 17 | 13 | 1.241820 | 2 CTA, 256 threads, FP32 SIMT |
| 18 | 12 | 1.293925 | 2 CTA, 128 threads, FP32 SIMT |

Variant 11 is the isolated target-shape Modal winner. The best cluster result
is valid but 56% slower than variant 11; at this matrix size its DSMEM
synchronization, duplicate B packing, and unused group-2 work outweigh the
extra-SM tensor throughput.

Popcorn test submission 898905 passed all 17 evaluator integration cases.
The full three-round leaderboard sweep retained its summary at
`artifacts/popcorn/b64_n256_popcorn_20260723T092114Z/summary.json`. All 18
variants received a successful score in every round and remained promotion
eligible. The leading medians were:

| Rank | ID | Median public B200 geomean, seconds |
|---:|---:|---:|
| 1 | 8 | 0.002144731481 |
| 2 | 9 | 0.002176213819 |
| 3 | 10 | 0.002176450700 |
| 4 | 16 | 0.002203943865 |
| 5 | 6 | 0.002209473398 |
| 6 | 2 | 0.002213606572 |
| 7 | 11 | 0.002216125701 |
| 8 | 0 | 0.002219099024 |

The sweep re-read an unchanged source hash and atomically promoted variant 8.
This differs from the isolated Modal winner because the public geomean also
contains the held-constant fallback shapes and normal remote timing noise.

## B200 results and NCU: 2026-07-28

The retained tuning payload is
`artifacts/tune/tuning_20260728T094014Z.json`. All eight requested variants
passed all six exact-shape input families.

| Rank | ID | Median ms | Experiment |
|---:|---:|---:|---|
| 1 | 25 | 0.235434 | LL8, scalar outer TRSM, tensor outer Schur |
| 2 | 28 | 0.257947 | LL16 plus tensor-blocked outer TRSM |
| 3 | 24 | 0.280321 | recursive factor plus tensor-blocked outer TRSM |
| 4 | 26 | 0.288754 | LL16 |
| 5 | 23 | 0.292595 | recursive factor plus SIMT-blocked outer TRSM |
| 6 | 22 | 0.300578 | padded recursive control with batched tensor slices |
| 7 | 18 | 0.305545 | padded recursive control |
| 8 | 27 | 0.368414 | LL32 |

Variant 25 is 22.9 percent faster than v18. The inner-width sweep has a clear
minimum at eight: v25 is 18.5 percent faster than LL16 and 36.1 percent faster
than LL32. Tensor blocking of the outer solve was useful in isolation, with
v24 8.3 percent faster than v18, but the previous round did not test it with
the winning LL8 factor.

The follow-up report is
`artifacts/ncu/b64_n256_brev_ncu_20260728T094217Z/`
`v25_cta512_ll8_tc_outer_refined_pad129/`
`profile.3-batch-64-n-256-cond-2-seed-41256/profile.ncu-rep`.
Nsight Compute 2026.2 measured 410.18 microseconds, versus 235.43
microseconds in the normal tuning harness; absolute NCU duration is therefore
not used as the tuning baseline.

The report measured 12.08 percent SM throughput, 12.59 percent memory
throughput, 24.99 percent achieved occupancy, and no spills. Each scheduler
had four active warps on average but only 0.25 eligible warps; no warp was
eligible in 80.62 percent of cycles. Warp sampling attributed 80,898 of
112,331 samples, or 72.0 percent, to barriers. The four dominant barrier
locations were:

| Generated CUDA line | Barrier samples | Phase |
|---:|---:|---|
| 680 | 40,431 | completion of the LL8 below-panel row solve |
| 1145 | 25,051 | output of solved `L10` before its Schur update |
| 653 | 8,545 | completion of the current left-looking panel update |
| 675 | 4,957 | completion of the warp-local 8-by-8 factor |

The `L10` output barrier alone accounts for 31.0 percent of all barrier
samples. The following Schur update only reads `L10` and writes `A11`, so
that barrier does not protect an address dependency. The LL8 row solve uses
at most 120 of 512 threads, explaining why its necessary completion barrier
is the largest remaining idle point.

Shared memory is secondary but measurable: the report records 344,192 shared
store requests, 130,686 store-bank conflicts, and 307,712 excessive shared
wavefronts, 4.0 percent of all shared wavefronts. The load/store pipeline
reached only 12.16 percent of peak for shared traffic, so bank remapping is
lower priority than shortening the synchronization critical path.

The native report displays a PM Sampling section, but
`veloq ncu metrics --counter 'pmsampling:*'` returns zero counters. No
equivalent metric was substituted, and no phase timeline is inferred from
that absent family. The source-attributed `warpsampling:*` counters above are
direct measurements present in the report.

The two papers reinforce the measured choices without being copied
architecture-blindly. Dong et al. found recursive left-looking panels inside
a right-looking trailing update, with outer block 32 and inner block 8, to be
their best K40 configuration. Kurzak et al. kept the mutable panel in fast
local storage, delayed writes, and autotuned both panel width and thread-block
shape; register residency and avoiding unnecessary conditionals were
essential. Their common 64-thread shape is not assumed optimal for B200, but
it motivates the explicit 128/256/512-thread sweep.

The new experiment matrix is:

| ID | Change from v25 |
|---:|---|
| 29 | defer all factor output to one final lower-triangle traversal |
| 30 | keep each LL8 row in eight registers during its solve |
| 31 | solve one LL8 row per four-lane subgroup |
| 32 | use 256 threads |
| 33 | use 128 threads |
| 34 | add the 32-column tensor-blocked outer solve |
| 35 | deferred output plus register rows |
| 36 | deferred output plus outer-32 |
| 37 | register rows plus outer-32 |
| 38 | deferred output, register rows, and outer-32 |

## Verification commands

Before this append-only pass, the Python compilation, diff check,
forbidden-token search, Modal tuning, Popcorn test, and three-round Popcorn
sweep commands had been run. NCU profiling remains for a subsequent
profiling pass.

For variants 22--28, Python compilation, `git diff --check`, the
forbidden-token search, a direct algebra/indexing check, CUDA 13.1 SM100a
translation-unit compilation, and B200 six-family tuning completed. The CUDA
compile reported only the existing PyTorch BFloat16 host/device constexpr
warnings. Variants 22, 25, 26, 27, and 28 use 47--48 registers; variant 23
uses 62 and variant 24 uses 71. None spills.

For variants 29--38, Python compilation, `git diff --check`, the
forbidden-token search, a direct FP32 algebra/indexing check, and a full CUDA
13.1 SM100a translation-unit compilation completed. The new kernels have no
stack frame and no spills. Variants 29, 30, 31, 32, 33, and 34 use 47, 64,
48, 48, 48, and 48 registers, respectively; combined variants 35--38 use 63,
48, 64, and 63 registers. The only compiler diagnostics are the existing
PyTorch BFloat16 constexpr warnings.

B200 execution remains required. The first required B200 command is:

```bash
.venv/bin/modal run cholesky/b64n256/cholesky_b64n256_modal.py \
  --action tune --variants 25,29,30,31,32,33,34,35,36,37,38

python3 -m py_compile \
  cholesky/b64n256/cholesky_b64n256.py \
  cholesky/b64n256/cholesky_b64n256_modal.py
git diff --check
rg -n 'stream' cholesky/b64n256/cholesky_b64n256.py

.venv/bin/modal run cholesky/b64n256/cholesky_b64n256_modal.py \
  --action tune

popcorn submit --leaderboard cholesky --gpu B200 --mode test --no-tui \
  cholesky/b64n256/cholesky_b64n256.py

.venv/bin/modal run cholesky/b64n256/cholesky_b64n256_modal.py \
  --action ncu --variants ID1,ID2 --benchmark-index 3

.venv/bin/modal run cholesky/b64n256/cholesky_b64n256_modal.py \
  --action popcorn --tuning-json PATH_TO_TUNING_JSON

.venv/bin/modal run cholesky/b64n256/cholesky_b64n256_modal.py \
  --action popcorn --variants ID1,ID2,ID3 --rounds 3 \
  --tuning-json PATH_TO_TUNING_JSON

popcorn submit --leaderboard cholesky --gpu B200 --mode leaderboard --no-tui \
  cholesky/b64n256/cholesky_b64n256.py
```

The Popcorn correctness grid does not include `(64, 256, 256)`. Its test mode
therefore checks fallback integration, while the six Modal cases are the
primary custom-path correctness gate.

## Current status

Variant 25 passes all six exact-shape validation families and is the current
isolated B200 winner at a 235.4-microsecond median. Variant 18 remains the
production default pending the next tuning round and normal promotion gates.
Variants 29--38 are the unmeasured append-only candidates.

## Cutlass-name clone experiment

A cutlass-named clone has been added for the current tracked default variant.
The public base variant is `37` and the cloned public variant is `39`.
The clone compiles identical CUDA algorithm source after renaming every custom
`__global__` kernel entry point and matching launch/configuration reference to
use a `cutlass_` prefix.

The 2026-07-28 Modal tune retained variant 37. Median time was 0.192456 ms for
variant 37 versus 0.197982 ms for variant 39, so the cutlass-named clone was
rejected before an authoritative Popcorn promotion run.
