# `b1n4096` B200 Cholesky design

## Status

The tracked default remains variant 0 (`torch.linalg.cholesky_ex`). The first
B200 gate completed on 2026-07-28 and retained it: the only fully passing
native candidate was substantially slower. The historical 1.53 ms baseline
was confirmed by the contemporaneous three-round median.

Variants 11--15 are a second-generation redesign based directly on Algorithm
3 of ICL-UTK-987-2017. They are implemented and compile for `sm_100a`. Their
isolated leaves reconstruct correctly, but the final variant-11 NSys duration
is 6.095 ms, including 4.112 ms in 32 serial M128 leaf calls. This is
decisively slower than the approximately 1.53 ms library baseline, so none is
promoted and no version is ported to `b2n4096`.

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
check only; the local machine cannot execute `sm_100a`.

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

Only a fully passing variant at or below `0.995 x` its contemporaneous
variant-0 median may become the default. No b2n4096 port is made until that
winner is known.
