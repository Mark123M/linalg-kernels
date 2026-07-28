# b1n8192: single-matrix n=8192 Cholesky specialization for B200

Benchmark entry 12:
`{"batch": 1, "n": 8192, "cond": 2, "seed": 48192}`.
The FP32 input contains 2^26 elements (256 MiB).

## Status

Implemented 2026-07-26. The initial default is variant 1, the direct
micro=128 port of the measured b1n32768 variant-14 winner. CUDA 13.1
host-side compilation for `sm_100a` succeeds for both tile widths; the
compiler reports no spills in either factor or fused kernel. Execution,
checker validation, and timing remain pending on B200 because the local
GPU is unavailable.

An NCU capture of variant 3's first four fused steps was analyzed on
2026-07-26. It showed a producer/consumer latency bottleneck and led to
append-only variants 8-10. Their `sm_100a` compilation also succeeds
without local memory or spills. The default is intentionally unchanged
until the new candidates pass the B200 checker and timing gate.

The existing public B200 measurements from unrelated submissions put the
unchanged torch/cuSOLVER row at approximately 6.37-6.40 ms. That is a
direct measurement, not a projection. A passing specialization must beat
6.37 ms before it is considered a performance win.

## Provenance and algorithm

The implementation ports the two strongest b1n32768 designs:

- Variant 13: one fused launch per micro step. CTA 0 factors the diagonal
  tile and builds its inverse while consumer CTAs prefetch their first
  sub-column tile, then a release/acquire flag hands off the inverse.
- Variant 14: the same wide factor and inverse build, followed by a
  TF32 `cublasGemmEx` for `X := X inv(L11)^T` into scratch and a float4
  copy-back. This replaced the latency-bound SIMT apply at n=32768.

Both use the left-looking panel schedule:

```text
copy lower triangle and write exact upper zeros
for each outer panel:
    apply earlier panels with one TF32 history GEMM
    for each micro tile:
        apply earlier micros in this panel with a TF32 inner GEMM
        factor the diagonal tile and construct its dense inverse
        apply the inverse with fused SIMT work or TF32 GEMM + copy-back
restore exact zeros in the upper wedges of outer diagonal panels
```

The matrix dimension is four times smaller than the source design, but
the diagonal factor chain scales only linearly with `n`. The added
micro=64 family is the shape-specific response: it halves the diagonal
tile width and factor/inverse critical path while testing NB in
{256, 512, 1024}. The micro=128 ports remain controls so the speculative
change cannot be promoted without measurement.

## Variant registry

IDs are stable and append-only.

| ID | name | NB | micro | apply | role |
|---:|---|---:|---:|---|---|
| 0 | `ll_nb1024_m128_microfused_tf32` | 1024 | 128 | fused SIMT | direct b1n32768 v13 control |
| 1 | `ll_nb1024_m128_invgemm_tf32` | 1024 | 128 | TF32 GEMM + copy | direct b1n32768 v14 control; initial default |
| 2 | `ll_nb256_m64_microfused_tf32` | 256 | 64 | fused SIMT | 4 micros/panel |
| 3 | `ll_nb512_m64_microfused_tf32` | 512 | 64 | fused SIMT | 8 micros/panel |
| 4 | `ll_nb1024_m64_microfused_tf32` | 1024 | 64 | fused SIMT | 16 micros/panel |
| 5 | `ll_nb256_m64_invgemm_tf32` | 256 | 64 | TF32 GEMM + copy | 4 micros/panel |
| 6 | `ll_nb512_m64_invgemm_tf32` | 512 | 64 | TF32 GEMM + copy | 8 micros/panel |
| 7 | `ll_nb1024_m64_invgemm_tf32` | 1024 | 64 | TF32 GEMM + copy | 16 micros/panel |
| 8 | `ll_nb512_m64_microfused_split2_tf32` | 512 | 64 | fused SIMT, 2 CTAs/tile | grid-size control |
| 9 | `ll_nb512_m64_microfused_compact_tf32` | 512 | 64 | fused SIMT | compact-factor control |
| 10 | `ll_nb512_m64_microfused_compact_split2_tf32` | 512 | 64 | fused SIMT, 2 CTAs/tile | combined optimization |

The autotuner promotes only a variant that passes every public benchmark
row in every requested round and has a median target-row mean below the
measured 6,371,008 ns baseline. Otherwise it retains variant 1. Ranking
uses the median target-row mean, then median best time, then ID.

## Kernel and workspace design

| micro | factor/fused threads | factor/fused dynamic shared | apply-tile shared |
|---:|---:|---:|---:|
| 128 | 512 | 153600 B | 132096 B |
| 64 | 256 | 39936 B | 33280 B |

- The factor uses redundant 8x8 corners in registers, four threads per
  sub-panel row, and quarter-split rank-8 updates. The same templated
  operation order is used at both widths.
- Inverse construction first inverts independent 32x32 diagonal blocks.
  Micro=64 needs one 32-to-64 combine; micro=128 additionally combines
  the two 64-wide halves.
- The micro=64 fused apply uses eight warps in a 2x4 layout. Each warp
  owns 32 rows by 16 columns and each thread accumulates an 8x2 tile.
  Micro=128 retains the source 4x4 layout and 8x4 thread tile.
- The split-apply candidates divide one 64x64 output tile by column
  across two CTAs. Each CTA uses a 4x2 warp layout and 4x2 thread tile,
  reads only its 32 inverse rows, and duplicates the 64x64 input-tile
  load. This changes neither output ownership nor arithmetic order
  within an output element.
- The compact-factor candidates factor each 8x8 corner with one
  cooperative warp, solve each trailing row with one thread, and return
  all 256 threads to the existing quarter-split rank-8 update. They keep
  exact FP32 square root/division and the original inverse builder.
- Fused grids are clamped to the queried co-resident CTA capacity after
  opting into dynamic shared memory. The last micro launches only CTA 0
  and skips inverse publication.
- Per call, `t_inv` contains `micro^2` FP32 values. Fused variants also
  allocate and zero `n/micro` int32 flags. GEMM-apply variants allocate
  `(n-micro)*micro` FP32 scratch (about 4.1 MiB at micro=128 and
  2.1 MiB at micro=64).
- The input is never written. The lower-copy kernel uses 512 CTAs and
  float4 traffic; copy-back uses 128 CTAs. All matrix indexing is int64.

## Correctness invariants

- History and inner GEMMs write rectangles whose invalid values are
  confined to the strict-upper wedge of an outer diagonal block.
- No later operand reads those wedges: all panel/history operands lie
  strictly below completed diagonal blocks, and factor loads are
  explicitly lower-triangular.
- The inverse workspace has exact zeros above its diagonal, so a dense
  inverse-apply GEMM is algebraically identical to the triangular
  product before TF32 rounding.
- One final wedge kernel restores exact strict-upper zeros.
- Fused consumers read the current inverse only after the producer's
  release and their acquire fence. All other inputs were completed by
  earlier work on the caller's execution queue.
- Only contiguous CUDA FP32 `(1,8192,8192)` inputs use the extension.
  Every other input retains `torch.linalg.cholesky_ex`.

## 2026-07-26 variant-3 NCU diagnosis

The retained Nsight Compute 2026.2.0 report contains only the first ten
launches, including the first four fused kernels. It does not contain a
complete factorization or an unprofiled end-to-end time. The full
analysis, metric-name note, and reproduction helper are in
[`artifacts/ncu/b1n8192_20260726T225216Z/v03_ll_nb512_m64_microfused_tf32/REPORT.md`](artifacts/ncu/b1n8192_20260726T225216Z/v03_ll_nb512_m64_microfused_tf32/REPORT.md).

Direct evidence for the early fused step:

- duration is stable at 69.440-69.824 us;
- grids shrink from 128 to 125 blocks on 148 SMs, only
  0.2883-0.2815 waves/SM;
- achieved occupancy is approximately 12.1% versus a register-limited
  37.5% theoretical occupancy at 80 registers/thread;
- 26,181 of 36,910 launch-2 samples (70.9%) are CTA-barrier stalls, and
  the NCU rule reports 25.639 stalled cycles per issued instruction;
- issue activity is 0.0562 instructions/cycle and eligible warps are
  only 0.0674/cycle;
- the busiest compute pipe is 2.65% of peak, while DRAM read, L1/TEX,
  and L2 throughput are 0.40%, 4.83%, and 0.84%.

Correlated SASS places the dominant barrier directly after the
consumer's flag-poll loop. The primary optimization target is therefore
the producer handoff and the secondary target is the sub-SM grid, not
memory bandwidth. The compiler-side resource changes are measured:

| Candidate | Registers/thread | Local bytes/thread |
|---|---:|---:|
| 3, original | 80 | 0 |
| 8, split apply | 75 | 0 |
| 9, compact factor | 40 | 0 |
| 10, compact + split | 40 | 0 |

The exact aggregate B200 stall metric is present as
`smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`.
VeloQ 0.4.1 labels its unit `inst`, whereas NCU's rule prose describes
the value in stalled cycles per issued instruction. The display alias
`smsp_average_barrier` in the rule payload is not queryable; no metric
was silently substituted.

## Validation and next actions

Sandbox-safe checks:

```bash
python3 -m py_compile \
  cholesky/b1n8192/cholesky_b1n8192.py \
  cholesky/b1n8192/cholesky_b1n8192_runner.py
rg -n 'stream' cholesky/b1n8192/cholesky_b1n8192.py
git diff --check -- cholesky/b1n8192
```

B200 validation:

```bash
# Exercises fallback behavior on the official test grid.
python3 cholesky/b1n8192/cholesky_b1n8192_runner.py \
  submit --mode test

# Exercises the native target row, verifies every public row, and
# atomically promotes the median winner.
python3 cholesky/b1n8192/cholesky_b1n8192_runner.py \
  autotune --variants 1,3,8,9,10 --rounds 3
```

The focused set compares the current default, the profiled control, and
all three isolated NCU-driven candidates. A later exhaustive sweep can
still use `--variants all`. If no passing candidate beats 6.37 ms, keep
variant 1 and capture the fastest passing candidate alongside it:

```bash
python3 cholesky/b1n8192/cholesky_b1n8192_runner.py \
  ncu --variants 1,<fastest-candidate-id>
```

Any NCU analysis must report missing or renamed metrics explicitly,
including expected and actual names, units, and normalization. Helpers
used for analysis must be retained under
`artifacts/helpers/ncu/`.

## Benchmark history

| Date | Variant | benchmark.12 mean | Status |
|---|---:|---:|---|
| 2026-07-26 | torch/cuSOLVER fallback | ~6.37-6.40 ms | measured in existing public artifacts |
| 2026-07-26 | 3 | NCU only; early fused step 69.440-69.824 us | passed profile job; no end-to-end timing |
| - | 0-10 | pending | B200 autotune required |

## Nsight Systems endpoint

The shape-local Modal launcher profiles one warmed factorization on B200:

```bash
.venv/bin/python -m modal run cholesky/b1n8192/cholesky_b1n8192_modal.py
# Optional comparison:
.venv/bin/python -m modal run cholesky/b1n8192/cholesky_b1n8192_modal.py --variant 8
```

`--variant -1` selects the tracked default, currently variant 8. For the
current micro=64, NB=512 split-consumer source, metadata records 258
algorithm launches per factorization. Input generation, extension
compilation, preparation, warmup, and correctness validation are outside the
capture; the NVTX range contains exactly one out-parameter factorization.

Artifacts are downloaded under `artifacts/nsys/`. `profile.nsys-rep` is the
forward-compatible UI/VeloQ input, `kernel-trace.csv` is the ordered GPU
timeline, `kernel-exec-trace.csv` separates API, queue, and execution time,
and `kernel-summary.csv` aggregates duration by kernel name. The SQLite
export, human-readable statistics, command, profiler version, environment,
preflight, stdout, and stderr are retained with the report.
