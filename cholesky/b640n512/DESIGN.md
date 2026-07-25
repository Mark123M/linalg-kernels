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

All dependencies are CTA-local. `__syncthreads()` separates shared-tile
producer and consumer phases. No inter-CTA communication, atomics, clusters,
or device-side work queue are required.

## Tile strategies

### `NB=64`

- 256 threads, eight warps.
- Two padded `64 x 65` FP32 shared tiles plus 64 reciprocals: 33,536 bytes.
- Recursive `16 -> 32 -> 64` diagonal factorization.
- Four-lane row groups for factor-internal `TRSM16/32`.
- Outer panel solve uses either one owner thread per row or a four-lane row
  group.
- The trailing update uses a 4x4 register microtile per lane.
- A solved left panel remains in shared memory while the CTA updates every
  destination in that tile row.

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

Only factor and solve arithmetic select a root mode. Every trailing update
uses FP32 FMA. The provisional default is the Newton-refined `NB=64`
variant; raw and precise remain direct controls until B640 validation and
autotuning are complete.

## Variant registry

| ID | Outer tile | Root | Outer solve | Update | Threads |
|---:|---:|---|---|---|---:|
| 0 | 64 | raw | scalar | 4x4 | 256 |
| 1 | 64 | Newton | scalar | 4x4 | 256 |
| 2 | 64 | precise | scalar | 4x4 | 256 |
| 3 | 64 | raw | sub4 | 4x4 | 256 |
| 4 | 32 | raw | scalar | 2x4 | 128 |
| 5 | 32 | Newton | scalar | 2x4 | 128 |

Native preparation reports registers, static shared memory, and local
memory. A candidate is rejected if the compiler creates a local frame above
8 bytes.

## Expected optimization sequence

1. Validate all six variants on benchmark index 5.
2. Compare `NB=64` and `NB=32` occupancy, barrier, shared-memory, and HBM
   behavior with Nsight Compute.
3. If `NB=64` is latency limited, evaluate a third shared tile and
   producer/consumer operand pipeline.
4. If FP32 FMA dominates, evaluate one-TMEM-allocation-per-CTA TCGen05
   updates. Earlier B16 tensor variants do not settle this question because
   they could not amortize setup across the complete matrix.

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

POPCORN_BREV_PROFILER_URL=URL \
python3 cholesky/b640n512/cholesky_b640n512_runner.py ncu \
  --variants WINNER,RUNNER_UP

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
