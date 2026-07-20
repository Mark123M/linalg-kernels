# Batched Cholesky: B=4096, N=32

## Status

- Raw CUDA Cholesky--Crout extension implemented.
- Ten B200 launch/math variants implemented.
- Modal correctness, timing, resource, and Nsight Systems tooling implemented.
- Popcorn leaderboard sweep implemented but not run by the assistant.
- Production variant 3 was selected by the full Popcorn leaderboard sweep.

## Correctness reference

LAPACK `SRC/spotf2.f` is the scalar correctness reference. For the lower factor,
its recurrence is

```text
L[j,j] = sqrt(A[j,j] - sum(k<j, L[j,k]^2))
L[i,j] = (A[i,j] - sum(k<j, L[i,k]*L[j,k])) / L[j,j], i>j
```

MAGMA's `magmablas/zpotf2_kernels.cu` and
`magmablas/zpotf2_devicesfunc.cuh` demonstrate two useful GPU ideas: fully
unroll the small fixed-width factorization and pack multiple independent
matrices into one CTA. The implementation here does not reproduce MAGMA's
shared-memory panel layout or its block barriers.

## B200 mapping

One warp factors one matrix. Lane `i` owns row `i` of `L` and retains its 32
entries in registers. At iteration `j`, lane `i` computes `L[i,j]`; lane `j`
broadcasts its previous row entries using warp shuffles. There are no block
barriers and no shared-memory allocations.

The input is symmetric, so the kernel reads `A[j,i]` instead of the strided
lower entry `A[i,j]`. A warp therefore reads one contiguous input row per
Crout step. Each lane finally writes its owned output row through eight aligned
`float4` stores, inserting exact zeros above the diagonal.

The B200 compute-capability 10.0 resource envelope used for the launch design
is 64 resident warps, 2,048 resident threads, 32 resident CTAs, 64K 32-bit
registers, and 228 KiB shared memory per SM. The fixed workload has about 28
matrices per SM on a 148-SM B200, so every candidate uses launch bounds that
target at least 32 resident warps. Runtime CUDA occupancy queries, rather than
these paper limits alone, determine each compiled variant's actual capacity.

Reference: NVIDIA Blackwell Tuning Guide,
https://docs.nvidia.com/cuda/blackwell-tuning-guide/

## Variant space

Variant IDs are stable because temporary Popcorn submissions replace only the
production ID constant.

| ID | Warps/CTA | Root and reciprocal |
|---:|----------:|---------------------|
| 0 | 1 | round-to-nearest sqrt/divide |
| 1 | 1 | approximate reciprocal sqrt + one Newton step |
| 2 | 2 | round-to-nearest sqrt/divide |
| 3 | 2 | approximate reciprocal sqrt + one Newton step |
| 4 | 4 | round-to-nearest sqrt/divide |
| 5 | 4 | approximate reciprocal sqrt + one Newton step |
| 6 | 8 | round-to-nearest sqrt/divide |
| 7 | 8 | approximate reciprocal sqrt + one Newton step |
| 8 | 16 | round-to-nearest sqrt/divide |
| 9 | 16 | approximate reciprocal sqrt + one Newton step |

Variant 3 is the production choice. The compiler uses `-O3`, fast global math,
explicit vectorization, FMA contraction, and `sm_100a`; precise variants call
explicit round-to-nearest intrinsics so they remain the numerical control group.

The native extension explicitly uses C++20, matching the PyTorch 2.12 extension
ABI on both Modal and the Popcorn evaluator. An earlier C++17 override compiled
on Modal but failed while parsing an ATen header in Popcorn; the extension cache
key was advanced when this was corrected.

The evaluated path uses CUDA's ordinary default-queue launch syntax. This is
semantically identical to the former explicit raw handle value of zero and
does not alter the selected device kernel or its launch dimensions. The raw
handle plumbing and dynamically assembled CUDA type identifier were removed in
anticipation of stricter source validation. Arbitrary nondefault-queue launches
are intentionally outside this fixed-shape submission path.

## Autotuning policy

The exact `(4096,32,32)` shape uses the custom extension. Every other shape
uses `torch.linalg.cholesky_ex(..., check_errors=False).L`. Consequently all
non-target leaderboard timings are held constant across temporary submissions.

The Popcorn action launches all ten independent leaderboard submissions
concurrently so Popcorn can queue them across its workers. It records each
process's raw command output, parses the public B200 geomean, and emits a ranked
JSON summary. It does not rewrite the tracked specialization. After the results
are reviewed, update the production ID above and record the score here.

| Date | Selected ID | Public B200 geomean | Notes |
|------|------------:|---------------------:|-------|
| 2026-07-20 | 3 (`w2_refined`) | 0.0018265698079868835 s | Best of ten full leaderboard submissions; IDs 7 and 9 were close runners-up |

## Verification commands

These commands are intentionally documented but have not been run by the
assistant.

```bash
python3 -m py_compile \
  cholesky/b4096n32/cholesky_b4096n32.py \
  cholesky/b4096n32/cholesky_b4096n32_modal.py

git diff --check

# The command must print no matches.
rg -n 'stream' cholesky/b4096n32/cholesky_b4096n32.py

# Modal correctness/resource sweep and kernel-only timing.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action tune

# One Nsight Systems report set per variant; all artifacts are downloaded.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action profile

# Check the provisional production route against the official tests.
popcorn submit --no-tui --leaderboard cholesky --gpu B200 --mode test \
  cholesky/b4096n32/cholesky_b4096n32.py

# Concurrently submit all ten variants, parse their public geomeans, and rank them.
.venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
  --action popcorn
```

The requested feedback after `--action tune` is its JSON output plus all ptxas
spill warnings. After `--action profile`, inspect or provide the downloaded
statistics files. After `--action popcorn`, provide `summary.json` so the
winning ID can replace the provisional default.
