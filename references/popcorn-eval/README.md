# Popcorn eval scripts (durable snapshot)

The **actual server-side** popcorn evaluation code for the `eigh` task, copied
out of ephemeral `/tmp` on 2026-07-14 so it survives reboots. Read these before
speculating about how benchmarks/correctness are measured — do not guess from the
local harness alone.

| file | source | role |
|---|---|---|
| `eval.py` | `/tmp/eigh-eval.py` | eval driver: modes `test` / `benchmark` / `leaderboard` / `profile`, input-count rule (`_benchmark_batch_count`), per-repeat recheck loop, `_make_data_batch` seed `+=42` |
| `reference.py` | `/tmp/eigh-reference.py` | `generate_input` + fp64 `check_implementation` (eigen/recon/orth/sort gates, per-matrix `.any()`). Tolerance factors verified identical to `eigh_bench_local_rtx4050.py`'s embedded snapshot. |
| `run_eval.py` | `kernelbot-src/src/libkernelbot/run_eval.py` | runner: invokes eval, maps outcomes → exit codes |
| `consts.py` | `kernelbot-src/src/libkernelbot/consts.py` | ExitCode enum: CUDA_FAIL=110, PIPE_FAILED=111, **VALIDATE_FAIL=112** (correctness fail), TEST_SPEC=113, TIMEOUT_EXPIRED=114 |
| `utils.py` | `kernelbot-src/examples/utils.py` | `clear_l2_cache`, `set_seed` used by `eval.py` |

Key asymmetry these reveal (why a kernel can pass `test` but fail `leaderboard`
with exit 112): **test** runs each case once on a clone vs the pristine input at
small batch; **leaderboard** runs full batch (640/60/8) up to 1000× reusing the
same inputs with `recheck=True` every repeat. Faithful B200 re-run:
`modal run eigh_bench_b200_modal.py --official-eigh --backends cublas`.

These are reference material only — not part of the `eigh.py` submission.
