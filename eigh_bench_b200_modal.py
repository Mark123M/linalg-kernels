"""Modal B200 launcher for the fair-benchmark core of eigh_bench_local_rtx4050.py.

Thin launcher: the bench logic (correctness gates, torch DLATRD baseline, CUDA-graph
+ L2-rotated timing) is imported from the local runner, which is mounted at runtime
(copy=False) together with eigh.py — every `modal run` benches the current working
tree with no image rebuild. This file is never submitted, so it may freely contain
the word "stream".

Usage:
    modal run eigh_bench_b200_modal.py --cases 640x512 --panel-size 8 \
        --backends cublas,cublasdx --repeats 3
    # full tridiagonalization (one timed call = one full factorization,
    # so keep --calls small):
    modal run eigh_bench_b200_modal.py --cases 640x512 --panel-size 16 \
        --backends cublas --tri --calls 20 --repeats 3

One-time setup: .venv/bin/pip install modal && .venv/bin/modal setup
"""

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.13")
    .entrypoint([])
    .pip_install(
        "nvidia-cutlass-dsl[cu13]==4.5.2",
        "apache-tvm-ffi",
        "torch==2.12.0",
        "ninja",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    # Exact header parity with local: bake the same MathDx tarball (incl. bundled
    # CUTLASS) into an image layer. doc/ and example/ are dead weight.
    .add_local_dir(
        "nvidia-mathdx-26.06.0-cuda13",
        "/opt/mathdx",
        copy=True,
        ignore=["**/doc/**", "**/example/**", "**/__pycache__"],
    )
    .env(
        {
            "MATHDX_ROOT": "/opt/mathdx/nvidia/mathdx/26.06",
            # Compile caches on the persistent volume: the CuTe DSL disk cache
            # (fingerprinted by eigh.py source + versions, so stale entries
            # self-invalidate) and, via TMPDIR, eigh.py's load_inline build root
            # (tempfile.gettempdir()/<user>/eigh_rank2k_native) — the ~90 s
            # cuBLASDx extension build survives cold starts.
            "EIGH_ENABLE_DISK_CACHE": "1",
            "EIGH_CUTE_CACHE_DIR": "/cache/cute",
            "TMPDIR": "/cache/tmp",
        }
    )
    # Runtime mounts: synced on every `modal run`, no rebuild.
    .add_local_file("eigh.py", "/workspace/eigh.py", copy=False)
    .add_local_file(
        "eigh_bench_local_rtx4050.py",
        "/workspace/eigh_bench_local_rtx4050.py",
        copy=False,
    )
    # torch-only module (pandas/triton imports are TYPE_CHECKING / unused paths);
    # mounted flat and re-registered as quack.bench.bench_utils inside bench().
    .add_local_file("quack/quack/bench/bench_utils.py", "/workspace/bench_utils.py", copy=False)
)

# IKET first ships with CuTe DSL 4.6. Keep it in a separate image so the
# performance benchmark above remains pinned to its established 4.5.2 compiler.
trace_image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.13")
    .entrypoint([])
    .pip_install(
        "nvidia-cutlass-dsl[cu13]==4.6.0",
        "apache-tvm-ffi",
        "torch==2.12.0",
        "ninja",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .add_local_dir(
        "nvidia-mathdx-26.06.0-cuda13",
        "/opt/mathdx",
        copy=True,
        ignore=["**/doc/**", "**/example/**", "**/__pycache__"],
    )
    .env(
        {
            "MATHDX_ROOT": "/opt/mathdx/nvidia/mathdx/26.06",
            # IKET re-runs the workload in separate compiler passes. Loading an
            # exported object would bypass instrumentation, so disk caching is off.
            "EIGH_ENABLE_DISK_CACHE": "0",
            "TMPDIR": "/cache/tmp",
        }
    )
    .add_local_file("eigh.py", "/workspace/eigh.py", copy=False)
    .add_local_file(
        "eigh_bench_local_rtx4050.py",
        "/workspace/eigh_bench_local_rtx4050.py",
        copy=False,
    )
    # Namespace-package placement lets the runner import this in its child process.
    .add_local_file(
        "quack/quack/bench/bench_utils.py",
        "/workspace/quack/bench/bench_utils.py",
        copy=False,
    )
)

app = modal.App("eigh-b200-bench", image=image)
cache_vol = modal.Volume.from_name("eigh-b200-cache", create_if_missing=True)


def _report_env(tag: str) -> None:
    import os
    import subprocess

    query = (
        "name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu"
    )
    smi = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"[env:{tag}] gpu: {smi.stdout.strip() or smi.stderr.strip()}", flush=True)
    if tag == "start":
        import cutlass
        import torch

        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        release = next(
            (line.strip() for line in nvcc.stdout.splitlines() if "release" in line), "?"
        )
        print(
            f"[env:start] torch={torch.__version__} cutlass={cutlass.__version__} "
            f"nvcc={release}",
            flush=True,
        )
        print(f"[env:start] MATHDX_ROOT={os.environ.get('MATHDX_ROOT')}", flush=True)


def _bench_torch_eigh(data, n_calls: int, repeats: int):
    # End-target yardstick. Plain CUDA-event loop over L2-rotated clones — NOT
    # graph-captured: cuSOLVER's host-side info checks sync and break capture.
    import torch
    from quack.bench.bench_utils import _clone_l2_rotate_inputs, _pick_l2_rotate_count

    n_sets = _pick_l2_rotate_count((data,), {})
    arg_sets, _ = _clone_l2_rotate_inputs((data,), {}, n_sets)
    for i in range(3):
        torch.linalg.eigh(arg_sets[i % n_sets][0])
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(n_calls):
            torch.linalg.eigh(arg_sets[i % n_sets][0])
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / n_calls)
    return samples, n_sets


@app.function(gpu="B200", timeout=3600, volumes={"/cache": cache_vol})
def bench(
    cases: list[tuple[int, int]],
    panel_size: int,
    k: int,
    backends: list[str],
    sets: int,
    calls: int,
    warmup_ms: float,
    repeats: int,
    stat: str,
    matmul_precision: str,
    seed: int,
    bench_eigh: bool,
    tri: bool,
) -> None:
    import os
    import sys
    import types

    # TMPDIR must exist before anything calls tempfile.gettempdir() (it silently
    # falls back to /tmp otherwise, losing the persistent load_inline build root).
    os.makedirs("/cache/tmp", exist_ok=True)
    os.makedirs("/cache/cute", exist_ok=True)
    sys.path.insert(0, "/workspace")

    import bench_utils
    import torch

    # The runner imports quack.bench.bench_utils from the repo checkout, which is
    # not mounted; satisfy the import from the flat-mounted module instead.
    for name, mod in (
        ("quack", types.ModuleType("quack")),
        ("quack.bench", types.ModuleType("quack.bench")),
        ("quack.bench.bench_utils", bench_utils),
    ):
        sys.modules.setdefault(name, mod)

    import eigh_bench_local_rtx4050 as runner

    _report_env("start")
    device = torch.device("cuda")
    summaries: list[str] = []
    failed = False

    for batch, n in cases:
        label = f"case {batch}x{n} ps={panel_size} k={k}"
        print(f"=== {label} ===", flush=True)
        runner.PANEL_SIZE = panel_size
        torch.manual_seed(seed)
        data = runner.make_input(batch, n, runner.DTYPE, device)

        # Gates run under strict fp32 matmul (reset per case: a previous case's
        # timed phase may have lowered it).
        torch.set_float32_matmul_precision("highest")

        if tri:
            # Full-tridiagonalization mode: D/E gate, then graph-captured
            # kernel driver vs the fair torch driver. One timed call is a full
            # factorization (tens of ms) — pass a small --calls (e.g. 20).
            tri_backends = backends or ["cublas"]
            gate_results = [runner.check_tridiag(data, be) for be in tri_backends]
            if not all(gate_results):
                print(f"{label}: TRIDIAG GATE FAILED — skipping timings", flush=True)
                summaries.append(f"{label}: TRIDIAG GATE FAILED")
                failed = True
                continue
            torch.set_float32_matmul_precision(matmul_precision)
            parts = []
            first_ms = None
            for be in tri_backends:
                samples, n_sets = runner.benchmark_tridiag(
                    data, be, sets, calls, warmup_ms, repeats
                )
                tri_ms = runner.select_sample(samples, stat)
                if first_ms is None:
                    first_ms = tri_ms
                print(
                    f"tridiag_{be} sets={n_sets} samples_ms={[f'{s:.4f}' for s in samples]}",
                    flush=True,
                )
                parts.append(f"tridiag_{be}={tri_ms:.4f}ms")
            samples, n_sets = runner.benchmark_torch_tridiag(
                data, sets, calls, warmup_ms, repeats
            )
            torch_tri_ms = runner.select_sample(samples, stat)
            print(
                f"torch_tridiag sets={n_sets} samples_ms={[f'{s:.4f}' for s in samples]}",
                flush=True,
            )
            parts.append(f"torch_tridiag={torch_tri_ms:.4f}ms")
            parts.append(f"ratio={torch_tri_ms / first_ms:.3f}x")
            if bench_eigh:
                samples, n_sets = _bench_torch_eigh(data, max(10, calls // 10), repeats)
                eigh_ms = runner.select_sample(samples, stat)
                print(
                    f"torch_eigh sets={n_sets} samples_ms={[f'{s:.4f}' for s in samples]}",
                    flush=True,
                )
                parts.append(f"torch_eigh={eigh_ms:.4f}ms")
            summaries.append(f"{label}: " + " ".join(parts))
            del data
            torch.cuda.empty_cache()
            continue

        if not runner.check_gemv_allclose(data, k):
            print(f"{label}: PANEL GATE FAILED — skipping timings", flush=True)
            summaries.append(f"{label}: PANEL GATE FAILED")
            failed = True
            continue
        bad_backend = False
        for backend in backends:
            if not runner.check_rank2k_backend(data, k, backend):
                print(f"{label}: rank2k {backend} GATE FAILED — skipping timings", flush=True)
                summaries.append(f"{label}: rank2k {backend} GATE FAILED")
                bad_backend = True
        if bad_backend:
            failed = True
            continue

        torch.set_float32_matmul_precision(matmul_precision)
        parts = []

        samples, n_sets = runner.benchmark_direct(data, k, sets, calls, warmup_ms, repeats)
        panel_ms = runner.select_sample(samples, stat)
        print(f"panel sets={n_sets} samples_ms={[f'{s:.4f}' for s in samples]}", flush=True)
        parts.append(f"panel={panel_ms:.4f}ms")

        samples, n_sets = runner.benchmark_torch_panel(data, k, sets, calls, warmup_ms, repeats)
        torch_ms = runner.select_sample(samples, stat)
        print(f"torch_panel sets={n_sets} samples_ms={[f'{s:.4f}' for s in samples]}", flush=True)
        parts.append(f"torch_panel={torch_ms:.4f}ms")
        parts.append(f"ratio={torch_ms / panel_ms:.3f}x")

        for backend in backends:
            samples, n_sets = runner.benchmark_rank2k(
                data, k, backend, sets, calls, warmup_ms, repeats
            )
            rank2k_ms = runner.select_sample(samples, stat)
            print(
                f"rank2k_{backend} sets={n_sets} samples_ms={[f'{s:.4f}' for s in samples]}",
                flush=True,
            )
            parts.append(f"rank2k_{backend}={rank2k_ms:.4f}ms")

            samples, n_sets = runner.benchmark_panel_with_update(
                data, k, backend, sets, calls, warmup_ms, repeats
            )
            combined_ms = runner.select_sample(samples, stat)
            print(
                f"panel_plus_{backend} sets={n_sets} samples_ms={[f'{s:.4f}' for s in samples]}",
                flush=True,
            )
            parts.append(f"panel_plus_{backend}={combined_ms:.4f}ms")

        if bench_eigh:
            samples, n_sets = _bench_torch_eigh(data, max(10, calls // 10), repeats)
            eigh_ms = runner.select_sample(samples, stat)
            print(
                f"torch_eigh sets={n_sets} samples_ms={[f'{s:.4f}' for s in samples]}",
                flush=True,
            )
            parts.append(f"torch_eigh={eigh_ms:.4f}ms")

        summaries.append(f"{label}: " + " ".join(parts))
        del data
        torch.cuda.empty_cache()

    _report_env("end")
    print("=== summary ===", flush=True)
    for line in summaries:
        print(line, flush=True)
    cache_vol.commit()
    if failed:
        raise SystemExit("one or more correctness gates failed")


@app.function(image=trace_image, gpu="B200", timeout=3600, volumes={"/cache": cache_vol})
def trace_panel(
    batch: int,
    n: int,
    panel_size: int,
    k: int,
    seed: int,
) -> tuple[str, list[tuple[str, str]]]:
    """Gate and trace one panel launch, returning volume paths for its artifacts."""
    from datetime import datetime, timezone
    import os
    from pathlib import Path
    import subprocess
    import sys

    os.makedirs("/cache/tmp", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"n{n}_ps{panel_size}_k{k}_{stamp}"
    output_dir = Path("/cache/iket") / run_name

    env = os.environ.copy()
    env["EIGH_ENABLE_DISK_CACHE"] = "0"
    env["PYTHONPATH"] = "/workspace"
    workload = [
        sys.executable,
        "/workspace/eigh_bench_local_rtx4050.py",
        "--batch",
        str(batch),
        "--n",
        str(n),
        "--panel-size",
        str(panel_size),
        "--k",
        str(k),
        "--seed",
        str(seed),
        "--skip-expected",
    ]

    _report_env("trace")
    print("=== CuTe 4.6 panel correctness preflight ===", flush=True)
    preflight = subprocess.run(workload + ["--check-panel-only"], env=env)
    if preflight.returncode != 0:
        raise RuntimeError(f"CuTe 4.6 panel correctness preflight failed: {preflight.returncode}")

    command = [
        "run-iket",
        "--output-dir",
        str(output_dir),
        "--clobber",
        "profile",
        "--postprocess",
        "all",
        "--max-ts-cnt-per-warp",
        # Sixteen nested ranges per column plus panel/setup metadata: panel
        # size 128 emits 4100 timestamps per warp, so reserve the next power of two.
        "8192",
        "--",
        *workload,
        "--trace-once",
    ]
    print("=== IKET panel trace ===", flush=True)
    print("command: " + " ".join(command), flush=True)
    profiled = subprocess.run(command, env=env)
    if profiled.returncode != 0:
        raise RuntimeError(f"run-iket failed: {profiled.returncode}")

    artifacts = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix in (".pftrace", ".json") and path.stat().st_size > 0
    )
    if not artifacts:
        raise RuntimeError(f"run-iket produced no nonempty trace artifacts under {output_dir}")

    cache_vol.commit()
    result = [
        (str(path.relative_to(output_dir)), str(path.relative_to("/cache")))
        for path in artifacts
    ]
    print(f"IKET artifacts: {[relative for relative, _ in result]}", flush=True)
    return run_name, result


@app.local_entrypoint()
def main(
    cases: str = "640x512",
    panel_size: int = 8,
    k: int = 0,
    backends: str = "cublas,cublasdx",
    sets: int = 0,
    calls: int = 200,
    warmup_ms: float = 200.0,
    repeats: int = 3,
    stat: str = "median",
    matmul_precision: str = "high",
    seed: int = 0,
    bench_eigh: bool = False,
    tri: bool = False,
    trace: bool = False,
    trace_batch: int = 1,
    trace_output: str = "profiles/eigh_iket",
) -> None:
    parsed_cases = []
    for item in cases.split(","):
        batch, _, n = item.strip().lower().partition("x")
        parsed_cases.append((int(batch), int(n)))
    backend_list = [b.strip() for b in backends.split(",") if b.strip()]
    for backend in backend_list:
        if backend not in ("cublas", "cublasdx"):
            raise SystemExit(f"unknown backend: {backend}")
    if stat not in ("min", "mean", "median"):
        raise SystemExit(f"unknown stat: {stat}")
    if trace:
        if len(parsed_cases) != 1:
            raise SystemExit("--trace requires exactly one --cases entry")
        if trace_batch <= 0:
            raise SystemExit("--trace-batch must be positive")
        _, n = parsed_cases[0]
        run_name, artifacts = trace_panel.remote(trace_batch, n, panel_size, k, seed)

        from pathlib import Path

        destination = Path(trace_output) / run_name
        downloaded: list[Path] = []
        for relative, remote_path in artifacts:
            local_path = destination / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("wb") as fileobj:
                cache_vol.read_file_into_fileobj(remote_path, fileobj)
            downloaded.append(local_path)
            print(f"downloaded {local_path}")
        for path in downloaded:
            if path.suffix == ".pftrace":
                print(f"Perfetto trace: {path} (open in https://ui.perfetto.dev/)")
        return
    bench.remote(
        parsed_cases,
        panel_size,
        k,
        backend_list,
        sets,
        calls,
        warmup_ms,
        repeats,
        stat,
        matmul_precision,
        seed,
        bench_eigh,
        tri,
    )
