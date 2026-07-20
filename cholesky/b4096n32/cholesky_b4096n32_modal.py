"""B200 tuning, profiling, and Popcorn orchestration for batched 4096x32.

The script deliberately contains no CUDA algorithm implementation. It imports
the exact source that is submitted to Popcorn, so profiling and leaderboard
runs exercise identical kernels.

Examples:
    .venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
        --action tune
    .venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
        --action profile
    .venv/bin/modal run cholesky/b4096n32/cholesky_b4096n32_modal.py \
        --action popcorn
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from types import ModuleType
from typing import Any

import modal


VARIANT_COUNT = 10
VARIANT_NAMES = (
    "w1_precise",
    "w1_refined",
    "w2_precise",
    "w2_refined",
    "w4_precise",
    "w4_refined",
    "w8_precise",
    "w8_refined",
    "w16_precise",
    "w16_refined",
)
LOCAL_SOLUTION = "cholesky/b4096n32/cholesky_b4096n32.py"
LOCAL_SCRIPT = "cholesky/b4096n32/cholesky_b4096n32_modal.py"
REMOTE_SOLUTION = "/workspace/cholesky_b4096n32.py"
REMOTE_SCRIPT = "/workspace/cholesky_b4096n32_modal.py"
_WORKER_PROCESS = (
    len(sys.argv) > 1 and sys.argv[1].startswith("_worker_")
)


def _base_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.13"
        )
        .entrypoint([])
        .pip_install(
            "torch==2.12.0",
            "ninja",
            extra_index_url="https://download.pytorch.org/whl/cu130",
        )
        .env(
            {
                "TORCH_EXTENSIONS_DIR": "/cache/torch_extensions",
                "TMPDIR": "/cache/tmp",
                "CC": "gcc",
                "CXX": "g++",
            }
        )
    )


def _mount_sources(image: modal.Image) -> modal.Image:
    return image.add_local_file(
        LOCAL_SOLUTION, REMOTE_SOLUTION, copy=False
    ).add_local_file(LOCAL_SCRIPT, REMOTE_SCRIPT, copy=False)


if _WORKER_PROCESS:
    # Child processes launched under nsys only need the workload helpers below.
    # Avoid asking Modal to resolve client-side mount paths inside the container.
    benchmark_image = modal.Image.debian_slim()
    nsys_image = benchmark_image
else:
    benchmark_image = _mount_sources(_base_image())
    nsys_image = _mount_sources(_base_image().apt_install("cuda-nsight-systems-13-1"))
app = modal.App("cholesky-b4096n32-b200", image=benchmark_image)
cache_volume = modal.Volume.from_name(
    "cholesky-b4096n32-cache", create_if_missing=True
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _install_task_stub() -> None:
    if "task" in sys.modules:
        return
    import torch

    task = ModuleType("task")
    task.input_t = torch.Tensor
    task.output_t = torch.Tensor
    sys.modules["task"] = task


def _load_solution():
    _install_task_stub()
    path = Path(REMOTE_SOLUTION)
    spec = importlib.util.spec_from_file_location("chol_b4096n32_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report_environment() -> dict[str, Any]:
    import torch

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,multiprocessor_count,clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    return {
        "gpu": smi.stdout.strip() or smi.stderr.strip(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability()),
        "nvcc": nvcc.stdout.strip(),
    }


def _generator(seed: int):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def _make_input(case: str, seed: int):
    import torch

    batch, n = 4096, 32
    device = torch.device("cuda")
    generator = _generator(seed)
    eye = torch.eye(n, device=device, dtype=torch.float32).expand(batch, n, n)

    if case == "diagonal":
        values = torch.logspace(0.0, math.log10(5.0), n, device=device)
        return torch.diag_embed(values.expand(batch, n)).contiguous()

    if case == "tridiagonal":
        diagonal = torch.full((batch, n), 2.0, device=device)
        off_diagonal = torch.full((batch, n - 1), -0.25, device=device)
        return (
            torch.diag_embed(diagonal)
            + torch.diag_embed(off_diagonal, offset=1)
            + torch.diag_embed(off_diagonal, offset=-1)
        ).contiguous()

    if case == "lowrank":
        factors = torch.randn(
            batch, n, 4, device=device, dtype=torch.float32, generator=generator
        )
        return (factors @ factors.transpose(1, 2) + 0.5 * eye).contiguous()

    source = torch.randn(
        batch, n, n, device=device, dtype=torch.float32, generator=generator
    )
    if case == "spectrum":
        q, _ = torch.linalg.qr(source)
        values = torch.logspace(0.0, math.log10(5.0), n, device=device)
        return ((q * values.view(1, 1, n)) @ q.transpose(1, 2)).contiguous()

    dense = source @ source.transpose(1, 2) + 4.0 * eye
    if case == "rowscale":
        scales = torch.logspace(
            0.0, 0.5 * math.log10(4.0), n, device=device
        ).view(1, n)
        dense = dense * scales.unsqueeze(2) * scales.unsqueeze(1)
    elif case != "dense":
        raise ValueError(f"unknown case: {case}")
    return dense.contiguous()


def _scaled_reconstruction_residual(data, factor) -> float:
    import torch

    reconstructed = factor @ factor.transpose(1, 2)
    residual_norm = (data - reconstructed).abs().sum(dim=2).amax(dim=1)
    data_norm = data.abs().sum(dim=2).amax(dim=1).clamp_min(1.0e-30)
    scaled = residual_norm / (torch.finfo(torch.float32).eps * 32 * data_norm)
    return float(scaled.amax().item())


def _validate_factor(data, factor, reference_scaled: float) -> dict[str, Any]:
    import torch

    finite = bool(torch.isfinite(factor).all().item())
    upper_max = float(torch.triu(factor, diagonal=1).abs().amax().item())
    diagonal_min = float(factor.diagonal(dim1=1, dim2=2).amin().item())
    scaled = _scaled_reconstruction_residual(data, factor) if finite else math.inf
    limit = max(16.0, 8.0 * reference_scaled)
    passed = finite and upper_max == 0.0 and diagonal_min > 0.0 and scaled <= limit
    return {
        "passed": passed,
        "finite": finite,
        "upper_max": upper_max,
        "diagonal_min": diagonal_min,
        "scaled_reconstruction_residual": scaled,
        "reference_scaled_reconstruction_residual": reference_scaled,
        "modal_gate": limit,
    }


def _resource_rows(solution) -> list[dict[str, Any]]:
    columns = (
        "variant",
        "warps_per_cta",
        "refined_root",
        "threads_per_cta",
        "registers_per_thread",
        "local_bytes_per_thread",
        "static_shared_bytes_per_cta",
        "active_ctas_per_sm",
        "resident_warps_per_sm",
    )
    values = solution._variant_metadata().tolist()
    return [dict(zip(columns, row, strict=True)) for row in values]


def _measure_variants(solution, calls: int, repeats: int) -> list[dict[str, Any]]:
    import torch

    ring_size = 8
    inputs = [_make_input("dense", 41032 + index) for index in range(ring_size)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(ring_size)]
    samples: dict[int, list[float]] = {variant: [] for variant in range(VARIANT_COUNT)}

    for variant in range(VARIANT_COUNT):
        for index in range(16):
            slot = index % ring_size
            solution._run_variant(inputs[slot], variant, outputs[slot])
    torch.cuda.synchronize()

    for repeat in range(repeats):
        order = list(range(VARIANT_COUNT))
        if repeat & 1:
            order.reverse()
        for variant in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for index in range(calls):
                slot = index % ring_size
                solution._run_variant(inputs[slot], variant, outputs[slot])
            end.record()
            end.synchronize()
            samples[variant].append(float(start.elapsed_time(end)) / calls)

    results = []
    for variant, values in samples.items():
        results.append(
            {
                "variant": variant,
                "name": VARIANT_NAMES[variant],
                "samples_ms": values,
                "median_ms": statistics.median(values),
                "minimum_ms": min(values),
            }
        )
    results.sort(key=lambda row: row["median_ms"])
    return results


def _run_tuning(calls: int, repeats: int) -> dict[str, Any]:
    import torch

    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)
    solution = _load_solution()
    validation: dict[str, list[dict[str, Any]]] = {
        str(variant): [] for variant in range(VARIANT_COUNT)
    }
    cases = ("dense", "spectrum", "diagonal", "lowrank", "rowscale", "tridiagonal")
    for case_index, case in enumerate(cases):
        data = _make_input(case, 53124 + case_index)
        reference = torch.linalg.cholesky_ex(data, check_errors=False).L
        reference_scaled = _scaled_reconstruction_residual(data, reference)
        for variant in range(VARIANT_COUNT):
            factor = solution._run_variant(data, variant)
            result = _validate_factor(data, factor, reference_scaled)
            result["case"] = case
            validation[str(variant)].append(result)
        del data, reference

    resources = _resource_rows(solution)
    timings = _measure_variants(solution, calls, repeats)
    for row in timings:
        checks = validation[str(row["variant"])]
        row["modal_validation_passed"] = all(check["passed"] for check in checks)
    return {
        "timestamp_utc": _timestamp(),
        "environment": _report_environment(),
        "calls_per_sample": calls,
        "repeats": repeats,
        "resources": resources,
        "validation": validation,
        "timings": timings,
    }


@app.function(gpu="B200", timeout=3600, volumes={"/cache": cache_volume})
def tune_remote(calls: int, repeats: int) -> dict[str, Any]:
    return _run_tuning(calls, repeats)


def _worker_preflight(variant: int) -> None:
    import torch

    solution = _load_solution()
    data = _make_input("dense", 41032)
    out = torch.empty_like(data)
    for _ in range(8):
        solution._run_variant(data, variant, out)
    torch.cuda.synchronize()
    resources = _resource_rows(solution)[variant]
    factor = solution._run_variant(data, variant)
    reference = torch.linalg.cholesky_ex(data, check_errors=False).L
    reference_scaled = _scaled_reconstruction_residual(data, reference)
    payload = {
        "variant": variant,
        "name": VARIANT_NAMES[variant],
        "resources": resources,
        "validation": _validate_factor(data, factor, reference_scaled),
        "environment": _report_environment(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _worker_profile(variant: int, calls: int) -> None:
    import torch

    solution = _load_solution()
    ring_size = 8
    inputs = [_make_input("dense", 41032 + index) for index in range(ring_size)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(ring_size)]
    for index in range(16):
        slot = index % ring_size
        solution._run_variant(inputs[slot], variant, outputs[slot])
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"variant_{variant}_{VARIANT_NAMES[variant]}")
    for index in range(calls):
        slot = index % ring_size
        solution._run_variant(inputs[slot], variant, outputs[slot])
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


@app.function(
    image=nsys_image,
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def profile_remote(variant: int, calls: int, run_name: str) -> list[str]:
    output_dir = Path("/cache/nsys") / run_name / f"v{variant}_{VARIANT_NAMES[variant]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)

    preflight = subprocess.run(
        [sys.executable, REMOTE_SCRIPT, "_worker_preflight", str(variant)],
        capture_output=True,
        text=True,
    )
    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(preflight.stdout)
    if preflight.returncode != 0:
        (output_dir / "preflight.stderr.txt").write_text(preflight.stderr)
        raise RuntimeError(
            f"variant {variant} preflight failed with code {preflight.returncode}: "
            f"{preflight.stderr[-2000:]}"
        )

    report_base = output_dir / "profile"
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "--export=sqlite",
        "--output",
        str(report_base),
        "--",
        sys.executable,
        REMOTE_SCRIPT,
        "_worker_profile",
        str(variant),
        str(calls),
    ]
    profiled = subprocess.run(command, capture_output=True, text=True)
    (output_dir / "nsys.stdout.txt").write_text(profiled.stdout)
    (output_dir / "nsys.stderr.txt").write_text(profiled.stderr)
    if profiled.returncode != 0:
        raise RuntimeError(
            f"variant {variant} nsys failed with code {profiled.returncode}: "
            f"{profiled.stderr[-2000:]}"
        )

    report_path = report_base.with_suffix(".nsys-rep")
    sqlite_path = report_base.with_suffix(".sqlite")
    for path in (report_path, sqlite_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty Nsight artifact: {path}")

    stats = subprocess.run(
        ["nsys", "stats", "--format", "table", "--output", "-", str(sqlite_path)],
        capture_output=True,
        text=True,
    )
    stats_path = output_dir / "stats.txt"
    stats_path.write_text(stats.stdout + stats.stderr)
    if stats.returncode != 0 or not stats.stdout.strip():
        raise RuntimeError(f"variant {variant} nsys stats failed")

    cache_volume.commit()
    return [
        str(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.stat().st_size > 0
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if path.stat().st_size == 0:
        raise RuntimeError(f"wrote empty result: {path}")


def _parse_variants(text: str) -> list[int]:
    if not text.strip():
        return list(range(VARIANT_COUNT))
    result = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not result or result[0] < 0 or result[-1] >= VARIANT_COUNT:
        raise ValueError(f"variants must be comma-separated IDs in [0, {VARIANT_COUNT - 1}]")
    return result


def _download_profile_artifacts(
    variants: list[int], calls: int, output_root: Path
) -> None:
    run_name = f"b4096_n32_{_timestamp()}"
    destination = output_root / run_name
    for variant in variants:
        remote_paths = profile_remote.remote(variant, calls, run_name)
        variant_dir = destination / f"v{variant}_{VARIANT_NAMES[variant]}"
        for remote_path in remote_paths:
            local_path = variant_dir / Path(remote_path).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("wb") as file_object:
                cache_volume.read_file_into_fileobj(remote_path, file_object)
            if local_path.stat().st_size == 0:
                raise RuntimeError(f"downloaded empty artifact: {local_path}")
            print(f"downloaded {local_path}")


def _variant_source(source: str, variant: int) -> str:
    pattern = r"^_DEFAULT_VARIANT = \d+  # POPCORN_VARIANT$"
    replacement = f"_DEFAULT_VARIANT = {variant}  # POPCORN_VARIANT"
    rendered, count = re.subn(pattern, replacement, source, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"expected one production variant marker, replaced {count}")
    ast.parse(rendered)
    rejected = "s" + "tream"
    if rejected in rendered:
        raise RuntimeError(f"temporary submission contains rejected token: {rejected}")
    return rendered


def _parse_public_score(text: str) -> float | None:
    match = re.search(
        r"Geomean score \(public\) on B200:\s*([0-9.eE+-]+)\s*s", text
    )
    return float(match.group(1)) if match else None


def _run_popcorn_sweep(output_root: Path) -> None:
    source_path = _repo_root() / LOCAL_SOLUTION
    source = source_path.read_text()
    run_dir = output_root / f"popcorn_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []

    def submit_one(
        variant: int,
        submission_path: Path,
        result_path: Path,
    ) -> dict[str, Any]:
        command = [
            "popcorn",
            "submit",
            "--no-tui",
            "--leaderboard",
            "cholesky",
            "--gpu",
            "B200",
            "--mode",
            "leaderboard",
            "--output",
            str(result_path),
            str(submission_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        (run_dir / f"v{variant}.stdout.txt").write_text(completed.stdout)
        (run_dir / f"v{variant}.stderr.txt").write_text(completed.stderr)
        result_text = result_path.read_text() if result_path.is_file() else ""
        combined = "\n".join((result_text, completed.stdout, completed.stderr))
        return {
            "variant": variant,
            "name": VARIANT_NAMES[variant],
            "returncode": completed.returncode,
            "public_b200_geomean_seconds": _parse_public_score(combined),
            "result_file": str(result_path),
        }

    with tempfile.TemporaryDirectory(prefix="chol_b4096n32_") as temporary:
        temporary_dir = Path(temporary)
        jobs: list[tuple[int, Path, Path]] = []
        for variant in range(VARIANT_COUNT):
            submission_path = temporary_dir / f"cholesky_b4096n32_v{variant}.py"
            submission_path.write_text(_variant_source(source, variant))
            result_path = run_dir / f"v{variant}_{VARIANT_NAMES[variant]}.result.txt"
            jobs.append((variant, submission_path, result_path))

        with ThreadPoolExecutor(max_workers=VARIANT_COUNT) as executor:
            pending = {}
            for variant, submission_path, result_path in jobs:
                future = executor.submit(
                    submit_one, variant, submission_path, result_path
                )
                pending[future] = variant
                print(
                    f"queued variant {variant}: {VARIANT_NAMES[variant]}",
                    flush=True,
                )

            for future in as_completed(pending):
                row = future.result()
                rows.append(row)
                rows.sort(key=lambda item: item["variant"])
                _write_json(run_dir / "progress.json", {"submissions": rows})
                print(
                    f"completed variant {row['variant']} "
                    f"returncode={row['returncode']} "
                    f"score={row['public_b200_geomean_seconds']}",
                    flush=True,
                )

    ranked = sorted(
        (row for row in rows if row["public_b200_geomean_seconds"] is not None),
        key=lambda row: row["public_b200_geomean_seconds"],
    )
    summary = {
        "timestamp_utc": _timestamp(),
        "policy": "all ten variants concurrently submitted through Popcorn leaderboard mode",
        "submissions": rows,
        "ranking": ranked,
        "winner": ranked[0] if ranked else None,
    }
    summary_path = run_dir / "summary.json"
    _write_json(summary_path, summary)
    if ranked:
        winner = ranked[0]
        print(
            f"winner: variant {winner['variant']} ({winner['name']}), "
            f"geomean={winner['public_b200_geomean_seconds']} s"
        )
    else:
        print("no successful public B200 score was parsed")
    print(f"Popcorn summary: {summary_path}")


@app.local_entrypoint()
def main(
    action: str = "tune",
    output: str = "",
    variants: str = "",
    calls: int = 200,
    repeats: int = 5,
    profile_calls: int = 64,
) -> None:
    selected = _parse_variants(variants)
    root = _repo_root()
    default_artifacts = root / "cholesky/b4096n32/artifacts"

    if action == "tune":
        if calls <= 0 or repeats <= 0:
            raise ValueError("calls and repeats must be positive")
        payload = tune_remote.remote(calls, repeats)
        output_path = (
            Path(output)
            if output
            else default_artifacts / f"tuning_{payload['timestamp_utc']}.json"
        )
        _write_json(output_path, payload)
        print(f"tuning JSON: {output_path}")
        best = payload["timings"][0]
        print(
            f"Modal winner: variant {best['variant']} ({best['name']}), "
            f"median={best['median_ms']:.6f} ms"
        )
        return

    if action == "profile":
        if profile_calls <= 0:
            raise ValueError("profile_calls must be positive")
        output_root = Path(output) if output else default_artifacts / "nsys"
        _download_profile_artifacts(selected, profile_calls, output_root)
        return

    if action == "popcorn":
        if variants.strip():
            raise ValueError("Popcorn policy requires all ten variants; omit --variants")
        output_root = Path(output) if output else default_artifacts
        _run_popcorn_sweep(output_root)
        return

    raise ValueError("action must be one of: tune, profile, popcorn")


def _worker_cli() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("worker mode requires a variant")
    mode = sys.argv[1]
    variant = int(sys.argv[2])
    if not 0 <= variant < VARIANT_COUNT:
        raise SystemExit(f"variant must be in [0, {VARIANT_COUNT - 1}]")
    if mode == "_worker_preflight":
        _worker_preflight(variant)
        return
    if mode == "_worker_profile":
        if len(sys.argv) != 4:
            raise SystemExit("profile worker requires call count")
        _worker_profile(variant, int(sys.argv[3]))
        return
    raise SystemExit(f"unknown worker mode: {mode}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1].startswith("_worker_"):
    _worker_cli()
