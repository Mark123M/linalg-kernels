"""Modal B200 Nsight Systems launcher for the b8n2048 Cholesky variants.

Examples:
    .venv/bin/python -m modal run cholesky/b8n2048/cholesky_b8n2048_modal.py
    .venv/bin/python -m modal run cholesky/b8n2048/cholesky_b8n2048_modal.py --variant 5
    .venv/bin/python -m modal run cholesky/b8n2048/cholesky_b8n2048_modal.py \
        --output profiles/cholesky_nsys

The default ``--variant -1`` resolves to the tracked ``_DEFAULT_VARIANT``.
Input construction, extension compilation, preparation, warmup, and
correctness validation all happen outside the profiler capture. The report
contains exactly one factorization into a preallocated output tensor.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import modal


BATCH = 8
N = 2048
SEED = 512048
LOW_RANK = 32

LOCAL_SOLUTION = "cholesky/b8n2048/cholesky_b8n2048.py"
LOCAL_SCRIPT = "cholesky/b8n2048/cholesky_b8n2048_modal.py"
REMOTE_SOLUTION = "/workspace/cholesky_b8n2048.py"
REMOTE_SCRIPT = "/workspace/cholesky_b8n2048_modal.py"

CUDA_IMAGE = "nvidia/cuda:13.3.0-devel-ubuntu24.04"
CUDA_UPDATE_PACKAGES = (
    "cuda-command-line-tools-13-3=13.3.1-1",
    "cuda-compiler-13-3=13.3.1-1",
    "cuda-libraries-dev-13-3=13.3.1-1",
    "libcublas-13-3=13.6.0.2-1",
    "libcublas-dev-13-3=13.6.0.2-1",
)
NSYS_DEB_URL = (
    "https://developer.nvidia.com/downloads/assets/tools/secure/"
    "nsight-systems/2026_3/"
    "NsightSystems-linux-cli-public-2026.3.1.157-3804839.deb"
)
NSYS_DEB_PATH = "/tmp/NsightSystems-linux-cli-public-2026.3.1.157-3804839.deb"
NSYS_DEB_SHA256 = (
    "3eb87ec08e5f8b8f153537847747bd5cfabb51b9c8793873b26a3c55dc813ad1"
)
_WORKER_PROCESS = len(sys.argv) > 1 and sys.argv[1].startswith("_worker_")


def _base_image() -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.13")
        .entrypoint([])
        .run_commands("apt-mark unhold libcublas-13-3 libcublas-dev-13-3")
        .apt_install(*CUDA_UPDATE_PACKAGES)
        .run_commands("apt-mark hold libcublas-13-3 libcublas-dev-13-3")
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
                "NV_CUDA_LIB_VERSION": "13.3.1-1",
                "NV_LIBCUBLAS_VERSION": "13.6.0.2-1",
                "NV_LIBCUBLAS_PACKAGE": "libcublas-13-3=13.6.0.2-1",
                "NV_LIBCUBLAS_DEV_VERSION": "13.6.0.2-1",
                "NV_LIBCUBLAS_DEV_PACKAGE": (
                    "libcublas-dev-13-3=13.6.0.2-1"
                ),
            }
        )
    )


def _nsys_base_image() -> modal.Image:
    return (
        _base_image()
        .apt_install("curl")
        .run_commands(
            f"curl -fL --retry 3 {NSYS_DEB_URL} -o {NSYS_DEB_PATH}",
            (
                f"echo '{NSYS_DEB_SHA256}  {NSYS_DEB_PATH}' "
                "| sha256sum --check --strict"
            ),
            (
                "apt-get update && apt-get install -y --no-install-recommends "
                f"{NSYS_DEB_PATH} && rm -f {NSYS_DEB_PATH}"
            ),
            "nsys --version | grep -F '2026.3.1'",
        )
    )


def _mount_sources(image: modal.Image) -> modal.Image:
    return image.add_local_file(
        LOCAL_SOLUTION, REMOTE_SOLUTION, copy=False
    ).add_local_file(LOCAL_SCRIPT, REMOTE_SCRIPT, copy=False)


if _WORKER_PROCESS:
    nsys_image = modal.Image.debian_slim()
else:
    nsys_image = _mount_sources(_nsys_base_image())

app = modal.App("cholesky-b8n2048-nsys-b200", image=nsys_image)
cache_volume = modal.Volume.from_name(
    "cholesky-b8n2048-cache", create_if_missing=True
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
    spec = importlib.util.spec_from_file_location("cholesky_profile_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solution from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _variant_ids(solution) -> tuple[int, ...]:
    values = getattr(solution, "_VARIANT_IDS", None)
    if values is None:
        values = getattr(solution, "_NATIVE_VARIANTS", None)
    if values is None:
        raise RuntimeError("solution exposes neither _VARIANT_IDS nor _NATIVE_VARIANTS")
    return tuple(int(value) for value in values)


def _resolve_variant(solution, requested: int) -> int:
    variant = int(solution._DEFAULT_VARIANT) if requested == -1 else requested
    valid = _variant_ids(solution)
    if variant not in valid:
        raise ValueError(f"variant must be -1 or one of {valid}, got {requested}")
    return variant


def _variant_name(solution, variant: int) -> str:
    names = tuple(str(value) for value in solution._VARIANT_NAMES)
    if not 0 <= variant < len(names):
        raise RuntimeError(f"variant {variant} has no registered name")
    return names[variant]


def _make_input():
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)
    factors = torch.randn(
        (BATCH, N, LOW_RANK),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    data = factors @ factors.transpose(1, 2)
    data.mul_(1.0 / LOW_RANK)
    data.diagonal(dim1=1, dim2=2).add_(4.0)
    return data.contiguous()


def _scaled_reconstruction_residual(data, factor) -> float:
    import torch

    reconstructed = factor @ factor.transpose(1, 2)
    residual_norm = (data - reconstructed).abs().sum(dim=2).amax(dim=1)
    data_norm = data.abs().sum(dim=2).amax(dim=1).clamp_min(1.0e-30)
    scaled = residual_norm / (torch.finfo(torch.float32).eps * N * data_norm)
    return float(scaled.amax().item())


def _validate_factor(data, factor, reference_scaled: float) -> dict[str, Any]:
    import math
    import torch

    shape_ok = tuple(factor.shape) == (BATCH, N, N)
    dtype_ok = factor.dtype == torch.float32
    device_ok = factor.is_cuda and factor.device == data.device
    finite = bool(torch.isfinite(factor).all().item())
    upper_max = (
        float(torch.triu(factor, diagonal=1).abs().amax().item())
        if finite
        else math.inf
    )
    diagonal_min = (
        float(factor.diagonal(dim1=1, dim2=2).amin().item())
        if finite
        else -math.inf
    )
    scaled = _scaled_reconstruction_residual(data, factor) if finite else math.inf
    limit = max(16.0, 8.0 * reference_scaled)
    passed = (
        shape_ok
        and dtype_ok
        and device_ok
        and finite
        and upper_max == 0.0
        and diagonal_min > 0.0
        and scaled <= limit
    )
    return {
        "passed": passed,
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "device_ok": device_ok,
        "finite": finite,
        "upper_max": upper_max,
        "diagonal_min": diagonal_min,
        "scaled_reconstruction_residual": scaled,
        "reference_scaled_reconstruction_residual": reference_scaled,
        "validation_limit": limit,
    }


def _environment() -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(0)
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    return {
        "gpu": smi.stdout.strip() or smi.stderr.strip(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability()),
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory": properties.total_memory,
        "nvcc": nvcc.stdout.strip() or nvcc.stderr.strip(),
    }


def _metadata(solution, variant: int) -> dict[str, int]:
    columns = tuple(str(value) for value in solution._METADATA_COLUMNS)
    values = solution._variant_metadata()[variant].tolist()
    if len(columns) != len(values):
        raise RuntimeError("variant metadata columns and values differ in length")
    return {column: int(value) for column, value in zip(columns, values)}


def _worker_preflight(requested: int) -> None:
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    solution = _load_solution()
    variant = _resolve_variant(solution, requested)
    data = _make_input()
    factor = solution._run_variant(data, variant)
    torch.cuda.synchronize()
    reference = torch.linalg.cholesky_ex(data, check_errors=False).L
    reference_scaled = _scaled_reconstruction_residual(data, reference)
    validation = _validate_factor(data, factor, reference_scaled)
    payload = {
        "shape": [BATCH, N, N],
        "seed": SEED,
        "input": f"damped-rank-{LOW_RANK}",
        "variant": variant,
        "name": _variant_name(solution, variant),
        "metadata": _metadata(solution, variant),
        "validation": validation,
        "environment": _environment(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not validation["passed"]:
        raise RuntimeError(f"variant {variant} failed profiling preflight")


def _worker_profile(requested: int) -> None:
    import torch

    solution = _load_solution()
    variant = _resolve_variant(solution, requested)
    name = _variant_name(solution, variant)
    data = _make_input()
    output = torch.empty_like(data)
    solution._run_variant(data, variant, output)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"b{BATCH}_n{N}_v{variant}_{name}")
    solution._run_variant(data, variant, output)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text if text else "<empty>\n")
    if path.stat().st_size == 0:
        raise RuntimeError(f"wrote empty artifact: {path}")


def _run_report(
    sqlite_path: Path,
    report: str | None,
    output_path: Path,
    output_format: str,
) -> None:
    command = ["nsys", "stats", "--quiet"]
    if report is not None:
        command.extend(("--report", report))
    command.extend(("--format", output_format, "--output", "-", str(sqlite_path)))
    completed = subprocess.run(command, capture_output=True, text=True)
    _write_text(output_path, completed.stdout)
    if completed.stderr:
        _write_text(
            output_path.with_suffix(output_path.suffix + ".stderr.txt"),
            completed.stderr,
        )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(
            f"nsys stats report {report or 'default'} failed: "
            f"{completed.stderr[-4000:]}"
        )


@app.function(
    image=nsys_image,
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def profile_nsys(
    requested_variant: int,
) -> tuple[str, int, str, list[str]]:
    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)
    solution = _load_solution()
    variant = _resolve_variant(solution, requested_variant)
    name = _variant_name(solution, variant)
    run_name = f"b{BATCH}_n{N}_{_timestamp()}"
    output_dir = Path("/cache/nsys") / run_name / f"v{variant}_{name}"
    output_dir.mkdir(parents=True, exist_ok=False)

    preflight = subprocess.run(
        [sys.executable, REMOTE_SCRIPT, "_worker_preflight", str(variant)],
        capture_output=True,
        text=True,
    )
    _write_text(output_dir / "preflight.json", preflight.stdout)
    _write_text(output_dir / "preflight.stderr.txt", preflight.stderr)
    if preflight.returncode != 0:
        raise RuntimeError(
            f"variant {variant} preflight failed with code {preflight.returncode}: "
            f"{preflight.stderr[-4000:]}"
        )
    try:
        preflight_payload = json.loads(preflight.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("preflight did not emit valid JSON") from error
    if not preflight_payload.get("validation", {}).get("passed", False):
        raise RuntimeError(f"variant {variant} preflight reported a failed validation")
    _write_text(
        output_dir / "environment.json",
        json.dumps(preflight_payload["environment"], indent=2, sort_keys=True) + "\n",
    )

    version = subprocess.run(["nsys", "--version"], capture_output=True, text=True)
    _write_text(output_dir / "nsys-version.txt", version.stdout + version.stderr)
    if version.returncode != 0:
        raise RuntimeError(f"nsys --version failed: {version.stderr[-4000:]}")

    report_base = output_dir / "profile"
    report_path = report_base.with_suffix(".nsys-rep")
    sqlite_path = report_base.with_suffix(".sqlite")
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,cublas",
        "--sample=none",
        "--cpuctxsw=none",
        "--cudabacktrace=none",
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
    ]
    _write_text(
        output_dir / "nsys-command.json",
        json.dumps(command, indent=2) + "\n",
    )
    profiled = subprocess.run(command, capture_output=True, text=True)
    _write_text(output_dir / "nsys.stdout.txt", profiled.stdout)
    _write_text(output_dir / "nsys.stderr.txt", profiled.stderr)
    if profiled.returncode != 0:
        raise RuntimeError(
            f"variant {variant} nsys failed with code {profiled.returncode}: "
            f"{profiled.stderr[-4000:]}"
        )
    for artifact in (report_path, sqlite_path):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"missing or empty Nsight Systems artifact: {artifact}")

    _run_report(
        sqlite_path, "cuda_gpu_kern_sum", output_dir / "kernel-summary.csv", "csv"
    )
    _run_report(
        sqlite_path, "cuda_gpu_trace", output_dir / "kernel-trace.csv", "csv"
    )
    _run_report(
        sqlite_path,
        "cuda_kern_exec_trace",
        output_dir / "kernel-exec-trace.csv",
        "csv",
    )
    _run_report(sqlite_path, None, output_dir / "stats.txt", "table")

    artifacts = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.stat().st_size > 0
    ]
    expected = {
        "profile.nsys-rep",
        "profile.sqlite",
        "kernel-summary.csv",
        "kernel-trace.csv",
        "kernel-exec-trace.csv",
        "stats.txt",
        "preflight.json",
        "preflight.stderr.txt",
        "environment.json",
        "nsys-version.txt",
        "nsys-command.json",
        "nsys.stdout.txt",
        "nsys.stderr.txt",
    }
    missing = sorted(expected.difference(path.name for path in artifacts))
    if missing:
        raise RuntimeError(f"missing profiling artifacts: {missing}")

    cache_volume.commit()
    return (
        run_name,
        variant,
        name,
        [str(path.relative_to("/cache")) for path in artifacts],
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@app.local_entrypoint()
def main(variant: int = -1, output: str = "") -> None:
    run_name, resolved_variant, name, remote_paths = profile_nsys.remote(variant)
    output_root = (
        Path(output)
        if output
        else _repo_root() / "cholesky/b8n2048/artifacts/nsys"
    )
    destination = output_root / run_name / f"v{resolved_variant}_{name}"
    for remote_path in remote_paths:
        local_path = destination / Path(remote_path).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as file_object:
            cache_volume.read_file_into_fileobj(remote_path, file_object)
        if local_path.stat().st_size == 0:
            raise RuntimeError(f"downloaded empty artifact: {local_path}")
        print(f"downloaded {local_path}")
    reports = list(destination.glob("*.nsys-rep"))
    if len(reports) != 1:
        raise RuntimeError(f"expected one Nsight Systems report, found {len(reports)}")
    print(f"Nsight Systems report: {reports[0]}")


if __name__ == "__main__" and _WORKER_PROCESS:
    mode = sys.argv[1]
    if len(sys.argv) != 3:
        raise SystemExit(f"{mode} requires one variant argument")
    requested = int(sys.argv[2])
    if mode == "_worker_preflight":
        _worker_preflight(requested)
    elif mode == "_worker_profile":
        _worker_profile(requested)
    else:
        raise SystemExit(f"unknown worker mode: {mode}")
