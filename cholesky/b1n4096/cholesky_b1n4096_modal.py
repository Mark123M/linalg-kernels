"""Modal B200 Nsight Systems launcher for the b1n4096 Cholesky variants.

Examples:
    .venv/bin/python -m modal run cholesky/b1n4096/cholesky_b1n4096_modal.py
    .venv/bin/python -m modal run cholesky/b1n4096/cholesky_b1n4096_modal.py --variant 5
    .venv/bin/python -m modal run cholesky/b1n4096/cholesky_b1n4096_modal.py \
        --output profiles/cholesky_nsys
    .venv/bin/python -m modal run cholesky/b1n4096/cholesky_b1n4096_modal.py \
        --panel-configs all
    .venv/bin/python -m modal run cholesky/b1n4096/cholesky_b1n4096_modal.py \
        --panel-ncu-configs 32,57
    .venv/bin/python -m modal run cholesky/b1n4096/cholesky_b1n4096_modal.py \
        --ncu-variant 0

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
import statistics
import subprocess
import sys
from types import ModuleType
from typing import Any

import modal


BATCH = 1
N = 4096
SEED = 48096
LOW_RANK = 32

LOCAL_SOLUTION = "cholesky/b1n4096/cholesky_b1n4096.py"
LOCAL_SCRIPT = "cholesky/b1n4096/cholesky_b1n4096_modal.py"
REMOTE_SOLUTION = "/workspace/cholesky_b1n4096.py"
REMOTE_SCRIPT = "/workspace/cholesky_b1n4096_modal.py"

CUDA_IMAGE = "nvidia/cuda:13.3.0-devel-ubuntu24.04"
CUDA_UPDATE_PACKAGES = (
    "cuda-command-line-tools-13-3=13.3.1-1",
    "cuda-compiler-13-3=13.3.1-1",
    "cuda-libraries-dev-13-3=13.3.1-1",
    "libcublas-13-3=13.6.0.2-1",
    "libcublas-dev-13-3=13.6.0.2-1",
)
NCU_APT_PACKAGE = "cuda-nsight-compute-13-3=13.3.1-1"
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


def _ncu_base_image() -> modal.Image:
    return (
        _base_image()
        .apt_install(NCU_APT_PACKAGE)
        .env(
            {
                "NV_CUDA_NSIGHT_COMPUTE_VERSION": "13.3.1-1",
                "NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE": NCU_APT_PACKAGE,
            }
        )
        .run_commands("ncu --version | grep -F 'Version 2026.2.1'")
    )


def _mount_sources(image: modal.Image) -> modal.Image:
    return image.add_local_file(
        LOCAL_SOLUTION, REMOTE_SOLUTION, copy=False
    ).add_local_file(LOCAL_SCRIPT, REMOTE_SCRIPT, copy=False)


if _WORKER_PROCESS:
    nsys_image = modal.Image.debian_slim()
    ncu_image = modal.Image.debian_slim()
else:
    nsys_image = _mount_sources(_nsys_base_image())
    ncu_image = _mount_sources(_ncu_base_image())

app = modal.App("cholesky-b1n4096-nsys-b200", image=nsys_image)
cache_volume = modal.Volume.from_name(
    "cholesky-b1n4096-cache", create_if_missing=True
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
    root = torch.randn(
        (BATCH, N, N),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    data = (root @ root.transpose(1, 2)).mul_(1.0 / N)
    data.diagonal(dim1=1, dim2=2).add_(1.0e-2)
    return (0.5 * (data + data.transpose(1, 2))).contiguous()


def _make_validation_case(case: str):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + sum(ord(char) for char in case))
    if case == "dense":
        root = torch.randn(
            (BATCH, N, N),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        data = (root @ root.transpose(1, 2)).mul_(1.0 / N)
        data.diagonal(dim1=1, dim2=2).add_(1.0e-2)
        data = 0.5 * (data + data.transpose(1, 2))
    elif case == "spectrum":
        basis = torch.randn(
            (BATCH, N, N),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        basis = torch.linalg.qr(basis).Q
        values = torch.logspace(
            0.0,
            -5.0,
            N,
            dtype=torch.float32,
            device="cuda",
        )
        data = (
            basis * values.unsqueeze(0).unsqueeze(1)
        ) @ basis.transpose(1, 2)
        data = 0.5 * (data + data.transpose(1, 2))
    elif case == "diagonal":
        values = torch.logspace(
            0.0,
            -5.0,
            N,
            dtype=torch.float32,
            device="cuda",
        )
        data = torch.diag_embed(values.expand(BATCH, -1))
    elif case in {"lowrank", "rowscale"}:
        if case == "lowrank":
            rank = 64
            factors = torch.randn(
                (BATCH, N, rank),
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            data = (factors @ factors.transpose(1, 2)).mul_(1.0 / rank)
            data.diagonal(dim1=1, dim2=2).add_(1.0e-4)
        else:
            root = torch.randn(
                (BATCH, N, N),
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            data = (root @ root.transpose(1, 2)).mul_(1.0 / N)
            data.diagonal(dim1=1, dim2=2).add_(1.0e-3)
            scale = torch.logspace(
                0.0,
                -2.0,
                N,
                dtype=torch.float32,
                device="cuda",
            )
            data.mul_(scale.view(1, N, 1))
            data.mul_(scale.view(1, 1, N))
            data.diagonal(dim1=1, dim2=2).add_(1.0e-6)
        data = 0.5 * (data + data.transpose(1, 2))
    elif case == "tridiagonal":
        diagonal = torch.empty(
            (BATCH, N), dtype=torch.float32, device="cuda"
        ).uniform_(1.5, 2.5, generator=generator)
        off_diagonal = torch.empty(
            (BATCH, N - 1), dtype=torch.float32, device="cuda"
        ).uniform_(-0.25, 0.25, generator=generator)
        data = (
            torch.diag_embed(diagonal)
            + torch.diag_embed(off_diagonal, offset=1)
            + torch.diag_embed(off_diagonal, offset=-1)
        )
    else:
        raise ValueError(f"unknown validation case: {case}")
    return data.contiguous()


def _scaled_reconstruction_residual(data, factor) -> float:
    import torch

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        reconstructed = factor @ factor.transpose(1, 2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    residual_norm = torch.linalg.matrix_norm(
        data - reconstructed, ord=1, dim=(-2, -1)
    )
    data_norm = torch.linalg.matrix_norm(
        data, ord=1, dim=(-2, -1)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    scaled = residual_norm / (torch.finfo(torch.float32).eps * N * data_norm)
    return float(scaled.amax().item())


def _validate_factor(data, factor, reference_scaled: float) -> dict[str, Any]:
    import math
    import torch

    shape_ok = tuple(factor.shape) == (BATCH, N, N)
    dtype_ok = factor.dtype == torch.float32
    device_ok = factor.is_cuda and factor.device == data.device
    finite = bool(torch.isfinite(factor).all().item())
    eps = torch.finfo(torch.float32).eps
    data_norm = torch.linalg.matrix_norm(
        data, ord=1, dim=(-2, -1)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    upper_norm = torch.linalg.matrix_norm(
        torch.triu(factor, diagonal=1), ord=1, dim=(-2, -1)
    )
    triangular_scaled = float(
        (upper_norm / (eps * N * data_norm)).amax().item()
    )
    diagonal_min = (
        float(factor.diagonal(dim1=1, dim2=2).amin().item())
        if finite
        else -math.inf
    )
    scaled = _scaled_reconstruction_residual(data, factor) if finite else math.inf
    limit = 20.0
    passed = (
        shape_ok
        and dtype_ok
        and device_ok
        and finite
        and triangular_scaled <= 8.0
        and diagonal_min > 0.0
        and scaled <= limit
    )
    return {
        "passed": passed,
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "device_ok": device_ok,
        "finite": finite,
        "scaled_triangular_residual": triangular_scaled,
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


def _make_leaf_input(solution, config: int):
    import torch

    m, _, _, _ = solution._LEAF_CONFIGS[config]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + config)
    root = torch.randn(
        (1, m, m),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    data = (root @ root.transpose(1, 2)).mul_(1.0 / m)
    data.diagonal(dim1=1, dim2=2).add_(0.25)
    return (0.5 * (data + data.transpose(1, 2))).contiguous()


def _validate_leaf(solution, config: int) -> dict[str, Any]:
    import math
    import torch

    m, ib, lb, threads = solution._LEAF_CONFIGS[config]
    data = _make_leaf_input(solution, config)
    factor = solution._run_leaf(data, config)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(factor).all().item())
    reconstructed = factor @ factor.transpose(1, 2)
    residual = torch.linalg.matrix_norm(
        data - reconstructed, ord=1, dim=(-2, -1)
    )
    data_norm = torch.linalg.matrix_norm(
        data, ord=1, dim=(-2, -1)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    scaled = (
        float(
            (
                residual /
                (torch.finfo(torch.float32).eps * m * data_norm)
            ).amax().item()
        )
        if finite
        else math.inf
    )
    upper_max = float(
        torch.triu(factor, diagonal=1).abs().amax().item()
    )
    diagonal_min = (
        float(factor.diagonal(dim1=1, dim2=2).amin().item())
        if finite
        else -math.inf
    )
    return {
        "config": config,
        "m": m,
        "ib": ib,
        "lb": lb,
        "threads": threads,
        "finite": finite,
        "upper_max": upper_max,
        "diagonal_min": diagonal_min,
        "scaled_reconstruction_residual": scaled,
        "passed": (
            finite
            and upper_max == 0.0
            and diagonal_min > 0.0
            and scaled <= 20.0
        ),
    }


def _parse_leaf_configs(text: str, count: int) -> tuple[int, ...]:
    if text.strip().lower() == "all":
        return tuple(range(count))
    values = tuple(
        int(token.strip())
        for token in text.split(",")
        if token.strip()
    )
    if not values:
        raise ValueError("leaf config list is empty")
    invalid = [value for value in values if not 0 <= value < count]
    if invalid:
        raise ValueError(
            f"leaf configs must be in [0, {count - 1}], got {invalid}"
        )
    return tuple(dict.fromkeys(values))


def _worker_leaf_ncu(config: int) -> None:
    import torch

    solution = _load_solution()
    validation = _validate_leaf(solution, config)
    if not validation["passed"]:
        raise RuntimeError(
            f"leaf config {config} failed preflight: {validation}"
        )
    data = _make_leaf_input(solution, config)
    solution._factor_leaf(data, config)
    torch.cuda.synchronize()
    target = _make_leaf_input(solution, config)
    torch.cuda.cudart().cudaProfilerStart()
    solution._factor_leaf(target, config)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _worker_variant_ncu(requested: int) -> None:
    import torch

    solution = _load_solution()
    variant = _resolve_variant(solution, requested)
    data = _make_input()
    solution._run_variant(data, variant)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    solution._run_variant(data, variant)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


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
        "input": "dense-cond-2",
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

    os.environ["CHOLESKY_PROFILE_NVTX"] = "1"
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
    image=ncu_image,
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def profile_leaf_sweep(
    configs: tuple[int, ...],
) -> tuple[str, str]:
    import torch

    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)
    solution = _load_solution()
    metadata_columns = tuple(solution._LEAF_METADATA_COLUMNS)
    metadata = solution._leaf_metadata().tolist()
    records: list[dict[str, Any]] = []
    for config in configs:
        row = {
            name: int(value)
            for name, value in zip(
                metadata_columns, metadata[config]
            )
        }
        row["ptxas_spill_free"] = (
            config not in solution._LEAF_PTXAS_STACK_CONFIGS
        )
        try:
            validation = _validate_leaf(solution, config)
        except Exception as error:
            records.append(
                {
                    "config": config,
                    "resource": row,
                    "validation": {
                        "passed": False,
                        "runtime_error": repr(error),
                    },
                }
            )
            continue
        timing_ms: list[float] = []
        if validation["passed"]:
            samples = 31
            targets = [
                _make_leaf_input(solution, config)
                for _ in range(samples + 3)
            ]
            for warm in targets[:3]:
                solution._factor_leaf(warm, config)
            torch.cuda.synchronize()
            starts = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(samples)
            ]
            ends = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(samples)
            ]
            for start, end, target in zip(
                starts, ends, targets[3:]
            ):
                start.record()
                solution._factor_leaf(target, config)
                end.record()
            torch.cuda.synchronize()
            timing_ms = [
                float(start.elapsed_time(end))
                for start, end in zip(starts, ends)
            ]
        records.append(
            {
                "config": config,
                "resource": row,
                "validation": validation,
                "latency_us": {
                    "samples": len(timing_ms),
                    "minimum": (
                        min(timing_ms) * 1000.0
                        if timing_ms else None
                    ),
                    "median": (
                        statistics.median(timing_ms) * 1000.0
                        if timing_ms else None
                    ),
                    "maximum": (
                        max(timing_ms) * 1000.0
                        if timing_ms else None
                    ),
                },
                "retained": (
                    validation["passed"]
                ),
            }
        )
    run_name = f"b{BATCH}_n{N}_leaf_{_timestamp()}"
    output_dir = Path("/cache/panel") / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "leaf-sweep.json"
    _write_text(
        result_path,
        json.dumps(
            {
                "shape": [BATCH, N, N],
                "configs": list(configs),
                "records": records,
                "timing_scope": (
                    "one initialized fused_ll_potf2 launch; "
                    "input preparation excluded"
                ),
                "retention_gate": (
                    "FP32 reconstruction passes; stack/local storage "
                    "is reported for tuning and never rejects a result"
                ),
                "environment": _environment(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    cache_volume.commit()
    return run_name, str(result_path.relative_to("/cache"))


@app.function(
    image=ncu_image,
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def profile_leaf_ncu(
    config: int, run_name: str,
) -> list[str]:
    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)
    solution = _load_solution()
    m, ib, lb, threads = solution._LEAF_CONFIGS[config]
    output_dir = (
        Path("/cache/panel_ncu") / run_name /
        f"c{config:02d}_m{m}_ib{ib}_lb{lb}_t{threads}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    validation = _validate_leaf(solution, config)
    _write_text(
        output_dir / "preflight.json",
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
    )
    if not validation["passed"]:
        raise RuntimeError(
            f"leaf config {config} failed profiling preflight"
        )
    version = subprocess.run(
        ["ncu", "--version"], capture_output=True, text=True
    )
    _write_text(
        output_dir / "ncu-version.txt",
        version.stdout + version.stderr,
    )
    if version.returncode != 0:
        raise RuntimeError(
            f"ncu --version failed: {version.stderr[-2000:]}"
        )
    report_base = output_dir / "profile"
    command = [
        "ncu",
        "--profile-from-start",
        "off",
        "--replay-mode",
        "kernel",
        "--cache-control",
        "all",
        "--kernel-name-base",
        "function",
        "--kernel-name",
        "fused_ll_potf2",
        "--launch-count",
        "1",
        "--set",
        "full",
        "--force-overwrite",
        "--export",
        str(report_base),
        sys.executable,
        REMOTE_SCRIPT,
        "_worker_leaf_ncu",
        str(config),
    ]
    _write_text(
        output_dir / "ncu-command.json",
        json.dumps(command, indent=2) + "\n",
    )
    profiler_scratch = (
        Path("/tmp") /
        f"cholesky_leaf_ncu_c{config}_{os.getpid()}"
    )
    profiler_scratch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TMPDIR"] = str(profiler_scratch)
    profiled = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
    )
    _write_text(output_dir / "ncu.stdout.txt", profiled.stdout)
    _write_text(output_dir / "ncu.stderr.txt", profiled.stderr)
    cache_volume.commit()
    if profiled.returncode != 0:
        raise RuntimeError(
            f"leaf config {config} ncu failed with code "
            f"{profiled.returncode}: {profiled.stderr[-8000:]}"
        )
    report_path = report_base.with_suffix(".ncu-rep")
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise RuntimeError(
            "ncu completed without a fused_ll_potf2 report"
        )
    for name, extra in (
        ("ncu-details.txt", ()),
        ("ncu-details.csv", ("--csv", "--print-units", "base")),
    ):
        details = subprocess.run(
            [
                "ncu",
                "--import",
                str(report_path),
                "--page",
                "details",
                "--print-details",
                "all",
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        _write_text(output_dir / name, details.stdout)
        if details.returncode != 0:
            raise RuntimeError(
                f"failed to export {name}: {details.stderr[-4000:]}"
            )
    _write_text(
        output_dir / "metric-policy.json",
        json.dumps(
            {
                "policy": (
                    "Missing or renamed metrics are reported verbatim; "
                    "no metric is silently substituted."
                ),
                "expected_scope": (
                    "instruction count, FP32 dependency stalls, "
                    "registers, shared traffic, spills, latency"
                ),
            },
            indent=2,
        )
        + "\n",
    )
    cache_volume.commit()
    return [
        str(path.relative_to("/cache"))
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.stat().st_size > 0
    ]


@app.function(
    image=ncu_image,
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def profile_variant_ncu(
    requested_variant: int, run_name: str,
) -> tuple[int, str, list[str]]:
    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)
    solution = _load_solution()
    variant = _resolve_variant(solution, requested_variant)
    name = _variant_name(solution, variant)
    if variant != 0:
        raise ValueError(
            "the full-factorization NCU endpoint currently targets "
            "variant 0's cuSOLVER kernel"
        )
    output_dir = (
        Path("/cache/ncu") / run_name / f"v{variant}_{name}"
    )
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
            f"variant {variant} preflight failed with code "
            f"{preflight.returncode}: {preflight.stderr[-4000:]}"
        )

    version = subprocess.run(
        ["ncu", "--version"], capture_output=True, text=True
    )
    _write_text(
        output_dir / "ncu-version.txt",
        version.stdout + version.stderr,
    )
    if version.returncode != 0:
        raise RuntimeError(
            f"ncu --version failed: {version.stderr[-2000:]}"
        )

    report_base = output_dir / "profile"
    command = [
        "ncu",
        "--profile-from-start",
        "off",
        "--replay-mode",
        "kernel",
        "--cache-control",
        "all",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        "regex:getrf_wo_pivot",
        "--launch-count",
        "1",
        "--set",
        "full",
        "--section",
        "PmSampling",
        "--section",
        "PmSampling_WarpStates",
        "--force-overwrite",
        "--export",
        str(report_base),
        sys.executable,
        REMOTE_SCRIPT,
        "_worker_variant_ncu",
        str(variant),
    ]
    _write_text(
        output_dir / "ncu-command.json",
        json.dumps(command, indent=2) + "\n",
    )
    profiler_scratch = (
        Path("/tmp") /
        f"cholesky_variant_ncu_v{variant}_{os.getpid()}"
    )
    profiler_scratch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TMPDIR"] = str(profiler_scratch)
    profiled = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
    )
    _write_text(output_dir / "ncu.stdout.txt", profiled.stdout)
    _write_text(output_dir / "ncu.stderr.txt", profiled.stderr)
    cache_volume.commit()
    if profiled.returncode != 0:
        raise RuntimeError(
            f"variant {variant} ncu failed with code "
            f"{profiled.returncode}: {profiled.stderr[-8000:]}"
        )

    report_path = report_base.with_suffix(".ncu-rep")
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise RuntimeError(
            "ncu completed without the cuSOLVER getrf_wo_pivot report"
        )
    for artifact_name, extra in (
        ("ncu-details.txt", ()),
        ("ncu-details.csv", ("--csv", "--print-units", "base")),
    ):
        details = subprocess.run(
            [
                "ncu",
                "--import",
                str(report_path),
                "--page",
                "details",
                "--print-details",
                "all",
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        _write_text(output_dir / artifact_name, details.stdout)
        if details.returncode != 0:
            raise RuntimeError(
                f"failed to export {artifact_name}: "
                f"{details.stderr[-4000:]}"
            )
    _write_text(
        output_dir / "metric-policy.json",
        json.dumps(
            {
                "policy": (
                    "Missing or renamed metrics are reported verbatim; "
                    "no metric is silently substituted."
                ),
                "target": (
                    "Torch/cuSOLVER fused getrf_wo_pivot factorization "
                    "kernel for exact shape (1,4096,4096)"
                ),
            },
            indent=2,
        )
        + "\n",
    )
    cache_volume.commit()
    return (
        variant,
        name,
        [
            str(path.relative_to("/cache"))
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.stat().st_size > 0
        ],
    )


@app.function(
    image=nsys_image,
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def validate_all() -> tuple[str, str]:
    import torch

    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)
    solution = _load_solution()
    variants = _variant_ids(solution)
    cases = (
        "dense",
        "spectrum",
        "diagonal",
        "lowrank",
        "rowscale",
        "tridiagonal",
    )
    records: list[dict[str, Any]] = []
    for case in cases:
        data = _make_validation_case(case)
        reference = torch.linalg.cholesky_ex(
            data, check_errors=False
        ).L
        reference_scaled = _scaled_reconstruction_residual(data, reference)
        for variant in variants:
            try:
                factor = solution._run_variant(data, variant)
                torch.cuda.synchronize()
            except Exception as error:
                records.append(
                    {
                        "case": case,
                        "variant": variant,
                        "name": _variant_name(solution, variant),
                        "validation": {
                            "passed": False,
                            "runtime_error": repr(error),
                        },
                    }
                )
                continue
            validation = _validate_factor(
                data, factor, reference_scaled
            )
            records.append(
                {
                    "case": case,
                    "variant": variant,
                    "name": _variant_name(solution, variant),
                    "validation": validation,
                }
            )
        del data, reference
        torch.cuda.empty_cache()
    run_name = f"b{BATCH}_n{N}_{_timestamp()}"
    output_dir = Path("/cache/validation") / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "results.json"
    _write_text(
        result_path,
        json.dumps(
            {
                "shape": [BATCH, N, N],
                "cases": list(cases),
                "variants": list(variants),
                "records": records,
                "all_passed": all(
                    record["validation"]["passed"] for record in records
                ),
                "failure_count": sum(
                    not record["validation"]["passed"] for record in records
                ),
                "environment": _environment(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    cache_volume.commit()
    return run_name, str(result_path.relative_to("/cache"))


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
def main(
    variant: int = -1,
    output: str = "",
    validate: bool = False,
    panel_configs: str = "",
    panel_ncu_configs: str = "",
    ncu_variant: int = -2,
) -> None:
    selected_actions = sum(
        bool(value)
        for value in (
            validate,
            panel_configs.strip(),
            panel_ncu_configs.strip(),
            ncu_variant != -2,
        )
    )
    if selected_actions > 1:
        raise ValueError(
            "--validate, --panel-configs, and "
            "--panel-ncu-configs, and --ncu-variant are mutually exclusive"
        )
    if panel_configs:
        configs = _parse_leaf_configs(panel_configs, 72)
        run_name, remote_path = profile_leaf_sweep.remote(configs)
        output_root = (
            Path(output)
            if output
            else _repo_root() /
            "cholesky/b1n4096/artifacts/tuning"
        )
        destination = (
            output_root / run_name / Path(remote_path).name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file_object:
            cache_volume.read_file_into_fileobj(
                remote_path, file_object
            )
        if destination.stat().st_size == 0:
            raise RuntimeError(
                f"downloaded empty artifact: {destination}"
            )
        print(f"leaf sweep: {destination}")
        return
    if panel_ncu_configs:
        configs = _parse_leaf_configs(panel_ncu_configs, 72)
        run_name = f"b{BATCH}_n{N}_leaf_ncu_{_timestamp()}"
        output_root = (
            Path(output)
            if output
            else _repo_root() /
            "cholesky/b1n4096/artifacts/ncu"
        )
        for config in configs:
            remote_paths = profile_leaf_ncu.remote(
                config, run_name
            )
            for remote_path in remote_paths:
                remote = Path(remote_path)
                local_path = (
                    output_root / run_name /
                    remote.parts[-2] / remote.name
                )
                local_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                with local_path.open("wb") as file_object:
                    cache_volume.read_file_into_fileobj(
                        remote_path, file_object
                    )
                if local_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"downloaded empty artifact: {local_path}"
                    )
                print(f"downloaded {local_path}")
        return
    if ncu_variant != -2:
        run_name = f"b{BATCH}_n{N}_torch_ncu_{_timestamp()}"
        resolved_variant, name, remote_paths = profile_variant_ncu.remote(
            ncu_variant, run_name
        )
        output_root = (
            Path(output)
            if output
            else _repo_root() /
            "cholesky/b1n4096/artifacts/ncu"
        )
        destination = (
            output_root / run_name /
            f"v{resolved_variant}_{name}"
        )
        for remote_path in remote_paths:
            local_path = destination / Path(remote_path).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("wb") as file_object:
                cache_volume.read_file_into_fileobj(
                    remote_path, file_object
                )
            if local_path.stat().st_size == 0:
                raise RuntimeError(
                    f"downloaded empty artifact: {local_path}"
                )
            print(f"downloaded {local_path}")
        reports = list(destination.glob("*.ncu-rep"))
        if len(reports) != 1:
            raise RuntimeError(
                "expected one Nsight Compute report, "
                f"found {len(reports)}"
            )
        print(f"Nsight Compute report: {reports[0]}")
        return
    if validate:
        run_name, remote_path = validate_all.remote()
        output_root = (
            Path(output)
            if output
            else _repo_root() / "cholesky/b1n4096/artifacts/validation"
        )
        destination = output_root / run_name / Path(remote_path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file_object:
            cache_volume.read_file_into_fileobj(remote_path, file_object)
        if destination.stat().st_size == 0:
            raise RuntimeError(f"downloaded empty artifact: {destination}")
        print(f"validation report: {destination}")
        return
    run_name, resolved_variant, name, remote_paths = profile_nsys.remote(variant)
    output_root = (
        Path(output)
        if output
        else _repo_root() / "cholesky/b1n4096/artifacts/nsys"
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
    elif mode == "_worker_leaf_ncu":
        _worker_leaf_ncu(requested)
    elif mode == "_worker_variant_ncu":
        _worker_variant_ncu(requested)
    else:
        raise SystemExit(f"unknown worker mode: {mode}")
