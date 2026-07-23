"""B200 tuning, profiling, and Popcorn orchestration for batched 256x128.

``--action ncu`` profiles through the hosted GPU Mode Brev B200 Nsight Compute
service (popcorn-cli ``--profile-brev``), because Modal does not reliably expose
hardware performance counters. ``--action ncu-modal`` keeps the older
self-hosted Modal NCU path. Both keep the ``--variants`` interface by baking each
variant into a temporary submission via the ``# POPCORN_VARIANT`` marker.

Examples:
    .venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
        --action tune
    .venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
        --action nsys --variants 9,20,21,22,23,24,25
    .venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
        --action ncu --variants 9,20,21,22,23,24,25
    .venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
        --action ncu-modal --variants 9,20,21
    .venv/bin/modal run cholesky/b256n128/cholesky_b256n128_modal.py \
        --action popcorn
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
import time
from types import ModuleType
from typing import Any

import modal


VARIANT_COUNT = 28
VARIANT_IDS = tuple(range(VARIANT_COUNT))
VARIANT_NAMES = (
    "phase_v6_precise_u8",
    "phase_v6_refined_u8",
    "phase_v6_raw_u8",
    "phase_v13_raw_full",
    "phase_v13_raw_u8",
    "phase_rows_raw",
    "rank1_rows_raw",
    "rank1_warptail_raw",
    "async_phase_v13_raw",
    "simt_tile_v13_raw",
    "tf32_wmma_v13_raw",
    "full128_refined",
    "full128_raw",
    "phase_shared_v6_precise_u8",
    "phase_shared_v6_refined_u8",
    "phase_shared_v6_raw_u8",
    "phase_shared_v13_raw_full",
    "phase_shared_v13_raw_u8",
    "simt_shared_v13_raw",
    "tf32_shared_v13_raw",
    "simt_tile_v13_raw_overlap",
    "simt_tile_v6_raw",
    "simt_balanced_v13_raw",
    "simt_balanced_v13_raw_overlap",
    "simt_tile_v6_raw_overlap",
    "simt_balanced_v6_raw_overlap",
    "simt_balanced_v13_warptail",
    "simt_balanced_v13_rows",
)

LOCAL_SOLUTION = "cholesky/b256n128/cholesky_b256n128.py"
LOCAL_SCRIPT = "cholesky/b256n128/cholesky_b256n128_modal.py"
REMOTE_SOLUTION = "/workspace/cholesky_b256n128.py"
REMOTE_SCRIPT = "/workspace/cholesky_b256n128_modal.py"
POPCORN_LEADERBOARD = "cholesky"
# ``--action ncu`` profiles the tuned kernel on the hosted GPU Mode Brev B200
# Nsight Compute service. custom_kernel only runs the specialized kernel for the
# exact (256, 128, 128) benchmark, so the profiler must target that entry:
# index 2 in the leaderboard benchmark grid (CLAUDE.md "Benchmark Shapes":
# 0=b4096n32, 1=b1024n64, 2=b256n128). Any other index profiles the torch
# fallback instead. Confirm the index against the live leaderboard before
# trusting a capture; override with --benchmark-index.
BREV_BENCHMARK_INDEX = 2
# Hosted profiler URL from popcorn-cli/docs/profiling.md. The popcorn CLI reads
# POPCORN_BREV_PROFILER_URL (or BREV_PROFILER_URL); this default is only used
# when neither is exported.
POPCORN_BREV_PROFILER_URL_DEFAULT = (
    "https://http--brev-profiler-proxy--dxfjds728w5v.code.run"
)
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
        # NVIDIA has not published a 13.3 Update 1 container tag. Upgrade the
        # compiler, command-line tools, and development libraries from its
        # signed Ubuntu repository without adding the visual/profiler bundle.
        # The 13.3.0 runtime and devel images deliberately hold their cuBLAS
        # packages, so release only those holds for this explicit upgrade and
        # restore them immediately afterward.
        .run_commands(
            "apt-mark unhold libcublas-13-3 libcublas-dev-13-3"
        )
        .apt_install(*CUDA_UPDATE_PACKAGES)
        .run_commands(
            "apt-mark hold libcublas-13-3 libcublas-dev-13-3"
        )
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
    return _base_image().apt_install(NCU_APT_PACKAGE).env(
        {
            "NV_CUDA_NSIGHT_COMPUTE_VERSION": "13.3.1-1",
            "NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE": NCU_APT_PACKAGE,
        }
    ).run_commands(
        "ncu --version | grep -F 'Version 2026.2.1'"
    )


def _mount_sources(image: modal.Image) -> modal.Image:
    return image.add_local_file(
        LOCAL_SOLUTION, REMOTE_SOLUTION, copy=False
    ).add_local_file(LOCAL_SCRIPT, REMOTE_SCRIPT, copy=False)


if _WORKER_PROCESS:
    benchmark_image = modal.Image.debian_slim()
    nsys_image = benchmark_image
    ncu_image = benchmark_image
else:
    benchmark_image = _mount_sources(_base_image())
    nsys_image = _mount_sources(_nsys_base_image())
    ncu_image = _mount_sources(_ncu_base_image())

app = modal.App("cholesky-b256n128-b200", image=benchmark_image)
cache_volume = modal.Volume.from_name(
    "cholesky-b256n128-cache", create_if_missing=True
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
    spec = importlib.util.spec_from_file_location("chol_b256n128_solution", path)
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
            "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,"
            "clocks.max.mem,power.limit",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": smi.stdout.strip() or smi.stderr.strip(),
        "multiprocessor_count": properties.multi_processor_count,
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

    batch, n = 256, 128
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
    scaled = residual_norm / (torch.finfo(torch.float32).eps * 128 * data_norm)
    return float(scaled.amax().item())


def _validate_factor(data, factor, reference_scaled: float) -> dict[str, Any]:
    import torch

    shape_ok = tuple(factor.shape) == tuple(data.shape)
    dtype_ok = factor.dtype == data.dtype
    device_ok = factor.device == data.device
    finite = bool(torch.isfinite(factor).all().item())
    upper_max = float(torch.triu(factor, diagonal=1).abs().amax().item())
    diagonal_min = float(factor.diagonal(dim1=1, dim2=2).amin().item())
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
        "modal_gate": limit,
    }


RESOURCE_COLUMNS = (
    "variant",
    "threads_per_cta",
    "registers_per_thread",
    "local_bytes_per_thread",
    "static_shared_bytes_per_cta",
    "dynamic_shared_bytes_per_cta",
    "active_ctas_per_sm",
    "resident_warps_per_sm",
    "first_factor",
    "root_mode",
    "update_mode",
    "last_factor",
    "async_load",
    "full_trsm_unroll",
    "unblocked",
    "tensor_update",
    "launch_min_ctas_per_sm",
    "matrices_per_cta",
)


def _resource_rows(solution) -> list[dict[str, Any]]:
    metadata = solution._variant_metadata()
    expected_shape = (VARIANT_COUNT, len(RESOURCE_COLUMNS))
    if tuple(metadata.shape) != expected_shape:
        raise RuntimeError(
            "native metadata ABI mismatch: "
            f"expected {expected_shape}, got {tuple(metadata.shape)}"
        )
    return [
        dict(zip(RESOURCE_COLUMNS, row, strict=True))
        for row in metadata.tolist()
    ]


def _measure_variants(
    solution, variants: list[int], calls: int, repeats: int
) -> list[dict[str, Any]]:
    import torch

    ring_size = 8
    inputs = [_make_input("dense", 41128 + index) for index in range(ring_size)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(ring_size)]
    samples: dict[int, list[float]] = {variant: [] for variant in variants}

    for variant in variants:
        for index in range(16):
            slot = index % ring_size
            solution._run_variant(inputs[slot], variant, outputs[slot])
    torch.cuda.synchronize()

    for repeat in range(repeats):
        order = list(variants)
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


def _run_tuning(
    variants: list[int], calls: int, repeats: int
) -> dict[str, Any]:
    import torch

    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)
    solution = _load_solution()
    resources_by_variant = {
        row["variant"]: row for row in _resource_rows(solution)
    }
    validation: dict[str, list[dict[str, Any]]] = {
        str(variant): [] for variant in variants
    }
    cases = (
        "dense",
        "spectrum",
        "diagonal",
        "lowrank",
        "rowscale",
        "tridiagonal",
    )
    for case_index, case in enumerate(cases):
        data = _make_input(case, 3321 + case_index)
        reference = torch.linalg.cholesky_ex(data, check_errors=False).L
        reference_scaled = _scaled_reconstruction_residual(data, reference)
        for variant in variants:
            factor = solution._run_variant(data, variant)
            result = _validate_factor(data, factor, reference_scaled)
            result["case"] = case
            validation[str(variant)].append(result)
        del data, reference

    timings = _measure_variants(solution, variants, calls, repeats)
    for row in timings:
        checks = validation[str(row["variant"])]
        row["modal_validation_passed"] = all(
            check["passed"] for check in checks
        )

    return {
        "timestamp_utc": _timestamp(),
        "environment": _report_environment(),
        "variants": variants,
        "calls_per_sample": calls,
        "repeats": repeats,
        "resources": [resources_by_variant[variant] for variant in variants],
        "validation": validation,
        "timings": timings,
    }


@app.function(gpu="B200", timeout=3600, volumes={"/cache": cache_volume})
def tune_remote(
    variants: list[int], calls: int, repeats: int
) -> dict[str, Any]:
    return _run_tuning(variants, calls, repeats)


def _worker_preflight(variant: int) -> None:
    import torch

    solution = _load_solution()
    data = _make_input("dense", 41128)
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


def _worker_nsys(variant: int, calls: int) -> None:
    import torch

    solution = _load_solution()
    ring_size = 8
    inputs = [_make_input("dense", 41128 + index) for index in range(ring_size)]
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


NCU_WARMUP_LAUNCHES = 16


def _ncu_kernel_name(variant: int) -> str:
    if variant in (11, 12):
        return "unblocked_128_kernel"
    # Only cases 13-19 launch shared_blocked_128_kernel; the simt_tile /
    # simt_balanced overlap family (variants 20-25) is templated on
    # blocked_128_kernel (the v13/v6 in their names is the kRight64/kCrout64
    # factorization mode, not the shared-memory kernel). Keep this aligned
    # with launch_variant() in cholesky_b256n128.py.
    if 13 <= variant <= 19:
        return "shared_blocked_128_kernel"
    return "blocked_128_kernel"


def _worker_ncu(variant: int) -> None:
    import torch

    solution = _load_solution()
    data = _make_input("dense", 41128)
    out = torch.empty_like(data)
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(NCU_WARMUP_LAUNCHES + 1):
        solution._run_variant(data, variant, out)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _write_preflight(output_dir: Path, variant: int) -> None:
    completed = subprocess.run(
        [sys.executable, REMOTE_SCRIPT, "_worker_preflight", str(variant)],
        capture_output=True,
        text=True,
    )
    (output_dir / "preflight.json").write_text(completed.stdout)
    if completed.stderr.strip():
        (output_dir / "preflight.stderr.txt").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"variant {variant} preflight failed with code "
            f"{completed.returncode}: {completed.stderr[-4000:]}"
        )


@app.function(
    image=nsys_image,
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def nsys_remote(variant: int, calls: int, run_name: str) -> list[str]:
    output_dir = Path("/cache/nsys") / run_name / (
        f"v{variant}_{VARIANT_NAMES[variant]}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)
    _write_preflight(output_dir, variant)

    version = subprocess.run(
        ["nsys", "--version"], capture_output=True, text=True
    )
    (output_dir / "nsys-version.txt").write_text(version.stdout + version.stderr)
    if version.returncode != 0:
        raise RuntimeError(f"nsys --version failed: {version.stderr[-2000:]}")

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
        "--output",
        str(report_base),
        "--",
        sys.executable,
        REMOTE_SCRIPT,
        "_worker_nsys",
        str(variant),
        str(calls),
    ]
    (output_dir / "nsys-command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    completed = subprocess.run(command, capture_output=True, text=True)
    (output_dir / "nsys.stdout.txt").write_text(completed.stdout)
    (output_dir / "nsys.stderr.txt").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"variant {variant} nsys failed with code {completed.returncode}: "
            f"{completed.stderr[-4000:]}"
        )

    report_path = report_base.with_suffix(".nsys-rep")
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty Nsight artifact: {report_path}")

    temporary_sqlite = Path("/tmp") / (
        f"cholesky_b256n128_nsys_v{variant}_{os.getpid()}.sqlite"
    )
    try:
        stats = subprocess.run(
            [
                "nsys",
                "stats",
                "--sqlite",
                str(temporary_sqlite),
                "--format",
                "table",
                "--output",
                "-",
                str(report_path),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        temporary_sqlite.unlink(missing_ok=True)
    (output_dir / "stats.txt").write_text(stats.stdout + stats.stderr)
    if stats.returncode != 0 or not stats.stdout.strip():
        raise RuntimeError(f"variant {variant} nsys stats failed")

    cache_volume.commit()
    return [
        str(path.relative_to("/cache"))
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.stat().st_size > 0
    ]


NCU_SECTIONS = (
    "ComputeWorkloadAnalysis",
    "InstructionStats",
    "LaunchStats",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "SchedulerStats",
    "SourceCounters",
    "SpeedOfLight",
    "SpeedOfLight_RooflineChart",
    "SpeedOfLight_HierarchicalSingleRooflineChart",
    "WarpStateStats",
)

# Named section presets. "fast" is the smoke test (basic set, ~3-5 replay
# passes): use it to confirm NCU can profile a kernel at all before escalating.
# "full" is the expensive custom list above; SourceCounters + the two roofline
# charts, collected in a single pass with --cache-control all, can stall NCU's
# CPU-side processing for many minutes on sm_100 / ncu 2026.2.1, especially on
# the newer heavily-unrolled variants (20-25).
NCU_SECTION_SETS: dict[str, tuple[str, ...]] = {
    "fast": ("LaunchStats", "Occupancy", "SpeedOfLight"),
    "full": NCU_SECTIONS,
}


def _subprocess_diagnostics(completed: subprocess.CompletedProcess[str]) -> str:
    parts = []
    if completed.stdout.strip():
        parts.append(f"stdout:\n{completed.stdout.strip()}")
    if completed.stderr.strip():
        parts.append(f"stderr:\n{completed.stderr.strip()}")
    return "\n".join(parts) or "profiler produced no stdout or stderr"


def _run_streaming(
    command: list[str],
    env: dict[str, str],
    echo_prefix: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, teeing combined output to our stdout so live progress
    (nsight's ``--verbose`` per-pass lines, or the popcorn CLI's job-status
    polling) no longer looks hung, and return a CompletedProcess so existing
    diagnostics keep working. stderr is folded into stdout; the returned
    ``stderr`` field is empty."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        chunks.append(line)
        sys.stdout.write(f"{echo_prefix}{line}")
        sys.stdout.flush()
    returncode = process.wait()
    return subprocess.CompletedProcess(
        command, returncode, stdout="".join(chunks), stderr=""
    )


@app.function(
    image=ncu_image,
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def ncu_remote(variant: int, run_name: str, sections: str = "full") -> list[str]:
    selected_sections = NCU_SECTION_SETS[sections]
    output_dir = Path("/cache/ncu") / run_name / (
        f"v{variant}_{VARIANT_NAMES[variant]}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    Path("/cache/tmp").mkdir(parents=True, exist_ok=True)
    # Preflight compiles the extension (cold: minutes of nvcc, 0% GPU) and warms
    # it for the NCU worker; time it separately so a slow run is attributable.
    preflight_start = time.monotonic()
    _write_preflight(output_dir, variant)
    preflight_seconds = time.monotonic() - preflight_start
    print(f"[ncu v{variant}] preflight/compile {preflight_seconds:.1f}s", flush=True)

    version = subprocess.run(["ncu", "--version"], capture_output=True, text=True)
    (output_dir / "ncu-version.txt").write_text(
        version.stdout + version.stderr
    )
    if version.returncode != 0:
        raise RuntimeError(f"ncu --version failed: {version.stderr[-2000:]}")

    report_base = output_dir / "profile"
    command = [
        "ncu",
        "--profile-from-start",
        "off",
        "--verbose",
        "--replay-mode",
        "kernel",
        "--cache-control",
        "all",
        "--kernel-name",
        _ncu_kernel_name(variant),
        "--launch-skip",
        str(NCU_WARMUP_LAUNCHES),
        "--launch-count",
        "1",
        "--force-overwrite",
        "--export",
        str(report_base),
    ]
    for section in selected_sections:
        command.extend(("--section", section))
    command.extend((sys.executable, REMOTE_SCRIPT, "_worker_ncu", str(variant)))
    (output_dir / "ncu-command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )

    profiler_scratch = Path("/tmp") / (
        f"cholesky_b256n128_ncu_v{variant}_{os.getpid()}"
    )
    profiler_scratch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TMPDIR"] = str(profiler_scratch)
    ncu_start = time.monotonic()
    completed = _run_streaming(command, environment, f"[ncu v{variant}] ")
    ncu_seconds = time.monotonic() - ncu_start
    print(
        f"[ncu v{variant}] phases: preflight/compile {preflight_seconds:.1f}s, "
        f"ncu profiling {ncu_seconds:.1f}s",
        flush=True,
    )
    (output_dir / "ncu.stdout.txt").write_text(completed.stdout)
    (output_dir / "ncu.stderr.txt").write_text(completed.stderr)
    (output_dir / "timing.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "name": VARIANT_NAMES[variant],
                "preflight_seconds": round(preflight_seconds, 3),
                "ncu_seconds": round(ncu_seconds, 3),
            },
            indent=2,
        )
        + "\n"
    )
    # Persist diagnostics before any failure raise so ncu.stdout/stderr
    # (e.g. "no kernels were profiled") survive on the /cache volume.
    cache_volume.commit()
    if completed.returncode != 0:
        raise RuntimeError(
            "Nsight Compute failed; requested sections are not replaced by "
            "different metrics.\n"
            f"command: {json.dumps(command)}\n"
            f"{_subprocess_diagnostics(completed)[-16000:]}"
        )

    report_path = report_base.with_suffix(".ncu-rep")
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise RuntimeError(
            f"missing or empty Nsight artifact: {report_path}\n"
            f"ncu exited 0 but profiled no kernel matching "
            f"--kernel-name {_ncu_kernel_name(variant)!r}; verify it matches "
            f"the kernel launch_variant() dispatches for variant {variant}."
        )

    detail_outputs = (
        (output_dir / "ncu-details.txt", ()),
        (output_dir / "ncu-details.csv", ("--csv", "--print-units", "base")),
    )
    for detail_path, extra_arguments in detail_outputs:
        details = subprocess.run(
            [
                "ncu",
                "--import",
                str(report_path),
                "--page",
                "details",
                "--print-details",
                "all",
                *extra_arguments,
            ],
            capture_output=True,
            text=True,
        )
        detail_path.write_text(details.stdout)
        if details.returncode != 0 or not details.stdout.strip():
            raise RuntimeError(
                f"failed to export {detail_path.name}:\n"
                f"{_subprocess_diagnostics(details)[-4000:]}"
            )

    cache_volume.commit()
    return [
        str(path.relative_to("/cache"))
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
        return list(VARIANT_IDS)
    result = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not result or any(variant not in VARIANT_IDS for variant in result):
        raise ValueError(f"variants must be comma-separated IDs from {VARIANT_IDS}")
    return result


def _download_artifacts(
    kind: str,
    variants: list[int],
    output_root: Path,
    calls: int = 0,
    sections: str = "full",
) -> None:
    run_name = f"b256_n128_{_timestamp()}"
    destination = output_root / run_name
    for variant in variants:
        if kind == "nsys":
            remote_paths = nsys_remote.remote(variant, calls, run_name)
        elif kind == "ncu":
            remote_paths = ncu_remote.remote(variant, run_name, sections)
        else:
            raise ValueError(f"unknown artifact kind: {kind}")
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


def _run_popcorn_sweep(
    output_root: Path, variants: list[int], rounds: int
) -> None:
    source_path = _repo_root() / LOCAL_SOLUTION
    source = source_path.read_text()
    run_dir = output_root / f"popcorn_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []

    def submit_one(
        round_index: int,
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
        prefix = run_dir / f"r{round_index}_v{variant}"
        prefix.with_suffix(".stdout.txt").write_text(completed.stdout)
        prefix.with_suffix(".stderr.txt").write_text(completed.stderr)
        result_text = result_path.read_text() if result_path.is_file() else ""
        combined = "\n".join((result_text, completed.stdout, completed.stderr))
        return {
            "round": round_index,
            "variant": variant,
            "name": VARIANT_NAMES[variant],
            "returncode": completed.returncode,
            "public_b200_geomean_seconds": _parse_public_score(combined),
            "result_file": str(result_path),
        }

    with tempfile.TemporaryDirectory(prefix="chol_b256n128_") as temporary:
        temporary_dir = Path(temporary)
        submissions: dict[int, Path] = {}
        for variant in variants:
            submission_path = temporary_dir / f"cholesky_b256n128_v{variant}.py"
            submission_path.write_text(_variant_source(source, variant))
            submissions[variant] = submission_path

        for round_index in range(1, rounds + 1):
            with ThreadPoolExecutor(max_workers=len(variants)) as executor:
                pending = {}
                for variant in variants:
                    result_path = run_dir / (
                        f"r{round_index}_v{variant}_{VARIANT_NAMES[variant]}.result.txt"
                    )
                    future = executor.submit(
                        submit_one,
                        round_index,
                        variant,
                        submissions[variant],
                        result_path,
                    )
                    pending[future] = variant
                    print(
                        f"queued round {round_index} variant {variant}: "
                        f"{VARIANT_NAMES[variant]}",
                        flush=True,
                    )

                for future in as_completed(pending):
                    row = future.result()
                    rows.append(row)
                    rows.sort(key=lambda item: (item["round"], item["variant"]))
                    _write_json(run_dir / "progress.json", {"submissions": rows})
                    print(
                        f"completed round {row['round']} variant {row['variant']} "
                        f"returncode={row['returncode']} "
                        f"score={row['public_b200_geomean_seconds']}",
                        flush=True,
                    )

    aggregates = []
    for variant in variants:
        scores = [
            row["public_b200_geomean_seconds"]
            for row in rows
            if row["variant"] == variant
            and row["public_b200_geomean_seconds"] is not None
        ]
        aggregates.append(
            {
                "variant": variant,
                "name": VARIANT_NAMES[variant],
                "successful_rounds": len(scores),
                "scores_seconds": scores,
                "median_seconds": statistics.median(scores) if scores else None,
                "minimum_seconds": min(scores) if scores else None,
            }
        )
    ranking = sorted(
        (row for row in aggregates if row["median_seconds"] is not None),
        key=lambda row: row["median_seconds"],
    )
    summary = {
        "timestamp_utc": _timestamp(),
        "policy": (
            "report-only median public geomean; tracked production ID is not changed"
        ),
        "requested_variants": variants,
        "rounds": rounds,
        "submissions": rows,
        "aggregates": aggregates,
        "ranking": ranking,
        "winner": ranking[0] if ranking else None,
    }
    summary_path = run_dir / "summary.json"
    _write_json(summary_path, summary)
    if ranking:
        winner = ranking[0]
        print(
            f"reported winner: variant {winner['variant']} ({winner['name']}), "
            f"median geomean={winner['median_seconds']} s"
        )
    else:
        print("no successful public B200 score was parsed")
    print(f"Popcorn summary: {summary_path}")


def _resolve_brev_profiler_url() -> str:
    """Prefer the caller's exported profiler URL, else the documented default."""
    for name in ("POPCORN_BREV_PROFILER_URL", "BREV_PROFILER_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return POPCORN_BREV_PROFILER_URL_DEFAULT


def _summarize_brev_artifacts(
    result_path: Path, variant_dir: Path
) -> dict[str, list[str]]:
    """Parse the popcorn --output JSON and resolve the extracted NCU artifacts
    (``ncu-details.txt``/``.csv`` and the ``.ncu-rep`` report) to absolute local
    paths. Artifact paths in the JSON are relative to the CLI's working
    directory, which we pin to ``variant_dir``."""
    details: list[str] = []
    reports: list[str] = []
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"details": details, "reports": reports}
    for artifact in payload.get("downloaded_artifacts", []) or []:
        for detail in artifact.get("details", []) or []:
            path = detail.get("path")
            if path:
                details.append(str((variant_dir / path).resolve()))
        for report in artifact.get("reports", []) or []:
            path = report.get("path")
            if path:
                reports.append(str((variant_dir / path).resolve()))
    return {"details": details, "reports": reports}


def _run_ncu_brev(
    output_root: Path, variants: list[int], benchmark_index: int
) -> None:
    """Profile each variant on the hosted GPU Mode Brev B200 Nsight Compute
    service. Runs entirely on the user's machine via the ``popcorn`` CLI (no
    Modal GPU); each variant is baked into a temporary submission through the
    ``# POPCORN_VARIANT`` marker, submitted with ``--profile-brev``, and its
    NCU artifacts are downloaded under ``output_root/<run>/v<id>_<name>/``."""
    profiler_url = _resolve_brev_profiler_url()
    source = (_repo_root() / LOCAL_SOLUTION).read_text()
    run_dir = output_root / f"b256_n128_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    environment = os.environ.copy()
    environment["POPCORN_BREV_PROFILER_URL"] = profiler_url

    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = run_dir / f"v{variant}_{VARIANT_NAMES[variant]}"
        variant_dir.mkdir(parents=True, exist_ok=False)
        submission_path = variant_dir / f"cholesky_b256n128_v{variant}.py"
        submission_path.write_text(_variant_source(source, variant))
        result_path = variant_dir / "brev_result.json"

        command = [
            "popcorn",
            "submit",
            str(submission_path),
            "--leaderboard",
            POPCORN_LEADERBOARD,
            "--profile-brev",
            "--benchmark-index",
            str(benchmark_index),
            "--no-tui",
            "--output",
            str(result_path),
        ]
        (variant_dir / "popcorn-command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        print(
            f"[ncu-brev v{variant}] {VARIANT_NAMES[variant]}: submitting to "
            f"{profiler_url} (leaderboard={POPCORN_LEADERBOARD}, "
            f"benchmark_index={benchmark_index})",
            flush=True,
        )
        # The popcorn CLI writes artifact zips into its working directory, so run
        # it inside variant_dir to keep every variant's captures separated.
        completed = _run_streaming(
            command, environment, f"[ncu-brev v{variant}] ", cwd=variant_dir
        )
        (variant_dir / "popcorn.log.txt").write_text(completed.stdout)

        artifacts = _summarize_brev_artifacts(result_path, variant_dir)
        passed = completed.returncode == 0 and bool(artifacts["details"])
        rows.append(
            {
                "variant": variant,
                "name": VARIANT_NAMES[variant],
                "returncode": completed.returncode,
                "details": artifacts["details"],
                "reports": artifacts["reports"],
                "result_file": (
                    str(result_path) if result_path.is_file() else None
                ),
                "passed": passed,
            }
        )
        if passed:
            for detail_path in artifacts["details"]:
                print(f"[ncu-brev v{variant}] details: {detail_path}", flush=True)
            for report_path in artifacts["reports"]:
                print(f"[ncu-brev v{variant}] report:  {report_path}", flush=True)
        else:
            print(
                f"[ncu-brev v{variant}] FAILED (returncode="
                f"{completed.returncode}); see {variant_dir / 'popcorn.log.txt'}",
                flush=True,
            )

    summary_path = run_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "timestamp_utc": _timestamp(),
            "profiler": "popcorn-brev-b200-ncu",
            "profiler_url": profiler_url,
            "leaderboard": POPCORN_LEADERBOARD,
            "benchmark_index": benchmark_index,
            "requested_variants": variants,
            "results": rows,
        },
    )
    print(f"NCU (Brev) summary: {summary_path}")

    failed = [row["variant"] for row in rows if not row["passed"]]
    if failed:
        raise RuntimeError(
            f"Brev NCU profiling failed for variants {failed}; inspect the "
            f"per-variant popcorn.log.txt under {run_dir}"
        )


@app.local_entrypoint()
def main(
    action: str = "tune",
    output: str = "",
    variants: str = "",
    calls: int = 200,
    repeats: int = 5,
    profile_calls: int = 64,
    rounds: int = 1,
    sections: str = "full",
    benchmark_index: int = BREV_BENCHMARK_INDEX,
) -> None:
    selected = _parse_variants(variants)
    root = _repo_root()
    default_artifacts = root / "cholesky/b256n128/artifacts"

    if action == "tune":
        if calls <= 0 or repeats <= 0:
            raise ValueError("calls and repeats must be positive")
        payload = tune_remote.remote(selected, calls, repeats)
        output_path = (
            Path(output)
            if output
            else default_artifacts / f"tuning_{payload['timestamp_utc']}.json"
        )
        _write_json(output_path, payload)
        print(f"tuning JSON: {output_path}")
        correct = [
            row for row in payload["timings"] if row["modal_validation_passed"]
        ]
        if not correct:
            failed_validation = [
                row["variant"]
                for row in payload["timings"]
                if not row["modal_validation_passed"]
            ]
            raise RuntimeError(
                "no variant passed numerical validation; "
                f"failed_validation={failed_validation}; details={output_path}"
            )
        best = correct[0]
        print(
            f"Modal fastest correct: variant {best['variant']} ({best['name']}), "
            f"median={best['median_ms']:.6f} ms"
        )
        return

    if action == "nsys":
        if profile_calls <= 0:
            raise ValueError("profile_calls must be positive")
        output_root = Path(output) if output else default_artifacts / "nsys"
        _download_artifacts(
            "nsys", selected, output_root, calls=profile_calls
        )
        return

    if action == "ncu":
        if not variants.strip():
            raise ValueError("--action ncu requires an explicit --variants list")
        if benchmark_index < 0:
            raise ValueError("--benchmark-index must be non-negative")
        output_root = Path(output) if output else default_artifacts / "ncu"
        _run_ncu_brev(output_root, selected, benchmark_index)
        return

    if action == "ncu-modal":
        if not variants.strip():
            raise ValueError(
                "--action ncu-modal requires an explicit --variants list"
            )
        if sections not in NCU_SECTION_SETS:
            raise ValueError(
                f"--sections must be one of {sorted(NCU_SECTION_SETS)}"
            )
        output_root = (
            Path(output) if output else default_artifacts / "ncu_modal"
        )
        _download_artifacts("ncu", selected, output_root, sections=sections)
        return

    if action == "popcorn":
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        output_root = Path(output) if output else default_artifacts
        _run_popcorn_sweep(output_root, selected, rounds)
        return

    raise ValueError(
        "action must be one of: tune, nsys, ncu, ncu-modal, popcorn"
    )


def _worker_cli() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("worker mode requires a variant")
    mode = sys.argv[1]
    variant = int(sys.argv[2])
    if variant not in VARIANT_IDS:
        raise SystemExit(f"variant must be one of {VARIANT_IDS}")
    if mode == "_worker_preflight":
        _worker_preflight(variant)
        return
    if mode == "_worker_nsys":
        if len(sys.argv) != 4:
            raise SystemExit("nsys worker requires call count")
        _worker_nsys(variant, int(sys.argv[3]))
        return
    if mode == "_worker_ncu":
        if len(sys.argv) != 3:
            raise SystemExit("ncu worker takes only a variant")
        _worker_ncu(variant)
        return
    raise SystemExit(f"unknown worker mode: {mode}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1].startswith(
    "_worker_"
):
    _worker_cli()
