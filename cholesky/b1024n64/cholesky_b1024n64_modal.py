"""B200 tuning, profiling, and Popcorn orchestration for batched 1024x64.

The script deliberately contains no CUDA algorithm implementation. It imports
the exact source that is submitted to Popcorn, so profiling and leaderboard
runs exercise identical kernels.

Examples:
    .venv/bin/modal run cholesky/b1024n64/cholesky_b1024n64_modal.py \
        --action tune --variants 6,13,22,23
    .venv/bin/modal run cholesky/b1024n64/cholesky_b1024n64_modal.py \
        --action profile
    .venv/bin/modal run cholesky/b1024n64/cholesky_b1024n64_modal.py \
        --action ncu --variants 4
    .venv/bin/modal run cholesky/b1024n64/cholesky_b1024n64_modal.py \
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


VARIANT_COUNT = 24
VARIANT_IDS = (*range(20), 22, 23)
VARIANT_NAMES = (
    "w4_blk_precise",
    "w4_blk_refined",
    "w4_blk_raw",
    "w4_int_f2_precise",
    "w4_int_f2_refined",
    "w4_int_f2_raw",
    "w4_int_scalar_raw",
    "w2_int_f2_raw",
    "w8_int_f2_raw",
    "w4_int_f2_raw_acc2",
    "w4_int_f2_raw_acc4",
    "w4_int_f2_raw_look2",
    "w4_int_f2_raw_right",
    "w4_int_f2_raw_right_rootlook",
    "w2_int_f2_raw_right",
    "smem64_m1_precise",
    "smem64_m1_raw",
    "smem64_m2_raw",
    "smem64_reg_raw",
    "cusolverdx_potrf",
    "retired_sync47",
    "retired_sync27",
    "w4_int_scalar_raw_shared_tail16",
    "w4_int_f2_raw_right_rootlook_shared_tail32",
)
LOCAL_SOLUTION = "cholesky/b1024n64/cholesky_b1024n64.py"
LOCAL_SCRIPT = "cholesky/b1024n64/cholesky_b1024n64_modal.py"
# Repo-root-relative like LOCAL_SOLUTION: `modal run` is invoked from the repo
# root, and the path must stay lazily resolvable inside the container.
LOCAL_MATHDX = "nvidia-mathdx-26.06.0-cuda13"
REMOTE_SOLUTION = "/workspace/cholesky_b1024n64.py"
REMOTE_SCRIPT = "/workspace/cholesky_b1024n64_modal.py"
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
_WORKER_PROCESS = (
    len(sys.argv) > 1 and sys.argv[1].startswith("_worker_")
)


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
        # Exact header parity with local: bake the same MathDx tarball
        # (including the bundled CUTLASS) into an image layer for variant 19.
        .add_local_dir(
            LOCAL_MATHDX,
            "/opt/mathdx",
            copy=True,
            ignore=["**/doc/**", "**/example/**", "**/__pycache__"],
        )
        .env(
            {
                "MATHDX_ROOT": "/opt/mathdx/nvidia/mathdx/26.06",
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
    # Child processes launched under a profiler only need the workload helpers
    # below.
    # Avoid asking Modal to resolve client-side mount paths inside the container.
    benchmark_image = modal.Image.debian_slim()
    nsys_image = benchmark_image
    ncu_image = benchmark_image
else:
    benchmark_image = _mount_sources(_base_image())
    nsys_image = _mount_sources(_nsys_base_image())
    ncu_image = _mount_sources(_ncu_base_image())
app = modal.App("cholesky-b1024n64-b200", image=benchmark_image)
cache_volume = modal.Volume.from_name(
    "cholesky-b1024n64-cache", create_if_missing=True
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
    spec = importlib.util.spec_from_file_location("chol_b1024n64_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report_environment() -> dict[str, Any]:
    import torch

    # Nsight Compute locks the SM clock to base while profiling, so recording
    # both the current and the maximum clocks is what lets its durations be
    # scaled back to the boost-clock numbers the timing worker reports. There
    # is no nvidia-smi field for the SM count; asking for one makes the whole
    # query fail, so it comes from torch instead.
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

    batch, n = 1024, 64
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
    scaled = residual_norm / (torch.finfo(torch.float32).eps * 64 * data_norm)
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
        "raw_root",
        "threads_per_cta",
        "registers_per_thread",
        "local_bytes_per_thread",
        "static_shared_bytes_per_cta",
        "active_ctas_per_sm",
        "resident_warps_per_sm",
        "row_layout",
        "launch_min_ctas_per_sm",
        "load_width",
        "dot_accumulators",
        "shuffle_lookahead",
        "right_looking",
        "root_lookahead",
        "threads_per_matrix",
        "matrices_per_cta",
    )
    metadata = solution._variant_metadata()
    expected_shape = (VARIANT_COUNT, len(columns))
    if tuple(metadata.shape) != expected_shape:
        raise RuntimeError(
            "native metadata ABI mismatch: "
            f"expected {expected_shape}, got {tuple(metadata.shape)}; "
            "bump the native extension cache version after changing metadata"
        )
    values = metadata.tolist()
    return [dict(zip(columns, row, strict=True)) for row in values]


def _measure_variants(
    solution, variants: list[int], calls: int, repeats: int
) -> list[dict[str, Any]]:
    import torch

    ring_size = 8
    inputs = [_make_input("dense", 41064 + index) for index in range(ring_size)]
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
    validation: dict[str, list[dict[str, Any]]] = {
        str(variant): [] for variant in variants
    }
    cases = ("dense", "spectrum", "diagonal", "lowrank", "rowscale", "tridiagonal")
    for case_index, case in enumerate(cases):
        data = _make_input(case, 53125 + case_index)
        reference = torch.linalg.cholesky_ex(data, check_errors=False).L
        reference_scaled = _scaled_reconstruction_residual(data, reference)
        for variant in variants:
            factor = solution._run_variant(data, variant)
            result = _validate_factor(data, factor, reference_scaled)
            result["case"] = case
            validation[str(variant)].append(result)
        del data, reference

    selected = set(variants)
    resources = [
        row for row in _resource_rows(solution) if row["variant"] in selected
    ]
    timings = _measure_variants(solution, variants, calls, repeats)
    for row in timings:
        checks = validation[str(row["variant"])]
        row["modal_validation_passed"] = all(check["passed"] for check in checks)
    return {
        "timestamp_utc": _timestamp(),
        "environment": _report_environment(),
        "variants": variants,
        "calls_per_sample": calls,
        "repeats": repeats,
        "resources": resources,
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
    data = _make_input("dense", 41064)
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
    inputs = [_make_input("dense", 41064 + index) for index in range(ring_size)]
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


# NCU selects the profiled launch itself. The variant-to-kernel mapping below
# chooses the function-name base per family; --launch-skip then counts only
# matching launches, so the warmup is discarded without the worker having to
# partition it. The timing worker's eight-tensor ring has no purpose here
# because kernel replay flushes every cache between passes, so one pair of
# buffers is enough.
NCU_WARMUP_LAUNCHES = 16


def _ncu_kernel_name(variant: int) -> str:
    if variant == 19:
        return "potrf_dx_kernel"
    if variant == 23:
        return "right_hybrid_64_kernel"
    if variant == 22:
        return "crout_hybrid_64_kernel"
    if variant == 18:
        return "smem_reg_64_kernel"
    if variant in (15, 16, 17):
        return "smem_64_kernel"
    if variant in (12, 13, 14):
        return "right_looking_64_kernel"
    return "crout_64_kernel"


def _worker_ncu(variant: int) -> None:
    import torch

    solution = _load_solution()
    data = _make_input("dense", 41064)
    out = torch.empty_like(data)

    # Profiling stays disabled until cudaProfilerStart, so the only launches
    # NCU can count are the ones below: the first NCU_WARMUP_LAUNCHES cover
    # module load and allocator reuse, and the last one is captured.
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(NCU_WARMUP_LAUNCHES + 1):
        solution._run_variant(data, variant, out)
    torch.cuda.synchronize()
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
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty Nsight artifact: {report_path}")

    # The report is the durable, forward-compatible artifact. nsys stats needs
    # a SQLite export internally, so direct that transient database to /tmp and
    # remove it immediately instead of storing/downloading one per variant.
    temporary_sqlite = Path("/tmp") / f"cholesky_nsys_v{variant}_{os.getpid()}.sqlite"
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
    stats_path = output_dir / "stats.txt"
    stats_path.write_text(stats.stdout + stats.stderr)
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


@app.function(
    image=ncu_image,
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def ncu_remote(variant: int, run_name: str) -> list[str]:
    output_dir = Path("/cache/ncu") / run_name / f"v{variant}_{VARIANT_NAMES[variant]}"
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

    version = subprocess.run(["ncu", "--version"], capture_output=True, text=True)
    version_path = output_dir / "ncu-version.txt"
    version_path.write_text(version.stdout + version.stderr)
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
    for section in NCU_SECTIONS:
        command.extend(("--section", section))
    # NCU expects the application directly after its options; unlike nsys,
    # inserting a `--` separator makes it try to launch that token as a file.
    command.extend(
        (
            sys.executable,
            REMOTE_SCRIPT,
            "_worker_ncu",
            str(variant),
        )
    )
    (output_dir / "ncu-command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    # The Nsight Compute tree launcher talks to the target application over
    # named pipes created under TMPDIR, and the Modal volume mounted at /cache
    # cannot create FIFOs. Only the profiler and its child need the override.
    profiler_scratch = Path("/tmp") / f"cholesky_ncu_v{variant}_{os.getpid()}"
    profiler_scratch.mkdir(parents=True, exist_ok=True)
    ncu_environment = os.environ.copy()
    ncu_environment["TMPDIR"] = str(profiler_scratch)
    profiled = subprocess.run(
        command, capture_output=True, text=True, env=ncu_environment
    )
    (output_dir / "ncu.stdout.txt").write_text(profiled.stdout)
    (output_dir / "ncu.stderr.txt").write_text(profiled.stderr)
    if profiled.returncode != 0:
        diagnostics = _subprocess_diagnostics(profiled)
        if "Launching the target application failed" in diagnostics:
            # NVIDIA recommends NVLOG for otherwise opaque launcher failures.
            # Retry only this fast-failing case and preserve the expanded log.
            nvlog_path = output_dir / "nvlog.config"
            nvlog_path.write_text(
                "UseStdout\n"
                "ForceFlush\n"
                "Format  $sev:${level:-3}|$proc|$name|$sfunc>> $text\n"
                "+ 0i 0w 100ef 0IW 100EF global\n"
            )
            debug_environment = ncu_environment.copy()
            debug_environment["NVLOG_CONFIG_FILE"] = str(nvlog_path)
            debugged = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=debug_environment,
            )
            (output_dir / "ncu-debug.stdout.txt").write_text(debugged.stdout)
            (output_dir / "ncu-debug.stderr.txt").write_text(debugged.stderr)
            diagnostics = (
                f"{diagnostics}\nNVLOG retry:\n"
                f"{_subprocess_diagnostics(debugged)}"
            )
        raise RuntimeError(
            f"variant {variant} ncu failed with code {profiled.returncode}:\n"
            f"command: {json.dumps(command)}\n"
            f"{diagnostics[-16000:]}"
        )

    report_path = report_base.with_suffix(".ncu-rep")
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty Nsight Compute artifact: {report_path}")

    detail_outputs = (
        (output_dir / "ncu-details.txt", ()),
        (
            output_dir / "ncu-details.csv",
            ("--csv", "--print-units", "base"),
        ),
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
            diagnostics = _subprocess_diagnostics(details)
            raise RuntimeError(
                f"failed to export {detail_path.name}:\n{diagnostics[-4000:]}"
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


def _subprocess_diagnostics(completed: subprocess.CompletedProcess[str]) -> str:
    parts = []
    if completed.stdout.strip():
        parts.append(f"stdout:\n{completed.stdout.strip()}")
    if completed.stderr.strip():
        parts.append(f"stderr:\n{completed.stderr.strip()}")
    return "\n".join(parts) or "profiler produced no stdout or stderr"


def _parse_variants(text: str) -> list[int]:
    if not text.strip():
        return list(VARIANT_IDS)
    result = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not result or any(variant not in VARIANT_IDS for variant in result):
        raise ValueError(f"variants must be comma-separated IDs from {VARIANT_IDS}")
    return result


def _download_profile_artifacts(
    variants: list[int], calls: int, output_root: Path
) -> None:
    run_name = f"b1024_n64_{_timestamp()}"
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


def _download_ncu_artifacts(variants: list[int], output_root: Path) -> None:
    run_name = f"b1024_n64_{_timestamp()}"
    destination = output_root / run_name
    for variant in variants:
        remote_paths = ncu_remote.remote(variant, run_name)
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


def _run_popcorn_sweep(output_root: Path, variants: list[int]) -> None:
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

    with tempfile.TemporaryDirectory(prefix="chol_b1024n64_") as temporary:
        temporary_dir = Path(temporary)
        jobs: list[tuple[int, Path, Path]] = []
        for variant in variants:
            submission_path = temporary_dir / f"cholesky_b1024n64_v{variant}.py"
            submission_path.write_text(_variant_source(source, variant))
            result_path = run_dir / f"v{variant}_{VARIANT_NAMES[variant]}.result.txt"
            jobs.append((variant, submission_path, result_path))

        with ThreadPoolExecutor(max_workers=len(VARIANT_IDS)) as executor:
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
        "policy": "selected variants concurrently submitted through Popcorn leaderboard mode",
        "requested_variants": variants,
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
    default_artifacts = root / "cholesky/b1024n64/artifacts"

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
        passing = [
            row for row in payload["timings"] if row["modal_validation_passed"]
        ]
        if not passing:
            raise RuntimeError("no variant passed the Modal correctness gate")
        best = passing[0]
        print(
            f"Modal passing winner: variant {best['variant']} ({best['name']}), "
            f"median={best['median_ms']:.6f} ms"
        )
        return

    if action == "profile":
        if profile_calls <= 0:
            raise ValueError("profile_calls must be positive")
        output_root = Path(output) if output else default_artifacts / "nsys"
        _download_profile_artifacts(selected, profile_calls, output_root)
        return

    if action == "ncu":
        if not variants.strip():
            raise ValueError("--action ncu requires an explicit --variants list")
        output_root = Path(output) if output else default_artifacts / "ncu"
        _download_ncu_artifacts(selected, output_root)
        return

    if action == "popcorn":
        output_root = Path(output) if output else default_artifacts
        _run_popcorn_sweep(output_root, selected)
        return

    raise ValueError("action must be one of: tune, profile, ncu, popcorn")


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
    if mode == "_worker_profile":
        if len(sys.argv) != 4:
            raise SystemExit("profile worker requires call count")
        _worker_profile(variant, int(sys.argv[3]))
        return
    if mode == "_worker_ncu":
        if len(sys.argv) != 3:
            raise SystemExit("ncu worker takes only a variant")
        _worker_ncu(variant)
        return
    raise SystemExit(f"unknown worker mode: {mode}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1].startswith("_worker_"):
    _worker_cli()
