"""B200 tuning and Popcorn orchestration for batched 64x256 Cholesky.

The only public actions are:

* ``tune``: validate and time every selected variant on a Modal B200.
* ``ncu``: submit selected variants to the Popcorn Brev NCU service locally.
* ``popcorn``: run repeated leaderboard submissions locally and promote the
  fastest eligible variant.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

import modal


VARIANT_COUNT = 22
VARIANT_IDS = tuple(range(VARIANT_COUNT))
VARIANT_NAMES = (
    "cta256_rec32_scalar_simt_precise",
    "cta256_rec32_scalar_simt_refined",
    "cta256_rec32_scalar_simt_raw",
    "cta256_rec32_sub8_simt_refined",
    "cta256_potf2_64_scalar_simt_refined",
    "cta256_rank1_64_scalar_simt_refined",
    "cta512_rec32_scalar_simt_refined",
    "cta256_rec32_scalar_tc_outer_refined",
    "cta256_rec32_scalar_tc_all_refined",
    "cta256_rec32_sub8_tc_all_refined",
    "cta256_rec32_scalar_tc_all_raw",
    "cta512_rec32_scalar_tc_all_refined",
    "cluster128_rec32_scalar_simt_refined",
    "cluster256_rec32_scalar_simt_refined",
    "cluster128_rec32_scalar_tc_outer_refined",
    "cluster128_rec32_scalar_tc_all_refined",
    "cluster256_rec32_scalar_tc_all_refined",
    "cluster256_rec32_scalar_tc_all_raw",
    "cta512_rec32_scalar_tc_all_refined_pad129",
    "cta512_rec32_outer_sub8_tc_all_refined",
    "cta512_rec32_scalar_tc_all_refined_potf2_128",
    "cta512_rec32_scalar_tc_all_refined_tc_batch",
)
METADATA_COLUMNS = (
    "variant",
    "threads",
    "registers",
    "local_bytes",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "active_ctas_per_sm",
    "active_warps_per_sm",
    "cluster_size",
    "uses_tcgen05",
    "recursive_base",
    "factor_mode",
    "trsm_mode",
    "update_mode",
    "root_mode",
    "launch_bounds",
    "a10_ld",
    "outer_trsm_mode",
    "potf2_threads",
    "tc_slice_batching",
)
CASES = (
    "dense",
    "spectrum",
    "diagonal",
    "lowrank",
    "rowscale",
    "tridiagonal",
)

LOCAL_SOLUTION = "cholesky/b64n256/cholesky_b64n256.py"
LOCAL_SCRIPT = "cholesky/b64n256/cholesky_b64n256_modal.py"
REMOTE_SOLUTION = "/workspace/cholesky_b64n256.py"
REMOTE_SCRIPT = "/workspace/cholesky_b64n256_modal.py"
POPCORN_LEADERBOARD = "cholesky"
BREV_BENCHMARK_INDEX = 3
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


def _base_image() -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.13")
        .entrypoint([])
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


benchmark_image = (
    _base_image()
    .add_local_file(LOCAL_SOLUTION, REMOTE_SOLUTION, copy=False)
    .add_local_file(LOCAL_SCRIPT, REMOTE_SCRIPT, copy=False)
)
app = modal.App("cholesky-b64n256-b200", image=benchmark_image)
cache_volume = modal.Volume.from_name(
    "cholesky-b64n256-cache", create_if_missing=True
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parse_variants(text: str) -> list[int]:
    if not text.strip():
        return list(VARIANT_IDS)
    selected: list[int] = []
    for item in text.split(","):
        variant = int(item.strip())
        if variant not in VARIANT_IDS:
            raise ValueError(f"variant must be in {VARIANT_IDS}, got {variant}")
        if variant not in selected:
            selected.append(variant)
    if not selected:
        raise ValueError("no variants selected")
    return selected


def _install_task_module() -> None:
    if "task" in sys.modules:
        return
    import torch

    task = ModuleType("task")
    task.input_t = torch.Tensor
    task.output_t = torch.Tensor
    sys.modules["task"] = task


def _load_solution():
    _install_task_module()
    path = Path(REMOTE_SOLUTION)
    spec = importlib.util.spec_from_file_location(
        "cholesky_b64n256_solution", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cuda_generator(seed: int):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def _make_input(case: str, seed: int):
    import torch

    batch = 64
    n = 256
    generator = _cuda_generator(seed)
    eye = torch.eye(n, device="cuda", dtype=torch.float32)

    if case == "dense":
        x = torch.randn(
            batch, n, n, generator=generator,
            device="cuda", dtype=torch.float32
        )
        return x @ x.transpose(-1, -2) / n + 0.75 * eye

    if case == "spectrum":
        x = torch.randn(
            batch, n, n, generator=generator,
            device="cuda", dtype=torch.float32
        )
        q = torch.linalg.qr(x).Q
        values = torch.logspace(
            0.0, 0.7, n, device="cuda", dtype=torch.float32
        )
        return (q * values) @ q.transpose(-1, -2)

    if case == "diagonal":
        values = torch.logspace(
            -0.3, 0.7, n, device="cuda", dtype=torch.float32
        )
        offsets = torch.arange(batch, device="cuda")[:, None]
        columns = torch.arange(n, device="cuda")[None, :]
        values = values[(columns + 17 * offsets) % n]
        return torch.diag_embed(values)

    if case == "lowrank":
        rank = 24
        u = torch.randn(
            batch, n, rank, generator=generator,
            device="cuda", dtype=torch.float32
        )
        return u @ u.transpose(-1, -2) / rank + 0.5 * eye

    if case == "rowscale":
        x = torch.randn(
            batch, n, n, generator=generator,
            device="cuda", dtype=torch.float32
        )
        base = x @ x.transpose(-1, -2) / n + eye
        scale = torch.logspace(
            -0.3, 0.3, n, device="cuda", dtype=torch.float32
        )
        offsets = torch.arange(batch, device="cuda")[:, None]
        columns = torch.arange(n, device="cuda")[None, :]
        scale = scale[(columns + 11 * offsets) % n]
        return base * scale[:, :, None] * scale[:, None, :]

    if case == "tridiagonal":
        diagonal = 2.25 + 0.25 * torch.rand(
            batch, n, generator=generator,
            device="cuda", dtype=torch.float32
        )
        off = -0.65 + 0.05 * torch.rand(
            batch, n - 1, generator=generator,
            device="cuda", dtype=torch.float32
        )
        return (
            torch.diag_embed(diagonal)
            + torch.diag_embed(off, offset=-1)
            + torch.diag_embed(off, offset=1)
        )

    raise ValueError(f"unknown validation case: {case}")


def _scaled_residual(data, factor) -> float:
    import torch

    residual = data - factor @ factor.transpose(-1, -2)
    numerator = torch.linalg.matrix_norm(
        residual, ord=float("inf")
    ).amax()
    denominator = (
        torch.finfo(torch.float32).eps
        * data.shape[-1]
        * torch.linalg.matrix_norm(data, ord=float("inf")).amax()
    )
    return float((numerator / denominator).item())


def _validate_factor(data, factor, reference_scaled: float) -> dict[str, Any]:
    import torch

    shape_ok = tuple(factor.shape) == tuple(data.shape)
    dtype_ok = factor.dtype == torch.float32
    device_ok = factor.device == data.device
    finite = bool(torch.isfinite(factor).all().item())
    strict_upper_zero = bool(
        (torch.triu(factor, diagonal=1) == 0).all().item()
    )
    positive_diagonal = bool(
        (torch.diagonal(factor, dim1=-2, dim2=-1) > 0).all().item()
    )
    scaled = _scaled_residual(data, factor) if finite else float("inf")
    limit = max(16.0, 8.0 * reference_scaled)
    residual_ok = scaled <= limit
    return {
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "device_ok": device_ok,
        "finite": finite,
        "strict_upper_zero": strict_upper_zero,
        "positive_diagonal": positive_diagonal,
        "scaled_residual": scaled,
        "reference_scaled_residual": reference_scaled,
        "scaled_residual_limit": limit,
        "residual_ok": residual_ok,
        "passed": all(
            (
                shape_ok,
                dtype_ok,
                device_ok,
                finite,
                strict_upper_zero,
                positive_diagonal,
                residual_ok,
            )
        ),
    }


def _outer_update_diagnostics(data, factor) -> dict[str, Any]:
    import torch

    panel = factor[:, 128:, :128]
    trailing_factor = factor[:, 128:, 128:]
    expected = panel @ panel.transpose(-1, -2)
    observed = (
        data[:, 128:, 128:]
        - trailing_factor @ trailing_factor.transpose(-1, -2)
    )
    expected_norm = torch.linalg.matrix_norm(expected).amax()
    observed_norm = torch.linalg.matrix_norm(observed).amax()
    difference_norm = torch.linalg.matrix_norm(observed - expected).amax()
    denominator = expected_norm.clamp_min(torch.finfo(torch.float32).tiny)
    quadrants = {}
    for row_block in range(2):
        for col_block in range(2):
            row_slice = slice(64 * row_block, 64 * (row_block + 1))
            col_slice = slice(64 * col_block, 64 * (col_block + 1))
            expected_block = expected[:, row_slice, col_slice]
            observed_block = observed[:, row_slice, col_slice]
            block_denominator = torch.linalg.matrix_norm(
                expected_block
            ).amax().clamp_min(torch.finfo(torch.float32).tiny)
            quadrants[f"{row_block}{col_block}"] = float(
                (
                    torch.linalg.matrix_norm(
                        observed_block - expected_block
                    ).amax()
                    / block_denominator
                ).item()
            )
    return {
        "expected_product_max_fro": float(expected_norm.item()),
        "observed_product_max_fro": float(observed_norm.item()),
        "observed_to_expected_norm_ratio": float(
            (observed_norm / denominator).item()
        ),
        "relative_product_error": float(
            (difference_norm / denominator).item()
        ),
        "relative_quadrant_errors": quadrants,
        "observed_symmetry_max_abs": float(
            (
                observed - observed.transpose(-1, -2)
            ).abs().amax().item()
        ),
    }


def _resource_rows(solution) -> list[dict[str, int | str]]:
    matrix = solution._variant_metadata().tolist()
    rows: list[dict[str, int | str]] = []
    for values in matrix:
        row: dict[str, int | str] = {
            key: int(value)
            for key, value in zip(METADATA_COLUMNS, values, strict=True)
        }
        variant = int(row["variant"])
        row["name"] = VARIANT_NAMES[variant]
        rows.append(row)
    return rows


def _environment() -> dict[str, Any]:
    import torch

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,"
            "clocks.mem,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "nvidia_smi": completed.stdout.strip(),
        "nvidia_smi_returncode": completed.returncode,
    }


def _time_variants(
    solution, variants: list[int], calls: int, repeats: int
) -> list[dict[str, Any]]:
    import torch

    ring_size = 8
    inputs = [
        _make_input("dense", 41256 + index) for index in range(ring_size)
    ]
    outputs = [torch.empty_like(inputs[0]) for _ in range(ring_size)]
    samples: dict[int, list[float]] = {variant: [] for variant in variants}

    for variant in variants:
        for index in range(16):
            slot = index % ring_size
            solution._run_variant(
                inputs[slot], variant, out=outputs[slot]
            )
    torch.cuda.synchronize()

    for repeat in range(repeats):
        order = variants if repeat % 2 == 0 else list(reversed(variants))
        for variant in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for index in range(calls):
                slot = index % ring_size
                solution._run_variant(
                    inputs[slot], variant, out=outputs[slot]
                )
            end.record()
            end.synchronize()
            samples[variant].append(float(start.elapsed_time(end)) / calls)

    rows = []
    for variant in variants:
        values = samples[variant]
        rows.append(
            {
                "variant": variant,
                "name": VARIANT_NAMES[variant],
                "samples_ms": values,
                "median_ms": statistics.median(values),
                "minimum_ms": min(values),
                "maximum_ms": max(values),
            }
        )
    rows.sort(key=lambda row: row["median_ms"])
    return rows


def _run_tuning(
    variants: list[int], calls: int, repeats: int
) -> dict[str, Any]:
    import torch

    temporary_root = Path(os.environ["TMPDIR"])
    temporary_root.mkdir(parents=True, exist_ok=True)
    solution = _load_solution()
    resources = _resource_rows(solution)
    resources_by_variant = {
        int(row["variant"]): row for row in resources
    }
    validation: dict[str, list[dict[str, Any]]] = {
        str(variant): [] for variant in variants
    }

    for case_index, case in enumerate(CASES):
        data = _make_input(case, 41256 + case_index)
        reference = torch.linalg.cholesky_ex(
            data, check_errors=False
        ).L
        reference_scaled = _scaled_residual(data, reference)
        for variant in variants:
            factor = solution._run_variant(data, variant)
            row = _validate_factor(data, factor, reference_scaled)
            row["case"] = case
            if not row["passed"]:
                row["outer_update_diagnostics"] = (
                    _outer_update_diagnostics(data, factor)
                )
            validation[str(variant)].append(row)
        del data, reference

    timings = _time_variants(solution, variants, calls, repeats)
    for row in timings:
        checks = validation[str(row["variant"])]
        row["modal_validation_passed"] = (
            len(checks) == len(CASES)
            and all(check["passed"] for check in checks)
        )

    cache_volume.commit()
    return {
        "schema_version": 1,
        "timestamp_utc": _timestamp(),
        "shape": [64, 256, 256],
        "environment": _environment(),
        "variants": variants,
        "cases": list(CASES),
        "ring_size": 8,
        "warmup_calls": 16,
        "calls_per_sample": calls,
        "repeats": repeats,
        "order_policy": "alternate forward and reverse",
        "resources": [
            resources_by_variant[variant] for variant in variants
        ],
        "validation": validation,
        "timings": timings,
    }


@app.function(
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def tune_remote(
    variants: list[int], calls: int = 200, repeats: int = 5
) -> dict[str, Any]:
    return _run_tuning(variants, calls, repeats)


def _variant_source(source: str, variant: int) -> str:
    pattern = r"^_DEFAULT_VARIANT = \d+  # POPCORN_VARIANT$"
    replacement = f"_DEFAULT_VARIANT = {variant}  # POPCORN_VARIANT"
    rendered, count = re.subn(pattern, replacement, source, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one production marker, replaced {count}"
        )
    ast.parse(rendered)
    rejected = "s" + "tream"
    if rejected in rendered:
        raise RuntimeError(
            f"temporary submission contains rejected token: {rejected}"
        )
    return rendered


def _parse_public_score(text: str) -> float | None:
    match = re.search(
        r"Geomean score \(public\) on B200:\s*"
        r"([0-9.eE+-]+)\s*s",
        text,
    )
    return float(match.group(1)) if match else None


def _resolve_profiler_url() -> str:
    for name in ("POPCORN_BREV_PROFILER_URL", "BREV_PROFILER_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return POPCORN_BREV_PROFILER_URL_DEFAULT


def _collect_ncu_paths(
    result_path: Path, variant_dir: Path
) -> dict[str, list[str]]:
    details: list[str] = []
    reports: list[str] = []
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        payload = {}
    for artifact in payload.get("downloaded_artifacts", []) or []:
        for detail in artifact.get("details", []) or []:
            item = detail.get("path")
            if item:
                details.append(str((variant_dir / item).resolve()))
        for report in artifact.get("reports", []) or []:
            item = report.get("path")
            if item:
                reports.append(str((variant_dir / item).resolve()))

    for path in variant_dir.rglob("*"):
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if path.suffix == ".ncu-rep" and resolved not in reports:
            reports.append(resolved)
        if (
            ("ncu" in path.name.lower())
            and path.suffix in (".txt", ".csv")
            and resolved not in details
        ):
            details.append(resolved)
    return {"details": sorted(details), "reports": sorted(reports)}


def _run_ncu(
    output_root: Path, variants: list[int], benchmark_index: int
) -> Path:
    source = (_repo_root() / LOCAL_SOLUTION).read_text()
    profiler_url = _resolve_profiler_url()
    run_dir = (
        output_root / f"b64_n256_brev_ncu_{_timestamp()}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["POPCORN_BREV_PROFILER_URL"] = profiler_url

    rows = []
    for variant in variants:
        variant_dir = (
            run_dir / f"v{variant}_{VARIANT_NAMES[variant]}"
        )
        variant_dir.mkdir(parents=True, exist_ok=False)
        submission = (
            variant_dir / f"cholesky_b64n256_v{variant}.py"
        ).resolve()
        submission.write_text(_variant_source(source, variant))
        result_path = (variant_dir / "brev-result.json").resolve()
        command = [
            "popcorn",
            "submit",
            str(submission),
            "--leaderboard",
            POPCORN_LEADERBOARD,
            "--profile-brev",
            "--benchmark-index",
            str(benchmark_index),
            "--no-tui",
            "--output",
            str(result_path),
        ]
        _write_json(variant_dir / "command.json", command)
        completed = subprocess.run(
            command,
            cwd=variant_dir,
            env=environment,
            capture_output=True,
            text=True,
        )
        (variant_dir / "stdout.txt").write_text(completed.stdout)
        (variant_dir / "stderr.txt").write_text(completed.stderr)
        paths = _collect_ncu_paths(result_path, variant_dir)
        passed = (
            completed.returncode == 0
            and bool(paths["details"])
            and bool(paths["reports"])
        )
        row = {
            "variant": variant,
            "name": VARIANT_NAMES[variant],
            "returncode": completed.returncode,
            "result": str(result_path) if result_path.is_file() else None,
            "details": paths["details"],
            "reports": paths["reports"],
            "passed": passed,
        }
        rows.append(row)
        _write_json(run_dir / "progress.json", {"results": rows})
        print(
            f"NCU variant {variant}: returncode={completed.returncode}, "
            f"details={len(paths['details'])}, reports={len(paths['reports'])}",
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
    failed = [row["variant"] for row in rows if not row["passed"]]
    if failed:
        raise RuntimeError(
            f"NCU artifacts incomplete for variants {failed}; "
            f"inspect {summary_path}"
        )
    return summary_path


def _modal_eligible_variants(
    tuning_path: Path, selected: list[int]
) -> tuple[set[int], dict[str, Any]]:
    payload = json.loads(tuning_path.read_text())
    if payload.get("shape") != [64, 256, 256]:
        raise ValueError(
            f"tuning JSON has wrong shape: {payload.get('shape')}"
        )
    cases = tuple(payload.get("cases", ()))
    if cases != CASES:
        raise ValueError(
            f"tuning JSON cases must be {CASES}, got {cases}"
        )
    validation = payload.get("validation", {})
    eligible: set[int] = set()
    for variant in selected:
        checks = validation.get(str(variant), [])
        observed = {check.get("case") for check in checks}
        if (
            observed == set(CASES)
            and len(checks) == len(CASES)
            and all(check.get("passed") is True for check in checks)
        ):
            eligible.add(variant)
    return eligible, payload


def _atomic_promote(
    source_path: Path,
    original_source: str,
    original_hash: str,
    variant: int,
) -> str:
    current = source_path.read_text()
    current_hash = hashlib.sha256(current.encode()).hexdigest()
    if current_hash != original_hash or current != original_source:
        raise RuntimeError(
            "tracked source changed during the sweep; refusing to promote"
        )
    promoted = _variant_source(current, variant)
    if promoted == current:
        return "already_default"
    temporary = source_path.with_name(
        f".{source_path.name}.{os.getpid()}.promote"
    )
    temporary.write_text(promoted)
    os.chmod(temporary, source_path.stat().st_mode)
    os.replace(temporary, source_path)
    return "updated"


def _run_popcorn(
    output_root: Path,
    variants: list[int],
    rounds: int,
    tuning_path: Path,
    max_workers: int,
) -> Path:
    source_path = (_repo_root() / LOCAL_SOLUTION).resolve()
    source = source_path.read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    modal_eligible, tuning_payload = _modal_eligible_variants(
        tuning_path, variants
    )
    run_dir = (
        output_root / f"b64_n256_popcorn_{_timestamp()}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    source_dir = run_dir / "submissions"
    source_dir.mkdir()

    submissions: dict[int, Path] = {}
    for variant in variants:
        path = source_dir / f"cholesky_b64n256_v{variant}.py"
        path.write_text(_variant_source(source, variant))
        submissions[variant] = path

    rows: list[dict[str, Any]] = []

    def submit_one(round_index: int, variant: int) -> dict[str, Any]:
        prefix = run_dir / f"round_{round_index}_variant_{variant}"
        result_path = prefix.with_suffix(".result.json")
        command = [
            "popcorn",
            "submit",
            "--leaderboard",
            POPCORN_LEADERBOARD,
            "--gpu",
            "B200",
            "--mode",
            "leaderboard",
            "--no-tui",
            "--output",
            str(result_path),
            str(submissions[variant]),
        ]
        _write_json(prefix.with_suffix(".command.json"), command)
        completed = subprocess.run(
            command, capture_output=True, text=True
        )
        prefix.with_suffix(".stdout.txt").write_text(completed.stdout)
        prefix.with_suffix(".stderr.txt").write_text(completed.stderr)
        result_text = (
            result_path.read_text() if result_path.is_file() else ""
        )
        combined = "\n".join(
            (result_text, completed.stdout, completed.stderr)
        )
        return {
            "round": round_index,
            "variant": variant,
            "name": VARIANT_NAMES[variant],
            "returncode": completed.returncode,
            "public_b200_geomean_seconds": _parse_public_score(combined),
            "result": str(result_path) if result_path.is_file() else None,
        }

    worker_count = min(max_workers, len(variants))
    for round_index in range(1, rounds + 1):
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending = {
                executor.submit(submit_one, round_index, variant): variant
                for variant in variants
            }
            for future in as_completed(pending):
                row = future.result()
                rows.append(row)
                rows.sort(key=lambda item: (item["round"], item["variant"]))
                _write_json(
                    run_dir / "progress.json",
                    {"submissions": rows},
                )
                print(
                    f"round {row['round']} variant {row['variant']}: "
                    f"returncode={row['returncode']} "
                    f"score={row['public_b200_geomean_seconds']}",
                    flush=True,
                )

    aggregates = []
    promotion_candidates = []
    for variant in variants:
        variant_rows = [
            row for row in rows if row["variant"] == variant
        ]
        scores = [
            row["public_b200_geomean_seconds"]
            for row in variant_rows
            if row["public_b200_geomean_seconds"] is not None
        ]
        every_round = (
            len(variant_rows) == rounds
            and len(scores) == rounds
            and all(row["returncode"] == 0 for row in variant_rows)
        )
        modal_passed = variant in modal_eligible
        aggregate = {
            "variant": variant,
            "name": VARIANT_NAMES[variant],
            "modal_validation_passed": modal_passed,
            "score_in_every_round": every_round,
            "scores_seconds": scores,
            "median_seconds": (
                statistics.median(scores) if scores else None
            ),
            "minimum_seconds": min(scores) if scores else None,
            "promotion_eligible": modal_passed and every_round,
        }
        aggregates.append(aggregate)
        if aggregate["promotion_eligible"]:
            promotion_candidates.append(aggregate)

    ranking = sorted(
        (
            row for row in aggregates
            if row["median_seconds"] is not None
        ),
        key=lambda row: row["median_seconds"],
    )
    eligible_ranking = sorted(
        promotion_candidates,
        key=lambda row: row["median_seconds"],
    )
    winner = eligible_ranking[0] if eligible_ranking else None
    promotion = {
        "status": "not_attempted",
        "variant": winner["variant"] if winner else None,
    }
    if winner is not None:
        winner_variant = int(winner["variant"])
        winning_source = (
            run_dir
            / f"winner_v{winner_variant}_{VARIANT_NAMES[winner_variant]}.py"
        )
        winning_source.write_text(submissions[winner_variant].read_text())
        try:
            promotion["status"] = _atomic_promote(
                source_path, source, source_hash, winner_variant
            )
        except RuntimeError as error:
            promotion["status"] = "refused_concurrent_edit"
            promotion["error"] = str(error)

    summary = {
        "schema_version": 1,
        "timestamp_utc": _timestamp(),
        "requested_variants": variants,
        "rounds": rounds,
        "max_workers": worker_count,
        "tuning_json": str(tuning_path.resolve()),
        "tuning_timestamp_utc": tuning_payload.get("timestamp_utc"),
        "modal_eligible_variants": sorted(modal_eligible),
        "submissions": rows,
        "aggregates": aggregates,
        "ranking": ranking,
        "eligible_ranking": eligible_ranking,
        "winner": winner,
        "promotion": promotion,
        "source_sha256_before": source_hash,
    }
    summary_path = run_dir / "summary.json"
    _write_json(summary_path, summary)
    if winner is None:
        raise RuntimeError(
            "no promotion-eligible winner: every winner must pass all Modal "
            f"cases and score in every round; inspect {summary_path}"
        )
    if promotion["status"] == "refused_concurrent_edit":
        raise RuntimeError(
            f"winner retained but default not updated; inspect {summary_path}"
        )
    return summary_path


@app.local_entrypoint()
def main(
    action: str = "tune",
    variants: str = "",
    output: str = "",
    tuning_json: str = "",
    rounds: int = 3,
    max_workers: int = 4,
    benchmark_index: int = BREV_BENCHMARK_INDEX,
) -> None:
    selected = _parse_variants(variants)
    artifacts = _repo_root() / "cholesky/b64n256/artifacts"
    output_root = Path(output).resolve() if output else artifacts

    if action == "tune":
        payload = tune_remote.remote(selected, 200, 5)
        output_path = (
            Path(output).resolve()
            if output
            else artifacts / "tune" / f"tuning_{payload['timestamp_utc']}.json"
        )
        _write_json(output_path, payload)
        passed = [
            row for row in payload["timings"]
            if row["modal_validation_passed"]
        ]
        print(f"tuning JSON: {output_path}")
        if not passed:
            raise RuntimeError(
                "no variant passed all six Modal validation cases"
            )
        best = min(passed, key=lambda row: row["median_ms"])
        print(
            f"fastest Modal-valid variant: {best['variant']} "
            f"({best['name']}), median={best['median_ms']:.6f} ms"
        )
        return

    if action == "ncu":
        if not variants.strip():
            raise ValueError("--action ncu requires --variants")
        if benchmark_index != BREV_BENCHMARK_INDEX:
            raise ValueError(
                "this specialization must be profiled at benchmark index 3"
            )
        summary = _run_ncu(
            output_root / "ncu", selected, benchmark_index
        )
        print(f"NCU summary: {summary}")
        return

    if action == "popcorn":
        if not tuning_json:
            raise ValueError("--action popcorn requires --tuning-json")
        if rounds <= 0:
            raise ValueError("--rounds must be positive")
        if max_workers <= 0:
            raise ValueError("--max-workers must be positive")
        summary = _run_popcorn(
            output_root / "popcorn",
            selected,
            rounds,
            Path(tuning_json).resolve(),
            max_workers,
        )
        print(f"Popcorn summary: {summary}")
        return

    raise ValueError("action must be one of: tune, ncu, popcorn")
