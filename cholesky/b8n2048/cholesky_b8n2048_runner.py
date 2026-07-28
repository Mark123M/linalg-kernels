#!/usr/bin/env python3
"""Popcorn/Brev driver for the B200 (8, 2048, 2048) specialization."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import datetime as dt
import decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


LEADERBOARD = "cholesky"
GPU = "B200"
TARGET_BENCHMARK_INDEX = 9
BASELINE_VARIANT = 11
DEFAULT_API_URL = "https://site--bot--dxfjds728w5v.code.run"
CLI_ID_HEADER = "X-Popcorn-Cli-Id"
API_TIMEOUT_SECONDS = 30
PROMOTION_RATIO = decimal.Decimal("0.995")
VARIANT_NAMES = (
    "torch_baseline",
    "rl_fixed128_custom_tf32",
    "rl_fixed512_hybrid_tf32",
    "rl_adaptive_hybrid_tf32",
    "rl_adaptive_cublas_tf32",
    "ll_fixed128_custom_tf32",
    "ll_fixed512_custom_tf32",
    "ll_adaptive_custom_tf32",
    "ll_adaptive_cublas_tf32",
    "ll_adaptive_custom_fp32",
    "rl_adaptive_hybrid_fp32",
    "ll_m128_to_m64_at_r1024_tf32",
    "ll_m128_m64_m32_at_r1024_r256_tf32",
)
VARIANT_COUNT = len(VARIANT_NAMES)
DEFAULT_MARKER = re.compile(
    r"^_DEFAULT_VARIANT = (\d+)  # POPCORN_VARIANT$", re.MULTILINE
)
RAW_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def _source_path() -> Path:
    return Path(__file__).with_name("cholesky_b8n2048.py").resolve()


def _default_artifact_root() -> Path:
    return Path(__file__).with_name("artifacts").resolve()


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )


def _tracked_default() -> int:
    match = DEFAULT_MARKER.search(_source_path().read_text())
    if match is None:
        raise RuntimeError("tracked source has no unique default marker")
    variant = int(match.group(1))
    if not 0 <= variant < VARIANT_COUNT:
        raise RuntimeError(f"tracked default {variant} is out of range")
    return variant


def _parse_variants(text: str) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(VARIANT_COUNT))
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.upper() == "WINNER":
            variant = _tracked_default()
        else:
            try:
                variant = int(item)
            except ValueError as error:
                raise ValueError(f"invalid variant {item!r}") from error
        if not 0 <= variant < VARIANT_COUNT:
            raise ValueError(
                f"variant must be in [0, {VARIANT_COUNT - 1}]"
            )
        if variant not in result:
            result.append(variant)
    if not result:
        raise ValueError("at least one variant is required")
    return result


def _variant_source(source: str, variant: int) -> str:
    replacement = f"_DEFAULT_VARIANT = {variant}  # POPCORN_VARIANT"
    rendered, count = DEFAULT_MARKER.subn(replacement, source)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one default marker, replaced {count}"
        )
    ast.parse(rendered)
    rejected = "stream"
    if rejected in rendered:
        raise RuntimeError(
            f"rendered submission contains rejected token: {rejected}"
        )
    return rendered


def _preflight_output(popcorn: str) -> str:
    completed = subprocess.run(
        [popcorn, "submit", "--help"],
        capture_output=True,
        text=True,
    )
    help_text = completed.stdout + completed.stderr
    if completed.returncode != 0 or "--output" not in help_text:
        raise RuntimeError(
            "the selected Popcorn CLI does not expose --output; provide "
            "the complete `popcorn submit --help` output"
        )
    return help_text


def _summary_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        direct = None
    if isinstance(direct, dict):
        return direct
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].rstrip() != "{":
            continue
        try:
            payload = json.loads("\n".join(lines[index:]))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(
        "Popcorn output did not contain a submission summary object"
    )


def _api_base() -> str:
    return (
        os.environ.get("POPCORN_API_URL", "").strip() or DEFAULT_API_URL
    ).rstrip("/")


def _cli_id() -> str:
    override = os.environ.get("POPCORN_CLI_ID", "").strip()
    if override:
        return override
    try:
        text = (Path.home() / ".popcorn.yaml").read_text()
    except OSError as error:
        raise RuntimeError(
            "cannot read ~/.popcorn.yaml; run `popcorn register` or set "
            "POPCORN_CLI_ID"
        ) from error
    match = re.search(
        r"^\s*cli_id\s*:\s*[\"']?([^\"'\s]+)",
        text,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(
            "~/.popcorn.yaml has no cli_id; run `popcorn register` or set "
            "POPCORN_CLI_ID"
        )
    return match.group(1)


def _fetch_submission(submission_id: int) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{_api_base()}/user/submissions/{submission_id}",
        headers={
            CLI_ID_HEADER: _cli_id(),
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=API_TIMEOUT_SECONDS
        ) as response:
            payload = json.loads(response.read().decode())
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
    ) as error:
        raise RuntimeError(
            f"authenticated Popcorn result fetch failed: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("Popcorn result API returned a non-object")
    return payload


def _result_maps(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "benchmark-count" in payload:
        return [payload]
    results: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        runs = payload.get("runs")
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict):
                    continue
                result = run.get("result")
                if (
                    run.get("secret", False) is not True
                    and str(run.get("mode", "benchmark")).lower()
                    == "benchmark"
                    and isinstance(result, dict)
                    and "benchmark-count" in result
                ):
                    results.append(result)
            if results:
                return results
        direct = payload.get("result")
        if isinstance(direct, dict) and "benchmark-count" in direct:
            return [direct]
        for value in payload.values():
            results.extend(_result_maps(value))
    elif isinstance(payload, list):
        for value in payload:
            results.extend(_result_maps(value))
    return results


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError(f"{label} must be an integer, got {value!r}")


def _nanoseconds(value: Any, label: str) -> decimal.Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not a raw nanosecond number")
    if isinstance(value, (int, float)):
        text = repr(value)
    elif isinstance(value, str) and RAW_NUMBER.fullmatch(value.strip()):
        text = value.strip()
    else:
        raise ValueError(
            f"{label} must be a raw numeric nanosecond value, "
            f"got {value!r}"
        )
    result = decimal.Decimal(text)
    if not result.is_finite() or result < 0:
        raise ValueError(
            f"{label} must be a finite nonnegative nanosecond value"
        )
    return result


def _extract_timings(payload: Any) -> dict[str, decimal.Decimal]:
    results = _result_maps(payload)
    if not results:
        raise ValueError("no public benchmark result was found")
    target_rows: list[dict[str, decimal.Decimal]] = []
    for result_index, result in enumerate(results):
        count = _integer(
            result.get("benchmark-count"),
            f"result[{result_index}].benchmark-count",
        )
        if count <= TARGET_BENCHMARK_INDEX:
            raise ValueError(
                f"target row {TARGET_BENCHMARK_INDEX} is missing"
            )
        for row_index in range(count):
            base = f"benchmark.{row_index}"
            status = result.get(f"{base}.status")
            mean = result.get(f"{base}.mean")
            if status == "fail" or mean is None:
                raise ValueError(
                    f"public row {row_index} did not pass: "
                    f"status={status!r}, "
                    f"spec={result.get(f'{base}.spec', '')!r}, "
                    f"error={result.get(f'{base}.error', '')!r}"
                )
            _nanoseconds(
                mean, f"result[{result_index}].{base}.mean"
            )
        if result.get("check") != "pass":
            raise ValueError(
                "public benchmark check did not pass: "
                f"{result.get('check')!r}"
            )
        base = f"benchmark.{TARGET_BENCHMARK_INDEX}"
        target_rows.append(
            {
                name: _nanoseconds(
                    result.get(f"{base}.{name}"),
                    f"result[{result_index}].{base}.{name}",
                )
                for name in ("mean", "err", "best", "worst")
            }
        )
    if len(target_rows) != 1:
        raise ValueError(
            "expected exactly one public target result, found "
            f"{len(target_rows)}"
        )
    return target_rows[0]


def _median(values: list[decimal.Decimal]) -> decimal.Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _submit_benchmark(
    *,
    popcorn: str,
    submission: Path,
    run_dir: Path,
    round_index: int,
    variant: int,
) -> dict[str, Any]:
    prefix = run_dir / f"round_{round_index:02d}_variant_{variant:02d}"
    output_path = prefix.with_suffix(".result.json")
    command = [
        popcorn,
        "submit",
        "--leaderboard",
        LEADERBOARD,
        "--gpu",
        GPU,
        "--mode",
        "benchmark",
        "--no-tui",
        "--output",
        str(output_path),
        str(submission),
    ]
    _write_json(prefix.with_suffix(".command.json"), command)
    completed = subprocess.run(command, capture_output=True, text=True)
    prefix.with_suffix(".stdout.txt").write_text(completed.stdout)
    prefix.with_suffix(".stderr.txt").write_text(completed.stderr)
    row: dict[str, Any] = {
        "round": round_index,
        "variant": variant,
        "name": VARIANT_NAMES[variant],
        "returncode": completed.returncode,
        "passed": False,
    }
    if completed.returncode != 0:
        row["error"] = "Popcorn process failed"
        return row
    try:
        result_text = (
            output_path.read_text()
            if output_path.is_file()
            else completed.stdout
        )
        summary = _summary_json(result_text)
        submission_id = _integer(
            summary.get("submission_id"), "submission_id"
        )
        row["submission_id"] = submission_id
        payload = _fetch_submission(submission_id)
        _write_json(prefix.with_suffix(".api.json"), payload)
        timings = _extract_timings(payload)
    except (
        RuntimeError,
        ValueError,
        decimal.InvalidOperation,
    ) as error:
        row["error"] = str(error)
        return row
    row["result_source"] = "authenticated-api"
    row["timings_ns"] = {
        name: str(value) for name, value in timings.items()
    }
    row["passed"] = True
    return row


def _atomic_promote(
    source_path: Path,
    original_source: str,
    original_hash: str,
    variant: int,
) -> str:
    current = source_path.read_text()
    if current != original_source or _sha256(current) != original_hash:
        raise RuntimeError(
            "tracked source changed during the sweep; refusing promotion"
        )
    promoted = _variant_source(current, variant)
    if promoted == current:
        return "already_default"
    temporary = source_path.with_name(
        f".{source_path.name}.{os.getpid()}.promote"
    )
    try:
        temporary.write_text(promoted)
        os.chmod(temporary, source_path.stat().st_mode)
        os.replace(temporary, source_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "updated"


def _autotune(args: argparse.Namespace) -> Path:
    variants = _parse_variants(args.variants)
    if args.rounds <= 0 or args.max_workers <= 0:
        raise ValueError("rounds and max-workers must be positive")
    if not args.no_promote and BASELINE_VARIANT not in variants:
        raise ValueError(
            f"promotion requires current default {BASELINE_VARIANT} in "
            "the sweep so the 0.5% gate uses a contemporaneous baseline"
        )
    help_text = _preflight_output(args.popcorn)
    _cli_id()
    source_path = _source_path()
    source = source_path.read_text()
    source_hash = _sha256(source)
    run_dir = (
        args.artifact_root / "autotune" /
        f"b8n2048_{_timestamp()}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "popcorn-submit-help.txt").write_text(help_text)
    submission_dir = run_dir / "submissions"
    submission_dir.mkdir()
    submissions: dict[int, Path] = {}
    for variant in variants:
        path = submission_dir / (
            f"cholesky_b8n2048_v{variant:02d}.py"
        )
        path.write_text(_variant_source(source, variant))
        submissions[variant] = path.resolve()

    rows: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        order = (
            variants if round_index % 2 == 0
            else list(reversed(variants))
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.max_workers, len(order))
        ) as executor:
            futures = {
                executor.submit(
                    _submit_benchmark,
                    popcorn=args.popcorn,
                    submission=submissions[variant],
                    run_dir=run_dir,
                    round_index=round_index,
                    variant=variant,
                ): variant
                for variant in order
            }
            for future in concurrent.futures.as_completed(futures):
                variant = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    row = {
                        "round": round_index,
                        "variant": variant,
                        "name": VARIANT_NAMES[variant],
                        "passed": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                rows.append(row)
                _write_json(
                    run_dir / "progress.json",
                    {
                        "source_hash": source_hash,
                        "variants": variants,
                        "rounds": args.rounds,
                        "results": sorted(
                            rows,
                            key=lambda item: (
                                item["round"], item["variant"]
                            ),
                        ),
                    },
                )
                status = "pass" if row["passed"] else "FAIL"
                timing = row.get("timings_ns", {}).get("mean", "-")
                print(
                    f"round={round_index} variant={variant} "
                    f"status={status} mean_ns={timing}",
                    flush=True,
                )

    ranking: list[dict[str, Any]] = []
    for variant in variants:
        candidate_rows = [
            row for row in rows if row["variant"] == variant
        ]
        if (
            len(candidate_rows) != args.rounds
            or not all(row.get("passed") is True for row in candidate_rows)
        ):
            continue
        means = [
            decimal.Decimal(row["timings_ns"]["mean"])
            for row in candidate_rows
        ]
        bests = [
            decimal.Decimal(row["timings_ns"]["best"])
            for row in candidate_rows
        ]
        ranking.append(
            {
                "variant": variant,
                "name": VARIANT_NAMES[variant],
                "median_mean_ns": str(_median(means)),
                "median_best_ns": str(_median(bests)),
            }
        )
    ranking.sort(
        key=lambda item: (
            decimal.Decimal(item["median_mean_ns"]),
            decimal.Decimal(item["median_best_ns"]),
            item["variant"],
        )
    )
    summary_path = run_dir / "summary.json"
    native_ranking = [
        item for item in ranking
        if item["variant"] != BASELINE_VARIANT
    ]
    baseline = next(
        (
            item for item in ranking
            if item["variant"] == BASELINE_VARIANT
        ),
        None,
    )
    promotion = "disabled"
    promoted_variant: int | None = None
    threshold: decimal.Decimal | None = None
    winner: dict[str, Any] | None = (
        native_ranking[0] if native_ranking else None
    )
    if not args.no_promote:
        if baseline is None:
            promotion = "retained_default_baseline_failed"
        elif winner is None:
            promotion = "retained_default_no_native_winner"
        else:
            baseline_mean = decimal.Decimal(
                baseline["median_mean_ns"]
            )
            threshold = baseline_mean * PROMOTION_RATIO
            winner_mean = decimal.Decimal(
                winner["median_mean_ns"]
            )
            if winner_mean <= threshold:
                promoted_variant = int(winner["variant"])
                promotion = _atomic_promote(
                    source_path, source, source_hash, promoted_variant
                )
            else:
                promotion = "retained_default_below_required_gain"
    if winner is not None:
        winner_path = run_dir / (
            f"best_native_v{winner['variant']:02d}_"
            f"{winner['name']}.py"
        )
        shutil.copy2(submissions[int(winner["variant"])], winner_path)
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "timestamp_utc": _timestamp(),
            "source_path": str(source_path),
            "source_hash_before": source_hash,
            "source_hash_after": _sha256(source_path.read_text()),
            "target_benchmark_index": TARGET_BENCHMARK_INDEX,
            "variants": variants,
            "rounds": args.rounds,
            "order_policy": "forward/reverse alternating rounds",
            "max_workers": args.max_workers,
            "results": sorted(
                rows,
                key=lambda item: (item["round"], item["variant"]),
            ),
            "ranking": ranking,
            "best_native": winner,
            "baseline": baseline,
            "promotion_ratio": str(PROMOTION_RATIO),
            "promotion_threshold_ns": (
                str(threshold) if threshold is not None else None
            ),
            "promoted_variant": promoted_variant,
            "promotion": promotion,
            "no_promote": args.no_promote,
        },
    )
    print(
        f"best_native={winner} baseline={baseline} "
        f"promotion={promotion}",
        flush=True,
    )
    return summary_path


def _profiler_url() -> str:
    for name in ("POPCORN_BREV_PROFILER_URL", "BREV_PROFILER_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            parsed = urllib.parse.urlparse(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
            ):
                raise RuntimeError(
                    f"{name} must be an absolute HTTP(S) URL"
                )
            return value
    raise RuntimeError(
        "set POPCORN_BREV_PROFILER_URL (or BREV_PROFILER_URL)"
    )


def _collect_ncu_artifacts(
    variant_dir: Path, result_path: Path
) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {
        "reports": [],
        "details": [],
        "archives": [],
        "raw": [],
        "sources": [],
        "logs": [],
        "commands": [],
    }
    for path in variant_dir.rglob("*"):
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        lower = path.name.lower()
        if path.suffix == ".ncu-rep":
            categories["reports"].append(resolved)
        elif path.suffix.lower() in {".zip", ".tar", ".gz"}:
            categories["archives"].append(resolved)
        elif path.suffix.lower() in {".txt", ".csv"} and (
            "ncu" in lower or "detail" in lower
        ):
            categories["details"].append(resolved)
        elif path.suffix == ".py":
            categories["sources"].append(resolved)
        elif "command" in lower:
            categories["commands"].append(resolved)
        elif path.suffix == ".txt":
            categories["logs"].append(resolved)
    if result_path.is_file():
        categories["raw"].append(str(result_path.resolve()))
        try:
            payload = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        for artifact in payload.get("downloaded_artifacts", []) or []:
            if not isinstance(artifact, dict):
                continue
            for key, category in (
                ("reports", "reports"),
                ("details", "details"),
            ):
                for item in artifact.get(key, []) or []:
                    if isinstance(item, dict) and item.get("path"):
                        path = (variant_dir / item["path"]).resolve()
                        categories[category].append(str(path))
    return {
        key: sorted(set(values)) for key, values in categories.items()
    }


def _ncu(args: argparse.Namespace) -> Path:
    variants = _parse_variants(args.variants)
    profiler_url = _profiler_url()
    source = _source_path().read_text()
    run_dir = (
        args.artifact_root / "ncu" /
        f"b8n2048_{_timestamp()}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (args.artifact_root / "helpers" / "ncu").mkdir(
        parents=True, exist_ok=True
    )
    environment = os.environ.copy()
    environment["POPCORN_BREV_PROFILER_URL"] = profiler_url
    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = run_dir / (
            f"v{variant:02d}_{VARIANT_NAMES[variant]}"
        )
        variant_dir.mkdir()
        submission = (
            variant_dir / f"cholesky_b8n2048_v{variant:02d}.py"
        ).resolve()
        submission.write_text(_variant_source(source, variant))
        result_path = (variant_dir / "brev-result.json").resolve()
        command = [
            args.popcorn,
            "submit",
            "--leaderboard",
            LEADERBOARD,
            "--profile-brev",
            "--benchmark-index",
            str(TARGET_BENCHMARK_INDEX),
            "--no-tui",
            "--output",
            str(result_path),
            str(submission),
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
        artifacts = _collect_ncu_artifacts(variant_dir, result_path)
        diagnostic_lines = [
            line.strip()
            for line in (
                completed.stderr + "\n" + completed.stdout
            ).splitlines()
            if line.strip()
        ]
        diagnostic = diagnostic_lines[-1] if diagnostic_lines else ""
        passed = (
            completed.returncode == 0
            and bool(artifacts["reports"])
            and bool(artifacts["details"])
            and bool(artifacts["raw"])
        )
        row = {
            "variant": variant,
            "name": VARIANT_NAMES[variant],
            "returncode": completed.returncode,
            "artifacts": artifacts,
            "diagnostic": diagnostic,
            "passed": passed,
        }
        rows.append(row)
        _write_json(run_dir / "progress.json", {"results": rows})
        print(
            f"variant={variant} returncode={completed.returncode} "
            f"reports={len(artifacts['reports'])} "
            f"details={len(artifacts['details'])}"
            + (f" diagnostic={diagnostic}" if diagnostic else ""),
            flush=True,
        )
    summary_path = run_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "timestamp_utc": _timestamp(),
            "profiler": "popcorn-brev-b200-ncu",
            "profiler_url": profiler_url,
            "leaderboard": LEADERBOARD,
            "benchmark_index": TARGET_BENCHMARK_INDEX,
            "variants": variants,
            "results": rows,
            "helper_policy": str(
                args.artifact_root / "helpers" / "ncu"
            ),
            "metric_policy": (
                "Never silently substitute missing or renamed metrics; "
                "record expected and actual names, units, normalization, "
                "and whether a conclusion is inferred."
            ),
        },
    )
    failed = [row["variant"] for row in rows if not row["passed"]]
    if failed:
        raise RuntimeError(
            f"incomplete NCU captures for variants {failed}; "
            f"inspect {summary_path}"
        )
    return summary_path


def _submit(args: argparse.Namespace) -> None:
    source = _source_path().read_text()
    _variant_source(source, _tracked_default())
    command = [
        args.popcorn,
        "submit",
        "--leaderboard",
        LEADERBOARD,
        "--gpu",
        GPU,
        "--mode",
        args.mode,
        "--no-tui",
        str(_source_path()),
    ]
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Autotune, profile, or submit the B200 "
            "(8,2048,2048) Cholesky specialization"
        )
    )
    parser.add_argument(
        "--popcorn",
        default="popcorn",
        help="Popcorn CLI executable (default: popcorn)",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=_default_artifact_root(),
        help="root directory for retained run artifacts",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    autotune = actions.add_parser(
        "autotune",
        help="benchmark selected variants and optionally promote",
    )
    autotune.add_argument("--variants", default="all")
    autotune.add_argument("--rounds", type=int, default=3)
    autotune.add_argument("--max-workers", type=int, default=4)
    autotune.add_argument(
        "--no-promote",
        action="store_true",
        help="retain artifacts and ranking without changing the default",
    )
    ncu = actions.add_parser(
        "ncu",
        help="capture hosted Brev B200 Nsight Compute reports",
    )
    ncu.add_argument("--variants", required=True)
    submit = actions.add_parser(
        "submit",
        help="submit the tracked production source",
    )
    submit.add_argument(
        "--mode",
        choices=("test", "benchmark", "leaderboard"),
        required=True,
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    if args.action == "autotune":
        print(_autotune(args))
    elif args.action == "ncu":
        print(_ncu(args))
    else:
        _submit(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
