#!/usr/bin/env python3
"""Profile one eigh panel with IKET on a Modal B200 and download its traces."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent
MODAL_LAUNCHER = REPO_ROOT / "eigh_bench_b200_modal.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile one eigh panel with IKET on B200 and download Perfetto/JSON artifacts."
    )
    parser.add_argument("--n", type=int, default=512, help="Matrix order (default: 512).")
    parser.add_argument(
        "--panel-size", type=int, default=32, help="Panel width in columns (default: 32)."
    )
    parser.add_argument(
        "--k", type=int, default=0, help="Panel index; panel_start = k * panel_size (default: 0)."
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Number of matrices/CTAs to trace (default: 1)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Input RNG seed (default: 0).")
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=1,
        help="Thread-block cluster width: CTAs cooperating per matrix (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/eigh_iket"),
        help="Local artifact root (default: profiles/eigh_iket).",
    )
    parser.add_argument(
        "--modal",
        type=Path,
        default=REPO_ROOT / ".venv/bin/modal",
        help="Modal executable (default: .venv/bin/modal).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without launching Modal."
    )
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.n <= 1:
        parser.error("--n must be greater than 1")
    if not 1 <= args.panel_size <= 128:
        parser.error("--panel-size must be in [1, 128]")
    if args.k < 0:
        parser.error("--k must be non-negative")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.k * args.panel_size + args.panel_size >= args.n:
        parser.error("k * panel_size + panel_size must be less than n")
    if not 1 <= args.cluster_size <= 16:
        parser.error("--cluster-size must be in [1, 16]")


def _resolve_modal(path: Path) -> str:
    if path.is_file():
        return str(path.resolve())
    found = shutil.which(str(path))
    if found is not None:
        return found
    raise SystemExit(f"Modal executable not found: {path}")


def _artifacts(run_dir: Path, suffix: str) -> list[Path]:
    return sorted(path for path in run_dir.rglob(f"*{suffix}") if path.stat().st_size > 0)


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    _validate(args, parser)

    modal = _resolve_modal(args.modal)
    output_root = args.output.expanduser().resolve()
    cs_tag = f"_cs{args.cluster_size}" if args.cluster_size > 1 else ""
    run_pattern = f"n{args.n}_ps{args.panel_size}_k{args.k}{cs_tag}_*"
    before = set(output_root.glob(run_pattern)) if output_root.exists() else set()

    command = [
        modal,
        "run",
        str(MODAL_LAUNCHER),
        "--trace",
        "--cases",
        f"{args.batch}x{args.n}",
        "--trace-batch",
        str(args.batch),
        "--panel-size",
        str(args.panel_size),
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
        "--cluster-size",
        str(args.cluster_size),
        "--trace-output",
        str(output_root),
    ]
    print("command:", " ".join(command), flush=True)
    if args.dry_run:
        return 0

    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"panel profiling failed with exit code {result.returncode}")

    candidates = set(output_root.glob(run_pattern))
    new_runs = candidates - before
    if new_runs:
        run_dir = max(new_runs, key=lambda path: path.stat().st_mtime_ns)
    elif candidates:
        # A same-second rerun can reuse the timestamped directory with --clobber.
        run_dir = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    else:
        raise SystemExit(f"profiling succeeded but no output directory matched {output_root / run_pattern}")

    perfetto = _artifacts(run_dir, ".pftrace")
    traces_json = _artifacts(run_dir, ".json")
    if not perfetto or not traces_json:
        raise SystemExit(
            f"incomplete trace download in {run_dir}: "
            f"pftrace={len(perfetto)} json={len(traces_json)}"
        )

    print(f"artifacts: {run_dir}")
    for path in perfetto + traces_json:
        print(f"  {path} ({path.stat().st_size} bytes)")
    print(f"open in Perfetto: {perfetto[0]} (https://ui.perfetto.dev/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
