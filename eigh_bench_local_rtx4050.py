import argparse
from pathlib import Path
from statistics import mean, median
import sys
import types

import torch


REPO_ROOT = Path(__file__).resolve().parent
QUACK_ROOT = REPO_ROOT / "quack"
if QUACK_ROOT.is_dir():
    sys.path.insert(0, str(QUACK_ROOT))

task = types.ModuleType("task")
task.input_t = torch.Tensor
task.output_t = object
sys.modules.setdefault("task", task)

import eigh  # noqa: E402
from quack.bench.bench_utils import (  # noqa: E402
    _bench_cuda_graph_l2_rotate,
    _clone_l2_rotate_inputs,
    _pick_l2_rotate_count,
)


# The kernel's cp.async path (num_copy_elems=1) needs >=4-byte transactions and the
# competition input is fp32, so the runner is fp32-only.
DTYPE = torch.float32
PANEL_SIZE = 1


def make_input(batch: int, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    data = torch.randn(batch, n, n, device=device, dtype=dtype)
    data = 0.5 * (data + data.transpose(-1, -2))
    return data.contiguous()


def make_workspace(data: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # V/W panel workspace in gmem; must be zero-allocated (see Eigh.workspace_shape).
    data_dtype = eigh.torch2cute_dtype_map[data.dtype]
    rows, cols = eigh.Eigh(data_dtype, data.size(1), k=k, panel_size=PANEL_SIZE).workspace_shape()
    v_ws = torch.zeros(data.size(0), rows, cols, device=data.device, dtype=torch.float32)
    return v_ws, torch.zeros_like(v_ws)


def run_direct(data: torch.Tensor, k: int = 0) -> torch.Tensor:
    data_dtype = eigh.torch2cute_dtype_map[data.dtype]
    tri = torch.full_like(data, float("nan"))
    v_ws, w_ws = make_workspace(data, k)
    compiled = eigh.Eigh.compile(data_dtype, data.size(1), k=k, panel_size=PANEL_SIZE)
    compiled(data, tri, v_ws, w_ws)
    return tri


def benchmark_direct(
    data: torch.Tensor,
    k: int,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    data_dtype = eigh.torch2cute_dtype_map[data.dtype]
    compiled = eigh.Eigh.compile(
        data_dtype,
        data.size(1),
        debug_printf=False,
        k=k,
        panel_size=PANEL_SIZE,
    )
    # Rotate only the input matrices: tri is untouched by the non-debug kernel, the
    # V/W workspace is fully rewritten in its valid region every call, and the torch
    # baseline's out is tiny — so sharing them keeps the auto-picked set counts and
    # per-set memory traffic identical between the kernel and torch benches.
    tri = torch.empty_like(data)
    v_ws, w_ws = make_workspace(data, k)
    base_args = (data,)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(bench_data: torch.Tensor) -> None:
        compiled(bench_data, tri, v_ws, w_ws)

    samples = [
        _bench_cuda_graph_l2_rotate(
            launch,
            arg_sets,
            kwarg_sets,
            extra_kwargs={},
            warmup_target_ms=warmup_target_ms,
            n_timed_calls=n_timed_calls,
        )
        for _ in range(repeats)
    ]
    return samples, n_sets


def torch_reflector_w(
    data: torch.Tensor,
    k: int,
    out: torch.Tensor | None = None,
):
    # Same ops as the kernel's i=0 column: reflector v from the panel's first subcolumn
    # with LAPACK's v[0] = 1 convention (tail = x[1:]/(alpha-beta)), b = trailing A @ v,
    # then w = tau*b + (-tau/2)(w^T v) v (deferred-update corrections are no-ops at i=0).
    # No zero-tail edge handling: benchmark inputs are dense random symmetric, and the
    # kernel's tau=2 fallback is a valid sign-flip reflector under residual gating.
    # Everything here must stay CUDA-graph-capture-safe (no .any()/boolean indexing).
    panel_start = k * PANEL_SIZE
    x = data[:, panel_start + 1 :, panel_start]
    norm = x.norm(dim=1)
    alpha = x[:, 0]
    beta = torch.where(alpha < 0, norm, -norm)
    tau = (beta - alpha) / beta
    v = x / (alpha - beta).unsqueeze(1)
    v[:, 0] = 1
    trailing = data[:, panel_start + 1 :, panel_start + 1 :]
    b = torch.bmm(trailing, v.unsqueeze(2)).squeeze(2)
    w = tau.unsqueeze(1) * b
    aw = -0.5 * tau * (w * v).sum(dim=1)
    w = w + aw.unsqueeze(1) * v
    if out is None:
        return w
    out.copy_(w)


def check_gemv_allclose(data: torch.Tensor, k: int) -> bool:
    tri = run_direct(data, k)
    torch.cuda.synchronize()
    panel_start = k * PANEL_SIZE
    w_kernel = tri[:, panel_start + 1 :, panel_start].float()
    w_ref = torch_reflector_w(data.float(), k)
    close = torch.allclose(w_kernel, w_ref, rtol=1e-3, atol=1e-2)
    max_rel = ((w_kernel - w_ref).abs().max() / w_ref.abs().max().clamp_min(1e-30)).item()
    print(f"w check: allclose={close} max_rel={max_rel:.3e}", flush=True)
    return close


def benchmark_torch_gemv(
    data: torch.Tensor,
    k: int,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    panel_start = k * PANEL_SIZE
    m = data.size(1) - panel_start - 1
    out = torch.empty(data.size(0), m, device=data.device, dtype=data.dtype)
    base_args = (data,)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(bench_data: torch.Tensor) -> None:
        torch_reflector_w(bench_data, k, out=out)

    samples = [
        _bench_cuda_graph_l2_rotate(
            launch,
            arg_sets,
            kwarg_sets,
            extra_kwargs={},
            warmup_target_ms=warmup_target_ms,
            n_timed_calls=n_timed_calls,
        )
        for _ in range(repeats)
    ]
    return samples, n_sets


def select_sample(samples: list[float], stat: str) -> float:
    if stat == "min":
        return min(samples)
    if stat == "mean":
        return mean(samples)
    if stat == "median":
        return median(samples)
    raise ValueError(f"Unsupported benchmark stat: {stat}")


def first_column_householder_reference(
    data: torch.Tensor,
    k: int,
    i: int = 0,
):
    panel_start = k * PANEL_SIZE
    col_idx = panel_start + i
    row_start = panel_start + i + 1
    col = data[0, row_start:, col_idx].float().cpu()
    alpha = col[0]
    norm_sq = torch.dot(col, col)
    norm = torch.sqrt(norm_sq)
    beta = norm if alpha < 0 else -norm
    tau = torch.tensor(0.0)
    if beta != 0:
        tau = (beta - alpha) / beta
    denom = alpha - beta
    vi = torch.zeros_like(col)
    if denom != 0:
        vi = col / denom
    return alpha, beta, tau, norm_sq, vi, row_start, col_idx


def print_householder_reference(
    data: torch.Tensor,
    k: int,
    preview: int,
    full: bool,
    print_vi: bool,
) -> None:
    i = 0
    alpha, beta, tau, norm_sq, vi, row_start, col_idx = first_column_householder_reference(
        data,
        k,
        i,
    )
    norm = torch.sqrt(norm_sq)
    print(
        f"expected subcolumn reflector k={k} i={i} A[{row_start}:N,{col_idx}]: "
        f"norm={norm.item():.9g} norm_sq={norm_sq.item():.9g} "
        f"alpha={alpha.item():.9g} beta={beta.item():.9g} tau={tau.item():.9g}",
        flush=True,
    )
    if not print_vi and not full:
        return
    limit = vi.numel() if full else min(preview, vi.numel())
    print(f"expected vi preview for A[{row_start}:N,{col_idx}] ({limit}/{vi.numel()}):", flush=True)
    for idx in range(limit):
        print(f"  vi[{idx}]={vi[idx].item():.9g}", flush=True)
    if not full and limit < vi.numel():
        print("  ... pass --expected-full to print all expected vi entries", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local runner for eigh.py CuTeDSL skeleton.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Compile-time panel index. The panel starts at k * PANEL_SIZE.",
    )
    parser.add_argument(
        "--expected-preview",
        type=int,
        default=16,
        help="Number of host-computed vi entries to print when --print-vi is set.",
    )
    parser.add_argument(
        "--print-vi",
        action="store_true",
        help="Print a host-computed vi preview before launching the kernel.",
    )
    parser.add_argument(
        "--expected-full",
        action="store_true",
        help="Print every host-computed vi entry. Useful when comparing against device printf.",
    )
    parser.add_argument(
        "--skip-expected",
        action="store_true",
        help="Do not print the host-side Householder reference.",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark the direct CuTe kernel with CUDA graph replay and L2-rotated tensors.",
    )
    parser.add_argument(
        "--bench-sets",
        type=int,
        default=0,
        help="Number of cloned input/output tensor sets to rotate through; 0 uses QuACK's heuristic.",
    )
    parser.add_argument(
        "--bench-calls",
        type=int,
        default=200,
        help="Approximate number of kernel calls captured in the timed CUDA graph.",
    )
    parser.add_argument(
        "--bench-warmup-ms",
        type=float,
        default=200.0,
        help="Target GPU warmup time before each timed graph capture.",
    )
    parser.add_argument(
        "--bench-repeats",
        type=int,
        default=1,
        help="Number of full benchmark samples to collect.",
    )
    parser.add_argument(
        "--bench-stat",
        choices=("min", "mean", "median"),
        default="median",
        help="Statistic to report when --bench-repeats is greater than one.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help="torch.set_float32_matmul_precision for the timed torch baseline "
        "(applied after the strict-fp32 allclose check).",
    )
    args = parser.parse_args()
    if args.bench_sets < 0 or args.bench_calls <= 0 or args.bench_repeats <= 0:
        parser.error("--bench-sets must be non-negative; --bench-calls and --bench-repeats must be positive")
    if args.bench_warmup_ms < 0:
        parser.error("--bench-warmup-ms must be non-negative")
    if args.k < 0:
        parser.error("--k must be non-negative")

    panel_start = args.k * PANEL_SIZE
    if panel_start + PANEL_SIZE >= args.n:
        parser.error("k * PANEL_SIZE + PANEL_SIZE must be less than n")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    try:
        torch.empty((), device=device)
    except Exception as exc:
        raise SystemExit(f"CUDA device allocation failed: {exc}") from exc
    data = make_input(args.batch, args.n, DTYPE, device)

    print(f"input: shape={tuple(data.shape)} dtype={data.dtype} device={data.device}", flush=True)
    print(f"panel debug: k={args.k} panel_size={PANEL_SIZE} panel_start={panel_start}", flush=True)
    if args.batch != 1:
        print("note: current debug kernel prints only the reflector for mData[0]", flush=True)
    # --bench implies skipping the host-side reference printout.
    if not args.bench and not args.skip_expected:
        print_householder_reference(
            data,
            args.k,
            args.expected_preview,
            args.expected_full,
            args.print_vi,
        )

    if args.bench:
        print(
            "bench: CUDA graph replay with L2-rotated tensor sets "
            f"(sets={args.bench_sets or 'auto'}, calls={args.bench_calls}, "
            f"warmup_ms={args.bench_warmup_ms:g}, repeats={args.bench_repeats})",
            flush=True,
        )
        check_gemv_allclose(data, args.k)
        torch.set_float32_matmul_precision(args.matmul_precision)
        print(f"bench float32_matmul_precision={args.matmul_precision}", flush=True)
        samples, bench_sets = benchmark_direct(
            data,
            args.k,
            args.bench_sets,
            args.bench_calls,
            args.bench_warmup_ms,
            args.bench_repeats,
        )
        selected = select_sample(samples, args.bench_stat)
        sample_text = ", ".join(f"{sample:.6f}" for sample in samples)
        print(f"bench sets={bench_sets}", flush=True)
        print(f"bench samples_ms=[{sample_text}]", flush=True)
        print(f"bench {args.bench_stat}_ms={selected:.6f}", flush=True)
        torch_samples, torch_sets = benchmark_torch_gemv(
            data,
            args.k,
            args.bench_sets,
            args.bench_calls,
            args.bench_warmup_ms,
            args.bench_repeats,
        )
        torch_selected = select_sample(torch_samples, args.bench_stat)
        torch_sample_text = ", ".join(f"{sample:.6f}" for sample in torch_samples)
        print(f"bench torch_gemv sets={torch_sets}", flush=True)
        print(f"bench torch_gemv samples_ms=[{torch_sample_text}]", flush=True)
        print(f"bench torch_gemv {args.bench_stat}_ms={torch_selected:.6f}", flush=True)
        print(f"bench kernel/torch_gemv ratio={torch_selected / selected:.3f}x", flush=True)
        print(f"compile cache: {eigh.Eigh.compile.cache_info()}")
        return

    out = run_direct(data, args.k)
    torch.cuda.synchronize()

    print(f"out: shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
    print(f"out nan_count={torch.isnan(out).sum().item()}")
    print(f"compile cache: {eigh.Eigh.compile.cache_info()}")


if __name__ == "__main__":
    main()
