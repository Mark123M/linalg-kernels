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


def factor_panel(
    data: torch.Tensor,
    k: int,
    *,
    debug_printf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_dtype = eigh.torch2cute_dtype_map[data.dtype]
    tri = torch.empty_like(data)
    v_ws, w_ws = make_workspace(data, k)
    compiled = eigh.Eigh.compile(
        data_dtype,
        data.size(1),
        debug_printf=debug_printf,
        k=k,
        panel_size=PANEL_SIZE,
    )
    compiled(data, tri, v_ws, w_ws)
    return tri, v_ws, w_ws


def check_rank2k_backend(data: torch.Tensor, k: int, backend: str) -> bool:
    _, v_ws, w_ws = factor_panel(data, k)
    panel_start = k * PANEL_SIZE
    panel_n = data.size(1) - panel_start - 1
    # dsytrd.f DSYR2K offsets: the trailing block starts at global row/col
    # panel_start + nb, i.e. workspace column p = nb - 1 (global = panel_start+1+p).
    offset = panel_start + PANEL_SIZE
    v_trailing = v_ws[:, :PANEL_SIZE, PANEL_SIZE - 1 : panel_n].transpose(1, 2)
    w_trailing = w_ws[:, :PANEL_SIZE, PANEL_SIZE - 1 : panel_n].transpose(1, 2)
    expected = data.clone()
    expected[:, offset:, offset:] -= (
        torch.bmm(v_trailing, w_trailing.transpose(1, 2))
        + torch.bmm(w_trailing, v_trailing.transpose(1, 2))
    )
    actual = data.clone()
    eigh.rank2k_update_(actual, v_ws, w_ws, k, PANEL_SIZE, backend)
    torch.cuda.synchronize()
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    scale = expected.abs().max().clamp_min(1e-30).item()
    outside = max(
        (actual[:, :offset] - data[:, :offset]).abs().max().item(),
        (actual[:, offset:, :offset] - data[:, offset:, :offset]).abs().max().item(),
    )
    close = torch.allclose(actual, expected, rtol=2e-4, atol=2e-4) and outside == 0.0
    print(
        f"rank2k {backend}: allclose={close} max_abs={max_abs:.3e} "
        f"max_rel={max_abs / scale:.3e} outside_max={outside:.3e}",
        flush=True,
    )
    return close


def benchmark_rank2k(
    data: torch.Tensor,
    k: int,
    backend: str,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    _, v_ws, w_ws = factor_panel(data, k)
    eigh.warm_rank2k_backend(PANEL_SIZE, backend)
    base_args = (data.clone(),)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(work: torch.Tensor) -> None:
        eigh.rank2k_update_(work, v_ws, w_ws, k, PANEL_SIZE, backend)

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


def benchmark_panel_with_update(
    data: torch.Tensor,
    k: int,
    backend: str,
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
    eigh.warm_rank2k_backend(PANEL_SIZE, backend)
    tri = torch.empty_like(data)
    v_ws, w_ws = make_workspace(data, k)
    # The public input must remain available to the eventual residual checker, so
    # combined timing includes the explicit source -> working-matrix copy.
    base_args = (data, data.clone())
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(source: torch.Tensor, work: torch.Tensor) -> None:
        work.copy_(source)
        compiled(work, tri, v_ws, w_ws)
        eigh.rank2k_update_(work, v_ws, w_ws, k, PANEL_SIZE, backend)

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


def torch_dlatrd_panel(
    data: torch.Tensor,
    k: int,
    V: torch.Tensor | None = None,
    W: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fair cuBLAS baseline: the same op sequence as the kernel's panel, per column i —
    # column refresh (i>0), reflector with LAPACK's v[i] = 1 convention, b = stale
    # trailing A @ v, inner GEMVs s_w = W^T v / s_v = V^T v, outer corrections
    # b -= V s_w + W s_v, then w = tau*b + (-tau/2)(w^T v) v — accumulating v/w into
    # V/W (batch, m, panel_size), mirroring the kernel's gmem workspace appends.
    # At panel_size=1 this reduces to the old reflector->GEMV->w baseline.
    # No zero-tail edge handling: benchmark inputs are dense random symmetric, and the
    # kernel's tau=2 fallback is a valid sign-flip reflector under residual gating.
    # Everything here must stay CUDA-graph-capture-safe (no .any()/boolean indexing;
    # slice bounds are Python ints, so the panel loop unrolls into the graph).
    panel_start = k * PANEL_SIZE
    m = data.size(1) - panel_start - 1
    if V is None:
        V = torch.empty(data.size(0), m, PANEL_SIZE, device=data.device, dtype=data.dtype)
    if W is None:
        W = torch.empty_like(V)
    trailing = data[:, panel_start + 1 :, panel_start + 1 :]
    for i in range(PANEL_SIZE):
        x = data[:, panel_start + 1 + i :, panel_start + i]
        if i > 0:
            Vp, Wp = V[:, :, :i], W[:, :, :i]
            corr = (
                torch.bmm(Vp[:, i:], W[:, i, :i].unsqueeze(2))
                + torch.bmm(Wp[:, i:], V[:, i, :i].unsqueeze(2))
            ).squeeze(2)
            x = x - corr
        norm = x.norm(dim=1)
        alpha = x[:, 0]
        beta = torch.where(alpha < 0, norm, -norm)
        tau = (beta - alpha) / beta
        v = V[:, :, i]
        v[:, :i] = 0
        v[:, i:] = x / (alpha - beta).unsqueeze(1)
        v[:, i] = 1
        b = torch.bmm(trailing, v.unsqueeze(2)).squeeze(2)
        if i > 0:
            s_w = torch.bmm(Wp.transpose(1, 2), v.unsqueeze(2))
            s_v = torch.bmm(Vp.transpose(1, 2), v.unsqueeze(2))
            b = b - (torch.bmm(Vp, s_w) + torch.bmm(Wp, s_v)).squeeze(2)
        w = tau.unsqueeze(1) * b
        aw = -0.5 * tau * (w * v).sum(dim=1)
        W[:, :, i] = w + aw.unsqueeze(1) * v
    return V, W


def check_gemv_allclose(data: torch.Tensor, k: int) -> bool:
    tri = run_direct(data, k)
    torch.cuda.synchronize()
    panel_start = k * PANEL_SIZE
    _, w_ref = torch_dlatrd_panel(data.float(), k)
    close = True
    max_rel = 0.0
    for i in range(PANEL_SIZE):
        w_kernel = tri[:, panel_start + 1 :, panel_start + i].float()
        close &= torch.allclose(w_kernel, w_ref[:, :, i], rtol=1e-3, atol=1e-2)
        rel = (
            (w_kernel - w_ref[:, :, i]).abs().max()
            / w_ref[:, :, i].abs().max().clamp_min(1e-30)
        ).item()
        max_rel = max(max_rel, rel)
    print(f"w check: allclose={close} max_rel={max_rel:.3e} (over {PANEL_SIZE} cols)", flush=True)
    return close


def benchmark_torch_panel(
    data: torch.Tensor,
    k: int,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    panel_start = k * PANEL_SIZE
    m = data.size(1) - panel_start - 1
    # Preallocated V/W accumulators, shared across rotated inputs like the kernel's
    # v_ws/w_ws workspace (fully rewritten every call).
    V = torch.empty(data.size(0), m, PANEL_SIZE, device=data.device, dtype=data.dtype)
    W = torch.empty_like(V)
    base_args = (data,)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(bench_data: torch.Tensor) -> None:
        torch_dlatrd_panel(bench_data, k, V=V, W=W)

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
    global PANEL_SIZE
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
        "--panel-size",
        type=int,
        default=PANEL_SIZE,
        help="Compile-time panel size (columns per panel) for both the kernel and "
        "the torch DLATRD-panel baseline.",
    )
    parser.add_argument(
        "--update-backend",
        choices=("none", "cublas", "cublasdx"),
        default="none",
        help="Post-panel trailing rank-2k backend to validate and benchmark.",
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
    if not 1 <= args.panel_size <= eigh.MAX_PANEL_SIZE:
        parser.error(f"--panel-size must be in [1, {eigh.MAX_PANEL_SIZE}]")
    PANEL_SIZE = args.panel_size

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
        torch_samples, torch_sets = benchmark_torch_panel(
            data,
            args.k,
            args.bench_sets,
            args.bench_calls,
            args.bench_warmup_ms,
            args.bench_repeats,
        )
        torch_selected = select_sample(torch_samples, args.bench_stat)
        torch_sample_text = ", ".join(f"{sample:.6f}" for sample in torch_samples)
        print(f"bench torch_panel sets={torch_sets}", flush=True)
        print(f"bench torch_panel samples_ms=[{torch_sample_text}]", flush=True)
        print(f"bench torch_panel {args.bench_stat}_ms={torch_selected:.6f}", flush=True)
        print(f"bench kernel/torch_panel ratio={torch_selected / selected:.3f}x", flush=True)
        if args.update_backend != "none":
            if not check_rank2k_backend(data, args.k, args.update_backend):
                raise SystemExit(f"rank2k {args.update_backend} correctness check failed")
            update_samples, update_sets = benchmark_rank2k(
                data,
                args.k,
                args.update_backend,
                args.bench_sets,
                args.bench_calls,
                args.bench_warmup_ms,
                args.bench_repeats,
            )
            update_selected = select_sample(update_samples, args.bench_stat)
            update_text = ", ".join(f"{sample:.6f}" for sample in update_samples)
            print(f"bench rank2k_{args.update_backend} sets={update_sets}", flush=True)
            print(f"bench rank2k_{args.update_backend} samples_ms=[{update_text}]", flush=True)
            print(
                f"bench rank2k_{args.update_backend} "
                f"{args.bench_stat}_ms={update_selected:.6f}",
                flush=True,
            )
            combined_samples, combined_sets = benchmark_panel_with_update(
                data,
                args.k,
                args.update_backend,
                args.bench_sets,
                args.bench_calls,
                args.bench_warmup_ms,
                args.bench_repeats,
            )
            combined_selected = select_sample(combined_samples, args.bench_stat)
            combined_text = ", ".join(f"{sample:.6f}" for sample in combined_samples)
            print(f"bench panel_plus_{args.update_backend} sets={combined_sets}", flush=True)
            print(f"bench panel_plus_{args.update_backend} samples_ms=[{combined_text}]", flush=True)
            print(
                f"bench panel_plus_{args.update_backend} "
                f"{args.bench_stat}_ms={combined_selected:.6f}",
                flush=True,
            )
        print(f"compile cache: {eigh.Eigh.compile.cache_info()}")
        return

    out = run_direct(data, args.k)
    torch.cuda.synchronize()

    if args.update_backend != "none" and not check_rank2k_backend(
        data, args.k, args.update_backend
    ):
        raise SystemExit(f"rank2k {args.update_backend} correctness check failed")

    print(f"out: shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
    print(f"out nan_count={torch.isnan(out).sum().item()}")
    print(f"compile cache: {eigh.Eigh.compile.cache_info()}")


if __name__ == "__main__":
    main()
