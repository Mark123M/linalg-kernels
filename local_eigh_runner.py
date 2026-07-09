import argparse
import sys
import types

import torch


task = types.ModuleType("task")
task.input_t = torch.Tensor
task.output_t = object
sys.modules.setdefault("task", task)

import eigh  # noqa: E402


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def make_input(batch: int, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    data = torch.randn(batch, n, n, device=device, dtype=dtype)
    data = 0.5 * (data + data.transpose(-1, -2))
    return data.contiguous()


def run_direct(data: torch.Tensor) -> torch.Tensor:
    data_dtype = eigh.torch2cute_dtype_map[data.dtype]
    tri = torch.full_like(data, float("nan"))
    eigh.Eigh.compile(data_dtype, data.size(1))(data, tri)
    return tri


def first_column_householder_reference(data: torch.Tensor):
    col = data[0, :, 0].float().cpu()
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
    return alpha, beta, tau, norm_sq, vi


def print_householder_reference(data: torch.Tensor, preview: int, full: bool, print_vi: bool) -> None:
    alpha, beta, tau, norm_sq, vi = first_column_householder_reference(data)
    norm = torch.sqrt(norm_sq)
    print(
        "expected first-column reflector: "
        f"norm={norm.item():.9g} norm_sq={norm_sq.item():.9g} "
        f"alpha={alpha.item():.9g} beta={beta.item():.9g} tau={tau.item():.9g}",
        flush=True,
    )
    if not print_vi and not full:
        return
    limit = vi.numel() if full else min(preview, vi.numel())
    print(f"expected vi preview ({limit}/{vi.numel()}):", flush=True)
    for idx in range(limit):
        print(f"  vi[{idx}]={vi[idx].item():.9g}", flush=True)
    if not full and limit < vi.numel():
        print("  ... pass --expected-full to print all expected vi entries", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local runner for eigh.py CuTeDSL skeleton.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--seed", type=int, default=0)
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
        "--custom",
        action="store_true",
        help="Call eigh.custom_kernel(data) instead of the direct CuTe kernel path.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    try:
        torch.empty((), device=device)
    except Exception as exc:
        raise SystemExit(f"CUDA device allocation failed: {exc}") from exc
    data = make_input(args.batch, args.n, DTYPES[args.dtype], device)

    print(f"input: shape={tuple(data.shape)} dtype={data.dtype} device={data.device}", flush=True)
    if args.batch != 1:
        print("note: current debug kernel reads only mData[0]", flush=True)
    if not args.skip_expected:
        print_householder_reference(
            data,
            args.expected_preview,
            args.expected_full,
            args.print_vi,
        )

    if args.custom:
        out = eigh.custom_kernel(data)
    else:
        out = run_direct(data)
    torch.cuda.synchronize()

    if isinstance(out, tuple):
        for idx, tensor in enumerate(out):
            print(f"out[{idx}]: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
    else:
        print(f"out: shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
        print(f"out nan_count={torch.isnan(out).sum().item()}")
    print(f"compile cache: {eigh.Eigh.compile.cache_info()}")


if __name__ == "__main__":
    main()
