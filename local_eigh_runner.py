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


def main() -> None:
    parser = argparse.ArgumentParser(description="Local runner for eigh.py CuTeDSL skeleton.")
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--seed", type=int, default=0)
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

    if args.custom:
        out = eigh.custom_kernel(data)
    else:
        out = run_direct(data)
    torch.cuda.synchronize()

    print(f"input: shape={tuple(data.shape)} dtype={data.dtype} device={data.device}")
    if isinstance(out, tuple):
        for idx, tensor in enumerate(out):
            print(f"out[{idx}]: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
    else:
        print(f"out: shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
        print(f"out nan_count={torch.isnan(out).sum().item()}")
    print(f"compile cache: {eigh.Eigh.compile.cache_info()}")


if __name__ == "__main__":
    main()
