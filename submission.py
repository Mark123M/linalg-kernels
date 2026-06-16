import torch

try:
    from task import input_t, output_t
except ModuleNotFoundError:
    input_t = torch.Tensor
    output_t = tuple[torch.Tensor, torch.Tensor]


def custom_kernel(data: input_t) -> output_t:
    return torch.geqrf(data)


def _dump_tensor(name: str, value: torch.Tensor) -> None:
    print(f"{name}:")
    print(f"  shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
    print(value)
    print()


def _main() -> None:
    torch.set_printoptions(precision=10, linewidth=140, sci_mode=False)

    matrix = torch.tensor(
        [
            [12.0, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0],
        ],
        dtype=torch.float64,
    )

    packed_householder, tau = custom_kernel(matrix.clone())
    r = torch.triu(packed_householder)
    q = torch.linalg.householder_product(packed_householder, tau)

    _dump_tensor("input A", matrix)
    _dump_tensor("geqrf packed Householder/R output", packed_householder)
    _dump_tensor("tau", tau)
    _dump_tensor("R = triu(packed output)", r)
    _dump_tensor("Q = torch.linalg.householder_product(packed, tau)", q)
    _dump_tensor("Q @ R", q @ r)
    _dump_tensor("A - Q @ R", matrix - q @ r)
    _dump_tensor("Q.T @ Q", q.T @ q)


if __name__ == "__main__":
    _main()
