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
BACKTRANSFORM_BLOCK_SIZE = 16
DC_LEAF_SIZE = 32
# Matvec cp.async prefetch depth K forwarded to tridiagonalize_; None -> eigh's
# own default. Set by the --stages sweep (and the modal runner) per run.
STAGE = None
# __launch_bounds__ min-CTAs-per-SM (nvvm.minctasm) forwarded to tridiagonalize_;
# None -> eigh's default (0, ptxas-chosen registers). Set by the --min-ctas sweep
# (and the modal runner) to force a register cap for higher occupancy.
MIN_CTAS = None

# Exact 2026-07-14 snapshot of the ranked task's public input generator and
# residual checker. Keep this harness copy aligned with reference-kernels'
# problems/linalg/eigh_py/reference.py.
import math

import torch
from task import input_t, output_t


# Intentionally broad, dimension-scaled residual gates. Eigh has sign and
# eigenspace non-uniqueness, and we want to admit reasonable approximate or
# low-bit internal strategies without comparing against reference eigenvectors.
_EIGEN_RTOL_FACTOR = 200.0
_RECON_RTOL_FACTOR = 400.0
_ORTH_RTOL_FACTOR = 100.0
_SORT_RTOL_FACTOR = 100.0


def _matrix_l1_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(value.double(), ord=1, dim=(-2, -1))


def _property_rtol(n: int, factor: float) -> float:
    eps = torch.finfo(torch.float32).eps
    return factor * max(n, 1) * eps


def _scaled_residual(
    residual: torch.Tensor,
    scale: torch.Tensor,
    n: int,
) -> torch.Tensor:
    eps = torch.finfo(torch.float32).eps
    return residual / (eps * max(n, 1) * scale.clamp_min(1e-30))


def _band_mask(n: int, bandwidth: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    return (idx[:, None] - idx[None, :]).abs() <= bandwidth


def _symmetrize(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.transpose(-1, -2))


def _signed_logspace(batch: int, n: int, cond: int, device: torch.device) -> torch.Tensor:
    span = max(cond, 1)
    magnitudes = torch.logspace(-float(span), 0.0, n, device=device, dtype=torch.float32)
    signs = torch.ones((n,), device=device, dtype=torch.float32)
    signs[::2] = -1.0
    values = magnitudes * signs
    return values.expand(batch, n).contiguous()


def _random_orthogonal(batch: int, n: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    x = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    q, r = torch.linalg.qr(x)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).clamp(min=0.0).mul(2.0).sub(1.0)
    return q * signs.unsqueeze(-2)


def _make_from_spectrum(values: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    batch, n = values.shape
    q = _random_orthogonal(batch, n, gen, values.device)
    a = (q * values.unsqueeze(-2)) @ q.transpose(-1, -2)
    return _symmetrize(a).contiguous()


def _lapack_scale(itype: int) -> float:
    if itype in (6, 11, 14, 17):
        return float(torch.finfo(torch.float32).max**0.5)
    if itype in (7, 12, 15, 18):
        return float(torch.finfo(torch.float32).tiny**0.5)
    return 1.0


def _lapack_signed_values(
    batch: int,
    n: int,
    mode: str,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    ulp = 2.0 * torch.finfo(torch.float32).eps
    if n == 1:
        values = torch.ones((1,), device=device, dtype=torch.float32)
    elif mode == "even":
        values = torch.linspace(1.0, ulp, n, device=device, dtype=torch.float32)
    elif mode == "geometric":
        values = torch.logspace(0.0, math.log10(ulp), n, device=device, dtype=torch.float32)
    elif mode == "clustered":
        values = torch.full((n,), ulp, device=device, dtype=torch.float32)
        values[0] = 1.0
    else:
        raise ValueError(f"unknown LAPACK spectrum mode: {mode}")

    signs = torch.randint(0, 2, (batch, n), device=device, generator=gen, dtype=torch.int64)
    return values.expand(batch, n) * signs.to(torch.float32).mul_(2.0).sub_(1.0)


_LAPACK_CASE_TYPES = {
    "lapack_zero": 1,
    "lapack_identity": 2,
    "lapack_diag_even_spectrum": 3,
    "lapack_diag_geometric_spectrum": 4,
    "lapack_diag_clustered_spectrum": 5,
    "lapack_diag_geometric_high_magnitude": 6,
    "lapack_diag_geometric_low_magnitude": 7,
    "lapack_dense_even_spectrum": 8,
    "lapack_dense_geometric_spectrum": 9,
    "lapack_dense_clustered_spectrum": 10,
    "lapack_dense_even_high_magnitude": 11,
    "lapack_dense_even_low_magnitude": 12,
    "lapack_random_symmetric": 13,
    "lapack_random_symmetric_high_magnitude": 14,
    "lapack_random_symmetric_low_magnitude": 15,
    "lapack_band_even_spectrum": 16,
    "lapack_band_even_high_magnitude": 17,
    "lapack_band_even_low_magnitude": 18,
}


def _generate_lapack(batch: int, n: int, itype: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    assert 1 <= itype <= 18, "LAPACK itype must be in [1, 18]"
    scale = _lapack_scale(itype)
    if itype == 1:
        return torch.zeros((batch, n, n), device=device, dtype=torch.float32)
    if itype == 2:
        return torch.eye(n, device=device, dtype=torch.float32).expand(batch, n, n).clone() * scale
    if itype == 3:
        return torch.diag_embed(_lapack_signed_values(batch, n, "even", gen, device) * scale)
    if itype in (4, 6, 7):
        return torch.diag_embed(_lapack_signed_values(batch, n, "geometric", gen, device) * scale)
    if itype == 5:
        return torch.diag_embed(_lapack_signed_values(batch, n, "clustered", gen, device) * scale)
    if itype in (8, 11, 12):
        return _make_from_spectrum(_lapack_signed_values(batch, n, "even", gen, device) * scale, gen)
    if itype == 9:
        return _make_from_spectrum(_lapack_signed_values(batch, n, "geometric", gen, device), gen)
    if itype == 10:
        return _make_from_spectrum(_lapack_signed_values(batch, n, "clustered", gen, device), gen)
    if itype in (13, 14, 15):
        a = torch.empty((batch, n, n), device=device, dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen)
        return _symmetrize(a) * scale
    if itype in (16, 17, 18):
        a = _make_from_spectrum(_lapack_signed_values(batch, n, "even", gen, device) * scale, gen)
        # DDRVST specifies a symmetric band matrix with eigenvalues. This
        # generator bands a planted-spectrum dense matrix, so the final banded
        # matrix's spectrum is perturbed; the checker validates the returned
        # eigendecomposition of the final FP32 input.
        bandwidth = torch.randint(0, n, (batch,), device=device, generator=gen)
        idx = torch.arange(n, device=device)
        mask = (idx[None, :, None] - idx[None, None, :]).abs() <= bandwidth[:, None, None]
        return (a * mask).contiguous()
    raise ValueError(f"unknown LAPACK matrix type: {itype}")


def _apply_case(a: torch.Tensor, case: str, cond: int, gen: torch.Generator) -> torch.Tensor:
    batch, n, _ = a.shape
    device = a.device

    if case == "dense":
        a = _symmetrize(a)
        if cond:
            scales = torch.logspace(0.0, -float(cond), n, device=device, dtype=torch.float32)
            a = scales.reshape(1, n, 1) * a * scales.reshape(1, 1, n)
    elif case == "spectrum":
        values = _signed_logspace(batch, n, cond, device)
        a = _make_from_spectrum(values, gen)
    elif case == "psd":
        scales = torch.logspace(0.0, -float(max(cond, 1)), n, device=device, dtype=torch.float32)
        g = a * scales.reshape(1, 1, n)
        a = (g @ g.transpose(-1, -2)) / float(n)
    elif case == "rankdef":
        rank = max(1, (3 * n) // 4)
        values = torch.zeros((batch, n), device=device, dtype=torch.float32)
        values[:, -rank:] = torch.logspace(
            -float(max(cond, 1)), 0.0, rank, device=device, dtype=torch.float32
        )
        a = _make_from_spectrum(values, gen)
    elif case == "nearrank":
        rank = max(1, (3 * n) // 4)
        values = torch.empty((batch, n), device=device, dtype=torch.float32)
        values[:, : n - rank] = 1.0e-6 * torch.logspace(
            -2.0, 0.0, n - rank, device=device, dtype=torch.float32
        )
        values[:, n - rank :] = torch.logspace(
            -float(max(cond, 1)), 0.0, rank, device=device, dtype=torch.float32
        )
        a = _make_from_spectrum(values, gen)
    elif case == "repeated":
        groups = max(1, min(16, n // 8))
        base = torch.linspace(-1.0, 1.0, groups, device=device, dtype=torch.float32)
        values = base.repeat_interleave((n + groups - 1) // groups)[:n]
        values = values.expand(batch, n).contiguous()
        a = _make_from_spectrum(values, gen)
    elif case == "clustered":
        center = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
        jitter = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
        values = center.sign().clamp(min=0.0).mul(2.0).sub(1.0) + 1.0e-5 * jitter
        values[n // 3 : 2 * n // 3] = 1.0 + 1.0e-6 * jitter[n // 3 : 2 * n // 3]
        values = values.sort().values.expand(batch, n).contiguous()
        a = _make_from_spectrum(values, gen)
    elif case == "diagonal":
        values = _signed_logspace(batch, n, cond, device)
        a = torch.diag_embed(values)
    elif case == "band":
        bandwidth = max(2, min(32, n // 32))
        a = _symmetrize(a) * _band_mask(n, bandwidth, device)
        diag_boost = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
        a.diagonal(dim1=-2, dim2=-1).add_(diag_boost)
    elif case == "rowscale":
        row_cond = max(cond, 4)
        scales = torch.logspace(0.0, -float(row_cond), n, device=device, dtype=torch.float32)
        a = scales.reshape(1, n, 1) * _symmetrize(a) * scales.reshape(1, 1, n)
    else:
        raise ValueError(f"unknown eigh test case: {case}")

    return _symmetrize(a).contiguous()


_MIXED_PROFILES = (
    "dense",
    "spectrum",
    "psd",
    "rankdef",
    "nearrank",
    "repeated",
    "clustered",
    "band",
    "rowscale",
)
_MIXED_WEIGHTS = (6.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _generate_mixed(a: torch.Tensor, cond: int, gen: torch.Generator) -> torch.Tensor:
    batch = a.shape[0]
    device = a.device
    weights = torch.tensor(_MIXED_WEIGHTS, dtype=torch.float32, device=device)
    labels = torch.multinomial(weights, batch, replacement=True, generator=gen)

    if batch >= 2:
        is_dense = labels == 0
        if not bool(is_dense.any()):
            labels[int(torch.randint(0, batch, (1,), device=device, generator=gen))] = 0
        elif bool(is_dense.all()):
            pos = int(torch.randint(0, batch, (1,), device=device, generator=gen))
            labels[pos] = int(torch.randint(1, len(_MIXED_PROFILES), (1,), device=device, generator=gen))

    for k, prof in enumerate(_MIXED_PROFILES):
        mask = labels == k
        if bool(mask.any()):
            a[mask] = _apply_case(a[mask], prof, cond, gen)
    return a


def generate_input(batch: int, n: int, cond: int, seed: int, case: str = "dense") -> input_t:
    assert batch > 0, "batch must be positive"
    assert n > 0, "n must be positive"
    assert cond >= 0, "cond must be non-negative"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    case = case.lower()
    if case in _LAPACK_CASE_TYPES:
        return _generate_lapack(batch, n, _LAPACK_CASE_TYPES[case], gen, torch.device(device)).contiguous()

    a = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    if case == "mixed":
        return _generate_mixed(a, cond, gen).contiguous()
    return _apply_case(a, case, cond, gen).contiguous()


def ref_kernel(data: input_t) -> output_t:
    values, vectors = torch.linalg.eigh(data)
    return vectors, values


def _check_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> str | None:
    if not isinstance(value, torch.Tensor):
        return f"{name} must be a torch.Tensor"
    if value.shape != shape:
        return f"{name} shape must be {shape}, got {tuple(value.shape)}"
    if value.dtype != torch.float32:
        return f"{name} dtype must be torch.float32, got {value.dtype}"
    if value.device != device:
        return f"{name} must be on {device}, got {value.device}"
    if not torch.isfinite(value).all().item():
        return f"{name} contains NaN or Inf"
    return None


def _check_ascending(values: torch.Tensor, n: int) -> tuple[bool, str]:
    if values.shape[-1] <= 1:
        return True, ""
    diffs = values[..., 1:] - values[..., :-1]
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    allowed = _property_rtol(n, _SORT_RTOL_FACTOR) * scale
    failed = diffs < -allowed
    if bool(failed.any().item()):
        matrix, col = torch.nonzero(failed, as_tuple=False)[0].tolist()
        return False, (
            "eigenvalues must be sorted in ascending order: "
            f"matrix={matrix}, index={col}, "
            f"left={values[matrix, col].item():.3g}, right={values[matrix, col + 1].item():.3g}"
        )
    return True, ""


def check_implementation(data: input_t, output: output_t) -> tuple[bool, str]:
    a = data
    batch, n, _ = a.shape
    eigen_rtol = _property_rtol(n, _EIGEN_RTOL_FACTOR)
    recon_rtol = _property_rtol(n, _RECON_RTOL_FACTOR)
    orth_rtol = _property_rtol(n, _ORTH_RTOL_FACTOR)

    if not isinstance(output, tuple) or len(output) != 2:
        return False, "output must be a tuple `(Q, L)`"

    q, values = output
    error = _check_tensor("Q", q, (batch, n, n), a.device)
    if error is not None:
        return False, error
    error = _check_tensor("L", values, (batch, n), a.device)
    if error is not None:
        return False, error

    good, message = _check_ascending(values, n)
    if not good:
        return False, message

    a_check = a.double()
    q_check = q.double()
    values_check = values.double()
    aq = a_check @ q_check
    ql = q_check * values_check.unsqueeze(-2)
    if not torch.isfinite(aq).all().item() or not torch.isfinite(ql).all().item():
        return False, "A @ Q or Q @ diag(L) contains NaN or Inf"

    eigen_residual = _matrix_l1_norm(aq - ql)
    eigen_scale = _matrix_l1_norm(a_check)
    eigen_allowed = eigen_rtol * eigen_scale
    eigen_scaled = _scaled_residual(eigen_residual, eigen_scale, n)
    if not torch.isfinite(eigen_scaled).all().item():
        return False, "A @ Q - Q @ diag(L) residual produced NaN or Inf"
    eigen_failed = eigen_residual > eigen_allowed
    if bool(eigen_failed.any().item()):
        worst = int(eigen_scaled.argmax().item())
        return False, (
            "A @ Q - Q @ diag(L) is too large: "
            f"matrix={worst}, residual={eigen_residual[worst].item():.3g}, "
            f"allowed={eigen_allowed[worst].item():.3g}, "
            f"scaled={eigen_scaled[worst].item():.3g}"
        )

    eye = torch.eye(n, device=a.device, dtype=torch.float64).expand(batch, n, n)
    qtq = q_check.transpose(-1, -2) @ q_check
    if not torch.isfinite(qtq).all().item():
        return False, "Q.T @ Q contains NaN or Inf"
    orth_residual = _matrix_l1_norm(qtq - eye).amax()
    orth_scale = _matrix_l1_norm(eye).amax()
    orth_allowed = orth_rtol * orth_scale
    orth_scaled = _scaled_residual(orth_residual, orth_scale, n)
    if orth_residual.item() > orth_allowed.item():
        return False, (
            "Q is not orthogonal enough: "
            f"residual={orth_residual.item():.3g}, allowed={orth_allowed.item():.3g}, "
            f"scaled={orth_scaled.item():.3g}"
        )

    recon = ql @ q_check.transpose(-1, -2)
    if not torch.isfinite(recon).all().item():
        return False, "Q @ diag(L) @ Q.T contains NaN or Inf"
    recon_residual = _matrix_l1_norm(recon - a_check)
    recon_scale = _matrix_l1_norm(a_check)
    recon_allowed = recon_rtol * recon_scale
    recon_scaled = _scaled_residual(recon_residual, recon_scale, n)
    recon_failed = recon_residual > recon_allowed
    if bool(recon_failed.any().item()):
        worst = int(recon_scaled.argmax().item())
        return False, (
            "Q @ diag(L) @ Q.T reconstruction is too large: "
            f"matrix={worst}, residual={recon_residual[worst].item():.3g}, "
            f"allowed={recon_allowed[worst].item():.3g}, "
            f"scaled={recon_scaled[worst].item():.3g}"
        )

    projected = q_check.transpose(-1, -2) @ a_check @ q_check
    offdiag = projected - torch.diag_embed(torch.diagonal(projected, dim1=-2, dim2=-1))
    diag_residual = _matrix_l1_norm(offdiag).amax()
    diag_scale = _matrix_l1_norm(a_check).amax()
    diag_scaled = _scaled_residual(diag_residual, diag_scale, n)

    return True, (
        f"eigen_rtol={eigen_rtol:.3g}; "
        f"recon_rtol={recon_rtol:.3g}; "
        f"orth_rtol={orth_rtol:.3g}; "
        f"scaled_eigen_residual={eigen_scaled.amax().item():.3g}; "
        f"scaled_reconstruction_residual={recon_scaled.amax().item():.3g}; "
        f"scaled_diagonalization_residual={diag_scaled.item():.3g}; "
        f"scaled_orthogonality_residual={orth_scaled.item():.3g}; "
        f"batch={batch}; n={n}"
    )


OFFICIAL_BENCHMARK_CASES = (
    {"batch": 20, "n": 32, "cond": 1, "seed": 43214},
    {"batch": 40, "n": 176, "cond": 1, "seed": 423011},
    {"batch": 40, "n": 352, "cond": 1, "seed": 123456},
    {"batch": 640, "n": 512, "cond": 2, "seed": 1029},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 75342},
    {"batch": 8, "n": 2048, "cond": 1, "seed": 224466},
    {"batch": 640, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 770002, "case": "mixed"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770003, "case": "rankdef"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770004, "case": "clustered"},
    {"batch": 60, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 780001, "case": "lapack_dense_even_spectrum"},
    {"batch": 60, "n": 1024, "cond": 0, "seed": 780007, "case": "lapack_dense_geometric_spectrum"},
)


def make_input(batch: int, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    data = torch.randn(batch, n, n, device=device, dtype=dtype)
    data = 0.5 * (data + data.transpose(-1, -2))
    return data.contiguous()


def make_workspace(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # V/W panel workspace in gmem; must be zero-allocated (see Eigh.workspace_shape).
    data_dtype = eigh.torch2cute_dtype_map[data.dtype]
    rows, cols = eigh.Eigh(data_dtype, data.size(1), panel_size=PANEL_SIZE).workspace_shape()
    v_ws = torch.zeros(data.size(0), rows, cols, device=data.device, dtype=torch.float32)
    return v_ws, torch.zeros_like(v_ws)


def make_de(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # LAPACK/cuSOLVER-style outputs: D (batch, N) diagonal, E (batch, N-1) subdiagonal.
    b, n = data.size(0), data.size(1)
    D = torch.empty(b, n, device=data.device, dtype=torch.float32)
    E = torch.empty(b, n - 1, device=data.device, dtype=torch.float32)
    return D, E


def make_htev_outputs(D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n = D.shape
    V = torch.empty(batch, n, n, device=D.device, dtype=torch.float32)
    info = torch.empty(batch, device=D.device, dtype=torch.int32)
    return V, info


def validate_htev_result(
    D_input: torch.Tensor,
    E_input: torch.Tensor,
    D: torch.Tensor,
    E: torch.Tensor,
    V: torch.Tensor,
    info: torch.Tensor,
    *,
    label: str,
) -> bool:
    bad = torch.nonzero(info, as_tuple=False).flatten().tolist()
    finite = bool(torch.isfinite(D).all().item() and torch.isfinite(V).all().item())
    sorted_values = bool((D[:, 1:] >= D[:, :-1]).all().item())
    e_zero = bool((E == 0).all().item())

    vd = V.double()
    tv = D_input.double().unsqueeze(2) * vd
    ed = E_input.double().unsqueeze(2)
    tv[:, 1:, :] += ed * vd[:, :-1, :]
    tv[:, :-1, :] += ed * vd[:, 1:, :]
    vl = vd * D.double().unsqueeze(1)
    eig_scale = torch.maximum(tv.abs().amax(), vl.abs().amax()).clamp_min(1e-30)
    eig_rel = ((tv - vl).abs().amax() / eig_scale).item()

    gram = torch.bmm(vd.transpose(1, 2), vd)
    identity = torch.eye(D.size(1), device=D.device, dtype=torch.float64).expand_as(gram)
    orth_rel = (gram - identity).abs().amax().item()
    ok = not bad and finite and sorted_values and e_zero and eig_rel <= 2e-3 and orth_rel <= 2e-3
    print(
        f"{label}: ok={ok} mode={eigh.htev_execution_mode(D.size(1))} "
        f"info_bad={bad} finite={finite} sorted={sorted_values} E_zero={e_zero} "
        f"eig_rel={eig_rel:.3e} orth_max={orth_rel:.3e}",
        flush=True,
    )
    return ok


def check_tridiag_htev(data: torch.Tensor, backend: str) -> bool:
    work = data.clone()
    D, E = make_de(data)
    v_ws, w_ws = make_workspace(data)
    eigh.tridiagonalize_(work, D, E, v_ws, w_ws, panel_size=PANEL_SIZE, backend=backend)
    D_tri, E_tri = D.clone(), E.clone()
    V, info = make_htev_outputs(D)
    eigh.htev_all_vectors_(D, E, V, info)
    torch.cuda.synchronize()
    htev_ok = validate_htev_result(D_tri, E_tri, D, E, V, info, label="tridiag+htev")

    spectrum_rel = 0.0
    for row in range(min(3, data.size(0))):
        expected = torch.linalg.eigvalsh(data[row].double())
        spectrum_rel = max(
            spectrum_rel,
            ((D[row].double() - expected).abs().max() / expected.abs().max().clamp_min(1e-30)).item(),
        )
    spectrum_ok = spectrum_rel <= 1e-4
    print(
        f"tridiag+htev spectrum: ok={spectrum_ok} rel={spectrum_rel:.3e} "
        f"(first {min(3, data.size(0))} matrices)",
        flush=True,
    )
    return htev_ok and spectrum_ok


def validate_dc_result(
    D_input: torch.Tensor,
    E_input: torch.Tensor,
    L: torch.Tensor,
    Z: torch.Tensor,
    *,
    label: str,
) -> bool:
    """Residual-gate the D&C output against the tridiagonal (D_input, E_input)."""
    finite = bool(torch.isfinite(L).all().item() and torch.isfinite(Z).all().item())
    sorted_values = bool((L[:, 1:] >= L[:, :-1]).all().item())

    vd = Z.double()
    tv = D_input.double().unsqueeze(2) * vd
    ed = E_input.double().unsqueeze(2)
    tv[:, 1:, :] += ed * vd[:, :-1, :]
    tv[:, :-1, :] += ed * vd[:, 1:, :]
    vl = vd * L.double().unsqueeze(1)
    eig_scale = torch.maximum(tv.abs().amax(), vl.abs().amax()).clamp_min(1e-30)
    eig_rel = ((tv - vl).abs().amax() / eig_scale).item()

    gram = torch.bmm(vd.transpose(1, 2), vd)
    identity = torch.eye(L.size(1), device=L.device, dtype=torch.float64).expand_as(gram)
    orth_rel = (gram - identity).abs().amax().item()
    ok = finite and sorted_values and eig_rel <= 2e-3 and orth_rel <= 2e-3
    print(
        f"{label}: ok={ok} leaf={DC_LEAF_SIZE} finite={finite} sorted={sorted_values} "
        f"eig_rel={eig_rel:.3e} orth_max={orth_rel:.3e}",
        flush=True,
    )
    return ok


def check_tridiag_dc(data: torch.Tensor, backend: str) -> bool:
    """Gate the full pipeline: tridiagonalize, then the batched D&C solve."""
    work = data.clone()
    D, E = make_de(data)
    v_ws, w_ws = make_workspace(data)
    eigh.tridiagonalize_(work, D, E, v_ws, w_ws, panel_size=PANEL_SIZE, backend=backend)
    D_tri, E_tri = D.clone(), E.clone()
    ws = eigh.dc_workspace(data.size(0), data.size(1), data.device, leaf_size=DC_LEAF_SIZE)
    Z = torch.zeros(data.size(0), data.size(1), data.size(1), device=data.device,
                    dtype=torch.float32)
    eigh.tridiag_eig_dc_(D, E, Z, ws, leaf_size=DC_LEAF_SIZE)
    torch.cuda.synchronize()
    dc_ok = validate_dc_result(D_tri, E_tri, D, Z, label="tridiag+dc")

    spectrum_rel = 0.0
    for row in range(min(3, data.size(0))):
        expected = torch.linalg.eigvalsh(data[row].double())
        spectrum_rel = max(
            spectrum_rel,
            ((D[row].double() - expected).abs().max() / expected.abs().max().clamp_min(1e-30)).item(),
        )
    spectrum_ok = spectrum_rel <= 1e-4
    print(
        f"tridiag+dc spectrum: ok={spectrum_ok} rel={spectrum_rel:.3e} "
        f"(first {min(3, data.size(0))} matrices)",
        flush=True,
    )
    return dc_ok and spectrum_ok


def factor_panel(
    data: torch.Tensor,
    k: int,
    *,
    debug_printf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data_dtype = eigh.torch2cute_dtype_map[data.dtype]
    D, E = make_de(data)
    tau = torch.empty_like(E)
    v_ws, w_ws = make_workspace(data)
    compiled = eigh.Eigh.compile(
        data_dtype,
        data.size(1),
        debug_printf=debug_printf,
        panel_size=PANEL_SIZE,
    )
    # Work copy: the panel kernel persists reflectors into its input's lower
    # triangle, and callers (w-check mirror, rank2k gate) reread `data` after.
    compiled(data.clone(), D, E, tau, v_ws, w_ws, k * PANEL_SIZE)
    return D, E, v_ws, w_ws


def check_rank2k_backend(data: torch.Tensor, k: int, backend: str) -> bool:
    _, _, v_ws, w_ws = factor_panel(data, k)
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
    eigh.rank2k_update_(actual, v_ws, w_ws, panel_start, PANEL_SIZE, backend)
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
    _, _, v_ws, w_ws = factor_panel(data, k)
    eigh.warm_rank2k_backend(PANEL_SIZE, backend)
    base_args = (data.clone(),)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(work: torch.Tensor) -> None:
        # Evaluated at capture time, so this resolves to the capture stream —
        # required for the cublasdx backend, whose default queue_handle=0 (legacy
        # stream) is not capturable in a CUDA graph.
        eigh.rank2k_update_(
            work,
            v_ws,
            w_ws,
            k * PANEL_SIZE,
            PANEL_SIZE,
            backend,
            queue_handle=torch.cuda.current_stream().cuda_stream,
        )

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
        panel_size=PANEL_SIZE,
    )
    eigh.warm_rank2k_backend(PANEL_SIZE, backend)
    D, E = make_de(data)
    tau = torch.empty_like(E)
    v_ws, w_ws = make_workspace(data)
    # The public input must remain available to the eventual residual checker, so
    # combined timing includes the explicit source -> working-matrix copy.
    base_args = (data, data.clone())
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(source: torch.Tensor, work: torch.Tensor) -> None:
        work.copy_(source)
        compiled(work, D, E, tau, v_ws, w_ws, k * PANEL_SIZE)
        # Capture-time current stream: see benchmark_rank2k.
        eigh.rank2k_update_(
            work,
            v_ws,
            w_ws,
            k * PANEL_SIZE,
            PANEL_SIZE,
            backend,
            queue_handle=torch.cuda.current_stream().cuda_stream,
        )

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
        panel_size=PANEL_SIZE,
    )
    # Rotate only the input matrices: D/E and the V/W workspace are fully
    # rewritten in their valid regions every call, and the torch baseline's
    # outputs are tiny — so sharing them keeps the auto-picked set counts and
    # per-set memory traffic identical between the kernel and torch benches.
    D, E = make_de(data)
    tau = torch.empty_like(E)
    v_ws, w_ws = make_workspace(data)
    base_args = (data,)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(bench_data: torch.Tensor) -> None:
        compiled(bench_data, D, E, tau, v_ws, w_ws, k * PANEL_SIZE)

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
    panel_start: int,
    panel_size: int,
    V: torch.Tensor | None = None,
    W: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    E: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fair cuBLAS baseline: the same op sequence as the kernel's panel, per column i —
    # column refresh (i>0), reflector with LAPACK's v[i] = 1 convention, b = stale
    # trailing A @ v, inner GEMVs s_w = W^T v / s_v = V^T v, outer corrections
    # b -= V s_w + W s_v, then w = tau*b + (-tau/2)(w^T v) v — accumulating v/w into
    # V/W (batch, m, panel_size), mirroring the kernel's gmem workspace appends.
    # When D/E are provided, the kernel's per-column D/E stores are mirrored too:
    # E[col] = beta; D[col] = raw diagonal minus its refresh correction (the
    # diagonal of column i sits at v-row i-1), raw copy at i = 0.
    # At panel_size=1 this reduces to the old reflector->GEMV->w baseline.
    # No zero-tail edge handling: benchmark inputs are dense random symmetric, and the
    # kernel's tau=2 fallback is a valid sign-flip reflector under residual gating.
    # Everything here must stay CUDA-graph-capture-safe (no .any()/boolean indexing;
    # slice bounds are Python ints, so the panel loop unrolls into the graph).
    m = data.size(1) - panel_start - 1
    if V is None:
        V = torch.empty(data.size(0), m, panel_size, device=data.device, dtype=data.dtype)
    if W is None:
        W = torch.empty_like(V)
    trailing = data[:, panel_start + 1 :, panel_start + 1 :]
    for i in range(panel_size):
        col = panel_start + i
        x = data[:, panel_start + 1 + i :, col]
        if i > 0:
            # Refresh coefficient row = the diagonal row of column i, v-space
            # row i-1 (dlatrd.f reads W(I,1)/A(I,1), local row I).
            Vp, Wp = V[:, :, :i], W[:, :, :i]
            corr = (
                torch.bmm(Vp[:, i:], W[:, i - 1, :i].unsqueeze(2))
                + torch.bmm(Wp[:, i:], V[:, i - 1, :i].unsqueeze(2))
            ).squeeze(2)
            x = x - corr
        if D is not None:
            if i == 0:
                D[:, col] = data[:, col, col]
            else:
                # Diagonal element: row and coefficient row coincide (both i-1).
                D[:, col] = data[:, col, col] - 2 * (
                    V[:, i - 1, :i] * W[:, i - 1, :i]
                ).sum(dim=1)
        norm = x.norm(dim=1)
        alpha = x[:, 0]
        beta = torch.where(alpha < 0, norm, -norm)
        tau = (beta - alpha) / beta
        if E is not None:
            E[:, col] = beta
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


def _panel_spans(n: int, panel_size: int) -> list[tuple[int, int]]:
    # The N-1 reflector columns split into full panels plus one tail panel.
    full, tail = divmod(n - 1, panel_size)
    spans = [(j * panel_size, panel_size) for j in range(full)]
    if tail:
        spans.append((full * panel_size, tail))
    return spans


def torch_backtransform_t_reference(
    data: torch.Tensor,
    tau: torch.Tensor,
    panel_start: int,
    panel_width: int,
    block_size: int,
) -> torch.Tensor:
    """Independent FP32 LARFT recurrence for one lower-reflector panel."""
    batch = data.size(0)
    V = torch.tril(
        data[:, panel_start + 1 :, panel_start : panel_start + panel_width]
    )
    T = torch.zeros(
        batch, block_size, block_size, device=data.device, dtype=torch.float32
    )
    for i in range(panel_width):
        tau_i = tau[:, panel_start + i]
        if i:
            dots = (V[:, :, :i] * V[:, :, i : i + 1]).sum(dim=1)
            x = -tau_i.unsqueeze(1) * dots
            for row in range(i):
                T[:, row, i] = (T[:, row, row:i] * x[:, row:i]).sum(dim=1)
        T[:, i, i] = tau_i
    return T


def validate_backtransform_t_factorization(
    data: torch.Tensor,
    backend: str,
    backtransform_block_size: int,
    *,
    label: str,
) -> bool:
    """Factor ``data`` and validate every reverse-panel compact-WY T."""
    work = data.clone()
    D, E = make_de(data)
    tau = torch.empty_like(E)
    v_ws, w_ws = make_workspace(data)
    eigh.tridiagonalize_(
        work,
        D,
        E,
        v_ws,
        w_ws,
        panel_size=PANEL_SIZE,
        backend=backend,
        tau=tau,
    )
    torch.cuda.synchronize()
    zero_tau_ok = label != "zero" or bool((tau == 0).all().item())

    expected_spans = tuple(
        reversed(_panel_spans(data.size(1), backtransform_block_size))
    )
    spans = eigh.backtransform_panel_spans(data.size(1), backtransform_block_size)
    spans_ok = spans == expected_spans
    active_rows = tuple(data.size(1) - start - 1 for start, _ in spans)
    growth_ok = all(a < b for a, b in zip(active_rows, active_rows[1:]))

    T = torch.empty(
        data.size(0),
        backtransform_block_size,
        backtransform_block_size,
        device=data.device,
        dtype=torch.float32,
    )
    work_before = work.clone()
    tau_before = tau.clone()
    finite = True
    triangular = True
    diagonal = True
    close = True
    max_abs = 0.0
    max_rel = 0.0
    max_action = 0.0
    for panel_start, panel_width in spans:
        eigh.form_backtransform_t_(
            work,
            tau,
            T,
            panel_start=panel_start,
            panel_width=panel_width,
        )
        torch.cuda.synchronize()
        expected = torch_backtransform_t_reference(
            work,
            tau,
            panel_start,
            panel_width,
            backtransform_block_size,
        )
        diff = (T - expected).abs()
        scale = expected.abs().max().clamp_min(1e-30)
        panel_abs = diff.max().item()
        panel_rel = (diff.max() / scale).item()
        max_abs = max(max_abs, panel_abs)
        max_rel = max(max_rel, panel_rel)
        close &= torch.allclose(T, expected, rtol=3e-4, atol=3e-5)
        finite &= bool(torch.isfinite(T).all().item())
        triangular &= bool((T == torch.triu(T)).all().item())
        diagonal &= torch.allclose(
            T[:, :panel_width, :panel_width].diagonal(dim1=1, dim2=2),
            tau[:, panel_start : panel_start + panel_width],
            rtol=0.0,
            atol=0.0,
        )
        if panel_width < backtransform_block_size:
            triangular &= bool(
                (T[:, panel_width:, :] == 0).all().item()
                and (T[:, :, panel_width:] == 0).all().item()
            )

        # Compare the compact-WY action with the elementary reflectors in FP64.
        V = torch.tril(
            work[:, panel_start + 1 :, panel_start : panel_start + panel_width]
        ).double()
        probe_cols = min(5, data.size(1))
        probe = torch.arange(
            1,
            V.size(1) * probe_cols + 1,
            device=data.device,
            dtype=torch.float64,
        ).reshape(1, V.size(1), probe_cols)
        probe = probe.expand(data.size(0), -1, -1) / max(V.size(1), 1)
        leading = T[:, :panel_width, :panel_width].double()
        block_action = probe - torch.bmm(
            V, torch.bmm(leading, torch.bmm(V.transpose(1, 2), probe))
        )
        elementary_action = probe.clone()
        for i in range(panel_width - 1, -1, -1):
            vi = V[:, :, i : i + 1]
            elementary_action -= (
                tau[:, panel_start + i].double().view(-1, 1, 1)
                * torch.bmm(
                    vi, torch.bmm(vi.transpose(1, 2), elementary_action)
                )
            )
        action_scale = elementary_action.abs().max().clamp_min(1e-30)
        action_rel = ((block_action - elementary_action).abs().max() / action_scale).item()
        max_action = max(max_action, action_rel)

    preserved = bool(torch.equal(work, work_before) and torch.equal(tau, tau_before))

    # Warm first, then verify that replay clears and rebuilds T deterministically.
    graph_ok = True
    if spans:
        panel_start, panel_width = spans[0]
        eigh.warm_backtransform_t(data.size(1), backtransform_block_size)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            eigh.form_backtransform_t_(
                work,
                tau,
                T,
                panel_start=panel_start,
                panel_width=panel_width,
            )
        graph.replay()
        torch.cuda.synchronize()
        first = T.clone()
        T.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        graph_ok = bool(torch.equal(T, first))

    ok = (
        spans_ok
        and growth_ok
        and zero_tau_ok
        and finite
        and triangular
        and diagonal
        and close
        and max_action <= 3e-5
        and preserved
        and graph_ok
    )
    print(
        f"backtransform T {label}: ok={ok} spans={spans_ok} growth={growth_ok} "
        f"zero_tau={zero_tau_ok} "
        f"finite={finite} triangular={triangular} diagonal={diagonal} "
        f"allclose={close} preserved={preserved} graph={graph_ok} "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"action_rel={max_action:.3e} spans_list={spans}",
        flush=True,
    )
    return ok


def check_backtransform_t_builder(
    data: torch.Tensor, backend: str, backtransform_block_size: int
) -> bool:
    random_ok = validate_backtransform_t_factorization(
        data,
        backend,
        backtransform_block_size,
        label="random",
    )
    zero_ok = validate_backtransform_t_factorization(
        torch.zeros_like(data),
        backend,
        backtransform_block_size,
        label="zero",
    )
    return random_ok and zero_ok


def torch_backtransform_reference(
    data: torch.Tensor, tau: torch.Tensor, Z: torch.Tensor
) -> torch.Tensor:
    """Independent FP64 application of every elementary reflector."""
    result = Z.double().clone()
    reflectors = data.double()
    tau64 = tau.double()
    for i in range(data.size(1) - 2, -1, -1):
        v = reflectors[:, i + 1 :, i : i + 1]
        active = result[:, i + 1 :, :]
        active -= tau64[:, i].view(-1, 1, 1) * torch.bmm(
            v, torch.bmm(v.transpose(1, 2), active)
        )
    return result


def prepare_backtransform_inputs(
    data: torch.Tensor, backend: str
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    eigh.DCWorkspace,
]:
    """Factor ``data`` and solve its tridiagonal eigensystem."""
    work = data.clone()
    D, E = make_de(data)
    tau = torch.empty_like(E)
    v_ws, w_ws = make_workspace(data)
    queue_handle = torch.cuda.current_stream().cuda_stream
    eigh.tridiagonalize_(
        work,
        D,
        E,
        v_ws,
        w_ws,
        panel_size=PANEL_SIZE,
        backend=backend,
        queue_handle=queue_handle,
        tau=tau,
    )
    dc_ws = eigh.dc_workspace(
        data.size(0), data.size(1), data.device, leaf_size=DC_LEAF_SIZE
    )
    Z = torch.zeros(
        data.size(0),
        data.size(1),
        data.size(1),
        device=data.device,
        dtype=torch.float32,
    )
    eigh.tridiag_eig_dc_(
        D,
        E,
        Z,
        dc_ws,
        leaf_size=DC_LEAF_SIZE,
        queue_handle=queue_handle,
    )
    return work, tau, D, Z, dc_ws


def validate_backtransform(
    data: torch.Tensor,
    backend: str,
    backtransform_block_size: int,
    *,
    label: str,
    check_graph: bool,
) -> bool:
    """Validate blocked reflector application and the final eigensystem."""
    work, tau, values, Z, dc_ws = prepare_backtransform_inputs(data, backend)
    torch.cuda.synchronize()
    Z_tri = Z.clone()
    row_zero = Z_tri[:, 0, :].clone()
    expected = torch_backtransform_reference(work, tau, Z_tri)
    work_before = work.clone()
    tau_before = tau.clone()
    T = torch.empty(
        data.size(0),
        backtransform_block_size,
        backtransform_block_size,
        device=data.device,
        dtype=torch.float32,
    )
    eigh.warm_backtransform(data.size(1), backtransform_block_size)
    eigh.backtransform_eigenvectors_(work, tau, Z, T, dc_ws)
    torch.cuda.synchronize()

    q64 = Z.double()
    action_scale = expected.abs().amax().clamp_min(1e-30)
    action_rel = ((q64 - expected).abs().amax() / action_scale).item()
    finite = bool(torch.isfinite(values).all().item() and torch.isfinite(Z).all().item())
    sorted_values = bool((values[:, 1:] >= values[:, :-1]).all().item())
    row_zero_ok = bool(torch.equal(Z[:, 0, :], row_zero))
    preserved = bool(torch.equal(work, work_before) and torch.equal(tau, tau_before))
    zero_tau_ok = label != "zero" or bool((tau == 0).all().item())

    a64 = data.double()
    l64 = values.double()
    aq = torch.bmm(a64, q64)
    ql = q64 * l64.unsqueeze(1)
    eig_scale = torch.maximum(aq.abs().amax(), ql.abs().amax()).clamp_min(1e-30)
    eig_rel = ((aq - ql).abs().amax() / eig_scale).item()
    recon = torch.bmm(q64 * l64.unsqueeze(1), q64.transpose(1, 2))
    recon_scale = a64.abs().amax().clamp_min(1e-30)
    recon_rel = ((recon - a64).abs().amax() / recon_scale).item()
    gram = torch.bmm(q64.transpose(1, 2), q64)
    identity = torch.eye(
        data.size(1), device=data.device, dtype=torch.float64
    ).expand_as(gram)
    orth_rel = (gram - identity).abs().amax().item()

    graph_ok = True
    if check_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            Z.copy_(Z_tri)
            eigh.backtransform_eigenvectors_(work, tau, Z, T, dc_ws)
        graph.replay()
        torch.cuda.synchronize()
        graph_result = Z.clone()
        Z.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        graph_ok = bool(torch.equal(Z, graph_result))

    spans = eigh.backtransform_panel_spans(data.size(1), backtransform_block_size)
    expected_spans = tuple(
        reversed(_panel_spans(data.size(1), backtransform_block_size))
    )
    spans_ok = spans == expected_spans
    ok = (
        spans_ok
        and finite
        and sorted_values
        and row_zero_ok
        and preserved
        and zero_tau_ok
        and graph_ok
        and action_rel <= 5e-4
        and eig_rel <= 2e-3
        and recon_rel <= 2e-3
        and orth_rel <= 2e-3
    )
    print(
        f"backtransform {label}: ok={ok} spans={spans_ok} finite={finite} "
        f"sorted={sorted_values} row0={row_zero_ok} preserved={preserved} "
        f"zero_tau={zero_tau_ok} graph={graph_ok} action_rel={action_rel:.3e} "
        f"eig_rel={eig_rel:.3e} recon_rel={recon_rel:.3e} "
        f"orth_max={orth_rel:.3e} spans_list={spans}",
        flush=True,
    )
    return ok


def check_backtransform(
    data: torch.Tensor, backend: str, backtransform_block_size: int
) -> bool:
    random_ok = validate_backtransform(
        data,
        backend,
        backtransform_block_size,
        label="random",
        check_graph=True,
    )
    zero_ok = validate_backtransform(
        torch.zeros_like(data),
        backend,
        backtransform_block_size,
        label="zero",
        check_graph=False,
    )
    return random_ok and zero_ok


def _official_eigh_residuals(
    data: torch.Tensor, Q: torch.Tensor, L: torch.Tensor
) -> tuple[bool, float, float, float]:
    """Dimension-scaled FP64 gates used by the ranked task checker."""
    n = data.size(1)
    eps = torch.finfo(torch.float32).eps
    a64 = data.double()
    q64 = Q.double()
    l64 = L.double()

    def norm1(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_norm(value, ord=1, dim=(-2, -1))

    aq = torch.bmm(a64, q64)
    ql = q64 * l64.unsqueeze(1)
    a_scale = norm1(a64)
    eig_resid = norm1(aq - ql)
    eig_allowed = (200.0 * n * eps) * a_scale

    eye = torch.eye(n, device=data.device, dtype=torch.float64).expand(
        data.size(0), n, n
    )
    gram = torch.bmm(q64.transpose(1, 2), q64)
    orth_resid = norm1(gram - eye).amax()
    orth_allowed = (100.0 * n * eps) * norm1(eye).amax()

    recon = torch.bmm(ql, q64.transpose(1, 2))
    recon_resid = norm1(recon - a64)
    recon_allowed = (400.0 * n * eps) * a_scale
    scale = (eps * n * a_scale.clamp_min(1e-30))
    eig_scaled = (eig_resid / scale).amax().item()
    recon_scaled = (recon_resid / scale).amax().item()
    orth_scaled = (
        orth_resid / (eps * n * norm1(eye).amax().clamp_min(1e-30))
    ).item()
    ok = bool(
        (eig_resid <= eig_allowed).all().item()
        and (recon_resid <= recon_allowed).all().item()
        and (orth_resid <= orth_allowed).item()
    )
    return ok, eig_scaled, recon_scaled, orth_scaled


def validate_full_eigh(
    data: torch.Tensor,
    backend: str,
    backtransform_block_size: int,
    *,
    label: str,
    check_graph: bool,
) -> bool:
    """Validate the connected production driver against task invariants."""
    batch, n, _ = data.shape
    panel_size = min(PANEL_SIZE, n - 1)
    bt_size = min(backtransform_block_size, n - 1)
    leaf_size = min(DC_LEAF_SIZE, n)
    workspace = eigh.full_eigh_workspace(
        batch,
        n,
        data.device,
        panel_size=panel_size,
        backtransform_block_size=bt_size,
        leaf_size=leaf_size,
    )
    Q = torch.empty_like(data)
    L = torch.empty(batch, n, device=data.device, dtype=torch.float32)
    source_before = data.clone()
    eigh.warm_full_eigh(
        n,
        panel_size=panel_size,
        backtransform_block_size=bt_size,
        leaf_size=leaf_size,
        backend=backend,
    )
    queue_handle = torch.cuda.current_stream().cuda_stream
    eigh.full_eigh_out_(
        data, Q, L, workspace, backend=backend, queue_handle=queue_handle
    )
    torch.cuda.synchronize()

    leaf_bad = []
    for group in workspace.dc.leaf_groups:
        bad = torch.nonzero(group.info, as_tuple=False).flatten().tolist()
        if bad:
            leaf_bad.append((group.size, bad))
    finite = bool(torch.isfinite(Q).all().item() and torch.isfinite(L).all().item())
    input_ok = bool(torch.equal(data, source_before))
    value_scale = L.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    sort_allowed = 100.0 * n * torch.finfo(torch.float32).eps * value_scale
    sorted_ok = bool(((L[:, 1:] - L[:, :-1]) >= -sort_allowed).all().item())
    residual_ok, eig_scaled, recon_scaled, orth_scaled = _official_eigh_residuals(
        data, Q, L
    )

    graph_ok = True
    if check_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            eigh.full_eigh_out_(
                data,
                Q,
                L,
                workspace,
                backend=backend,
                queue_handle=torch.cuda.current_stream().cuda_stream,
            )
        graph.replay()
        torch.cuda.synchronize()
        graph_q, graph_l = Q.clone(), L.clone()
        Q.fill_(float("nan"))
        L.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        graph_ok = bool(torch.equal(Q, graph_q) and torch.equal(L, graph_l))

    ok = (
        not leaf_bad
        and finite
        and input_ok
        and sorted_ok
        and residual_ok
        and graph_ok
    )
    print(
        f"full eigh {label}: ok={ok} finite={finite} sorted={sorted_ok} "
        f"input_preserved={input_ok} leaf_bad={leaf_bad} graph={graph_ok} "
        f"scaled_eig={eig_scaled:.3e} scaled_recon={recon_scaled:.3e} "
        f"scaled_orth={orth_scaled:.3e}",
        flush=True,
    )
    return ok


def check_full_eigh(
    data: torch.Tensor, backend: str, backtransform_block_size: int
) -> bool:
    random_ok = validate_full_eigh(
        data,
        backend,
        backtransform_block_size,
        label="random-tail",
        check_graph=True,
    )
    zero_ok = validate_full_eigh(
        torch.zeros_like(data),
        backend,
        backtransform_block_size,
        label="zero",
        check_graph=False,
    )
    div_n = max(2, ((data.size(1) - 1) // backtransform_block_size) * backtransform_block_size + 1)
    if div_n == data.size(1):
        div_n = data.size(1) + backtransform_block_size
    diag_values = torch.linspace(
        -1.0, 1.0, div_n, device=data.device, dtype=torch.float32
    ).expand(data.size(0), div_n)
    diagonal = torch.diag_embed(diag_values).contiguous()
    divisible_ok = validate_full_eigh(
        diagonal,
        backend,
        backtransform_block_size,
        label=f"diagonal-divisible-n{div_n}",
        check_graph=False,
    )
    return random_ok and zero_ok and divisible_ok


def torch_tridiag_reference(data: torch.Tensor, panel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """float64 D/E ground truth: the full blocked DLATRD + DSYR2K recurrence."""
    A = data.double().clone()
    b, n = A.size(0), A.size(1)
    D = torch.empty(b, n, device=A.device, dtype=torch.float64)
    E = torch.empty(b, n - 1, device=A.device, dtype=torch.float64)
    for start, size in _panel_spans(n, panel_size):
        V, W = torch_dlatrd_panel(A, start, size, D=D, E=E)
        off = start + size
        # Trailing rows of the panel vectors: v-row p maps to global row
        # start+1+p, so the block starting at global `off` is p >= size-1.
        Vt = V[:, size - 1 :, :]
        Wt = W[:, size - 1 :, :]
        A[:, off:, off:] -= torch.bmm(Vt, Wt.transpose(1, 2)) + torch.bmm(
            Wt, Vt.transpose(1, 2)
        )
    D[:, -1] = A[:, -1, -1]
    return D, E


def torch_tridiag_driver(
    source: torch.Tensor,
    work: torch.Tensor,
    D: torch.Tensor,
    E: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
) -> None:
    # Fair cuBLAS baseline for the full tridiagonalization: the kernel path's
    # exact op sequence — source copy, per-panel DLATRD chain (incl. D/E
    # writes), two bmm rank-2k updates per panel, final D gather. Capture-safe.
    n = source.size(1)
    work.copy_(source)
    for start, size in _panel_spans(n, PANEL_SIZE):
        m = n - start - 1
        torch_dlatrd_panel(
            work, start, size, V=V[:, :m, :size], W=W[:, :m, :size], D=D, E=E
        )
        off = start + size
        Vt = V[:, size - 1 : m, :size]
        Wt = W[:, size - 1 : m, :size]
        work[:, off:, off:] -= torch.bmm(Vt, Wt.transpose(1, 2)) + torch.bmm(
            Wt, Vt.transpose(1, 2)
        )
    D[:, -1] = work[:, -1, -1]


def check_gemv_allclose(data: torch.Tensor, k: int) -> bool:
    D, E, _, w_ws = factor_panel(data, k)
    torch.cuda.synchronize()
    panel_start = k * PANEL_SIZE
    panel_n = data.size(1) - panel_start - 1
    D_ref, E_ref = make_de(data)
    _, w_ref = torch_dlatrd_panel(data.float(), panel_start, PANEL_SIZE, D=D_ref, E=E_ref)
    close = True
    max_rel = 0.0
    for i in range(PANEL_SIZE):
        # The kernel's w columns live in the gmem workspace rows (mTri is gone).
        w_kernel = w_ws[:, i, :panel_n]
        close &= torch.allclose(w_kernel, w_ref[:, :, i], rtol=1e-3, atol=1e-2)
        rel = (
            (w_kernel - w_ref[:, :, i]).abs().max()
            / w_ref[:, :, i].abs().max().clamp_min(1e-30)
        ).item()
        max_rel = max(max_rel, rel)
    cols = slice(panel_start, panel_start + PANEL_SIZE)
    d_close = torch.allclose(D[:, cols], D_ref[:, cols], rtol=1e-3, atol=1e-2)
    e_close = torch.allclose(E[:, cols], E_ref[:, cols], rtol=1e-3, atol=1e-2)
    close &= d_close and e_close
    print(
        f"w check: allclose={close} max_rel={max_rel:.3e} (over {PANEL_SIZE} cols) "
        f"D={d_close} E={e_close}",
        flush=True,
    )
    return close


def check_tridiag(data: torch.Tensor, backend: str) -> bool:
    work = data.clone()
    D, E = make_de(data)
    v_ws, w_ws = make_workspace(data)
    eigh.tridiagonalize_(work, D, E, v_ws, w_ws, panel_size=PANEL_SIZE, backend=backend)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(D).all().item() and torch.isfinite(E).all().item())
    D_ref, E_ref = torch_tridiag_reference(data, PANEL_SIZE)
    d_scale = D_ref.abs().max().clamp_min(1e-30)
    per_mat = (D.double() - D_ref).abs().max(dim=1).values / d_scale
    d_med, d_max = per_mat.median().item(), per_mat.max().item()
    e_max = ((E.double() - E_ref).abs().max() / E_ref.abs().max().clamp_min(1e-30)).item()
    # fp32 and fp64 Householder trajectories legitimately bifurcate at
    # near-cancellations (both remain valid factorizations), so elementwise
    # divergence vs the fp64 mirror is heavy-tailed (max ~4e-1 at 640x512 with
    # eigenvalues still matching to 5e-8). The hard gate is semantic:
    # eigenvalues of the kernel's T vs A in fp64 on the elementwise-worst
    # matrices — a recurrence bug measures ~1e-1 there, healthy runs ~5e-8.
    eig_rel = 0.0
    for row in per_mat.topk(min(3, data.size(0))).indices.tolist():
        T = (
            torch.diag(D[row].double())
            + torch.diag(E[row].double(), 1)
            + torch.diag(E[row].double(), -1)
        )
        ev_t = torch.linalg.eigvalsh(T)
        ev_a = torch.linalg.eigvalsh(data[row].double())
        eig_rel = max(eig_rel, ((ev_t - ev_a).abs().max() / ev_a.abs().max()).item())
    close = finite and eig_rel <= 1e-4 and d_med <= 1e-2
    print(
        f"tridiag {backend}: ok={close} eig_rel={eig_rel:.3e} (worst-3 mats) "
        f"D rel median={d_med:.3e} max={d_max:.3e} E max={e_max:.3e} (ps={PANEL_SIZE})",
        flush=True,
    )
    return close


def sweep_tridiag_stages(data, backend, stage_list, args) -> None:
    """Time the full tridiagonalization at each matvec prefetch depth K in
    ``stage_list`` and print one line per K plus the best. Correctness is
    K-invariant (verified separately), so only the timings are swept."""
    global STAGE
    print(f"bench tridiag stage sweep {stage_list} (backend={backend})", flush=True)
    results = []
    try:
        for st in stage_list:
            STAGE = st
            samples, n_sets = benchmark_tridiag(
                data, backend, args.bench_sets, args.bench_calls,
                args.bench_warmup_ms, args.bench_repeats,
            )
            selected = select_sample(samples, args.bench_stat)
            results.append((st, selected))
            sample_text = ", ".join(f"{s:.4f}" for s in samples)
            print(
                f"bench tridiag_{backend} stage={st} sets={n_sets} "
                f"samples_ms=[{sample_text}] {args.bench_stat}_ms={selected:.6f}",
                flush=True,
            )
    finally:
        STAGE = None
    best = min(results, key=lambda r: r[1])
    print(
        f"bench tridiag stage sweep best: stage={best[0]} "
        f"{args.bench_stat}_ms={best[1]:.4f}",
        flush=True,
    )


def sweep_tridiag_min_ctas(data, backend, min_ctas_list, args) -> None:
    """Time the full tridiagonalization at each __launch_bounds__ min-CTAs-per-SM
    in ``min_ctas_list`` (0 = ptxas-default registers) and print one line each
    plus the best. Occupancy only affects timing, not correctness."""
    global MIN_CTAS
    print(f"bench tridiag min-ctas sweep {min_ctas_list} (backend={backend})", flush=True)
    results = []
    try:
        for mc in min_ctas_list:
            MIN_CTAS = mc
            samples, n_sets = benchmark_tridiag(
                data, backend, args.bench_sets, args.bench_calls,
                args.bench_warmup_ms, args.bench_repeats,
            )
            selected = select_sample(samples, args.bench_stat)
            results.append((mc, selected))
            sample_text = ", ".join(f"{s:.4f}" for s in samples)
            print(
                f"bench tridiag_{backend} min_ctas={mc} sets={n_sets} "
                f"samples_ms=[{sample_text}] {args.bench_stat}_ms={selected:.6f}",
                flush=True,
            )
    finally:
        MIN_CTAS = None
    best = min(results, key=lambda r: r[1])
    print(
        f"bench tridiag min-ctas sweep best: min_ctas={best[0]} "
        f"{args.bench_stat}_ms={best[1]:.4f}",
        flush=True,
    )


def benchmark_tridiag(
    data: torch.Tensor,
    backend: str,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    D, E = make_de(data)
    v_ws, w_ws = make_workspace(data)
    eigh.warm_rank2k_backend(PANEL_SIZE, backend)
    tail = (data.size(1) - 1) % PANEL_SIZE
    if tail and backend == "cublasdx":
        eigh.warm_rank2k_backend(tail, backend)
    base_args = (data, data.clone())
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    # STAGE=None defers to tridiagonalize_'s own default; a set value sweeps K.
    stage_kwargs = {} if STAGE is None else {"stage": STAGE}
    # MIN_CTAS=None -> default (ptxas-chosen regs); a set value forces the cap.
    if MIN_CTAS is not None:
        stage_kwargs["min_blocks_per_mp"] = MIN_CTAS

    def launch(source: torch.Tensor, work: torch.Tensor) -> None:
        work.copy_(source)
        # Capture-time current stream: see benchmark_rank2k.
        eigh.tridiagonalize_(
            work,
            D,
            E,
            v_ws,
            w_ws,
            panel_size=PANEL_SIZE,
            backend=backend,
            queue_handle=torch.cuda.current_stream().cuda_stream,
            **stage_kwargs,
        )

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


def benchmark_htev(
    D_input: torch.Tensor,
    E_input: torch.Tensor,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    eigh.warm_htev(D_input.size(1))
    D = D_input.clone()
    E = E_input.clone()
    V, info = make_htev_outputs(D)
    base_args = (D_input, E_input, D, E, V, info)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(
        source_d: torch.Tensor,
        source_e: torch.Tensor,
        work_d: torch.Tensor,
        work_e: torch.Tensor,
        vectors: torch.Tensor,
        status: torch.Tensor,
    ) -> None:
        work_d.copy_(source_d)
        work_e.copy_(source_e)
        eigh.htev_all_vectors_(
            work_d,
            work_e,
            vectors,
            status,
            queue_handle=torch.cuda.current_stream().cuda_stream,
        )

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


def benchmark_tridiag_htev(
    data: torch.Tensor,
    backend: str,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    D, E = make_de(data)
    V, info = make_htev_outputs(D)
    v_ws, w_ws = make_workspace(data)
    eigh.warm_rank2k_backend(PANEL_SIZE, backend)
    tail = (data.size(1) - 1) % PANEL_SIZE
    if tail and backend == "cublasdx":
        eigh.warm_rank2k_backend(tail, backend)
    eigh.warm_htev(data.size(1))
    base_args = (data, data.clone(), D, E, V, info, v_ws, w_ws)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(
        source: torch.Tensor,
        work: torch.Tensor,
        values: torch.Tensor,
        offdiag: torch.Tensor,
        vectors: torch.Tensor,
        status: torch.Tensor,
        panel_v: torch.Tensor,
        panel_w: torch.Tensor,
    ) -> None:
        queue_handle = torch.cuda.current_stream().cuda_stream
        work.copy_(source)
        eigh.tridiagonalize_(
            work,
            values,
            offdiag,
            panel_v,
            panel_w,
            panel_size=PANEL_SIZE,
            backend=backend,
            queue_handle=queue_handle,
        )
        eigh.htev_all_vectors_(
            values,
            offdiag,
            vectors,
            status,
            queue_handle=queue_handle,
        )

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


def benchmark_dc(
    D_input: torch.Tensor,
    E_input: torch.Tensor,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    """Graph-captured D&C tridiagonal eigensolve alone (inputs pre-factored)."""
    batch, n = D_input.shape
    eigh.warm_dc(n, DC_LEAF_SIZE)
    # The workspace is scratch traffic (fully rewritten each call), so one
    # instance is shared across the L2-rotated input sets.
    ws = eigh.dc_workspace(batch, n, D_input.device, leaf_size=DC_LEAF_SIZE)
    D = D_input.clone()
    Z = torch.zeros(batch, n, n, device=D_input.device, dtype=torch.float32)
    base_args = (D_input, E_input, D, Z)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(
        source_d: torch.Tensor,
        source_e: torch.Tensor,
        work_d: torch.Tensor,
        vectors: torch.Tensor,
    ) -> None:
        work_d.copy_(source_d)
        eigh.tridiag_eig_dc_(
            work_d,
            source_e,
            vectors,
            ws,
            leaf_size=DC_LEAF_SIZE,
            queue_handle=torch.cuda.current_stream().cuda_stream,
        )

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


def benchmark_tridiag_dc(
    data: torch.Tensor,
    backend: str,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    """Graph-captured full pipeline: tridiagonalization + D&C eigensolve."""
    batch, n = data.size(0), data.size(1)
    D, E = make_de(data)
    Z = torch.zeros(batch, n, n, device=data.device, dtype=torch.float32)
    v_ws, w_ws = make_workspace(data)
    eigh.warm_rank2k_backend(PANEL_SIZE, backend)
    tail = (n - 1) % PANEL_SIZE
    if tail and backend == "cublasdx":
        eigh.warm_rank2k_backend(tail, backend)
    eigh.warm_dc(n, DC_LEAF_SIZE)
    ws = eigh.dc_workspace(batch, n, data.device, leaf_size=DC_LEAF_SIZE)
    base_args = (data, data.clone(), D, E, Z, v_ws, w_ws)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(
        source: torch.Tensor,
        work: torch.Tensor,
        values: torch.Tensor,
        offdiag: torch.Tensor,
        vectors: torch.Tensor,
        panel_v: torch.Tensor,
        panel_w: torch.Tensor,
    ) -> None:
        queue_handle = torch.cuda.current_stream().cuda_stream
        work.copy_(source)
        eigh.tridiagonalize_(
            work,
            values,
            offdiag,
            panel_v,
            panel_w,
            panel_size=PANEL_SIZE,
            backend=backend,
            queue_handle=queue_handle,
        )
        eigh.tridiag_eig_dc_(
            values,
            offdiag,
            vectors,
            ws,
            leaf_size=DC_LEAF_SIZE,
            queue_handle=queue_handle,
        )

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


def benchmark_full_eigh(
    data: torch.Tensor,
    backend: str,
    backtransform_block_size: int,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    """Graph diagnostic for all three stages with preallocated outputs."""
    batch, n = data.size(0), data.size(1)
    panel_size = min(PANEL_SIZE, n - 1)
    bt_size = min(backtransform_block_size, n - 1)
    leaf_size = min(DC_LEAF_SIZE, n)
    workspace = eigh.full_eigh_workspace(
        batch,
        n,
        data.device,
        panel_size=panel_size,
        backtransform_block_size=bt_size,
        leaf_size=leaf_size,
    )
    eigh.warm_full_eigh(
        n,
        panel_size=panel_size,
        backtransform_block_size=bt_size,
        leaf_size=leaf_size,
        backend=backend,
    )
    Q = torch.empty_like(data)
    L = torch.empty(batch, n, device=data.device, dtype=torch.float32)
    base_args = (data, Q, L)
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, {})
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, {}, n_sets)

    def launch(source: torch.Tensor, vectors: torch.Tensor, values: torch.Tensor) -> None:
        eigh.full_eigh_out_(
            source,
            vectors,
            values,
            workspace,
            backend=backend,
            queue_handle=torch.cuda.current_stream().cuda_stream,
        )

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


def benchmark_backtransform(
    data: torch.Tensor,
    backend: str,
    backtransform_block_size: int,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    """Graph-captured blocked backtransform on precomputed tridiagonal vectors."""
    work, tau, _, Z, dc_ws = prepare_backtransform_inputs(data, backend)
    T = torch.empty(
        data.size(0),
        backtransform_block_size,
        backtransform_block_size,
        device=data.device,
        dtype=torch.float32,
    )
    eigh.warm_backtransform(data.size(1), backtransform_block_size)
    base_args = (work, tau, Z)
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(
        base_args, base_kwargs, n_sets
    )

    def launch(
        reflectors: torch.Tensor,
        reflector_tau: torch.Tensor,
        vectors: torch.Tensor,
    ) -> None:
        eigh.backtransform_eigenvectors_(
            reflectors, reflector_tau, vectors, T, dc_ws
        )

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


def profile_backtransform_t_once(
    data: torch.Tensor, backend: str, backtransform_block_size: int
) -> None:
    """Expose one warmed reverse T-formation sweep to an external profiler."""
    D, E = make_de(data)
    tau = torch.empty_like(E)
    v_ws, w_ws = make_workspace(data)
    work = data.clone()
    T = torch.empty(
        data.size(0),
        backtransform_block_size,
        backtransform_block_size,
        device=data.device,
        dtype=torch.float32,
    )
    spans = eigh.backtransform_panel_spans(
        data.size(1), backtransform_block_size
    )

    eigh.warm_rank2k_backend(PANEL_SIZE, backend)
    tri_tail = (data.size(1) - 1) % PANEL_SIZE
    if tri_tail and backend == "cublasdx":
        eigh.warm_rank2k_backend(tri_tail, backend)
    queue_handle = torch.cuda.current_stream().cuda_stream
    eigh.tridiagonalize_(
        work,
        D,
        E,
        v_ws,
        w_ws,
        panel_size=PANEL_SIZE,
        backend=backend,
        queue_handle=queue_handle,
        tau=tau,
    )
    eigh.warm_backtransform_t(data.size(1), backtransform_block_size)

    def launch(*, annotate: bool) -> None:
        from contextlib import nullcontext

        for panel_start, panel_width in spans:
            active_rows = data.size(1) - panel_start - 1
            label = (
                f"form_T_start{panel_start}_width{panel_width}_rows{active_rows}"
            )
            with torch.cuda.nvtx.range(label) if annotate else nullcontext():
                eigh.form_backtransform_t_(
                    work,
                    tau,
                    T,
                    panel_start=panel_start,
                    panel_width=panel_width,
                )

    # Compile, execute, and validate outside the profiler range.
    launch(annotate=False)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(T).all().item()):
        raise RuntimeError("T-formation warmup produced non-finite output")

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        with torch.cuda.nvtx.range("backtransform_T_sweep"):
            launch(annotate=True)
            torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()

    print(
        "profile workload: one backtransform T sweep complete "
        f"(batch={data.size(0)} n={data.size(1)} tri_ps={PANEL_SIZE} "
        f"bt_bs={backtransform_block_size} backend={backend} panels={len(spans)})",
        flush=True,
    )


def profile_backtransform_once(
    data: torch.Tensor, backend: str, backtransform_block_size: int
) -> None:
    """Expose one warmed blocked eigenvector backtransform to a profiler."""
    work, tau, _, Z, dc_ws = prepare_backtransform_inputs(data, backend)
    Z_seed = Z.clone()
    T = torch.empty(
        data.size(0),
        backtransform_block_size,
        backtransform_block_size,
        device=data.device,
        dtype=torch.float32,
    )
    eigh.warm_backtransform(data.size(1), backtransform_block_size)

    eigh.backtransform_eigenvectors_(work, tau, Z, T, dc_ws)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(Z).all().item()):
        raise RuntimeError("backtransform warmup produced non-finite eigenvectors")

    Z.copy_(Z_seed)
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        eigh.BT_NVTX = True
        with torch.cuda.nvtx.range("backtransform_sweep"):
            eigh.backtransform_eigenvectors_(work, tau, Z, T, dc_ws)
            torch.cuda.synchronize()
    finally:
        eigh.BT_NVTX = False
        cudart.cudaProfilerStop()

    if not bool(torch.isfinite(Z).all().item()):
        raise RuntimeError("profiled backtransform produced non-finite eigenvectors")
    print(
        "profile workload: one eigenvector backtransform complete "
        f"(batch={data.size(0)} n={data.size(1)} tri_ps={PANEL_SIZE} "
        f"bt_bs={backtransform_block_size} leaf={DC_LEAF_SIZE} "
        f"backend={backend})",
        flush=True,
    )


def profile_pipeline_once(
    data: torch.Tensor,
    backend: str,
    stage: str,
    backtransform_block_size: int,
) -> None:
    """Warm, then expose exactly one requested pipeline to an external profiler."""
    from contextlib import nullcontext

    D, E = make_de(data)
    tau = torch.empty_like(E)
    v_ws, w_ws = make_workspace(data)
    work = data.clone()
    Z = dc_ws = T = None
    if stage == "full":
        Z = torch.zeros(
            data.size(0), data.size(1), data.size(1),
            device=data.device, dtype=torch.float32,
        )
        T = torch.empty(
            data.size(0),
            backtransform_block_size,
            backtransform_block_size,
            device=data.device,
            dtype=torch.float32,
        )

    if stage == "full":
        eigh.warm_full_eigh(
            data.size(1),
            panel_size=PANEL_SIZE,
            backtransform_block_size=backtransform_block_size,
            leaf_size=DC_LEAF_SIZE,
            backend=backend,
        )
        dc_ws = eigh.dc_workspace(
            data.size(0), data.size(1), data.device, leaf_size=DC_LEAF_SIZE
        )
    else:
        eigh.warm_rank2k_backend(PANEL_SIZE, backend)
        tail = (data.size(1) - 1) % PANEL_SIZE
        if tail and backend == "cublasdx":
            eigh.warm_rank2k_backend(tail, backend)

    def check_dc_leaf_info(label: str) -> None:
        if dc_ws is None:
            return
        failures = []
        for group in dc_ws.leaf_groups:
            bad = torch.nonzero(group.info, as_tuple=False).flatten().tolist()
            if bad:
                failures.append(f"size={group.size} flat_batch_leaf={bad}")
        if failures:
            raise RuntimeError(f"{label} D&C leaf HTEV failed: " + "; ".join(failures))

    def launch(*, annotate: bool) -> None:
        queue_handle = torch.cuda.current_stream().cuda_stream

        with torch.cuda.nvtx.range("input_copy") if annotate else nullcontext():
            work.copy_(data)

        with torch.cuda.nvtx.range("tridiagonalization") if annotate else nullcontext():
            eigh.tridiagonalize_(
                work,
                D,
                E,
                v_ws,
                w_ws,
                panel_size=PANEL_SIZE,
                backend=backend,
                queue_handle=queue_handle,
                tau=tau,
            )

        if stage == "full":
            assert Z is not None and dc_ws is not None and T is not None
            # Fine-grained D&C ranges (leaves/prep/bmm) live inside eigh.py,
            # gated off by default so eval calls never pay for them.
            eigh.DC_NVTX = annotate
            try:
                with torch.cuda.nvtx.range("dc_solve") if annotate else nullcontext():
                    eigh.tridiag_eig_dc_(
                        D, E, Z, dc_ws, leaf_size=DC_LEAF_SIZE, queue_handle=queue_handle
                    )
            finally:
                eigh.DC_NVTX = False
            eigh.BT_NVTX = annotate
            try:
                with torch.cuda.nvtx.range("backtransform") if annotate else nullcontext():
                    eigh.backtransform_eigenvectors_(work, tau, Z, T, dc_ws)
            finally:
                eigh.BT_NVTX = False

    # Compile every specialization and bring the GPU out of its idle state before
    # cudaProfilerStart. This execution is intentionally outside the captured range.
    launch(annotate=False)
    torch.cuda.synchronize()
    check_dc_leaf_info("warmup")
    if Z is not None:
        if not bool(torch.isfinite(Z).all().item()):
            raise RuntimeError("full eigensolver warmup produced non-finite eigenvectors")

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        pipeline_range = "full_pipeline" if stage == "full" else "tridiag_pipeline"
        with torch.cuda.nvtx.range(pipeline_range):
            launch(annotate=True)
            # Keep collection active until every GPU operation in the pipeline has
            # completed; the D&C solve and all preceding launches are asynchronous.
            torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()

    check_dc_leaf_info("profiled")
    print(
        f"profile workload: one {stage} pipeline launch complete "
        f"(batch={data.size(0)} n={data.size(1)} ps={PANEL_SIZE} backend={backend})",
        flush=True,
    )


def benchmark_torch_tridiag(
    data: torch.Tensor,
    n_sets: int,
    n_timed_calls: int,
    warmup_target_ms: float,
    repeats: int,
) -> tuple[list[float], int]:
    b, n = data.size(0), data.size(1)
    D, E = make_de(data)
    V = torch.empty(b, n - 1, PANEL_SIZE, device=data.device, dtype=data.dtype)
    W = torch.empty_like(V)
    base_args = (data, data.clone())
    base_kwargs = {}
    if n_sets == 0:
        n_sets = _pick_l2_rotate_count(base_args, base_kwargs)
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(base_args, base_kwargs, n_sets)

    def launch(source: torch.Tensor, work: torch.Tensor) -> None:
        torch_tridiag_driver(source, work, D, E, V, W)

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

    # The kernel path writes D/E; the fair baseline must produce them too.
    D, E = make_de(data)

    def launch(bench_data: torch.Tensor) -> None:
        torch_dlatrd_panel(bench_data, panel_start, PANEL_SIZE, V=V, W=W, D=D, E=E)

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
    global PANEL_SIZE, DC_LEAF_SIZE
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
        "--tri",
        action="store_true",
        help="Full-tridiagonalization mode: D/E gate vs the float64 blocked "
        "recurrence, plus (with --bench) kernel driver vs fair torch driver. "
        "One timed call is a full factorization, so use small --bench-calls.",
    )
    parser.add_argument(
        "--tri-solve",
        action="store_true",
        help="Run full tridiagonalization followed by the batched divide-and-"
        "conquer tridiagonal eigensolver (block-mode HTEV leaves + GEMM merges).",
    )
    parser.add_argument(
        "--full-eigh",
        action="store_true",
        help="Run and validate the connected tridiagonalization, D&C solve, "
        "and blocked eigenvector backtransform.",
    )
    parser.add_argument(
        "--backtransform-t",
        action="store_true",
        help="Factor the input and validate the reverse-blocked compact-WY "
        "T builder for every reflector panel, including graph replay.",
    )
    parser.add_argument(
        "--backtransform",
        action="store_true",
        help="Factor and solve the input, then validate the complete blocked "
        "eigenvector backtransform; combine with --bench to time it alone.",
    )
    parser.add_argument(
        "--backtransform-block-size",
        type=int,
        default=BACKTRANSFORM_BLOCK_SIZE,
        help="Independent reflector block size for --backtransform-t and "
        "--backtransform.",
    )
    parser.add_argument(
        "--leaf-size",
        type=int,
        default=DC_LEAF_SIZE,
        help="D&C leaf size (subproblems this small are solved by block-mode "
        "cuSolverDx HTEV; must stay under the arch's smem cap, ~157 on sm89, "
        "~240 on B200).",
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
        "--trace-once",
        action="store_true",
        help="Launch exactly one panel kernel for an external intra-kernel profiler.",
    )
    parser.add_argument(
        "--check-panel-only",
        action="store_true",
        help="Run only the panel correctness gate and exit.",
    )
    parser.add_argument(
        "--profile-pipeline-once",
        action="store_true",
        help="Warm, then launch exactly one selected pipeline between CUDA profiler "
        "start/stop calls.",
    )
    parser.add_argument(
        "--profile-stage",
        choices=("full", "tridiag", "backtransform-t", "backtransform"),
        default="full",
        help="Pipeline exposed by --profile-pipeline-once.",
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
    parser.add_argument(
        "--stages",
        default="",
        help="Comma-separated matvec cp.async prefetch depths K to sweep in "
        "--tri --bench mode (e.g. '2,3,4'). Empty -> single run at the kernel "
        "default. Each K reports its own tridiag time; correctness is K-invariant.",
    )
    parser.add_argument(
        "--min-ctas",
        default="",
        help="Comma-separated __launch_bounds__ min-CTAs-per-SM to sweep in "
        "--tri --bench mode (e.g. '0,7,8'; 0 = ptxas-default registers). Empty -> "
        "single run at the kernel default. Forces a register cap for occupancy.",
    )
    args = parser.parse_args()
    if sum(
        (
            args.tri,
            args.tri_solve,
            args.full_eigh,
            args.backtransform_t,
            args.backtransform,
        )
    ) > 1:
        parser.error(
            "--tri, --tri-solve, --full-eigh, --backtransform-t, and "
            "--backtransform are mutually exclusive"
        )
    if args.backtransform_t and any(
        (args.bench, args.trace_once, args.check_panel_only)
    ):
        parser.error(
            "--backtransform-t cannot be combined with benchmark, trace, panel-only, "
            "or profile modes other than --profile-stage=backtransform-t"
        )
    if args.backtransform_t and args.profile_pipeline_once:
        if args.profile_stage != "backtransform-t":
            parser.error(
                "--backtransform-t with --profile-pipeline-once requires "
                "--profile-stage=backtransform-t"
            )
    if args.backtransform and args.profile_pipeline_once:
        if args.profile_stage != "backtransform":
            parser.error(
                "--backtransform with --profile-pipeline-once requires "
                "--profile-stage=backtransform"
            )
    if sum((args.bench, args.trace_once, args.check_panel_only, args.profile_pipeline_once)) > 1:
        parser.error(
            "--bench, --trace-once, --check-panel-only, and "
            "--profile-pipeline-once are mutually exclusive"
        )
    if args.profile_pipeline_once:
        required_mode = {
            "full": args.full_eigh,
            "tridiag": args.tri,
            "backtransform-t": args.backtransform_t,
            "backtransform": args.backtransform,
        }[args.profile_stage]
        if not required_mode:
            required_flag = {
                "full": "full-eigh",
                "tridiag": "tri",
                "backtransform-t": "backtransform-t",
                "backtransform": "backtransform",
            }[args.profile_stage]
            parser.error(
                f"--profile-pipeline-once --profile-stage={args.profile_stage} requires "
                f"--{required_flag}"
            )
    if args.bench_sets < 0 or args.bench_calls <= 0 or args.bench_repeats <= 0:
        parser.error("--bench-sets must be non-negative; --bench-calls and --bench-repeats must be positive")
    if args.bench_warmup_ms < 0:
        parser.error("--bench-warmup-ms must be non-negative")
    if args.k < 0:
        parser.error("--k must be non-negative")
    if not 1 <= args.panel_size <= eigh.MAX_PANEL_SIZE:
        parser.error(f"--panel-size must be in [1, {eigh.MAX_PANEL_SIZE}]")
    PANEL_SIZE = args.panel_size
    if not 1 <= args.backtransform_block_size <= eigh.MAX_PANEL_SIZE:
        parser.error(
            f"--backtransform-block-size must be in [1, {eigh.MAX_PANEL_SIZE}]"
        )
    if args.leaf_size < 3:
        parser.error("--leaf-size must be at least 3")
    DC_LEAF_SIZE = args.leaf_size

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

    if args.full_eigh and not args.profile_pipeline_once:
        backend = args.update_backend if args.update_backend != "none" else "cublas"
        if not check_full_eigh(data, backend, args.backtransform_block_size):
            raise SystemExit("full eigensolver correctness gate failed")
        print("full eigensolver correctness gate passed", flush=True)
        if args.bench:
            samples, n_sets = benchmark_full_eigh(
                data,
                backend,
                args.backtransform_block_size,
                args.bench_sets,
                args.bench_calls,
                args.bench_warmup_ms,
                args.bench_repeats,
            )
            selected = select_sample(samples, args.bench_stat)
            print(
                f"bench full_eigh sets={n_sets} samples_ms={samples} "
                f"{args.bench_stat}_ms={selected:.6f}",
                flush=True,
            )
        return

    if args.backtransform_t and not args.profile_pipeline_once:
        backend = args.update_backend if args.update_backend != "none" else "cublas"
        if not check_backtransform_t_builder(
            data, backend, args.backtransform_block_size
        ):
            raise SystemExit("backtransform T correctness gate failed")
        print("backtransform T correctness gate passed", flush=True)
        return

    if args.backtransform and not args.profile_pipeline_once:
        backend = args.update_backend if args.update_backend != "none" else "cublas"
        if not check_backtransform(
            data, backend, args.backtransform_block_size
        ):
            raise SystemExit("backtransform correctness gate failed")
        print("backtransform correctness gate passed", flush=True)
        if args.bench:
            samples, n_sets = benchmark_backtransform(
                data,
                backend,
                args.backtransform_block_size,
                args.bench_sets,
                args.bench_calls,
                args.bench_warmup_ms,
                args.bench_repeats,
            )
            selected = select_sample(samples, args.bench_stat)
            print(
                f"bench backtransform sets={n_sets} samples_ms={samples} "
                f"{args.bench_stat}_ms={selected:.6f}",
                flush=True,
            )
        return

    if args.trace_once:
        factor_panel(data, args.k, debug_printf=False)
        torch.cuda.synchronize()
        print("trace workload: one panel launch complete", flush=True)
        return

    if args.check_panel_only:
        if not check_gemv_allclose(data, args.k):
            raise SystemExit("panel correctness gate failed")
        print("panel correctness gate passed", flush=True)
        return

    if args.profile_pipeline_once:
        backend = args.update_backend if args.update_backend != "none" else "cublas"
        if args.profile_stage == "backtransform-t":
            profile_backtransform_t_once(
                data, backend, args.backtransform_block_size
            )
        elif args.profile_stage == "backtransform":
            profile_backtransform_once(
                data, backend, args.backtransform_block_size
            )
        else:
            profile_pipeline_once(
                data, backend, args.profile_stage, args.backtransform_block_size
            )
        return

    if args.tri_solve:
        backend = args.update_backend if args.update_backend != "none" else "cublas"
        if not check_tridiag(data, backend):
            raise SystemExit(f"tridiag {backend} correctness check failed")
        if not check_tridiag_dc(data, backend):
            raise SystemExit(f"tridiag+D&C {backend} correctness check failed")
        if args.bench:
            work = data.clone()
            D_input, E_input = make_de(data)
            v_ws, w_ws = make_workspace(data)
            eigh.tridiagonalize_(
                work,
                D_input,
                E_input,
                v_ws,
                w_ws,
                panel_size=PANEL_SIZE,
                backend=backend,
            )
            dc_samples, dc_sets = benchmark_dc(
                D_input.clone(),
                E_input.clone(),
                args.bench_sets,
                args.bench_calls,
                args.bench_warmup_ms,
                args.bench_repeats,
            )
            combined_samples, combined_sets = benchmark_tridiag_dc(
                data,
                backend,
                args.bench_sets,
                args.bench_calls,
                args.bench_warmup_ms,
                args.bench_repeats,
            )
            dc_selected = select_sample(dc_samples, args.bench_stat)
            combined_selected = select_sample(combined_samples, args.bench_stat)
            print(
                f"bench dc sets={dc_sets} samples_ms={dc_samples} "
                f"{args.bench_stat}_ms={dc_selected:.6f}",
                flush=True,
            )
            print(
                f"bench tridiag_dc_{backend} sets={combined_sets} "
                f"samples_ms={combined_samples} {args.bench_stat}_ms={combined_selected:.6f}",
                flush=True,
            )
        return

    # --bench implies skipping the host-side reference printout.
    if args.tri:
        # Full-tridiagonalization mode: D/E gate against the float64 blocked
        # recurrence, then (with --bench) kernel driver vs the fair torch driver.
        backend = args.update_backend if args.update_backend != "none" else "cublas"
        if not check_tridiag(data, backend):
            raise SystemExit(f"tridiag {backend} correctness check failed")
        if args.bench:
            torch.set_float32_matmul_precision(args.matmul_precision)
            print(f"bench float32_matmul_precision={args.matmul_precision}", flush=True)
            stage_list = [int(x) for x in args.stages.split(",") if x.strip()]
            if stage_list:
                sweep_tridiag_stages(data, backend, stage_list, args)
                print(f"compile cache: {eigh.Eigh.compile.cache_info()}")
                return
            min_ctas_list = [int(x) for x in args.min_ctas.split(",") if x.strip()]
            if min_ctas_list:
                sweep_tridiag_min_ctas(data, backend, min_ctas_list, args)
                print(f"compile cache: {eigh.Eigh.compile.cache_info()}")
                return
            samples, bench_sets = benchmark_tridiag(
                data,
                backend,
                args.bench_sets,
                args.bench_calls,
                args.bench_warmup_ms,
                args.bench_repeats,
            )
            selected = select_sample(samples, args.bench_stat)
            sample_text = ", ".join(f"{sample:.6f}" for sample in samples)
            print(f"bench tridiag_{backend} sets={bench_sets}", flush=True)
            print(f"bench tridiag_{backend} samples_ms=[{sample_text}]", flush=True)
            print(
                f"bench tridiag_{backend} {args.bench_stat}_ms={selected:.6f}", flush=True
            )
            torch_samples, torch_sets = benchmark_torch_tridiag(
                data,
                args.bench_sets,
                args.bench_calls,
                args.bench_warmup_ms,
                args.bench_repeats,
            )
            torch_selected = select_sample(torch_samples, args.bench_stat)
            torch_text = ", ".join(f"{sample:.6f}" for sample in torch_samples)
            print(f"bench torch_tridiag sets={torch_sets}", flush=True)
            print(f"bench torch_tridiag samples_ms=[{torch_text}]", flush=True)
            print(
                f"bench torch_tridiag {args.bench_stat}_ms={torch_selected:.6f}",
                flush=True,
            )
            print(f"bench tridiag/torch ratio={torch_selected / selected:.3f}x", flush=True)
        print(f"compile cache: {eigh.Eigh.compile.cache_info()}")
        return

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

    D, E, _, w_ws = factor_panel(data, args.k, debug_printf=True)
    torch.cuda.synchronize()

    if args.update_backend != "none" and not check_rank2k_backend(
        data, args.k, args.update_backend
    ):
        raise SystemExit(f"rank2k {args.update_backend} correctness check failed")

    cols = slice(args.k * PANEL_SIZE, args.k * PANEL_SIZE + PANEL_SIZE)
    print(f"D cols: {D[0, cols].tolist()}")
    print(f"E cols: {E[0, cols].tolist()}")
    print(f"w nan_count={torch.isnan(w_ws).sum().item()}")
    print(f"compile cache: {eigh.Eigh.compile.cache_info()}")


if __name__ == "__main__":
    main()
