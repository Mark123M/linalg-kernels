from typing import Any

import ctypes

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from task import input_t, output_t


_SMALL_QR = None
_SMALL_QR_COMPILED = {}
_N32_QR = {}
_N32_QR_COMPILED = {}
_N64_QR = {}
_N64_QR_COMPILED = {}
_N176_QR = None
_N176_QR_COMPILED = {}
_N176_WY_FACTOR = {}
_N176_WY_FACTOR_COMPILED = {}
_N176_WY32_FACTOR = {}
_N176_WY32_FACTOR_COMPILED = {}
_N176_TAIL = {}
_N176_TAIL_COMPILED = {}
_N176_TAIL0 = 80
_N176_TAIL_N = 96
_N176_TAIL_NWARPS = 32
_N176_TAIL_UNROLL = 1
_N352_QR = None
_N352_QR_COMPILED = {}
_N352_WY32_FACTOR = {}
_N352_WY32_FACTOR_COMPILED = {}
_PANEL_QR = None
_PANEL_QR_COMPILED = {}
_WY512_FACTOR = {}
_WY512_FACTOR_COMPILED = {}
_WY_FACTORS = {}
_WY_FACTOR_COMPILED = {}
_WY24_NORMS_FACTOR = None
_WY24_NORMS_FACTOR_COMPILED = None
_WY24_MICRO3_FACTOR = None
_WY24_MICRO3_FACTOR_COMPILED = None
_WY16_OUTER64_MERGE = None
_WY16_OUTER64_MERGE_COMPILED = None
_R4096_BUILD_M0 = None
_R4096_INIT_Y_TOP = None
_R4096_PACK_PANEL = None
_R4096_BUILD_M0_COMPILED = None
_R4096_INIT_Y_TOP_COMPILED = None
_R4096_PACK_PANEL_COMPILED = None
_CUBLAS = None
_CUBLAS_HANDLE = None
_R4096_N = 4096
_R4096_WIDTH = 512


def _load_cublas() -> tuple[Any, ctypes.c_void_p]:
    global _CUBLAS, _CUBLAS_HANDLE
    if _CUBLAS is None:
        lib = ctypes.CDLL("libcublas.so")
        lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cublasCreate_v2.restype = ctypes.c_int
        lib.cublasSetMathMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.cublasSetMathMode.restype = ctypes.c_int
        lib.cublasSgemmStridedBatched.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_int,
        ]
        lib.cublasSgemmStridedBatched.restype = ctypes.c_int
        handle = ctypes.c_void_p()
        status = lib.cublasCreate_v2(ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate failed with status {status}")
        _CUBLAS = lib
        _CUBLAS_HANDLE = handle
    return _CUBLAS, _CUBLAS_HANDLE


def _cublas_check(status: int, name: str) -> None:
    if status != 0:
        raise RuntimeError(f"{name} failed with cuBLAS status {status}")


def _load_r4096_build_m0() -> Any:
    global _R4096_BUILD_M0
    if _R4096_BUILD_M0 is not None:
        return _R4096_BUILD_M0

    @cute.kernel
    def r4096_build_m0_kernel(
        work: cute.Tensor,
        r: cute.Tensor,
        sign: cute.Tensor,
        m0: cute.Tensor,
        k: cutlass.Int32,
        total: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        p = block * 256 + tidx
        stride = cute.arch.grid_dim()[0] * 256
        n = _R4096_N
        width = _R4096_WIDTH
        panel_elems = width * width

        while p < total:
            b = p // panel_elems
            rem = p - b * panel_elems
            i = rem // width
            j = rem - i * width
            work_base = b * n * n
            panel_base = b * panel_elems
            alpha = work[work_base + (k + i) * n + k + i]
            s = -1.0 if alpha >= 0.0 else 1.0
            if j == 0:
                sign[b * width + i] = s
            m0[p] = work[work_base + (k + i) * n + k + j] - s * r[panel_base + i * width + j]
            p += stride

    @cute.jit
    def r4096_build_m0(
        work: cute.Tensor,
        r: cute.Tensor,
        sign: cute.Tensor,
        m0: cute.Tensor,
        batch: cutlass.Int32,
        k: cutlass.Int32,
    ):
        total = batch * _R4096_WIDTH * _R4096_WIDTH
        r4096_build_m0_kernel(work, r, sign, m0, k, total).launch(
            grid=(cute.ceil_div(total, 256), 1, 1),
            block=(256, 1, 1),
        )

    _R4096_BUILD_M0 = r4096_build_m0
    return r4096_build_m0


def _load_r4096_init_y_top() -> Any:
    global _R4096_INIT_Y_TOP
    if _R4096_INIT_Y_TOP is not None:
        return _R4096_INIT_Y_TOP

    @cute.kernel
    def r4096_init_y_top_kernel(
        y: cute.Tensor,
        lu: cute.Tensor,
        total: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        p = block * 256 + tidx
        stride = cute.arch.grid_dim()[0] * 256
        n = _R4096_N
        width = _R4096_WIDTH
        panel_elems = width * width
        y_stride = n * width

        while p < total:
            b = p // panel_elems
            rem = p - b * panel_elems
            i = rem // width
            j = rem - i * width
            val = cutlass.Float32(0.0)
            if i == j:
                val = 1.0
            elif i > j:
                val = lu[b * panel_elems + i * width + j]
            y[b * y_stride + i * width + j] = val
            p += stride

    @cute.jit
    def r4096_init_y_top(y: cute.Tensor, lu: cute.Tensor, batch: cutlass.Int32):
        total = batch * _R4096_WIDTH * _R4096_WIDTH
        r4096_init_y_top_kernel(y, lu, total).launch(
            grid=(cute.ceil_div(total, 256), 1, 1),
            block=(256, 1, 1),
        )

    _R4096_INIT_Y_TOP = r4096_init_y_top
    return r4096_init_y_top


def _load_r4096_pack_panel() -> Any:
    global _R4096_PACK_PANEL
    if _R4096_PACK_PANEL is not None:
        return _R4096_PACK_PANEL

    @cute.kernel
    def r4096_pack_panel_kernel(
        work: cute.Tensor,
        tau: cute.Tensor,
        y: cute.Tensor,
        r: cute.Tensor,
        sign: cute.Tensor,
        tmat: cute.Tensor,
        k: cutlass.Int32,
        rows: cutlass.Int32,
        total: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        p = block * 256 + tidx
        stride = cute.arch.grid_dim()[0] * 256
        n = _R4096_N
        width = _R4096_WIDTH
        panel_elems = width * width
        y_stride = n * width
        work_stride = n * n

        while p < total:
            panel = rows * width
            b = p // panel
            rem = p - b * panel
            rr = rem // width
            c = rem - rr * width
            val = cutlass.Float32(0.0)
            if rr > c:
                val = y[b * y_stride + rr * width + c]
            else:
                val = sign[b * width + rr] * r[b * panel_elems + rr * width + c]
            work[b * work_stride + (k + rr) * n + k + c] = val
            if rr == 0:
                tau[b * n + k + c] = tmat[b * panel_elems + c * width + c]
            p += stride

    @cute.jit
    def r4096_pack_panel(
        work: cute.Tensor,
        tau: cute.Tensor,
        y: cute.Tensor,
        r: cute.Tensor,
        sign: cute.Tensor,
        tmat: cute.Tensor,
        batch: cutlass.Int32,
        k: cutlass.Int32,
        rows: cutlass.Int32,
    ):
        total = batch * rows * _R4096_WIDTH
        r4096_pack_panel_kernel(work, tau, y, r, sign, tmat, k, rows, total).launch(
            grid=(cute.ceil_div(total, 256), 1, 1),
            block=(256, 1, 1),
        )

    _R4096_PACK_PANEL = r4096_pack_panel
    return r4096_pack_panel


def _run_r4096_build_m0(
    work: torch.Tensor,
    r: torch.Tensor,
    sign: torch.Tensor,
    m0: torch.Tensor,
    k: int,
) -> None:
    global _R4096_BUILD_M0_COMPILED
    batch = work.shape[0]
    if _R4096_BUILD_M0_COMPILED is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_work = torch.empty((batch * _R4096_N * _R4096_N,), device="cuda", dtype=torch.float32)
            fake_r = torch.empty((batch * _R4096_WIDTH * _R4096_WIDTH,), device="cuda", dtype=torch.float32)
            fake_sign = torch.empty((batch * _R4096_WIDTH,), device="cuda", dtype=torch.float32)
            fake_m0 = torch.empty((batch * _R4096_WIDTH * _R4096_WIDTH,), device="cuda", dtype=torch.float32)
            _R4096_BUILD_M0_COMPILED = cute.compile(
                _load_r4096_build_m0(),
                from_dlpack(fake_work),
                from_dlpack(fake_r),
                from_dlpack(fake_sign),
                from_dlpack(fake_m0),
                batch,
                k,
            )
    _R4096_BUILD_M0_COMPILED(
        from_dlpack(work.reshape(-1)),
        from_dlpack(r.reshape(-1)),
        from_dlpack(sign.reshape(-1)),
        from_dlpack(m0.reshape(-1)),
        batch,
        k,
    )


def _run_r4096_init_y_top(y: torch.Tensor, lu: torch.Tensor) -> None:
    global _R4096_INIT_Y_TOP_COMPILED
    batch = y.shape[0]
    if _R4096_INIT_Y_TOP_COMPILED is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_y = torch.empty((batch * _R4096_N * _R4096_WIDTH,), device="cuda", dtype=torch.float32)
            fake_lu = torch.empty((batch * _R4096_WIDTH * _R4096_WIDTH,), device="cuda", dtype=torch.float32)
            _R4096_INIT_Y_TOP_COMPILED = cute.compile(
                _load_r4096_init_y_top(),
                from_dlpack(fake_y),
                from_dlpack(fake_lu),
                batch,
            )
    _R4096_INIT_Y_TOP_COMPILED(from_dlpack(y.reshape(-1)), from_dlpack(lu.reshape(-1)), batch)


def _run_r4096_pack_panel(
    work: torch.Tensor,
    tau: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    sign: torch.Tensor,
    tmat: torch.Tensor,
    k: int,
    rows: int,
) -> None:
    global _R4096_PACK_PANEL_COMPILED
    batch = work.shape[0]
    if _R4096_PACK_PANEL_COMPILED is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_work = torch.empty((batch * _R4096_N * _R4096_N,), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * _R4096_N,), device="cuda", dtype=torch.float32)
            fake_y = torch.empty((batch * _R4096_N * _R4096_WIDTH,), device="cuda", dtype=torch.float32)
            fake_r = torch.empty((batch * _R4096_WIDTH * _R4096_WIDTH,), device="cuda", dtype=torch.float32)
            fake_sign = torch.empty((batch * _R4096_WIDTH,), device="cuda", dtype=torch.float32)
            fake_t = torch.empty((batch * _R4096_WIDTH * _R4096_WIDTH,), device="cuda", dtype=torch.float32)
            _R4096_PACK_PANEL_COMPILED = cute.compile(
                _load_r4096_pack_panel(),
                from_dlpack(fake_work),
                from_dlpack(fake_tau),
                from_dlpack(fake_y),
                from_dlpack(fake_r),
                from_dlpack(fake_sign),
                from_dlpack(fake_t),
                batch,
                k,
                rows,
            )
    _R4096_PACK_PANEL_COMPILED(
        from_dlpack(work.reshape(-1)),
        from_dlpack(tau.reshape(-1)),
        from_dlpack(y.reshape(-1)),
        from_dlpack(r.reshape(-1)),
        from_dlpack(sign.reshape(-1)),
        from_dlpack(tmat.reshape(-1)),
        batch,
        k,
        rows,
    )


def _structured_t_packed(lu: torch.Tensor, r: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    urinv = torch.linalg.solve_triangular(r.mT, torch.triu(lu).mT, upper=False).mT
    urinv_s = urinv * sign[:, None, :]
    return -torch.linalg.solve_triangular(lu, urinv_s.mT, upper=False, unitriangular=True).mT


def _apply_wy(c: torch.Tensor, y: torch.Tensor, tmat: torch.Tensor) -> None:
    w = torch.bmm(y.transpose(1, 2), c)
    w = torch.bmm(tmat.transpose(1, 2), w)
    c.baddbmm_(y, w, beta=1.0, alpha=-1.0)


def _try_4096_rfirst(data: torch.Tensor) -> output_t | None:
    if (
        not data.is_cuda
        or data.dtype != torch.float32
        or data.ndim != 3
        or data.shape != (2, 4096, 4096)
    ):
        return None

    batch, n, _ = data.shape
    width = _R4096_WIDTH
    work = data.contiguous().clone()
    tau = torch.empty((batch, n), device=data.device, dtype=data.dtype)
    sign = torch.empty((batch, width), device=data.device, dtype=data.dtype)
    m0 = torch.empty((batch, width, width), device=data.device, dtype=data.dtype)
    y_full = torch.empty((batch, n, width), device=data.device, dtype=data.dtype)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        k = 0
        while k < n:
            rows = n - k
            if rows <= width:
                h_tail, tau_tail = torch.geqrf(work[:, k:, k:].contiguous())
                work[:, k:, k:].copy_(h_tail)
                tau[:, k:].copy_(tau_tail)
                break

            panel = work[:, k:, k : k + width].contiguous()
            r = torch.linalg.cholesky_ex(panel.mT @ panel, upper=True)[0].contiguous()
            _run_r4096_build_m0(work, r, sign, m0, k)
            lu = torch.linalg.lu_factor_ex(m0, pivot=False)[0]
            tmat = _structured_t_packed(lu, r, sign)
            _run_r4096_init_y_top(y_full, lu)
            y_full[:, width:rows, :].copy_(
                torch.linalg.solve_triangular(
                    lu.mT,
                    panel[:, width:, :].mT,
                    upper=False,
                ).mT
            )

            _apply_wy(work[:, k:, k + width :], y_full[:, :rows, :], tmat)
            _run_r4096_pack_panel(work, tau, y_full, r, sign, tmat, k, rows)
            k += width
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    return work, tau


def _load_small_qr() -> Any:
    global _SMALL_QR
    if _SMALL_QR is not None:
        return _SMALL_QR

    @cute.kernel
    def small_qr_kernel(
        src: cute.Tensor,
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        n: cutlass.Int32,
        total: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        mat_elems = n * n
        base = bidx * mat_elems
        tau_base = bidx * n

        allocator = cutlass.utils.SmemAllocator()
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((128,)),
            byte_alignment=16,
            swizzle=None,
        )
        vcol = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((64,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)
        scale = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < mat_elems:
            dst_h[base + p] = src[base + p]
            p += 128
        i = tidx
        while i < n:
            dst_tau[tau_base + i] = 0.0
            i += 128
        cute.arch.sync_threads()

        k = 0
        while k < n:
            ss = cutlass.Float32(0.0)
            i = k + 1 + tidx
            while i < n:
                x = dst_h[base + i * n + k]
                ss = ss + x * x
                i += 128
            red[tidx] = ss
            cute.arch.sync_threads()

            step = 64
            while step > 0:
                if tidx < step:
                    red[tidx] = red[tidx] + red[tidx + step]
                cute.arch.sync_threads()
                step = step // 2

            if tidx == 0:
                sigma = red[0]
                alpha = dst_h[base + k * n + k]
                if sigma == 0.0:
                    dst_tau[tau_base + k] = 0.0
                    tau_k.store(0.0)
                    scale.store(0.0)
                else:
                    nr = cute.sqrt(alpha * alpha + sigma)
                    beta = -nr if alpha >= 0.0 else nr
                    local_tau = (beta - alpha) / beta
                    local_scale = 1.0 / (alpha - beta)
                    dst_h[base + k * n + k] = beta
                    dst_tau[tau_base + k] = local_tau
                    tau_k.store(local_tau)
                    scale.store(local_scale)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            if local_tau != 0.0:
                i = k + 1 + tidx
                local_scale = scale.load()
                while i < n:
                    dst_h[base + i * n + k] = dst_h[base + i * n + k] * local_scale
                    i += 128
            cute.arch.sync_threads()

            if local_tau != 0.0:
                if tidx == 0:
                    vcol[k] = 1.0
                i = k + 1 + tidx
                while i < n:
                    vcol[i] = dst_h[base + i * n + k]
                    i += 128
            cute.arch.sync_threads()

            if local_tau != 0.0:
                j = k + 1 + tidx
                while j < n:
                    dot = dst_h[base + k * n + j]
                    i = k + 1
                    while i < n:
                        dot = dot + vcol[i] * dst_h[base + i * n + j]
                        i += 1
                    w = local_tau * dot
                    dst_h[base + k * n + j] = dst_h[base + k * n + j] - w
                    i = k + 1
                    while i < n:
                        dst_h[base + i * n + j] = dst_h[base + i * n + j] - vcol[i] * w
                        i += 1
                    j += 128
            cute.arch.sync_threads()
            k += 1

    @cute.jit
    def small_qr(src: cute.Tensor, dst_h: cute.Tensor, dst_tau: cute.Tensor, n: cutlass.Int32, total: cutlass.Int32):
        small_qr_kernel(src, dst_h, dst_tau, n, total).launch(
            grid=(cute.ceil_div(total, n * n), 1, 1),
            block=(128, 1, 1),
        )

    _SMALL_QR = small_qr
    return _SMALL_QR


def _try_small_cutedsl(data: torch.Tensor) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, n, m = data.shape
    if batch <= 0 or n <= 0 or n != m or n > 64:
        return None
    if n == 32:
        return _try_n32_shared_cutedsl(data, 32)
    if n == 64:
        return _try_n64_shared_cutedsl(data, 32)
    x = data if data.is_contiguous() else data.contiguous()
    h = torch.empty_like(x)
    tau = torch.empty((batch, n), device=x.device, dtype=torch.float32)
    key = (batch, n)
    compiled = _SMALL_QR_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_x = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_h = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * n,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_small_qr(),
                from_dlpack(fake_x),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                n,
                x.numel(),
            )
        _SMALL_QR_COMPILED[key] = compiled
    compiled(from_dlpack(x.reshape(-1)), from_dlpack(h.reshape(-1)), from_dlpack(tau.reshape(-1)), n, x.numel())
    return h, tau


def _load_n32_qr(nwarps: int) -> Any:
    global _N32_QR
    compiled_kernel = _N32_QR.get(nwarps)
    if compiled_kernel is not None:
        return compiled_kernel
    nthreads = nwarps * 32

    @cute.kernel
    def n32_qr_kernel(src: cute.Tensor, dst_h: cute.Tensor, dst_tau: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        base = bidx * 32 * 32
        tau_base = bidx * 32

        allocator = cutlass.utils.SmemAllocator()
        hs = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32 * 33,)),
            byte_alignment=16,
            swizzle=None,
        )
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((nwarps,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
            swizzle=None,
        )
        scale_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < 32 * 32:
            row = p // 32
            col = p - row * 32
            hs[row * 33 + col] = src[base + p]
            p += nthreads
        cute.arch.sync_threads()

        for k in cutlass.range_constexpr(32):
            ss = cutlass.Float32(0.0)
            i = k + 1 + tidx
            while i < 32:
                x = hs[i * 33 + k]
                ss = ss + x * x
                i += nthreads

            off = 16
            while off > 0:
                ss = ss + cute.arch.shuffle_sync_down(ss, off)
                off = off // 2
            if lane == 0:
                red[warp] = ss
            cute.arch.sync_threads()

            if warp == 0:
                total = red[lane] if lane < nwarps else 0.0
                off = 16
                while off > 0:
                    total = total + cute.arch.shuffle_sync_down(total, off)
                    off = off // 2
                if lane == 0:
                    alpha = hs[k * 33 + k]
                    if total == 0.0:
                        tau_s[k] = 0.0
                        scale_s[k] = 0.0
                        tau_k.store(0.0)
                    else:
                        nr = cute.sqrt(alpha * alpha + total)
                        beta = -nr if alpha >= 0.0 else nr
                        local_tau = (beta - alpha) / beta
                        hs[k * 33 + k] = beta
                        tau_s[k] = local_tau
                        scale_s[k] = 1.0 / (alpha - beta)
                        tau_k.store(local_tau)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            local_scale = scale_s[k]
            j = k + 1 + warp
            while j < 32:
                acc = cutlass.Float32(0.0)
                i = k + 1 + lane
                while i < 32:
                    acc = acc + hs[i * 33 + k] * hs[i * 33 + j]
                    i += 32

                off = 16
                while off > 0:
                    acc = acc + cute.arch.shuffle_sync_down(acc, off)
                    off = off // 2

                sw = cutlass.Float32(0.0)
                if lane == 0:
                    w = local_tau * (hs[k * 33 + j] + local_scale * acc)
                    hs[k * 33 + j] = hs[k * 33 + j] - w
                    sw = local_scale * w
                sw = cute.arch.shuffle_sync(sw, 0)

                i = k + 1 + lane
                while i < 32:
                    hs[i * 33 + j] = hs[i * 33 + j] - hs[i * 33 + k] * sw
                    i += 32
                j += nwarps
            cute.arch.sync_threads()

        p = tidx
        while p < 32:
            dst_tau[tau_base + p] = tau_s[p]
            p += nthreads

        p = tidx
        while p < 32 * 32:
            row = p // 32
            col = p - row * 32
            hval = hs[row * 33 + col]
            if row > col:
                hval = hval * scale_s[col]
            dst_h[base + p] = hval
            p += nthreads

    @cute.jit
    def n32_qr(src: cute.Tensor, dst_h: cute.Tensor, dst_tau: cute.Tensor, batch: cutlass.Int32):
        n32_qr_kernel(src, dst_h, dst_tau).launch(grid=(batch, 1, 1), block=(nthreads, 1, 1))

    _N32_QR[nwarps] = n32_qr
    return n32_qr


def _try_n32_shared_cutedsl(data: torch.Tensor, nwarps: int) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, n, m = data.shape
    if batch <= 0 or n != 32 or m != 32:
        return None
    x = data if data.is_contiguous() else data.contiguous()
    h = torch.empty_like(x)
    tau = torch.empty((batch, 32), device=x.device, dtype=torch.float32)
    key = (batch, nwarps)
    compiled = _N32_QR_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_x = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_h = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 32,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_n32_qr(nwarps),
                from_dlpack(fake_x),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                batch,
            )
        _N32_QR_COMPILED[key] = compiled
    compiled(from_dlpack(x.reshape(-1)), from_dlpack(h.reshape(-1)), from_dlpack(tau.reshape(-1)), batch)
    return h, tau


def _load_n64_qr(nwarps: int) -> Any:
    global _N64_QR
    compiled_kernel = _N64_QR.get(nwarps)
    if compiled_kernel is not None:
        return compiled_kernel
    nthreads = nwarps * 32

    @cute.kernel
    def n64_qr_kernel(src: cute.Tensor, dst_h: cute.Tensor, dst_tau: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        base = bidx * 64 * 64
        tau_base = bidx * 64

        allocator = cutlass.utils.SmemAllocator()
        hs = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((64 * 65,)),
            byte_alignment=16,
            swizzle=None,
        )
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((nwarps,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((64,)),
            byte_alignment=16,
            swizzle=None,
        )
        scale_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((64,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < 64 * 64:
            row = p // 64
            col = p - row * 64
            hs[row * 65 + col] = src[base + p]
            p += nthreads
        cute.arch.sync_threads()

        for k in cutlass.range_constexpr(64):
            ss = cutlass.Float32(0.0)
            i = k + 1 + tidx
            while i < 64:
                x = hs[i * 65 + k]
                ss = ss + x * x
                i += nthreads

            off = 16
            while off > 0:
                ss = ss + cute.arch.shuffle_sync_down(ss, off)
                off = off // 2
            if lane == 0:
                red[warp] = ss
            cute.arch.sync_threads()

            if warp == 0:
                total = red[lane] if lane < nwarps else 0.0
                off = 16
                while off > 0:
                    total = total + cute.arch.shuffle_sync_down(total, off)
                    off = off // 2
                if lane == 0:
                    alpha = hs[k * 65 + k]
                    if total == 0.0:
                        tau_s[k] = 0.0
                        scale_s[k] = 0.0
                        tau_k.store(0.0)
                    else:
                        nr = cute.sqrt(alpha * alpha + total)
                        beta = -nr if alpha >= 0.0 else nr
                        local_tau = (beta - alpha) / beta
                        hs[k * 65 + k] = beta
                        tau_s[k] = local_tau
                        scale_s[k] = 1.0 / (alpha - beta)
                        tau_k.store(local_tau)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            local_scale = scale_s[k]
            j = k + 1 + warp
            while j < 64:
                acc = cutlass.Float32(0.0)
                i = k + 1 + lane
                while i < 64:
                    acc = acc + hs[i * 65 + k] * hs[i * 65 + j]
                    i += 32

                off = 16
                while off > 0:
                    acc = acc + cute.arch.shuffle_sync_down(acc, off)
                    off = off // 2

                sw = cutlass.Float32(0.0)
                if lane == 0:
                    w = local_tau * (hs[k * 65 + j] + local_scale * acc)
                    hs[k * 65 + j] = hs[k * 65 + j] - w
                    sw = local_scale * w
                sw = cute.arch.shuffle_sync(sw, 0)

                i = k + 1 + lane
                while i < 64:
                    hs[i * 65 + j] = hs[i * 65 + j] - hs[i * 65 + k] * sw
                    i += 32
                j += nwarps
            cute.arch.sync_threads()

        p = tidx
        while p < 64:
            dst_tau[tau_base + p] = tau_s[p]
            p += nthreads

        p = tidx
        while p < 64 * 64:
            row = p // 64
            col = p - row * 64
            hval = hs[row * 65 + col]
            if row > col:
                hval = hval * scale_s[col]
            dst_h[base + p] = hval
            p += nthreads

    @cute.jit
    def n64_qr(src: cute.Tensor, dst_h: cute.Tensor, dst_tau: cute.Tensor, batch: cutlass.Int32):
        n64_qr_kernel(src, dst_h, dst_tau).launch(grid=(batch, 1, 1), block=(nthreads, 1, 1))

    _N64_QR[nwarps] = n64_qr
    return n64_qr


def _try_n64_shared_cutedsl(data: torch.Tensor, nwarps: int) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, n, m = data.shape
    if batch <= 0 or n != 64 or m != 64:
        return None
    x = data if data.is_contiguous() else data.contiguous()
    h = torch.empty_like(x)
    tau = torch.empty((batch, 64), device=x.device, dtype=torch.float32)
    key = (batch, nwarps)
    compiled = _N64_QR_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_x = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_h = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 64,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_n64_qr(nwarps),
                from_dlpack(fake_x),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                batch,
            )
        _N64_QR_COMPILED[key] = compiled
    compiled(from_dlpack(x.reshape(-1)), from_dlpack(h.reshape(-1)), from_dlpack(tau.reshape(-1)), batch)
    return h, tau


def _load_n176_qr() -> Any:
    global _N176_QR
    if _N176_QR is not None:
        return _N176_QR

    @cute.kernel
    def n176_qr_kernel(src: cute.Tensor, dst_h: cute.Tensor, dst_tau: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        elems = 176 * 176
        base = bidx * elems
        tau_base = bidx * 176

        allocator = cutlass.utils.SmemAllocator()
        hs = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((176 * 176,)),
            byte_alignment=16,
            swizzle=None,
        )
        vcol = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((176,)),
            byte_alignment=16,
            swizzle=None,
        )
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((256,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)
        scale = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < elems:
            hs[p] = src[base + p]
            p += 256
        i = tidx
        while i < 176:
            dst_tau[tau_base + i] = 0.0
            i += 256
        cute.arch.sync_threads()

        k = 0
        while k < 176:
            ss = cutlass.Float32(0.0)
            i = k + 1 + tidx
            while i < 176:
                x = hs[i * 176 + k]
                ss = ss + x * x
                i += 256
            red[tidx] = ss
            cute.arch.sync_threads()

            step = 128
            while step > 0:
                if tidx < step:
                    red[tidx] = red[tidx] + red[tidx + step]
                cute.arch.sync_threads()
                step = step // 2

            if tidx == 0:
                sigma = red[0]
                alpha = hs[k * 176 + k]
                if sigma == 0.0:
                    dst_tau[tau_base + k] = 0.0
                    tau_k.store(0.0)
                    scale.store(0.0)
                else:
                    nr = cute.sqrt(alpha * alpha + sigma)
                    beta = -nr if alpha >= 0.0 else nr
                    local_tau = (beta - alpha) / beta
                    local_scale = 1.0 / (alpha - beta)
                    hs[k * 176 + k] = beta
                    dst_tau[tau_base + k] = local_tau
                    tau_k.store(local_tau)
                    scale.store(local_scale)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            if local_tau != 0.0:
                if tidx == 0:
                    vcol[k] = 1.0
                i = k + 1 + tidx
                local_scale = scale.load()
                while i < 176:
                    v = hs[i * 176 + k] * local_scale
                    hs[i * 176 + k] = v
                    vcol[i] = v
                    i += 256
            cute.arch.sync_threads()

            if local_tau != 0.0:
                j = k + 1 + tidx
                while j < 176:
                    dot = hs[k * 176 + j]
                    i = k + 1
                    while i < 176:
                        dot = dot + vcol[i] * hs[i * 176 + j]
                        i += 1
                    w = local_tau * dot
                    hs[k * 176 + j] = hs[k * 176 + j] - w
                    i = k + 1
                    while i < 176:
                        hs[i * 176 + j] = hs[i * 176 + j] - vcol[i] * w
                        i += 1
                    j += 256
            cute.arch.sync_threads()
            k += 1

        p = tidx
        while p < elems:
            dst_h[base + p] = hs[p]
            p += 256

    @cute.jit
    def n176_qr(src: cute.Tensor, dst_h: cute.Tensor, dst_tau: cute.Tensor, batch: cutlass.Int32):
        n176_qr_kernel(src, dst_h, dst_tau).launch(grid=(batch, 1, 1), block=(256, 1, 1))

    _N176_QR = n176_qr
    return _N176_QR


def _try_n176_cutedsl(data: torch.Tensor) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, n, m = data.shape
    if batch <= 0 or n != 176 or m != 176:
        return None
    x = data if data.is_contiguous() else data.contiguous()
    h = torch.empty_like(x)
    tau = torch.empty((batch, 176), device=x.device, dtype=torch.float32)
    compiled = _N176_QR_COMPILED.get(batch)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_x = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_h = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 176,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_n176_qr(),
                from_dlpack(fake_x),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                batch,
            )
        _N176_QR_COMPILED[batch] = compiled
    compiled(from_dlpack(x.reshape(-1)), from_dlpack(h.reshape(-1)), from_dlpack(tau.reshape(-1)), batch)
    return h, tau


def _load_n352_qr() -> Any:
    global _N352_QR
    if _N352_QR is not None:
        return _N352_QR

    @cute.kernel
    def n352_qr_kernel(src: cute.Tensor, dst_h: cute.Tensor, dst_tau: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        elems = 352 * 352
        base = bidx * elems
        tau_base = bidx * 352

        allocator = cutlass.utils.SmemAllocator()
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((512,)),
            byte_alignment=16,
            swizzle=None,
        )
        vcol = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((352,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)
        scale = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < elems:
            dst_h[base + p] = src[base + p]
            p += 512
        i = tidx
        while i < 352:
            dst_tau[tau_base + i] = 0.0
            i += 512
        cute.arch.sync_threads()

        panel0 = 0
        while panel0 < 352:
            panel_end = panel0 + 16 if panel0 + 16 < 352 else 352

            k = panel0
            while k < panel_end:
                ss = cutlass.Float32(0.0)
                i = k + 1 + tidx
                while i < 352:
                    x = dst_h[base + i * 352 + k]
                    ss = ss + x * x
                    i += 512
                red[tidx] = ss
                cute.arch.sync_threads()

                step = 256
                while step > 0:
                    if tidx < step:
                        red[tidx] = red[tidx] + red[tidx + step]
                    cute.arch.sync_threads()
                    step = step // 2

                if tidx == 0:
                    sigma = red[0]
                    alpha = dst_h[base + k * 352 + k]
                    if sigma == 0.0:
                        dst_tau[tau_base + k] = 0.0
                        tau_k.store(0.0)
                        scale.store(0.0)
                    else:
                        nr = cute.sqrt(alpha * alpha + sigma)
                        beta = -nr if alpha >= 0.0 else nr
                        local_tau = (beta - alpha) / beta
                        local_scale = 1.0 / (alpha - beta)
                        dst_h[base + k * 352 + k] = beta
                        dst_tau[tau_base + k] = local_tau
                        tau_k.store(local_tau)
                        scale.store(local_scale)
                cute.arch.sync_threads()

                local_tau = tau_k.load()
                if local_tau != 0.0:
                    if tidx == 0:
                        vcol[k] = 1.0
                    i = k + 1 + tidx
                    local_scale = scale.load()
                    while i < 352:
                        v = dst_h[base + i * 352 + k] * local_scale
                        dst_h[base + i * 352 + k] = v
                        vcol[i] = v
                        i += 512
                cute.arch.sync_threads()

                if local_tau != 0.0:
                    j = k + 1 + tidx
                    while j < panel_end:
                        dot = dst_h[base + k * 352 + j]
                        i = k + 1
                        while i < 352:
                            dot = dot + vcol[i] * dst_h[base + i * 352 + j]
                            i += 1
                        w = local_tau * dot
                        dst_h[base + k * 352 + j] = dst_h[base + k * 352 + j] - w
                        i = k + 1
                        while i < 352:
                            dst_h[base + i * 352 + j] = dst_h[base + i * 352 + j] - vcol[i] * w
                            i += 1
                        j += 512
                cute.arch.sync_threads()
                k += 1

            k = panel0
            while k < panel_end:
                local_tau = dst_tau[tau_base + k]
                if local_tau != 0.0:
                    if tidx == 0:
                        vcol[k] = 1.0
                    i = k + 1 + tidx
                    while i < 352:
                        vcol[i] = dst_h[base + i * 352 + k]
                        i += 512
                cute.arch.sync_threads()

                if local_tau != 0.0:
                    j = panel_end + tidx
                    while j < 352:
                        dot = dst_h[base + k * 352 + j]
                        i = k + 1
                        while i < 352:
                            dot = dot + vcol[i] * dst_h[base + i * 352 + j]
                            i += 1
                        w = local_tau * dot
                        dst_h[base + k * 352 + j] = dst_h[base + k * 352 + j] - w
                        i = k + 1
                        while i < 352:
                            dst_h[base + i * 352 + j] = dst_h[base + i * 352 + j] - vcol[i] * w
                            i += 1
                        j += 512
                cute.arch.sync_threads()
                k += 1

            panel0 += 16

    @cute.jit
    def n352_qr(src: cute.Tensor, dst_h: cute.Tensor, dst_tau: cute.Tensor, batch: cutlass.Int32):
        n352_qr_kernel(src, dst_h, dst_tau).launch(grid=(batch, 1, 1), block=(512, 1, 1))

    _N352_QR = n352_qr
    return _N352_QR


def _try_n352_cutedsl(data: torch.Tensor) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, n, m = data.shape
    if batch <= 0 or n != 352 or m != 352:
        return None
    x = data if data.is_contiguous() else data.contiguous()
    h = torch.empty_like(x)
    tau = torch.empty((batch, 352), device=x.device, dtype=torch.float32)
    compiled = _N352_QR_COMPILED.get(batch)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_x = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_h = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 352,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_n352_qr(),
                from_dlpack(fake_x),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                batch,
            )
        _N352_QR_COMPILED[batch] = compiled
    compiled(from_dlpack(x.reshape(-1)), from_dlpack(h.reshape(-1)), from_dlpack(tau.reshape(-1)), batch)
    return h, tau


def _load_panel_qr() -> Any:
    global _PANEL_QR
    if _PANEL_QR is not None:
        return _PANEL_QR

    @cute.kernel
    def panel_qr_kernel(
        src: cute.Tensor,
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        rows: cutlass.Int32,
        cols: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        elems = rows * cols
        base = bidx * elems
        tau_base = bidx * cols

        allocator = cutlass.utils.SmemAllocator()
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((512,)),
            byte_alignment=16,
            swizzle=None,
        )
        vcol = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((2048,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)
        scale = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < elems:
            dst_h[base + p] = src[base + p]
            p += 512
        i = tidx
        while i < cols:
            dst_tau[tau_base + i] = 0.0
            i += 512
        cute.arch.sync_threads()

        k = 0
        while k < cols:
            ss = cutlass.Float32(0.0)
            i = k + 1 + tidx
            while i < rows:
                x = dst_h[base + i * cols + k]
                ss = ss + x * x
                i += 512
            red[tidx] = ss
            cute.arch.sync_threads()

            step = 256
            while step > 0:
                if tidx < step:
                    red[tidx] = red[tidx] + red[tidx + step]
                cute.arch.sync_threads()
                step = step // 2

            if tidx == 0:
                sigma = red[0]
                alpha = dst_h[base + k * cols + k]
                if sigma == 0.0:
                    dst_tau[tau_base + k] = 0.0
                    tau_k.store(0.0)
                    scale.store(0.0)
                else:
                    nr = cute.sqrt(alpha * alpha + sigma)
                    beta = -nr if alpha >= 0.0 else nr
                    local_tau = (beta - alpha) / beta
                    local_scale = 1.0 / (alpha - beta)
                    dst_h[base + k * cols + k] = beta
                    dst_tau[tau_base + k] = local_tau
                    tau_k.store(local_tau)
                    scale.store(local_scale)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            if local_tau != 0.0:
                if tidx == 0:
                    vcol[k] = 1.0
                i = k + 1 + tidx
                local_scale = scale.load()
                while i < rows:
                    v = dst_h[base + i * cols + k] * local_scale
                    dst_h[base + i * cols + k] = v
                    vcol[i] = v
                    i += 512
            cute.arch.sync_threads()

            if local_tau != 0.0:
                j = k + 1 + tidx
                while j < cols:
                    dot = dst_h[base + k * cols + j]
                    i = k + 1
                    while i < rows:
                        dot = dot + vcol[i] * dst_h[base + i * cols + j]
                        i += 1
                    w = local_tau * dot
                    dst_h[base + k * cols + j] = dst_h[base + k * cols + j] - w
                    i = k + 1
                    while i < rows:
                        dst_h[base + i * cols + j] = dst_h[base + i * cols + j] - vcol[i] * w
                        i += 1
                    j += 512
            cute.arch.sync_threads()
            k += 1

    @cute.jit
    def panel_qr(
        src: cute.Tensor,
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        batch: cutlass.Int32,
        rows: cutlass.Int32,
        cols: cutlass.Int32,
    ):
        panel_qr_kernel(src, dst_h, dst_tau, rows, cols).launch(grid=(batch, 1, 1), block=(512, 1, 1))

    _PANEL_QR = panel_qr
    return _PANEL_QR


def _try_panel_cutedsl(data: torch.Tensor) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, rows, cols = data.shape
    if batch <= 0 or rows <= 0 or rows > 2048 or cols <= 0 or cols > 64 or cols > rows:
        return None
    x = data if data.is_contiguous() else data.contiguous()
    h = torch.empty_like(x)
    tau = torch.empty((batch, cols), device=x.device, dtype=torch.float32)
    key = (batch, rows, cols)
    compiled = _PANEL_QR_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_x = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_h = torch.empty((x.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * cols,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_panel_qr(),
                from_dlpack(fake_x),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                batch,
                rows,
                cols,
            )
        _PANEL_QR_COMPILED[key] = compiled
    compiled(from_dlpack(x.reshape(-1)), from_dlpack(h.reshape(-1)), from_dlpack(tau.reshape(-1)), batch, rows, cols)
    return h, tau


def _load_wy512_factor(ld_panel: int, threads: int = 256) -> Any:
    key = (ld_panel, threads)
    cached = _WY512_FACTOR.get(key)
    if cached is not None:
        return cached

    warps = threads // 32

    @cute.kernel
    def wy512_factor_kernel(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        panel0: cutlass.Int32,
        need_t: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        rows = 512 - panel0
        h_base = bidx * 512 * 512
        tau_base = bidx * 512
        t_base = bidx * 32 * 32
        v_base = bidx * 512 * 32

        allocator = cutlass.utils.SmemAllocator()
        pbuf = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((ld_panel * 32,)),
            byte_alignment=16,
            swizzle=None,
        )
        warp_sums = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((warps,)),
            byte_alignment=16,
            swizzle=None,
        )
        z = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
            swizzle=None,
        )
        ts = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32 * 33,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)
        scale = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < rows * 32:
            r = p // 32
            j = p - r * 32
            pbuf[j * ld_panel + r] = dst_h[h_base + (panel0 + r) * 512 + panel0 + j]
            p += threads
        cute.arch.sync_threads()

        for kk in cutlass.range_constexpr(32):
            ss = cutlass.Float32(0.0)
            r = kk + 1 + tidx
            while r < rows:
                x = pbuf[kk * ld_panel + r]
                ss = ss + x * x
                r += threads

            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            acc = ss
            off = 16
            while off > 0:
                acc = acc + cute.arch.shuffle_sync_down(acc, off)
                off = off // 2
            if lane == 0:
                warp_sums[warp] = acc
            cute.arch.sync_threads()

            if warp == 0:
                sigma_acc = warp_sums[lane] if lane < warps else cutlass.Float32(0.0)
                off = 16
                while off > 0:
                    sigma_acc = sigma_acc + cute.arch.shuffle_sync_down(sigma_acc, off)
                    off = off // 2

                if lane == 0:
                    sigma = sigma_acc
                    alpha = pbuf[kk * ld_panel + kk]
                    if sigma == 0.0:
                        dst_tau[tau_base + panel0 + kk] = 0.0
                        tau_k.store(0.0)
                        scale.store(0.0)
                    else:
                        nr = cute.sqrt(alpha * alpha + sigma)
                        beta = -nr if alpha >= 0.0 else nr
                        local_tau = (beta - alpha) / beta
                        local_scale = 1.0 / (alpha - beta)
                        pbuf[kk * ld_panel + kk] = beta
                        dst_tau[tau_base + panel0 + kk] = local_tau
                        tau_k.store(local_tau)
                        scale.store(local_scale)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            if local_tau != 0.0:
                r = kk + 1 + tidx
                local_scale = scale.load()
                while r < rows:
                    pbuf[kk * ld_panel + r] = pbuf[kk * ld_panel + r] * local_scale
                    r += threads
            cute.arch.sync_threads()

            if local_tau != 0.0:
                lane = cute.arch.lane_idx()
                warp = cute.arch.warp_idx()
                jj = kk + 1 + warp * 2
                while jj < 32:
                    has1 = jj + 1 < 32
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    r = kk + 1 + lane
                    while r < rows:
                        v = pbuf[kk * ld_panel + r]
                        acc0 = acc0 + v * pbuf[jj * ld_panel + r]
                        if has1:
                            acc1 = acc1 + v * pbuf[(jj + 1) * ld_panel + r]
                        r += 32

                    off = 16
                    while off > 0:
                        acc0 = acc0 + cute.arch.shuffle_sync_down(acc0, off)
                        acc1 = acc1 + cute.arch.shuffle_sync_down(acc1, off)
                        off = off // 2

                    w0 = cutlass.Float32(0.0)
                    w1 = cutlass.Float32(0.0)
                    if lane == 0:
                        w0 = local_tau * (pbuf[jj * ld_panel + kk] + acc0)
                        pbuf[jj * ld_panel + kk] = pbuf[jj * ld_panel + kk] - w0
                        if has1:
                            w1 = local_tau * (pbuf[(jj + 1) * ld_panel + kk] + acc1)
                            pbuf[(jj + 1) * ld_panel + kk] = pbuf[(jj + 1) * ld_panel + kk] - w1
                    w0 = cute.arch.shuffle_sync(w0, 0)
                    w1 = cute.arch.shuffle_sync(w1, 0)

                    r = kk + 1 + lane
                    while r < rows:
                        v = pbuf[kk * ld_panel + r]
                        pbuf[jj * ld_panel + r] = pbuf[jj * ld_panel + r] - v * w0
                        if has1:
                            pbuf[(jj + 1) * ld_panel + r] = pbuf[(jj + 1) * ld_panel + r] - v * w1
                        r += 32
                    jj += warps * 2
            cute.arch.sync_threads()

        if need_t != 0:
            p = tidx
            while p < 32 * 33:
                ts[p] = 0.0
                p += threads
            cute.arch.sync_threads()

            lane_t = cute.arch.lane_idx()
            warp_t = cute.arch.warp_idx()
            j = 0
            while j < 32:
                tau_j = dst_tau[tau_base + panel0 + j]
                if tidx == 0:
                    ts[j * 33 + j] = tau_j

                i = warp_t
                while i < j:
                    acc = cutlass.Float32(0.0)
                    r = j + lane_t
                    while r < rows:
                        vi = pbuf[i * ld_panel + r]
                        vj = 1.0 if r == j else pbuf[j * ld_panel + r]
                        acc = acc + vi * vj
                        r += 32

                    off = 16
                    while off > 0:
                        acc = acc + cute.arch.shuffle_sync_down(acc, off)
                        off = off // 2

                    if lane_t == 0:
                        z[i] = acc
                    i += warps
                cute.arch.sync_threads()

                if tidx < j:
                    y = cutlass.Float32(0.0)
                    l = 0
                    while l < j:
                        y = y + ts[tidx * 33 + l] * z[l]
                        l += 1
                    ts[tidx * 33 + j] = -tau_j * y
                cute.arch.sync_threads()
                j += 1

            p = tidx
            while p < 32 * 32:
                i = p // 32
                j2 = p - i * 32
                tmat[t_base + p] = ts[i * 33 + j2]
                p += threads

        p = tidx
        while p < rows * 32:
            r = p // 32
            j = p - r * 32
            hval = pbuf[j * ld_panel + r]
            dst_h[h_base + (panel0 + r) * 512 + panel0 + j] = hval
            if need_t != 0:
                v = cutlass.Float32(0.0)
                if r == j:
                    v = 1.0
                elif r > j:
                    v = hval
                vmat[v_base + p] = v
            p += threads

    @cute.jit
    def wy512_factor(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        batch: cutlass.Int32,
        panel0: cutlass.Int32,
        need_t: cutlass.Int32,
    ):
        wy512_factor_kernel(dst_h, dst_tau, tmat, vmat, panel0, need_t).launch(
            grid=(batch, 1, 1),
            block=(threads, 1, 1),
        )

    _WY512_FACTOR[key] = wy512_factor
    return wy512_factor


def _run_wy512_factor(
    h: torch.Tensor,
    tau: torch.Tensor,
    tmat: torch.Tensor,
    vmat: torch.Tensor,
    panel0: int,
    threads: int = 256,
) -> None:
    batch = h.shape[0]
    need_t = 1 if panel0 + 32 < 512 else 0
    rows = 512 - panel0
    if rows > 448:
        ld_panel = 513
    elif rows > 384:
        ld_panel = 449
    elif rows > 320:
        ld_panel = 385
    elif rows > 256:
        ld_panel = 321
    elif rows > 192:
        ld_panel = 257
    elif rows > 128:
        ld_panel = 193
    elif rows > 64:
        ld_panel = 129
    else:
        ld_panel = 65
    key = (batch, need_t, ld_panel, threads)
    compiled = _WY512_FACTOR_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_h = torch.empty((h.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 512,), device="cuda", dtype=torch.float32)
            fake_t = torch.empty((batch * 32 * 32,), device="cuda", dtype=torch.float32)
            fake_v = torch.empty((batch * 512 * 32,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_wy512_factor(ld_panel, threads),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                from_dlpack(fake_t),
                from_dlpack(fake_v),
                batch,
                panel0,
                need_t,
            )
        _WY512_FACTOR_COMPILED[key] = compiled
    compiled(
        from_dlpack(h.reshape(-1)),
        from_dlpack(tau.reshape(-1)),
        from_dlpack(tmat.reshape(-1)),
        from_dlpack(vmat.reshape(-1)),
        batch,
        panel0,
        need_t,
    )


def _apply_wy512_cublas(
    h: torch.Tensor,
    vmat: torch.Tensor,
    tmat: torch.Tensor,
    wmat: torch.Tensor,
    ymat: torch.Tensor,
    panel0: int,
) -> None:
    batch = h.shape[0]
    rows = 512 - panel0
    col0 = panel0 + 32
    trailing = 512 - col0
    if trailing <= 0:
        return
    lib, handle = _load_cublas()
    _cublas_check(lib.cublasSetMathMode(handle, 2), "cublasSetMathMode")
    one = ctypes.c_float(1.0)
    zero = ctypes.c_float(0.0)
    minus_one = ctypes.c_float(-1.0)
    h_stride = 512 * 512
    v_stride = 512 * 32
    t_stride = 32 * 32
    wy_stride = 32 * 512
    c_ptr = h.data_ptr() + (panel0 * 512 + col0) * 4

    _cublas_check(
        lib.cublasSgemmStridedBatched(
            handle,
            0,
            1,
            trailing,
            32,
            rows,
            ctypes.byref(one),
            ctypes.c_void_p(c_ptr),
            512,
            h_stride,
            ctypes.c_void_p(vmat.data_ptr()),
            32,
            v_stride,
            ctypes.byref(zero),
            ctypes.c_void_p(wmat.data_ptr()),
            512,
            wy_stride,
            batch,
        ),
        "W = V^T C",
    )
    _cublas_check(
        lib.cublasSgemmStridedBatched(
            handle,
            0,
            1,
            trailing,
            32,
            32,
            ctypes.byref(one),
            ctypes.c_void_p(wmat.data_ptr()),
            512,
            wy_stride,
            ctypes.c_void_p(tmat.data_ptr()),
            32,
            t_stride,
            ctypes.byref(zero),
            ctypes.c_void_p(ymat.data_ptr()),
            512,
            wy_stride,
            batch,
        ),
        "Y = T^T W",
    )
    _cublas_check(
        lib.cublasSgemmStridedBatched(
            handle,
            0,
            0,
            trailing,
            rows,
            32,
            ctypes.byref(minus_one),
            ctypes.c_void_p(ymat.data_ptr()),
            512,
            wy_stride,
            ctypes.c_void_p(vmat.data_ptr()),
            32,
            v_stride,
            ctypes.byref(one),
            ctypes.c_void_p(c_ptr),
            512,
            h_stride,
            batch,
        ),
        "C -= VY",
    )


def _try_n512_wy_cutedsl_cublas(data: torch.Tensor) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, n, m = data.shape
    if batch <= 0 or n != 512 or m != 512:
        return None
    h = data.clone(memory_format=torch.contiguous_format)
    tau = torch.empty((batch, 512), device=data.device, dtype=torch.float32)
    tmat = torch.empty((batch, 32, 32), device=data.device, dtype=torch.float32)
    vmat = torch.empty((batch, 512, 32), device=data.device, dtype=torch.float32)
    wmat = torch.empty((batch, 32, 512), device=data.device, dtype=torch.float32)
    ymat = torch.empty((batch, 32, 512), device=data.device, dtype=torch.float32)
    for panel0 in range(0, 512, 32):
        _run_wy512_factor(h, tau, tmat, vmat, panel0)
        _apply_wy512_cublas(h, vmat, tmat, wmat, ymat, panel0)
    return h, tau


def _load_wy_factor(n: int, ib: int, ld: int) -> Any:
    key = (n, ib, ld)
    cached = _WY_FACTORS.get(key)
    if cached is not None:
        return cached

    @cute.kernel
    def wy_factor_kernel(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        panel0: cutlass.Int32,
        need_t: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        rows = n - panel0
        h_base = bidx * n * n
        tau_base = bidx * n
        t_base = bidx * ib * ib
        v_base = bidx * n * ib

        allocator = cutlass.utils.SmemAllocator()
        pbuf = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((ld * ib,)),
            byte_alignment=16,
            swizzle=None,
        )
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((512,)),
            byte_alignment=16,
            swizzle=None,
        )
        z = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((ib,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)
        scale = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < rows * ib:
            r = p // ib
            j = p - r * ib
            pbuf[j * ld + r] = dst_h[h_base + (panel0 + r) * n + panel0 + j]
            p += 512
        cute.arch.sync_threads()

        kk = 0
        while kk < ib:
            ss = cutlass.Float32(0.0)
            r = kk + 1 + tidx
            while r < rows:
                x = pbuf[kk * ld + r]
                ss = ss + x * x
                r += 512
            red[tidx] = ss
            cute.arch.sync_threads()

            step = 256
            while step > 0:
                if tidx < step:
                    red[tidx] = red[tidx] + red[tidx + step]
                cute.arch.sync_threads()
                step = step // 2

            if tidx == 0:
                sigma = red[0]
                alpha = pbuf[kk * ld + kk]
                if sigma == 0.0:
                    dst_tau[tau_base + panel0 + kk] = 0.0
                    tau_k.store(0.0)
                    scale.store(0.0)
                else:
                    nr = cute.sqrt(alpha * alpha + sigma)
                    beta = -nr if alpha >= 0.0 else nr
                    local_tau = (beta - alpha) / beta
                    local_scale = 1.0 / (alpha - beta)
                    pbuf[kk * ld + kk] = beta
                    dst_tau[tau_base + panel0 + kk] = local_tau
                    tau_k.store(local_tau)
                    scale.store(local_scale)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            if local_tau != 0.0:
                r = kk + 1 + tidx
                local_scale = scale.load()
                while r < rows:
                    pbuf[kk * ld + r] = pbuf[kk * ld + r] * local_scale
                    r += 512
            cute.arch.sync_threads()

            if local_tau != 0.0:
                lane = cute.arch.lane_idx()
                warp = cute.arch.warp_idx()
                jj = kk + 1 + warp * 2
                while jj < ib:
                    has1 = jj + 1 < ib
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    r = kk + 1 + lane
                    while r < rows:
                        v = pbuf[kk * ld + r]
                        acc0 = acc0 + v * pbuf[jj * ld + r]
                        if has1:
                            acc1 = acc1 + v * pbuf[(jj + 1) * ld + r]
                        r += 32

                    off = 16
                    while off > 0:
                        acc0 = acc0 + cute.arch.shuffle_sync_down(acc0, off)
                        acc1 = acc1 + cute.arch.shuffle_sync_down(acc1, off)
                        off = off // 2

                    w0 = cutlass.Float32(0.0)
                    w1 = cutlass.Float32(0.0)
                    if lane == 0:
                        w0 = local_tau * (pbuf[jj * ld + kk] + acc0)
                        pbuf[jj * ld + kk] = pbuf[jj * ld + kk] - w0
                        if has1:
                            w1 = local_tau * (pbuf[(jj + 1) * ld + kk] + acc1)
                            pbuf[(jj + 1) * ld + kk] = pbuf[(jj + 1) * ld + kk] - w1
                    w0 = cute.arch.shuffle_sync(w0, 0)
                    w1 = cute.arch.shuffle_sync(w1, 0)

                    r = kk + 1 + lane
                    while r < rows:
                        v = pbuf[kk * ld + r]
                        pbuf[jj * ld + r] = pbuf[jj * ld + r] - v * w0
                        if has1:
                            pbuf[(jj + 1) * ld + r] = pbuf[(jj + 1) * ld + r] - v * w1
                        r += 32
                    jj += 32
            cute.arch.sync_threads()
            kk += 1

        if need_t != 0:
            p = tidx
            while p < ib * ib:
                tmat[t_base + p] = 0.0
                p += 512
            cute.arch.sync_threads()

            lane_t = cute.arch.lane_idx()
            warp_t = cute.arch.warp_idx()
            j = 0
            while j < ib:
                tau_j = dst_tau[tau_base + panel0 + j]
                if tidx == 0:
                    tmat[t_base + j * ib + j] = tau_j

                i = warp_t
                while i < j:
                    acc = cutlass.Float32(0.0)
                    r = j + lane_t
                    while r < rows:
                        vi = pbuf[i * ld + r]
                        vj = 1.0 if r == j else pbuf[j * ld + r]
                        acc = acc + vi * vj
                        r += 32

                    off = 16
                    while off > 0:
                        acc = acc + cute.arch.shuffle_sync_down(acc, off)
                        off = off // 2

                    if lane_t == 0:
                        z[i] = acc
                    i += 16
                cute.arch.sync_threads()

                if tidx < j:
                    y = cutlass.Float32(0.0)
                    l = 0
                    while l < j:
                        y = y + tmat[t_base + tidx * ib + l] * z[l]
                        l += 1
                    tmat[t_base + tidx * ib + j] = -tau_j * y
                cute.arch.sync_threads()
                j += 1

        p = tidx
        while p < rows * ib:
            r = p // ib
            j = p - r * ib
            hval = pbuf[j * ld + r]
            dst_h[h_base + (panel0 + r) * n + panel0 + j] = hval
            if need_t != 0:
                v = cutlass.Float32(0.0)
                if r == j:
                    v = 1.0
                elif r > j:
                    v = hval
                vmat[v_base + p] = v
            p += 512

    @cute.jit
    def wy_factor(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        batch: cutlass.Int32,
        panel0: cutlass.Int32,
        need_t: cutlass.Int32,
    ):
        wy_factor_kernel(dst_h, dst_tau, tmat, vmat, panel0, need_t).launch(
            grid=(batch, 1, 1),
            block=(512, 1, 1),
        )

    _WY_FACTORS[key] = wy_factor
    return wy_factor


def _run_wy_factor(
    h: torch.Tensor,
    tau: torch.Tensor,
    tmat: torch.Tensor,
    vmat: torch.Tensor,
    panel0: int,
    n: int,
    ib: int,
    ld: int,
) -> None:
    batch = h.shape[0]
    need_t = 1 if panel0 + ib < n else 0
    key = (n, ib, ld, batch, need_t)
    compiled = _WY_FACTOR_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_h = torch.empty((h.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * n,), device="cuda", dtype=torch.float32)
            fake_t = torch.empty((batch * ib * ib,), device="cuda", dtype=torch.float32)
            fake_v = torch.empty((batch * n * ib,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_wy_factor(n, ib, ld),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                from_dlpack(fake_t),
                from_dlpack(fake_v),
                batch,
                panel0,
                need_t,
            )
        _WY_FACTOR_COMPILED[key] = compiled
    compiled(
        from_dlpack(h.reshape(-1)),
        from_dlpack(tau.reshape(-1)),
        from_dlpack(tmat.reshape(-1)),
        from_dlpack(vmat.reshape(-1)),
        batch,
        panel0,
        need_t,
    )


def _load_wy24_norms_factor() -> Any:
    global _WY24_NORMS_FACTOR
    if _WY24_NORMS_FACTOR is not None:
        return _WY24_NORMS_FACTOR

    n = 2048
    ib = 24
    ld = 2055

    @cute.kernel
    def wy24_norms_factor_kernel(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        panel0: cutlass.Int32,
        need_t: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        rows = n - panel0
        h_base = bidx * n * n
        tau_base = bidx * n
        t_base = bidx * ib * ib
        v_base = bidx * n * ib

        allocator = cutlass.utils.SmemAllocator()
        pbuf = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((ld * ib,)),
            byte_alignment=16,
            swizzle=None,
        )
        norms = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((ib,)),
            byte_alignment=16,
            swizzle=None,
        )
        z = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((ib,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)
        scale = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < rows * ib:
            r = p // ib
            j = p - r * ib
            pbuf[j * ld + r] = dst_h[h_base + (panel0 + r) * n + panel0 + j]
            p += 512
        cute.arch.sync_threads()

        lane0 = cute.arch.lane_idx()
        warp0 = cute.arch.warp_idx()
        j0 = warp0
        while j0 < ib:
            acc0 = cutlass.Float32(0.0)
            r0 = lane0
            while r0 < rows:
                x0 = pbuf[j0 * ld + r0]
                acc0 = acc0 + x0 * x0
                r0 += 32

            off0 = 16
            while off0 > 0:
                acc0 = acc0 + cute.arch.shuffle_sync_down(acc0, off0)
                off0 = off0 // 2

            if lane0 == 0:
                norms[j0] = acc0
            j0 += 16
        cute.arch.sync_threads()

        kk = 0
        while kk < ib:
            if tidx == 0:
                alpha = pbuf[kk * ld + kk]
                sigma = norms[kk] - alpha * alpha
                if sigma < 0.0:
                    sigma = 0.0
                if sigma == 0.0:
                    dst_tau[tau_base + panel0 + kk] = 0.0
                    tau_k.store(0.0)
                    scale.store(0.0)
                else:
                    nr = cute.sqrt(alpha * alpha + sigma)
                    beta = -nr if alpha >= 0.0 else nr
                    local_tau = (beta - alpha) / beta
                    local_scale = 1.0 / (alpha - beta)
                    pbuf[kk * ld + kk] = beta
                    dst_tau[tau_base + panel0 + kk] = local_tau
                    tau_k.store(local_tau)
                    scale.store(local_scale)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            if local_tau != 0.0:
                r = kk + 1 + tidx
                local_scale = scale.load()
                while r < rows:
                    pbuf[kk * ld + r] = pbuf[kk * ld + r] * local_scale
                    r += 512
            cute.arch.sync_threads()

            if local_tau != 0.0:
                lane = cute.arch.lane_idx()
                warp = cute.arch.warp_idx()
                remaining = ib - kk - 1
                if remaining <= 8:
                    warps_per_col = 2
                    if remaining <= 4:
                        warps_per_col = 4
                    if remaining <= 2:
                        warps_per_col = 8
                    if remaining <= 1:
                        warps_per_col = 16

                    col_group = warp // warps_per_col
                    part = warp - col_group * warps_per_col
                    jj = kk + 1 + col_group
                    if jj < ib:
                        acc = cutlass.Float32(0.0)
                        r = kk + 1 + part * 32 + lane
                        while r < rows:
                            v = pbuf[kk * ld + r]
                            acc = acc + v * pbuf[jj * ld + r]
                            r += 32 * warps_per_col

                        off = 16
                        while off > 0:
                            acc = acc + cute.arch.shuffle_sync_down(acc, off)
                            off = off // 2

                        if lane == 0:
                            z[warp] = acc
                    cute.arch.sync_threads()

                    if jj < ib and lane == 0 and part == 0:
                        total = cutlass.Float32(0.0)
                        ppart = 0
                        while ppart < warps_per_col:
                            total = total + z[col_group * warps_per_col + ppart]
                            ppart += 1
                        w = local_tau * (pbuf[jj * ld + kk] + total)
                        pbuf[jj * ld + kk] = pbuf[jj * ld + kk] - w
                        z[16 + col_group] = w
                    cute.arch.sync_threads()

                    if jj < ib:
                        w = z[16 + col_group]
                        r = kk + 1 + part * 32 + lane
                        while r < rows:
                            v = pbuf[kk * ld + r]
                            pbuf[jj * ld + r] = pbuf[jj * ld + r] - v * w
                            r += 32 * warps_per_col
                else:
                    jj = kk + 1 + warp * 2
                    while jj < ib:
                        has1 = jj + 1 < ib
                        acc0 = cutlass.Float32(0.0)
                        acc1 = cutlass.Float32(0.0)
                        r = kk + 1 + lane
                        while r < rows:
                            v = pbuf[kk * ld + r]
                            acc0 = acc0 + v * pbuf[jj * ld + r]
                            if has1:
                                acc1 = acc1 + v * pbuf[(jj + 1) * ld + r]
                            r += 32

                        off = 16
                        while off > 0:
                            acc0 = acc0 + cute.arch.shuffle_sync_down(acc0, off)
                            acc1 = acc1 + cute.arch.shuffle_sync_down(acc1, off)
                            off = off // 2

                        w0 = cutlass.Float32(0.0)
                        w1 = cutlass.Float32(0.0)
                        if lane == 0:
                            w0 = local_tau * (pbuf[jj * ld + kk] + acc0)
                            pbuf[jj * ld + kk] = pbuf[jj * ld + kk] - w0
                            if has1:
                                w1 = local_tau * (pbuf[(jj + 1) * ld + kk] + acc1)
                                pbuf[(jj + 1) * ld + kk] = pbuf[(jj + 1) * ld + kk] - w1
                        w0 = cute.arch.shuffle_sync(w0, 0)
                        w1 = cute.arch.shuffle_sync(w1, 0)

                        r = kk + 1 + lane
                        while r < rows:
                            v = pbuf[kk * ld + r]
                            pbuf[jj * ld + r] = pbuf[jj * ld + r] - v * w0
                            if has1:
                                pbuf[(jj + 1) * ld + r] = pbuf[(jj + 1) * ld + r] - v * w1
                            r += 32
                        jj += 32
            cute.arch.sync_threads()

            jn = kk + 1 + tidx
            while jn < ib:
                top = pbuf[jn * ld + kk]
                norms[jn] = norms[jn] - top * top
                jn += 512
            cute.arch.sync_threads()
            kk += 1

        if need_t != 0:
            p = tidx
            while p < ib * ib:
                tmat[t_base + p] = 0.0
                p += 512
            cute.arch.sync_threads()

            lane_t = cute.arch.lane_idx()
            warp_t = cute.arch.warp_idx()
            j = 0
            while j < ib:
                tau_j = dst_tau[tau_base + panel0 + j]
                if tidx == 0:
                    tmat[t_base + j * ib + j] = tau_j

                i = warp_t
                while i < j:
                    acc_t = cutlass.Float32(0.0)
                    r_t = j + lane_t
                    while r_t < rows:
                        vi = pbuf[i * ld + r_t]
                        vj = 1.0 if r_t == j else pbuf[j * ld + r_t]
                        acc_t = acc_t + vi * vj
                        r_t += 32

                    off_t = 16
                    while off_t > 0:
                        acc_t = acc_t + cute.arch.shuffle_sync_down(acc_t, off_t)
                        off_t = off_t // 2

                    if lane_t == 0:
                        z[i] = acc_t
                    i += 16
                cute.arch.sync_threads()

                if tidx < j:
                    y = cutlass.Float32(0.0)
                    l = 0
                    while l < j:
                        y = y + tmat[t_base + tidx * ib + l] * z[l]
                        l += 1
                    tmat[t_base + tidx * ib + j] = -tau_j * y
                cute.arch.sync_threads()
                j += 1

        p = tidx
        while p < rows * ib:
            r = p // ib
            jv = p - r * ib
            hval = pbuf[jv * ld + r]
            dst_h[h_base + (panel0 + r) * n + panel0 + jv] = hval
            if need_t != 0:
                v = cutlass.Float32(0.0)
                if r == jv:
                    v = 1.0
                elif r > jv:
                    v = hval
                vmat[v_base + p] = v
            p += 512

    @cute.jit
    def wy24_norms_factor(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        batch: cutlass.Int32,
        panel0: cutlass.Int32,
        need_t: cutlass.Int32,
    ):
        wy24_norms_factor_kernel(dst_h, dst_tau, tmat, vmat, panel0, need_t).launch(
            grid=(batch, 1, 1),
            block=(512, 1, 1),
        )

    _WY24_NORMS_FACTOR = wy24_norms_factor
    return wy24_norms_factor


def _run_wy24_norms_factor(
    h: torch.Tensor,
    tau: torch.Tensor,
    tmat: torch.Tensor,
    vmat: torch.Tensor,
    panel0: int,
) -> None:
    global _WY24_NORMS_FACTOR_COMPILED
    batch = h.shape[0]
    need_t = 1
    if _WY24_NORMS_FACTOR_COMPILED is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_h = torch.empty((h.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 2048,), device="cuda", dtype=torch.float32)
            fake_t = torch.empty((batch * 24 * 24,), device="cuda", dtype=torch.float32)
            fake_v = torch.empty((batch * 2048 * 24,), device="cuda", dtype=torch.float32)
            _WY24_NORMS_FACTOR_COMPILED = cute.compile(
                _load_wy24_norms_factor(),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                from_dlpack(fake_t),
                from_dlpack(fake_v),
                batch,
                panel0,
                need_t,
            )
    _WY24_NORMS_FACTOR_COMPILED(
        from_dlpack(h.reshape(-1)),
        from_dlpack(tau.reshape(-1)),
        from_dlpack(tmat.reshape(-1)),
        from_dlpack(vmat.reshape(-1)),
        batch,
        panel0,
        need_t,
    )


def _load_wy24_micro3_factor() -> Any:
    global _WY24_MICRO3_FACTOR
    if _WY24_MICRO3_FACTOR is not None:
        return _WY24_MICRO3_FACTOR

    n = 2048
    ib = 24
    ld = 2055

    @cute.kernel
    def wy24_micro3_factor_kernel(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        panel0: cutlass.Int32,
        need_t: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        rows = n - panel0
        h_base = bidx * n * n
        tau_base = bidx * n
        t_base = bidx * ib * ib
        v_base = bidx * n * ib

        allocator = cutlass.utils.SmemAllocator()
        pbuf = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((ld * ib,)),
            byte_alignment=16,
            swizzle=None,
        )
        norms = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((ib,)),
            byte_alignment=16,
            swizzle=None,
        )
        z = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((64,)),
            byte_alignment=16,
            swizzle=None,
        )

        p = tidx
        while p < rows * ib:
            r = p // ib
            j = p - r * ib
            pbuf[j * ld + r] = dst_h[h_base + (panel0 + r) * n + panel0 + j]
            p += 512
        cute.arch.sync_threads()

        lane0 = cute.arch.lane_idx()
        warp0 = cute.arch.warp_idx()
        j0 = warp0
        while j0 < ib:
            acc0 = cutlass.Float32(0.0)
            r0 = lane0
            while r0 < rows:
                x0 = pbuf[j0 * ld + r0]
                acc0 = acc0 + x0 * x0
                r0 += 32

            off0 = 16
            while off0 > 0:
                acc0 = acc0 + cute.arch.shuffle_sync_down(acc0, off0)
                off0 = off0 // 2

            if lane0 == 0:
                norms[j0] = acc0
            j0 += 16
        cute.arch.sync_threads()

        kk = 0
        while kk < ib:
            ii = 0
            while ii < 3:
                cur = kk + ii
                pp = 0
                while pp < ii:
                    prev = kk + pp
                    tau_p = z[40 + pp]
                    if tau_p != 0.0:
                        lane = cute.arch.lane_idx()
                        warp = cute.arch.warp_idx()
                        acc = cutlass.Float32(0.0)
                        r = prev + 1 + warp * 32 + lane
                        while r < rows:
                            acc = acc + pbuf[prev * ld + r] * pbuf[cur * ld + r]
                            r += 512

                        off = 16
                        while off > 0:
                            acc = acc + cute.arch.shuffle_sync_down(acc, off)
                            off = off // 2

                        if lane == 0:
                            z[warp] = acc
                    cute.arch.sync_threads()

                    if tau_p != 0.0 and tidx == 0:
                        total = pbuf[cur * ld + prev]
                        part = 0
                        while part < 16:
                            total = total + z[part]
                            part += 1
                        w_prev = tau_p * total
                        pbuf[cur * ld + prev] = pbuf[cur * ld + prev] - w_prev
                        top_cur = pbuf[cur * ld + prev]
                        norms[cur] = norms[cur] - top_cur * top_cur
                        z[16] = w_prev
                    cute.arch.sync_threads()

                    if tau_p != 0.0:
                        w_prev_load = z[16]
                        r_upd = prev + 1 + tidx
                        while r_upd < rows:
                            pbuf[cur * ld + r_upd] = pbuf[cur * ld + r_upd] - pbuf[prev * ld + r_upd] * w_prev_load
                            r_upd += 512
                    cute.arch.sync_threads()
                    pp += 1

                if tidx == 0:
                    alpha = pbuf[cur * ld + cur]
                    sigma = norms[cur] - alpha * alpha
                    if sigma < 0.0:
                        sigma = 0.0
                    if sigma == 0.0:
                        dst_tau[tau_base + panel0 + cur] = 0.0
                        z[40 + ii] = 0.0
                        z[44 + ii] = 0.0
                    else:
                        nr = cute.sqrt(alpha * alpha + sigma)
                        beta = -nr if alpha >= 0.0 else nr
                        tau = (beta - alpha) / beta
                        scale = 1.0 / (alpha - beta)
                        pbuf[cur * ld + cur] = beta
                        dst_tau[tau_base + panel0 + cur] = tau
                        z[40 + ii] = tau
                        z[44 + ii] = scale
                cute.arch.sync_threads()

                tau_cur = z[40 + ii]
                if tau_cur != 0.0:
                    scale_cur = z[44 + ii]
                    r_scale = cur + 1 + tidx
                    while r_scale < rows:
                        pbuf[cur * ld + r_scale] = pbuf[cur * ld + r_scale] * scale_cur
                        r_scale += 512
                cute.arch.sync_threads()
                ii += 1

            lane_c = cute.arch.lane_idx()
            warp_c = cute.arch.warp_idx()
            pair = warp_c
            if pair < 3:
                a = 1
                b = 0
                if pair == 1:
                    a = 2
                    b = 0
                if pair == 2:
                    a = 2
                    b = 1

                ca = kk + a
                cb = kk + b
                acc_c = cutlass.Float32(0.0)
                r_c = ca + 1 + lane_c
                while r_c < rows:
                    acc_c = acc_c + pbuf[ca * ld + r_c] * pbuf[cb * ld + r_c]
                    r_c += 32

                off_c = 16
                while off_c > 0:
                    acc_c = acc_c + cute.arch.shuffle_sync_down(acc_c, off_c)
                    off_c = off_c // 2

                if lane_c == 0:
                    z[20 + pair] = pbuf[cb * ld + ca] + acc_c
            cute.arch.sync_threads()

            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            jj = kk + 3 + warp
            while jj < ib:
                acc0 = cutlass.Float32(0.0)
                acc1 = cutlass.Float32(0.0)
                acc2 = cutlass.Float32(0.0)
                if lane == 0:
                    x1 = pbuf[jj * ld + kk + 1]
                    x2 = pbuf[jj * ld + kk + 2]
                    acc0 = pbuf[jj * ld + kk] + pbuf[kk * ld + kk + 1] * x1 + pbuf[kk * ld + kk + 2] * x2
                    acc1 = x1 + pbuf[(kk + 1) * ld + kk + 2] * x2
                    acc2 = x2
                r_dot = kk + 3 + lane
                while r_dot < rows:
                    x = pbuf[jj * ld + r_dot]
                    acc0 = acc0 + pbuf[kk * ld + r_dot] * x
                    acc1 = acc1 + pbuf[(kk + 1) * ld + r_dot] * x
                    acc2 = acc2 + pbuf[(kk + 2) * ld + r_dot] * x
                    r_dot += 32

                off = 16
                while off > 0:
                    acc0 = acc0 + cute.arch.shuffle_sync_down(acc0, off)
                    acc1 = acc1 + cute.arch.shuffle_sync_down(acc1, off)
                    acc2 = acc2 + cute.arch.shuffle_sync_down(acc2, off)
                    off = off // 2

                w0 = cutlass.Float32(0.0)
                w1 = cutlass.Float32(0.0)
                w2 = cutlass.Float32(0.0)
                if lane == 0:
                    w0 = z[40] * acc0
                    w1 = z[41] * (acc1 - z[20] * w0)
                    w2 = z[42] * (acc2 - z[21] * w0 - z[22] * w1)
                    new0 = pbuf[jj * ld + kk] - w0
                    new1 = pbuf[jj * ld + kk + 1] - pbuf[kk * ld + kk + 1] * w0 - w1
                    new2 = pbuf[jj * ld + kk + 2] - pbuf[kk * ld + kk + 2] * w0 - pbuf[(kk + 1) * ld + kk + 2] * w1 - w2
                    pbuf[jj * ld + kk] = new0
                    pbuf[jj * ld + kk + 1] = new1
                    pbuf[jj * ld + kk + 2] = new2
                    norms[jj] = norms[jj] - new0 * new0 - new1 * new1 - new2 * new2
                w0 = cute.arch.shuffle_sync(w0, 0)
                w1 = cute.arch.shuffle_sync(w1, 0)
                w2 = cute.arch.shuffle_sync(w2, 0)

                r = kk + 3 + lane
                while r < rows:
                    pbuf[jj * ld + r] = pbuf[jj * ld + r] - pbuf[kk * ld + r] * w0 - pbuf[(kk + 1) * ld + r] * w1 - pbuf[(kk + 2) * ld + r] * w2
                    r += 32
                jj += 16
            cute.arch.sync_threads()
            kk += 3

        if need_t != 0:
            p = tidx
            while p < ib * ib:
                tmat[t_base + p] = 0.0
                p += 512
            cute.arch.sync_threads()

            lane_t = cute.arch.lane_idx()
            warp_t = cute.arch.warp_idx()
            j = 0
            while j < ib:
                tau_j = dst_tau[tau_base + panel0 + j]
                if tidx == 0:
                    tmat[t_base + j * ib + j] = tau_j

                i = warp_t
                while i < j:
                    acc_t = cutlass.Float32(0.0)
                    r_t = j + lane_t
                    while r_t < rows:
                        vi = pbuf[i * ld + r_t]
                        vj = 1.0 if r_t == j else pbuf[j * ld + r_t]
                        acc_t = acc_t + vi * vj
                        r_t += 32

                    off_t = 16
                    while off_t > 0:
                        acc_t = acc_t + cute.arch.shuffle_sync_down(acc_t, off_t)
                        off_t = off_t // 2

                    if lane_t == 0:
                        z[i] = acc_t
                    i += 16
                cute.arch.sync_threads()

                if tidx < j:
                    y = cutlass.Float32(0.0)
                    l = 0
                    while l < j:
                        y = y + tmat[t_base + tidx * ib + l] * z[l]
                        l += 1
                    tmat[t_base + tidx * ib + j] = -tau_j * y
                cute.arch.sync_threads()
                j += 1

        p = tidx
        while p < rows * ib:
            r = p // ib
            jv = p - r * ib
            hval = pbuf[jv * ld + r]
            dst_h[h_base + (panel0 + r) * n + panel0 + jv] = hval
            if need_t != 0:
                v = cutlass.Float32(0.0)
                if r == jv:
                    v = 1.0
                elif r > jv:
                    v = hval
                vmat[v_base + p] = v
            p += 512

    @cute.jit
    def wy24_micro3_factor(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        batch: cutlass.Int32,
        panel0: cutlass.Int32,
        need_t: cutlass.Int32,
    ):
        wy24_micro3_factor_kernel(dst_h, dst_tau, tmat, vmat, panel0, need_t).launch(
            grid=(batch, 1, 1),
            block=(512, 1, 1),
        )

    _WY24_MICRO3_FACTOR = wy24_micro3_factor
    return wy24_micro3_factor


def _run_wy24_micro3_factor(
    h: torch.Tensor,
    tau: torch.Tensor,
    tmat: torch.Tensor,
    vmat: torch.Tensor,
    panel0: int,
) -> None:
    global _WY24_MICRO3_FACTOR_COMPILED
    batch = h.shape[0]
    need_t = 1
    if _WY24_MICRO3_FACTOR_COMPILED is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_h = torch.empty((h.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 2048,), device="cuda", dtype=torch.float32)
            fake_t = torch.empty((batch * 24 * 24,), device="cuda", dtype=torch.float32)
            fake_v = torch.empty((batch * 2048 * 24,), device="cuda", dtype=torch.float32)
            _WY24_MICRO3_FACTOR_COMPILED = cute.compile(
                _load_wy24_micro3_factor(),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                from_dlpack(fake_t),
                from_dlpack(fake_v),
                batch,
                panel0,
                need_t,
            )
    _WY24_MICRO3_FACTOR_COMPILED(
        from_dlpack(h.reshape(-1)),
        from_dlpack(tau.reshape(-1)),
        from_dlpack(tmat.reshape(-1)),
        from_dlpack(vmat.reshape(-1)),
        batch,
        panel0,
        need_t,
    )


def _apply_wy_cublas(
    h: torch.Tensor,
    vmat: torch.Tensor,
    tmat: torch.Tensor,
    wmat: torch.Tensor,
    ymat: torch.Tensor,
    panel0: int,
    n: int,
    ib: int,
    math_mode: int,
) -> None:
    batch = h.shape[0]
    rows = n - panel0
    col0 = panel0 + ib
    trailing = n - col0
    if trailing <= 0:
        return
    lib, handle = _load_cublas()
    _cublas_check(lib.cublasSetMathMode(handle, math_mode), "cublasSetMathMode")
    one = ctypes.c_float(1.0)
    zero = ctypes.c_float(0.0)
    minus_one = ctypes.c_float(-1.0)
    h_stride = n * n
    v_stride = n * ib
    t_stride = ib * ib
    wy_stride = ib * n
    c_ptr = h.data_ptr() + (panel0 * n + col0) * 4

    _cublas_check(
        lib.cublasSgemmStridedBatched(
            handle,
            0,
            1,
            trailing,
            ib,
            rows,
            ctypes.byref(one),
            ctypes.c_void_p(c_ptr),
            n,
            h_stride,
            ctypes.c_void_p(vmat.data_ptr()),
            ib,
            v_stride,
            ctypes.byref(zero),
            ctypes.c_void_p(wmat.data_ptr()),
            n,
            wy_stride,
            batch,
        ),
        "W = V^T C",
    )
    _cublas_check(
        lib.cublasSgemmStridedBatched(
            handle,
            0,
            1,
            trailing,
            ib,
            ib,
            ctypes.byref(one),
            ctypes.c_void_p(wmat.data_ptr()),
            n,
            wy_stride,
            ctypes.c_void_p(tmat.data_ptr()),
            ib,
            t_stride,
            ctypes.byref(zero),
            ctypes.c_void_p(ymat.data_ptr()),
            n,
            wy_stride,
            batch,
        ),
        "Y = T^T W",
    )
    _cublas_check(
        lib.cublasSgemmStridedBatched(
            handle,
            0,
            0,
            trailing,
            rows,
            ib,
            ctypes.byref(minus_one),
            ctypes.c_void_p(ymat.data_ptr()),
            n,
            wy_stride,
            ctypes.c_void_p(vmat.data_ptr()),
            ib,
            v_stride,
            ctypes.byref(one),
            ctypes.c_void_p(c_ptr),
            n,
            h_stride,
            batch,
        ),
        "C -= VY",
    )


def _apply_wy_cublas_range(
    h: torch.Tensor,
    vmat: torch.Tensor,
    tmat: torch.Tensor,
    wmat: torch.Tensor,
    ymat: torch.Tensor,
    panel0: int,
    n: int,
    ib: int,
    col0: int,
    trailing: int,
    math_mode: int,
) -> None:
    if trailing <= 0:
        return
    batch = h.shape[0]
    rows = n - panel0
    lib, handle = _load_cublas()
    _cublas_check(lib.cublasSetMathMode(handle, math_mode), "cublasSetMathMode")
    one = ctypes.c_float(1.0)
    zero = ctypes.c_float(0.0)
    minus_one = ctypes.c_float(-1.0)
    h_stride = n * n
    v_stride = n * ib
    t_stride = ib * ib
    wy_stride = ib * n
    c_ptr = h.data_ptr() + (panel0 * n + col0) * 4

    _cublas_check(
        lib.cublasSgemmStridedBatched(
            handle,
            0,
            1,
            trailing,
            ib,
            rows,
            ctypes.byref(one),
            ctypes.c_void_p(c_ptr),
            n,
            h_stride,
            ctypes.c_void_p(vmat.data_ptr()),
            ib,
            v_stride,
            ctypes.byref(zero),
            ctypes.c_void_p(wmat.data_ptr()),
            n,
            wy_stride,
            batch,
        ),
        "W = V^T C",
    )
    _cublas_check(
        lib.cublasSgemmStridedBatched(
            handle,
            0,
            1,
            trailing,
            ib,
            ib,
            ctypes.byref(one),
            ctypes.c_void_p(wmat.data_ptr()),
            n,
            wy_stride,
            ctypes.c_void_p(tmat.data_ptr()),
            ib,
            t_stride,
            ctypes.byref(zero),
            ctypes.c_void_p(ymat.data_ptr()),
            n,
            wy_stride,
            batch,
        ),
        "Y = T^T W",
    )
    _cublas_check(
        lib.cublasSgemmStridedBatched(
            handle,
            0,
            0,
            trailing,
            rows,
            ib,
            ctypes.byref(minus_one),
            ctypes.c_void_p(ymat.data_ptr()),
            n,
            wy_stride,
            ctypes.c_void_p(vmat.data_ptr()),
            ib,
            v_stride,
            ctypes.byref(one),
            ctypes.c_void_p(c_ptr),
            n,
            h_stride,
            batch,
        ),
        "C -= VY",
    )


def _load_wy16_outer64_merge() -> Any:
    global _WY16_OUTER64_MERGE
    if _WY16_OUTER64_MERGE is not None:
        return _WY16_OUTER64_MERGE

    n = 2048
    inner = 16
    outer = 64

    @cute.kernel
    def wy16_outer64_merge_kernel(
        t16: cute.Tensor,
        v16: cute.Tensor,
        t64: cute.Tensor,
        v64: cute.Tensor,
        outer0: cutlass.Int32,
        rel: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        rows_outer = n - outer0
        active_rows = rows_outer - rel
        t16_base = bidx * inner * inner
        v16_base = bidx * n * inner
        t64_base = bidx * outer * outer
        v64_base = bidx * n * outer

        allocator = cutlass.utils.SmemAllocator()
        gram = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((outer * inner,)),
            byte_alignment=16,
            swizzle=None,
        )
        tmp = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((outer * inner,)),
            byte_alignment=16,
            swizzle=None,
        )

        if rel == 0:
            p = tidx
            while p < rows_outer * outer:
                v64[v64_base + p] = 0.0
                p += 512
            p = tidx
            while p < outer * outer:
                t64[t64_base + p] = 0.0
                p += 512
        cute.arch.sync_threads()

        p = tidx
        while p < active_rows * inner:
            r = p // inner
            j = p - r * inner
            v64[v64_base + (rel + r) * outer + rel + j] = v16[v16_base + r * inner + j]
            p += 512

        p = tidx
        while p < inner * inner:
            r = p // inner
            j = p - r * inner
            t64[t64_base + (rel + r) * outer + rel + j] = t16[t16_base + p]
            p += 512
        cute.arch.sync_threads()

        if rel != 0:
            p = tidx
            while p < rel * inner:
                a = p // inner
                b = p - a * inner
                acc = cutlass.Float32(0.0)
                r = 0
                while r < rows_outer:
                    acc = acc + (
                        v64[v64_base + r * outer + a] * v64[v64_base + r * outer + rel + b]
                    )
                    r += 1
                gram[p] = acc
                p += 512
            cute.arch.sync_threads()

            p = tidx
            while p < rel * inner:
                a = p // inner
                j = p - a * inner
                acc = cutlass.Float32(0.0)
                b = 0
                while b < inner:
                    acc = acc + gram[a * inner + b] * t16[t16_base + b * inner + j]
                    b += 1
                tmp[p] = acc
                p += 512
            cute.arch.sync_threads()

            p = tidx
            while p < rel * inner:
                i = p // inner
                j = p - i * inner
                acc = cutlass.Float32(0.0)
                a = 0
                while a < rel:
                    acc = acc + t64[t64_base + i * outer + a] * tmp[a * inner + j]
                    a += 1
                t64[t64_base + i * outer + rel + j] = -acc
                p += 512

    @cute.jit
    def wy16_outer64_merge(
        t16: cute.Tensor,
        v16: cute.Tensor,
        t64: cute.Tensor,
        v64: cute.Tensor,
        batch: cutlass.Int32,
        outer0: cutlass.Int32,
        rel: cutlass.Int32,
    ):
        wy16_outer64_merge_kernel(t16, v16, t64, v64, outer0, rel).launch(
            grid=(batch, 1, 1),
            block=(512, 1, 1),
        )

    _WY16_OUTER64_MERGE = wy16_outer64_merge
    return wy16_outer64_merge


def _run_wy16_outer64_merge(
    t16: torch.Tensor,
    v16: torch.Tensor,
    t64: torch.Tensor,
    v64: torch.Tensor,
    outer0: int,
    rel: int,
) -> None:
    global _WY16_OUTER64_MERGE_COMPILED
    batch = t16.shape[0]
    if _WY16_OUTER64_MERGE_COMPILED is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_t16 = torch.empty((batch * 16 * 16,), device="cuda", dtype=torch.float32)
            fake_v16 = torch.empty((batch * 2048 * 16,), device="cuda", dtype=torch.float32)
            fake_t64 = torch.empty((batch * 64 * 64,), device="cuda", dtype=torch.float32)
            fake_v64 = torch.empty((batch * 2048 * 64,), device="cuda", dtype=torch.float32)
            _WY16_OUTER64_MERGE_COMPILED = cute.compile(
                _load_wy16_outer64_merge(),
                from_dlpack(fake_t16),
                from_dlpack(fake_v16),
                from_dlpack(fake_t64),
                from_dlpack(fake_v64),
                batch,
                outer0,
                rel,
            )
    _WY16_OUTER64_MERGE_COMPILED(
        from_dlpack(t16.reshape(-1)),
        from_dlpack(v16.reshape(-1)),
        from_dlpack(t64.reshape(-1)),
        from_dlpack(v64.reshape(-1)),
        batch,
        outer0,
        rel,
    )


def _try_wy_cutedsl_cublas(data: torch.Tensor, n: int, ib: int, ld: int, math_mode: int) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, rows, cols = data.shape
    if batch <= 0 or rows != n or cols != n:
        return None
    h = data.clone(memory_format=torch.contiguous_format)
    tau = torch.empty((batch, n), device=data.device, dtype=torch.float32)
    tmat = torch.empty((batch, ib, ib), device=data.device, dtype=torch.float32)
    vmat = torch.empty((batch, n, ib), device=data.device, dtype=torch.float32)
    wmat = torch.empty((batch, ib, n), device=data.device, dtype=torch.float32)
    ymat = torch.empty((batch, ib, n), device=data.device, dtype=torch.float32)
    for panel0 in range(0, n, ib):
        _run_wy_factor(h, tau, tmat, vmat, panel0, n, ib, ld)
        _apply_wy_cublas(h, vmat, tmat, wmat, ymat, panel0, n, ib, math_mode)
    return h, tau


def _load_n352_wy32_factor(panel0: int) -> Any:
    global _N352_WY32_FACTOR
    compiled_kernel = _N352_WY32_FACTOR.get(panel0)
    if compiled_kernel is not None:
        return compiled_kernel
    need_t = panel0 + 32 < 352

    @cute.kernel
    def n352_wy32_factor_kernel(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        rows = 352 - panel0
        h_base = bidx * 352 * 352
        tau_base = bidx * 352
        t_base = bidx * 32 * 32
        v_base = bidx * 352 * 32

        allocator = cutlass.utils.SmemAllocator()
        pbuf = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((353 * 32,)),
            byte_alignment=16,
            swizzle=None,
        )
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((16,)),
            byte_alignment=16,
            swizzle=None,
        )
        z = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
            swizzle=None,
        )
        scale_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < rows * 32:
            r = p // 32
            j = p - r * 32
            pbuf[j * 353 + r] = dst_h[h_base + (panel0 + r) * 352 + panel0 + j]
            p += 512
        cute.arch.sync_threads()

        for kk in cutlass.range(32, unroll=2):
            ss = cutlass.Float32(0.0)
            r = kk + 1 + tidx
            while r < rows:
                x = pbuf[kk * 353 + r]
                ss = ss + x * x
                r += 512

            off = 16
            while off > 0:
                ss = ss + cute.arch.shuffle_sync_down(ss, off)
                off = off // 2
            if lane == 0:
                red[warp] = ss
            cute.arch.sync_threads()

            if warp == 0:
                total = red[lane] if lane < 16 else 0.0
                off = 16
                while off > 0:
                    total = total + cute.arch.shuffle_sync_down(total, off)
                    off = off // 2
                if lane == 0:
                    sigma = total
                    alpha = pbuf[kk * 353 + kk]
                    if sigma == 0.0:
                        tau_s[kk] = 0.0
                        scale_s[kk] = 0.0
                        tau_k.store(0.0)
                    else:
                        nr = cute.sqrt(alpha * alpha + sigma)
                        beta = -nr if alpha >= 0.0 else nr
                        local_tau = (beta - alpha) / beta
                        pbuf[kk * 353 + kk] = beta
                        tau_s[kk] = local_tau
                        scale_s[kk] = 1.0 / (alpha - beta)
                        tau_k.store(local_tau)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            local_scale = scale_s[kk]
            jj = kk + 1 + warp
            while jj < 32:
                acc = cutlass.Float32(0.0)
                r = kk + 1 + lane
                while r < rows:
                    acc = acc + pbuf[kk * 353 + r] * pbuf[jj * 353 + r]
                    r += 32

                off = 16
                while off > 0:
                    acc = acc + cute.arch.shuffle_sync_down(acc, off)
                    off = off // 2

                sw = cutlass.Float32(0.0)
                if lane == 0:
                    w = local_tau * (pbuf[jj * 353 + kk] + local_scale * acc)
                    pbuf[jj * 353 + kk] = pbuf[jj * 353 + kk] - w
                    sw = local_scale * w
                sw = cute.arch.shuffle_sync(sw, 0)

                r = kk + 1 + lane
                while r < rows:
                    pbuf[jj * 353 + r] = pbuf[jj * 353 + r] - pbuf[kk * 353 + r] * sw
                    r += 32
                jj += 16
            cute.arch.sync_threads()

        if need_t:
            p = tidx
            while p < 32 * 32:
                tmat[t_base + p] = 0.0
                p += 512
            cute.arch.sync_threads()

            j = 0
            while j < 32:
                tau_j = tau_s[j]
                if tidx == 0:
                    tmat[t_base + j * 32 + j] = tau_j

                i = warp
                while i < j:
                    acc = cutlass.Float32(0.0)
                    r = j + lane
                    while r < rows:
                        vi = pbuf[i * 353 + r] * scale_s[i]
                        vj = 1.0 if r == j else pbuf[j * 353 + r] * scale_s[j]
                        acc = acc + vi * vj
                        r += 32

                    off = 16
                    while off > 0:
                        acc = acc + cute.arch.shuffle_sync_down(acc, off)
                        off = off // 2

                    if lane == 0:
                        z[i] = acc
                    i += 16
                cute.arch.sync_threads()

                if tidx < j:
                    y = cutlass.Float32(0.0)
                    l = 0
                    while l < j:
                        y = y + tmat[t_base + tidx * 32 + l] * z[l]
                        l += 1
                    tmat[t_base + tidx * 32 + j] = -tau_j * y
                cute.arch.sync_threads()
                j += 1

        if tidx < 32:
            dst_tau[tau_base + panel0 + tidx] = tau_s[tidx]

        p = tidx
        while p < rows * 32:
            r = p // 32
            j = p - r * 32
            hval = pbuf[j * 353 + r]
            if r > j:
                hval = hval * scale_s[j]
            dst_h[h_base + (panel0 + r) * 352 + panel0 + j] = hval
            if need_t:
                v = cutlass.Float32(0.0)
                if r == j:
                    v = 1.0
                elif r > j:
                    v = hval
                vmat[v_base + p] = v
            p += 512

    @cute.jit
    def n352_wy32_factor(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        batch: cutlass.Int32,
    ):
        n352_wy32_factor_kernel(dst_h, dst_tau, tmat, vmat).launch(
            grid=(batch, 1, 1),
            block=(512, 1, 1),
        )

    _N352_WY32_FACTOR[panel0] = n352_wy32_factor
    return n352_wy32_factor


def _run_n352_wy32_factor(
    h: torch.Tensor,
    tau: torch.Tensor,
    tmat: torch.Tensor,
    vmat: torch.Tensor,
    panel0: int,
) -> None:
    batch = h.shape[0]
    key = (batch, panel0)
    compiled = _N352_WY32_FACTOR_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_h = torch.empty((h.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 352,), device="cuda", dtype=torch.float32)
            fake_t = torch.empty((batch * 32 * 32,), device="cuda", dtype=torch.float32)
            fake_v = torch.empty((batch * 352 * 32,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_n352_wy32_factor(panel0),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                from_dlpack(fake_t),
                from_dlpack(fake_v),
                batch,
            )
        _N352_WY32_FACTOR_COMPILED[key] = compiled
    compiled(
        from_dlpack(h.reshape(-1)),
        from_dlpack(tau.reshape(-1)),
        from_dlpack(tmat.reshape(-1)),
        from_dlpack(vmat.reshape(-1)),
        batch,
    )


def _try_n352_wy32_static_cutedsl(data: torch.Tensor) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, rows, cols = data.shape
    if batch <= 0 or rows != 352 or cols != 352:
        return None
    h = data.clone(memory_format=torch.contiguous_format)
    tau = torch.empty((batch, 352), device=data.device, dtype=torch.float32)
    tmat = torch.empty((batch, 32, 32), device=data.device, dtype=torch.float32)
    vmat = torch.empty((batch, 352, 32), device=data.device, dtype=torch.float32)
    wmat = torch.empty((batch, 32, 352), device=data.device, dtype=torch.float32)
    ymat = torch.empty((batch, 32, 352), device=data.device, dtype=torch.float32)
    for panel0 in range(0, 352, 32):
        _run_n352_wy32_factor(h, tau, tmat, vmat, panel0)
        _apply_wy_cublas(h, vmat, tmat, wmat, ymat, panel0, 352, 32, 2)
    return h, tau


def _load_n176_wy_factor(panel0: int) -> Any:
    global _N176_WY_FACTOR
    compiled_kernel = _N176_WY_FACTOR.get(panel0)
    if compiled_kernel is not None:
        return compiled_kernel

    @cute.kernel
    def n176_wy_factor_kernel(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        rows = 176 - panel0
        h_base = bidx * 176 * 176
        tau_base = bidx * 176
        t_base = bidx * 16 * 16
        v_base = bidx * 176 * 16

        allocator = cutlass.utils.SmemAllocator()
        pbuf = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((177 * 16,)),
            byte_alignment=16,
            swizzle=None,
        )
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((512,)),
            byte_alignment=16,
            swizzle=None,
        )
        z = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((16,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((16,)),
            byte_alignment=16,
            swizzle=None,
        )
        scale_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((16,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < rows * 16:
            r = p // 16
            j = p - r * 16
            pbuf[j * 177 + r] = dst_h[h_base + (panel0 + r) * 176 + panel0 + j]
            p += 512
        cute.arch.sync_threads()

        for kk in cutlass.range_constexpr(16):
            ss = cutlass.Float32(0.0)
            r = kk + 1 + tidx
            while r < rows:
                x = pbuf[kk * 177 + r]
                ss = ss + x * x
                r += 512

            off = 16
            while off > 0:
                ss = ss + cute.arch.shuffle_sync_down(ss, off)
                off = off // 2
            if lane == 0:
                red[warp] = ss
            cute.arch.sync_threads()

            if warp == 0:
                total = red[lane] if lane < 16 else 0.0
                off = 16
                while off > 0:
                    total = total + cute.arch.shuffle_sync_down(total, off)
                    off = off // 2
                if lane == 0:
                    sigma = total
                    alpha = pbuf[kk * 177 + kk]
                    if sigma == 0.0:
                        tau_s[kk] = 0.0
                        scale_s[kk] = 0.0
                        tau_k.store(0.0)
                    else:
                        nr = cute.sqrt(alpha * alpha + sigma)
                        beta = -nr if alpha >= 0.0 else nr
                        local_tau = (beta - alpha) / beta
                        pbuf[kk * 177 + kk] = beta
                        tau_s[kk] = local_tau
                        scale_s[kk] = 1.0 / (alpha - beta)
                        tau_k.store(local_tau)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            local_scale = scale_s[kk]
            jj = kk + 1 + warp
            while jj < 16:
                acc = cutlass.Float32(0.0)
                r = kk + 1 + lane
                while r < rows:
                    acc = acc + pbuf[kk * 177 + r] * pbuf[jj * 177 + r]
                    r += 32

                off = 16
                while off > 0:
                    acc = acc + cute.arch.shuffle_sync_down(acc, off)
                    off = off // 2

                sw = cutlass.Float32(0.0)
                if lane == 0:
                    w = local_tau * (pbuf[jj * 177 + kk] + local_scale * acc)
                    pbuf[jj * 177 + kk] = pbuf[jj * 177 + kk] - w
                    sw = local_scale * w
                sw = cute.arch.shuffle_sync(sw, 0)

                r = kk + 1 + lane
                while r < rows:
                    pbuf[jj * 177 + r] = pbuf[jj * 177 + r] - pbuf[kk * 177 + r] * sw
                    r += 32
                jj += 16
            cute.arch.sync_threads()

        p = tidx
        while p < 16 * 16:
            tmat[t_base + p] = 0.0
            p += 512
        cute.arch.sync_threads()

        j = 0
        while j < 16:
            tau_j = tau_s[j]
            if tidx == 0:
                tmat[t_base + j * 16 + j] = tau_j

            i = warp
            while i < j:
                acc = cutlass.Float32(0.0)
                r = j + lane
                while r < rows:
                    vi = pbuf[i * 177 + r] * scale_s[i]
                    vj = 1.0 if r == j else pbuf[j * 177 + r] * scale_s[j]
                    acc = acc + vi * vj
                    r += 32

                off = 16
                while off > 0:
                    acc = acc + cute.arch.shuffle_sync_down(acc, off)
                    off = off // 2

                if lane == 0:
                    z[i] = acc
                i += 16
            cute.arch.sync_threads()

            if tidx < j:
                y = cutlass.Float32(0.0)
                l = 0
                while l < j:
                    y = y + tmat[t_base + tidx * 16 + l] * z[l]
                    l += 1
                tmat[t_base + tidx * 16 + j] = -tau_j * y
            cute.arch.sync_threads()
            j += 1

        if tidx < 16:
            dst_tau[tau_base + panel0 + tidx] = tau_s[tidx]

        p = tidx
        while p < rows * 16:
            r = p // 16
            j = p - r * 16
            hval = pbuf[j * 177 + r]
            if r > j:
                hval = hval * scale_s[j]
            dst_h[h_base + (panel0 + r) * 176 + panel0 + j] = hval
            v = cutlass.Float32(0.0)
            if r == j:
                v = 1.0
            elif r > j:
                v = hval
            vmat[v_base + p] = v
            p += 512

    @cute.jit
    def n176_wy_factor(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        batch: cutlass.Int32,
    ):
        n176_wy_factor_kernel(dst_h, dst_tau, tmat, vmat).launch(
            grid=(batch, 1, 1),
            block=(512, 1, 1),
        )

    _N176_WY_FACTOR[panel0] = n176_wy_factor
    return n176_wy_factor


def _run_n176_wy_factor(
    h: torch.Tensor,
    tau: torch.Tensor,
    tmat: torch.Tensor,
    vmat: torch.Tensor,
    panel0: int,
) -> None:
    batch = h.shape[0]
    key = (batch, panel0)
    compiled = _N176_WY_FACTOR_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_h = torch.empty((h.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 176,), device="cuda", dtype=torch.float32)
            fake_t = torch.empty((batch * 16 * 16,), device="cuda", dtype=torch.float32)
            fake_v = torch.empty((batch * 176 * 16,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_n176_wy_factor(panel0),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                from_dlpack(fake_t),
                from_dlpack(fake_v),
                batch,
            )
        _N176_WY_FACTOR_COMPILED[key] = compiled
    compiled(
        from_dlpack(h.reshape(-1)),
        from_dlpack(tau.reshape(-1)),
        from_dlpack(tmat.reshape(-1)),
        from_dlpack(vmat.reshape(-1)),
        batch,
    )


def _load_n176_wy32_factor(panel0: int) -> Any:
    global _N176_WY32_FACTOR
    compiled_kernel = _N176_WY32_FACTOR.get(panel0)
    if compiled_kernel is not None:
        return compiled_kernel

    @cute.kernel
    def n176_wy32_factor_kernel(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        rows = 176 - panel0
        h_base = bidx * 176 * 176
        tau_base = bidx * 176
        t_base = bidx * 32 * 32
        v_base = bidx * 176 * 32

        allocator = cutlass.utils.SmemAllocator()
        pbuf = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((177 * 32,)),
            byte_alignment=16,
            swizzle=None,
        )
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((16,)),
            byte_alignment=16,
            swizzle=None,
        )
        z = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
            swizzle=None,
        )
        scale_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < rows * 32:
            r = p // 32
            j = p - r * 32
            pbuf[j * 177 + r] = dst_h[h_base + (panel0 + r) * 176 + panel0 + j]
            p += 512
        cute.arch.sync_threads()

        for kk in cutlass.range_constexpr(32):
            ss = cutlass.Float32(0.0)
            r = kk + 1 + tidx
            while r < rows:
                x = pbuf[kk * 177 + r]
                ss = ss + x * x
                r += 512

            off = 16
            while off > 0:
                ss = ss + cute.arch.shuffle_sync_down(ss, off)
                off = off // 2
            if lane == 0:
                red[warp] = ss
            cute.arch.sync_threads()

            if warp == 0:
                total = red[lane] if lane < 16 else 0.0
                off = 16
                while off > 0:
                    total = total + cute.arch.shuffle_sync_down(total, off)
                    off = off // 2
                if lane == 0:
                    sigma = total
                    alpha = pbuf[kk * 177 + kk]
                    if sigma == 0.0:
                        tau_s[kk] = 0.0
                        scale_s[kk] = 0.0
                        tau_k.store(0.0)
                    else:
                        nr = cute.sqrt(alpha * alpha + sigma)
                        beta = -nr if alpha >= 0.0 else nr
                        local_tau = (beta - alpha) / beta
                        pbuf[kk * 177 + kk] = beta
                        tau_s[kk] = local_tau
                        scale_s[kk] = 1.0 / (alpha - beta)
                        tau_k.store(local_tau)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            local_scale = scale_s[kk]
            jj = kk + 1 + warp
            while jj < 32:
                acc = cutlass.Float32(0.0)
                r = kk + 1 + lane
                while r < rows:
                    acc = acc + pbuf[kk * 177 + r] * pbuf[jj * 177 + r]
                    r += 32

                off = 16
                while off > 0:
                    acc = acc + cute.arch.shuffle_sync_down(acc, off)
                    off = off // 2

                sw = cutlass.Float32(0.0)
                if lane == 0:
                    w = local_tau * (pbuf[jj * 177 + kk] + local_scale * acc)
                    pbuf[jj * 177 + kk] = pbuf[jj * 177 + kk] - w
                    sw = local_scale * w
                sw = cute.arch.shuffle_sync(sw, 0)

                r = kk + 1 + lane
                while r < rows:
                    pbuf[jj * 177 + r] = pbuf[jj * 177 + r] - pbuf[kk * 177 + r] * sw
                    r += 32
                jj += 16
            cute.arch.sync_threads()

        p = tidx
        while p < 32 * 32:
            tmat[t_base + p] = 0.0
            p += 512
        cute.arch.sync_threads()

        j = 0
        while j < 32:
            tau_j = tau_s[j]
            if tidx == 0:
                tmat[t_base + j * 32 + j] = tau_j

            i = warp
            while i < j:
                acc = cutlass.Float32(0.0)
                r = j + lane
                while r < rows:
                    vi = pbuf[i * 177 + r] * scale_s[i]
                    vj = 1.0 if r == j else pbuf[j * 177 + r] * scale_s[j]
                    acc = acc + vi * vj
                    r += 32

                off = 16
                while off > 0:
                    acc = acc + cute.arch.shuffle_sync_down(acc, off)
                    off = off // 2

                if lane == 0:
                    z[i] = acc
                i += 16
            cute.arch.sync_threads()

            if tidx < j:
                y = cutlass.Float32(0.0)
                l = 0
                while l < j:
                    y = y + tmat[t_base + tidx * 32 + l] * z[l]
                    l += 1
                tmat[t_base + tidx * 32 + j] = -tau_j * y
            cute.arch.sync_threads()
            j += 1

        if tidx < 32:
            dst_tau[tau_base + panel0 + tidx] = tau_s[tidx]

        p = tidx
        while p < rows * 32:
            r = p // 32
            j = p - r * 32
            hval = pbuf[j * 177 + r]
            if r > j:
                hval = hval * scale_s[j]
            dst_h[h_base + (panel0 + r) * 176 + panel0 + j] = hval
            v = cutlass.Float32(0.0)
            if r == j:
                v = 1.0
            elif r > j:
                v = hval
            vmat[v_base + p] = v
            p += 512

    @cute.jit
    def n176_wy32_factor(
        dst_h: cute.Tensor,
        dst_tau: cute.Tensor,
        tmat: cute.Tensor,
        vmat: cute.Tensor,
        batch: cutlass.Int32,
    ):
        n176_wy32_factor_kernel(dst_h, dst_tau, tmat, vmat).launch(
            grid=(batch, 1, 1),
            block=(512, 1, 1),
        )

    _N176_WY32_FACTOR[panel0] = n176_wy32_factor
    return n176_wy32_factor


def _run_n176_wy32_factor(
    h: torch.Tensor,
    tau: torch.Tensor,
    tmat: torch.Tensor,
    vmat: torch.Tensor,
    panel0: int,
) -> None:
    batch = h.shape[0]
    key = (batch, panel0)
    compiled = _N176_WY32_FACTOR_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_h = torch.empty((h.numel(),), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 176,), device="cuda", dtype=torch.float32)
            fake_t = torch.empty((batch * 32 * 32,), device="cuda", dtype=torch.float32)
            fake_v = torch.empty((batch * 176 * 32,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_n176_wy32_factor(panel0),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                from_dlpack(fake_t),
                from_dlpack(fake_v),
                batch,
            )
        _N176_WY32_FACTOR_COMPILED[key] = compiled
    compiled(
        from_dlpack(h.reshape(-1)),
        from_dlpack(tau.reshape(-1)),
        from_dlpack(tmat.reshape(-1)),
        from_dlpack(vmat.reshape(-1)),
        batch,
    )


def _load_n176_tail(tail0: int, tail_n: int, nwarps: int, unroll: int) -> Any:
    global _N176_TAIL
    key = (tail0, tail_n, nwarps, unroll)
    compiled_kernel = _N176_TAIL.get(key)
    if compiled_kernel is not None:
        return compiled_kernel
    nthreads = nwarps * 32

    @cute.kernel
    def n176_tail_kernel(dst_h: cute.Tensor, dst_tau: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        base = bidx * 176 * 176
        tau_base = bidx * 176

        allocator = cutlass.utils.SmemAllocator()
        s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((tail_n * (tail_n + 1),)),
            byte_alignment=16,
            swizzle=None,
        )
        red = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((nwarps,)),
            byte_alignment=16,
            swizzle=None,
        )
        scale_s = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((tail_n,)),
            byte_alignment=16,
            swizzle=None,
        )
        tau_k = allocator.allocate(cutlass.Float32)

        p = tidx
        while p < tail_n * tail_n:
            row = p // tail_n
            col = p - row * tail_n
            s[row * (tail_n + 1) + col] = dst_h[base + (tail0 + row) * 176 + tail0 + col]
            p += nthreads
        cute.arch.sync_threads()

        for k in cutlass.range(tail_n, unroll=unroll):
            ss = cutlass.Float32(0.0)
            i = k + 1 + tidx
            while i < tail_n:
                x = s[i * (tail_n + 1) + k]
                ss = ss + x * x
                i += nthreads

            off = 16
            while off > 0:
                ss = ss + cute.arch.shuffle_sync_down(ss, off)
                off = off // 2
            if lane == 0:
                red[warp] = ss
            cute.arch.sync_threads()

            if warp == 0:
                block_ss = red[lane] if lane < nwarps else 0.0
                off = 16
                while off > 0:
                    block_ss = block_ss + cute.arch.shuffle_sync_down(block_ss, off)
                    off = off // 2
                if lane == 0:
                    alpha = s[k * (tail_n + 1) + k]
                    if block_ss == 0.0:
                        dst_tau[tau_base + tail0 + k] = 0.0
                        scale_s[k] = 0.0
                        tau_k.store(0.0)
                    else:
                        nr = cute.sqrt(alpha * alpha + block_ss)
                        beta = -nr if alpha >= 0.0 else nr
                        local_tau = (beta - alpha) / beta
                        s[k * (tail_n + 1) + k] = beta
                        dst_tau[tau_base + tail0 + k] = local_tau
                        scale_s[k] = 1.0 / (alpha - beta)
                        tau_k.store(local_tau)
            cute.arch.sync_threads()

            local_tau = tau_k.load()
            local_scale = scale_s[k]
            j = k + 1 + warp
            while j < tail_n:
                acc = cutlass.Float32(0.0)
                i = k + 1 + lane
                while i < tail_n:
                    acc = acc + s[i * (tail_n + 1) + k] * s[i * (tail_n + 1) + j]
                    i += 32

                off = 16
                while off > 0:
                    acc = acc + cute.arch.shuffle_sync_down(acc, off)
                    off = off // 2

                sw = cutlass.Float32(0.0)
                if lane == 0:
                    w = local_tau * (s[k * (tail_n + 1) + j] + local_scale * acc)
                    s[k * (tail_n + 1) + j] = s[k * (tail_n + 1) + j] - w
                    sw = local_scale * w
                sw = cute.arch.shuffle_sync(sw, 0)

                i = k + 1 + lane
                while i < tail_n:
                    s[i * (tail_n + 1) + j] = s[i * (tail_n + 1) + j] - s[i * (tail_n + 1) + k] * sw
                    i += 32
                j += nwarps
            cute.arch.sync_threads()
        p = tidx
        while p < tail_n * tail_n:
            row = p // tail_n
            col = p - row * tail_n
            hval = s[row * (tail_n + 1) + col]
            if row > col:
                hval = hval * scale_s[col]
            dst_h[base + (tail0 + row) * 176 + tail0 + col] = hval
            p += nthreads

    @cute.jit
    def n176_tail(dst_h: cute.Tensor, dst_tau: cute.Tensor, batch: cutlass.Int32):
        n176_tail_kernel(dst_h, dst_tau).launch(grid=(batch, 1, 1), block=(nthreads, 1, 1))

    _N176_TAIL[key] = n176_tail
    return n176_tail


def _run_n176_tail(h: torch.Tensor, tau: torch.Tensor, tail0: int, tail_n: int, nwarps: int, unroll: int) -> None:
    batch = h.shape[0]
    key = (batch, tail0, tail_n, nwarps, unroll)
    compiled = _N176_TAIL_COMPILED.get(key)
    if compiled is None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_h = torch.empty((batch * 176 * 176,), device="cuda", dtype=torch.float32)
            fake_tau = torch.empty((batch * 176,), device="cuda", dtype=torch.float32)
            compiled = cute.compile(
                _load_n176_tail(tail0, tail_n, nwarps, unroll),
                from_dlpack(fake_h),
                from_dlpack(fake_tau),
                batch,
            )
        _N176_TAIL_COMPILED[key] = compiled
    compiled(from_dlpack(h.reshape(-1)), from_dlpack(tau.reshape(-1)), batch)


def _try_n176_wy_tail_cutedsl(data: torch.Tensor) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, rows, cols = data.shape
    if batch <= 0 or rows != 176 or cols != 176:
        return None

    n = 176
    ib = 16
    h = data.clone(memory_format=torch.contiguous_format)
    tau = torch.empty((batch, n), device=data.device, dtype=torch.float32)
    tmat = torch.empty((batch, ib, ib), device=data.device, dtype=torch.float32)
    vmat = torch.empty((batch, n, ib), device=data.device, dtype=torch.float32)
    wmat = torch.empty((batch, ib, n), device=data.device, dtype=torch.float32)
    ymat = torch.empty((batch, ib, n), device=data.device, dtype=torch.float32)
    for panel0 in range(0, _N176_TAIL0, ib):
        _run_n176_wy_factor(h, tau, tmat, vmat, panel0)
        _apply_wy_cublas(h, vmat, tmat, wmat, ymat, panel0, n, ib, 2)
    _run_n176_tail(h, tau, _N176_TAIL0, _N176_TAIL_N, _N176_TAIL_NWARPS, _N176_TAIL_UNROLL)
    return h, tau


def _try_wy2048_ib24_tail8_cutedsl(data: torch.Tensor) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, rows, cols = data.shape
    n = 2048
    if batch <= 0 or rows != n or cols != n:
        return None

    ib = 24
    split = 2040
    h = data.clone(memory_format=torch.contiguous_format)
    tau = torch.empty((batch, n), device=data.device, dtype=torch.float32)
    tmat = torch.empty((batch, ib, ib), device=data.device, dtype=torch.float32)
    vmat = torch.empty((batch, n, ib), device=data.device, dtype=torch.float32)
    wmat = torch.empty((batch, ib, n), device=data.device, dtype=torch.float32)
    ymat = torch.empty((batch, ib, n), device=data.device, dtype=torch.float32)
    for panel0 in range(0, split, ib):
        if batch == 8:
            _run_wy24_micro3_factor(h, tau, tmat, vmat, panel0)
        else:
            _run_wy_factor(h, tau, tmat, vmat, panel0, n, ib, 2055)
        _apply_wy_cublas(h, vmat, tmat, wmat, ymat, panel0, n, ib, 2)

    tail_out = _try_small_cutedsl(h[:, split:, split:].contiguous())
    if tail_out is None:
        return None
    tail_h, tail_tau = tail_out
    h[:, split:, split:].copy_(tail_h)
    tau[:, split:].copy_(tail_tau)
    return h, tau


def _try_wy2048_inner16_outer64_cutedsl(data: torch.Tensor) -> output_t | None:
    if (not data.is_cuda) or data.dtype != torch.float32 or data.ndim != 3:
        return None
    batch, rows, cols = data.shape
    n = 2048
    if batch <= 0 or rows != n or cols != n:
        return None

    inner = 16
    outer = 64
    ld = 2050
    h = data.clone(memory_format=torch.contiguous_format)
    tau = torch.empty((batch, n), device=data.device, dtype=torch.float32)
    t16 = torch.empty((batch, inner, inner), device=data.device, dtype=torch.float32)
    v16 = torch.empty((batch, n, inner), device=data.device, dtype=torch.float32)
    w16 = torch.empty((batch, inner, n), device=data.device, dtype=torch.float32)
    y16 = torch.empty((batch, inner, n), device=data.device, dtype=torch.float32)
    t64 = torch.empty((batch, outer, outer), device=data.device, dtype=torch.float32)
    v64 = torch.empty((batch, n, outer), device=data.device, dtype=torch.float32)
    w64 = torch.empty((batch, outer, n), device=data.device, dtype=torch.float32)
    y64 = torch.empty((batch, outer, n), device=data.device, dtype=torch.float32)

    for outer0 in range(0, n, outer):
        outer_end = outer0 + outer
        need_far_update = outer_end < n
        for panel0 in range(outer0, outer_end, inner):
            _run_wy_factor(h, tau, t16, v16, panel0, n, inner, ld)
            rel = panel0 - outer0
            if need_far_update:
                _run_wy16_outer64_merge(t16, v16, t64, v64, outer0, rel)
            local_col0 = panel0 + inner
            _apply_wy_cublas_range(
                h,
                v16,
                t16,
                w16,
                y16,
                panel0,
                n,
                inner,
                local_col0,
                outer_end - local_col0,
                2,
            )
        if need_far_update:
            _apply_wy_cublas_range(h, v64, t64, w64, y64, outer0, n, outer, outer_end, n - outer_end, 2)
    return h, tau


def _blocked_size(n: int) -> int:
    if n <= 192:
        return 16
    if n <= 768:
        return 32
    return 64


def _blocked_torch_qr(data: torch.Tensor, block: int) -> output_t:
    batch, n, _ = data.shape
    h = data.clone(memory_format=torch.contiguous_format)
    tau = torch.empty((batch, n), device=data.device, dtype=torch.float32)

    old_tf32 = None
    if data.is_cuda:
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        idx_block = torch.arange(block, device=data.device)
        for k in range(0, n, block):
            ib = min(block, n - k)
            panel = h[:, k:, k : k + ib].contiguous()
            panel_out = _try_panel_cutedsl(panel)
            hp, tp = panel_out if panel_out is not None else torch.geqrf(panel)
            h[:, k:, k : k + ib].copy_(hp)
            tau[:, k : k + ib].copy_(tp)

            if k + ib >= n:
                continue

            v = torch.tril(hp, diagonal=-1)
            idx = idx_block[:ib]
            v[:, idx, idx] = 1.0

            t = torch.zeros((batch, ib, ib), device=data.device, dtype=torch.float32)
            for j in range(ib):
                t[:, j, j] = tp[:, j]
                if j > 0:
                    z = torch.bmm(v[:, j:, :j].transpose(1, 2), v[:, j:, j : j + 1])
                    t[:, :j, j : j + 1] = -tp[:, j].reshape(batch, 1, 1) * torch.bmm(t[:, :j, :j], z)

            c = h[:, k:, k + ib :]
            w = torch.bmm(v.transpose(1, 2), c)
            w = torch.bmm(t.transpose(1, 2), w)
            c.baddbmm_(v, w, beta=1.0, alpha=-1.0)
    finally:
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    return h, tau


@torch.no_grad()
def qr_kernel(data: input_t) -> output_t:
    if isinstance(data, torch.Tensor):
        small = _try_small_cutedsl(data)
        if small is not None:
            return small
        n176 = _try_n176_wy_tail_cutedsl(data)
        if n176 is not None:
            return n176
        wy352 = _try_n352_wy32_static_cutedsl(data)
        if wy352 is not None:
            return wy352
        wy352 = _try_wy_cutedsl_cublas(data, 352, 32, 353, 2)
        if wy352 is not None:
            return wy352
        n352 = _try_n352_cutedsl(data)
        if n352 is not None:
            return n352
        wy512 = _try_n512_wy_cutedsl_cublas(data)
        if wy512 is not None:
            return wy512
        wy1024 = _try_wy_cutedsl_cublas(data, 1024, 32, 1025, 2)
        if wy1024 is not None:
            return wy1024
        wy2048 = _try_wy2048_ib24_tail8_cutedsl(data)
        if wy2048 is not None:
            return wy2048
        wy2048 = _try_wy_cutedsl_cublas(data, 2048, 16, 2050, 2)
        if wy2048 is not None:
            return wy2048
        rfirst4096 = _try_4096_rfirst(data)
        if rfirst4096 is not None:
            return rfirst4096
        batch, n, m = data.shape
        if batch > 0 and n == m and n in (512, 1024, 2048):
            return _blocked_torch_qr(data, _blocked_size(n))
    return torch.geqrf(data)


def kernel(data: input_t) -> output_t:
    return qr_kernel(data)


def custom_kernel(data: input_t) -> output_t:
    return qr_kernel(data)


def ref_kernel(data: input_t) -> output_t:
    return qr_kernel(data)


def solve(data: input_t) -> output_t:
    return qr_kernel(data)
