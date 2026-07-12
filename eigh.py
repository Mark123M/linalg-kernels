import fcntl
import hashlib
import math
import operator
import os
import pickle
import sys
import tempfile
import time
from collections import namedtuple
from getpass import getuser
from pathlib import Path
from typing import Type, Optional
from functools import lru_cache, wraps

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float16, BFloat16, Float32, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05, warp
from cutlass.cute.nvgpu.tcgen05.mma import CtaGroup  # noqa
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.pipeline
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
import cuda.bindings.driver as cuda
from task import input_t, output_t


EXPORT_FUNC_NAME = "func"
LOCK_TIMEOUT = 60
CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def disk_cache_enabled() -> bool:
    return _env_flag("EIGH_ENABLE_DISK_CACHE")


def get_cache_path() -> Path:
    cache_dir = os.environ.get("EIGH_CUTE_CACHE_DIR")
    path = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / getuser() / "eigh_cute_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={getattr(cutlass, '__version__', 'unknown')}".encode())
    try:
        import tvm_ffi

        h.update(f"tvm_ffi={getattr(tvm_ffi, '__version__', 'unknown')}".encode())
    except ModuleNotFoundError:
        h.update(b"tvm_ffi=missing")
    src = Path(__file__).resolve()
    h.update(src.name.encode())
    content = src.read_bytes()
    h.update(len(content).to_bytes(8, "little"))
    h.update(content)
    return h.hexdigest()


def _key_to_hash(key: tuple) -> str:
    try:
        payload = pickle.dumps(key)
    except Exception:
        payload = repr(key).encode()
    return hashlib.sha256(payload).hexdigest()


class FileLock:
    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = LOCK_TIMEOUT):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self._fd = -1

    def __enter__(self):
        flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        self._fd = os.open(str(self.lock_path), flags)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.1)
        os.close(self._fd)
        self._fd = -1
        raise RuntimeError(f"Timed out waiting for lock: {self.lock_path}")

    def __exit__(self, *exc):
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


def jit_cache(fn):
    cache = {}
    hits = 0
    misses = 0

    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = args + tuple(sorted(kwargs.items())) if kwargs else args
        if cache_key in cache:
            hits += 1
            return cache[cache_key]

        if not disk_cache_enabled():
            misses += 1
            compiled_fn = fn(*args, **kwargs)
            cache[cache_key] = compiled_fn
            return compiled_fn

        sha = _key_to_hash((fn.__qualname__,) + cache_key)
        cache_path = get_cache_path() / _compute_source_fingerprint()
        cache_path.mkdir(parents=True, exist_ok=True)
        o_path = cache_path / f"{sha}.o"
        lock_path = cache_path / f"{sha}.lock"

        def load_cached():
            module = cute.runtime.load_module(str(o_path), enable_tvm_ffi=True)
            return module[EXPORT_FUNC_NAME]

        if o_path.exists():
            try:
                with FileLock(lock_path, exclusive=False):
                    if o_path.exists():
                        loaded = load_cached()
                        cache[cache_key] = loaded
                        hits += 1
                        return loaded
            except RuntimeError:
                pass

        try:
            with FileLock(lock_path, exclusive=True):
                if o_path.exists():
                    loaded = load_cached()
                    cache[cache_key] = loaded
                    hits += 1
                    return loaded

                misses += 1
                compiled_fn = fn(*args, **kwargs)
                try:
                    compiled_fn.export_to_c(
                        object_file_path=str(o_path),
                        function_name=EXPORT_FUNC_NAME,
                    )
                except Exception as exc:
                    print(f"eigh cache: export failed for key {sha}: {exc}")
                cache[cache_key] = compiled_fn
                return compiled_fn
        except RuntimeError as exc:
            print(
                f"eigh cache: lock timeout for key {sha}: {exc}; "
                "falling back to in-process compile without disk cache"
            )
            misses += 1
            compiled_fn = fn(*args, **kwargs)
            cache[cache_key] = compiled_fn
            return compiled_fn

    def cache_clear():
        nonlocal hits, misses
        cache.clear()
        hits = 0
        misses = 0

    def cache_info():
        return CacheInfo(hits=hits, misses=misses, maxsize=None, currsize=len(cache))

    wrapper.cache = cache
    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper

torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


@cute.jit
def block_sum(val: cute.Numeric, reduction_buffer: cute.Tensor) -> cute.Numeric:
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = Float32(0.0)
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(block_reduce_val, operator.add)


@cute.jit
def row_sum(
    x: cute.TensorSSA | cute.Numeric,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
) -> cute.Numeric:
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
    else:
        val = x
    val = cute.arch.warp_reduction(
        val,
        operator.add,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if const_expr(reduction_buffer is not None):
        warps_per_row = cute.size(reduction_buffer.shape[1])
        if const_expr(warps_per_row > 1):
            val = block_sum(val, reduction_buffer)
    return val


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


class Eigh:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        k: int = 0,
        panel_size: int = 1,
        stage: int = 1,
        reduction_dtype=Float32,
        debug_printf: bool = True,
    ):
        assert k >= 0
        assert panel_size > 0
        assert k * panel_size + panel_size < N
        self.dtype = dtype
        self.N = N
        self.k = k
        self.panel_size = panel_size
        self.stage = stage
        self.reduction_dtype = reduction_dtype
        self.debug_printf = debug_printf

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (8192, 128)]:
            if N <= limit:
                return threads
        return 256

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def tiled_copy_2d(
        self,
        dtype: Type[cutlass.Numeric],
        threads_per_row: int,
        num_threads: int,
        num_copy_elems: int = 1,
        is_async: bool = False,
    ) -> cute.TiledCopy:
        num_copy_bits = num_copy_elems * dtype.width
        copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
        assert num_threads % threads_per_row == 0
        thr_layout = cute.make_ordered_layout(
            (num_threads // threads_per_row, threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, num_copy_elems))
        return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    def tiled_copy_1d(
        self,
        dtype: Type[cutlass.Numeric],
        num_threads: int,
        num_copy_elems: int = 1,
        is_async: bool = False,
    ) -> cute.TiledCopy:
        num_copy_bits = num_copy_elems * dtype.width
        copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
        thr_layout = cute.make_layout(num_threads)
        val_layout = cute.make_layout(num_copy_elems)
        return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    def _get_tiled_copy(self, vecsize: int = 1):
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = self.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _get_column_tiled_copy(self):
        num_threads = self._num_threads()
        threads_per_row = self._threads_per_row()
        assert threads_per_row <= num_threads
        assert num_threads % threads_per_row == 0
        panel_start = self.k * self.panel_size
        max_panel_n = self.N - panel_start - 1
        tiler_n = cute.ceil_div(max_panel_n, num_threads) * num_threads
        tiled_copy = self.tiled_copy_1d(self.dtype, num_threads, is_async=True)
        matvec_tiled_copy = self.tiled_copy_2d(
            self.dtype, threads_per_row, num_threads, is_async=True
        )
        return tiled_copy, matvec_tiled_copy, tiler_n, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout: cute.Layout):
        num_warps = cute.ceil_div(cute.size(tv_layout, mode=[0]), cute.arch.WARP_SIZE)
        warps_per_row = (
            num_warps
            if cute.rank(tv_layout.shape[0]) == 1
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, warps_per_row, self.stage),
            order=(1, 0, 2),
        )
            
    @staticmethod
    def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1) -> Optional[cute.Tensor]:
        if leading_dim < 0:
            leading_dim = len(shape) + leading_dim
        if dtype is None:
            return None
        stride = tuple(
            cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
            for i in range(len(shape))
        )
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
        )
    
    @cute.kernel
    def kernel(
        self,
        mData: cute.Tensor,
        mTri: cute.Tensor,
        tiler_n: cutlass.Constexpr[int],
        tiled_copy: cute.TiledCopy,
        matvec_tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tv_layout = matvec_tiled_copy.layout_tv_tiled

        panel_start = const_expr(self.k * self.panel_size)
        panel_row_base = const_expr(panel_start + 1)
        panel_n = self.N - panel_row_base
        num_threads = const_expr(tiled_copy.size)
        rows_per_tile = const_expr(num_threads // threads_per_row)
        warps_per_row = const_expr(max(threads_per_row // cute.arch.WARP_SIZE, 1))
        num_tiles = const_expr(cute.ceil_div(panel_n, rows_per_tile))

        smem = cutlass.utils.SmemAllocator()
        sCol = smem.allocate_tensor(
            mData.element_type,
            cute.make_layout(tiler_n),
            byte_alignment=16,
        )
        # A @ v
        sB = smem.allocate_tensor(
            self.reduction_dtype,
            cute.make_layout(tiler_n),
            byte_alignment=16,
        )
        sA = smem.allocate_tensor(
            mData.element_type,
            cute.make_ordered_layout((rows_per_tile, tiler_n, 2), order=(1, 0, 2)),
            byte_alignment=16,
        )
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout),
            byte_alignment=8,
        )

        thr_copy = tiled_copy.get_slice(tidx)
        tCsCol = thr_copy.partition_D(sCol)
        reduction_tidx = tidx % threads_per_row

        thr_matvec = matvec_tiled_copy.get_slice(tidx)
        # Each row group reads the whole column with the same per-thread layout as a tile row,
        # so v registers line up elementwise with tile rows in the matvec below.
        sColBcast = cute.make_tensor(
            sCol.iterator, cute.make_layout((rows_per_tile, tiler_n), stride=(0, 1))
        )
        tVsV = thr_matvec.partition_S(sColBcast)
        tVrCol = cute.make_rmem_tensor_like(tVsV)
        tVrV = cute.make_rmem_tensor(tVsV.shape, Float32)
        tAsA = thr_matvec.partition_D(sA)
        tArA = cute.make_rmem_tensor_like(tAsA[None, None, None, 0])

        cA = cute.make_identity_tensor((rows_per_tile, tiler_n))
        tAcA = thr_matvec.partition_S(cA)
        tApA = predicate_k(tAcA, limit=panel_n)
        local_row = tAcA[0][0]

        mA = cute.domain_offset((panel_row_base, panel_row_base), mData[bidx, None, None])

        for i in cutlass.range(self.panel_size, unroll_full=True):
            col = panel_start + i
            mCol = cute.domain_offset((panel_row_base,), mData[bidx, None, col])
            gCol = cute.local_tile(mCol, (tiler_n,), (0,))
            tCgCol = thr_copy.partition_S(gCol)

            pred = cute.make_rmem_tensor((1, cute.size(tCsCol, mode=[1])), Boolean)
            for m in cutlass.range(cute.size(tCsCol, mode=[1]), unroll_full=True):
                col_idx = tidx + m * tiled_copy.size
                pred[0, m] = col_idx >= i and col_idx < panel_n

            cute.copy(thr_copy, tCgCol, tCsCol, pred=pred)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            alpha = Float32(sCol[i])

            cute.autovec_copy(tVsV, tVrCol)
            col_values = tVrCol.load().to(Float32)
            norm_sq = row_sum(
                col_values * col_values,
                threads_per_row,
                reduction_buffer[None, None, 0],
            )

            norm = Float32(0.0)
            if norm_sq > 0.0:
                norm = cute.math.sqrt(norm_sq)

            beta = -norm
            if alpha < 0.0:
                beta = norm

            tau = Float32(0.0)
            if beta != 0.0:
                tau = (beta - alpha) / beta

            denom = alpha - beta
            inv_denom = Float32(0.0)
            if denom != 0.0:
                inv_denom = Float32(1.0) / denom

            # LAPACK dlarfg convention: v[i] = 1 (col_values gives alpha/denom there).
            # Dynamic condition => `if` statement; a ternary would bake a branch at trace time.
            tVrV.store(col_values * inv_denom)
            for elem in cutlass.range(cute.size(tVrV), unroll_full=True):
                if tAcA[elem][1] == i:
                    tVrV[elem] = Float32(1.0)

            if const_expr(self.debug_printf):
                if bidx == 0 and tidx == 0:
                    cute.printf(
                        "eigh tridiag debug: k=%d i=%d norm=%f norm_sq=%f alpha=%f beta=%f tau=%f\n",
                        self.k,
                        i,
                        norm,
                        norm_sq,
                        alpha,
                        beta,
                        tau,
                    )

            # b = A' @ v over the trailing (panel_n x panel_n) block, pipelined over row tiles:
            # prefetch tile t+1 while reducing tile t. Each thread reads back exactly the smem
            # elements it copied, so no barrier is needed around sA. Columns < i are loaded
            # but multiply v's zeros; masking them off the load measured slower (a dynamic
            # predicate defeats the constant folding a static limit gets) for <2% traffic.
            gA0 = cute.local_tile(mA, (rows_per_tile, tiler_n), (0, 0))
            tAgA0 = thr_matvec.partition_S(gA0)
            if local_row < panel_n:
                cute.copy(thr_matvec, tAgA0, tAsA[None, None, None, 0], pred=tApA)
            cute.arch.cp_async_commit_group()

            for t in cutlass.range(num_tiles):
                if t + 1 < num_tiles:
                    gA_next = cute.local_tile(mA, (rows_per_tile, tiler_n), (t + 1, 0))
                    tAgA_next = thr_matvec.partition_S(gA_next)
                    if (t + 1) * rows_per_tile + local_row < panel_n:
                        cute.copy(
                            thr_matvec,
                            tAgA_next,
                            tAsA[None, None, None, (t + 1) % 2],
                            pred=tApA,
                        )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)

                if const_expr(warps_per_row > 1):
                    # Keep a fast row group from overwriting reduction_buffer before slow
                    # warps have read the previous reduction.
                    cute.arch.barrier()
                cute.autovec_copy(tAsA[None, None, None, t % 2], tArA)
                a = tArA.load().to(Float32)
                b = row_sum(
                    a * tVrV.load(),
                    threads_per_row,
                    reduction_buffer[None, None, 0],
                )
                row_global = t * rows_per_tile + local_row
                if reduction_tidx == 0:
                    if row_global < panel_n:
                        sB[row_global] = b

            # All threads read sCol/reduction_buffer during the matvec; sync before the next
            # column overwrites them (and before reading sB below).
            cute.arch.barrier()

            if const_expr(self.debug_printf):
                for m in cutlass.range(tiler_n // num_threads, unroll_full=True):
                    j = tidx + m * num_threads
                    if j < panel_n:
                        mTri[bidx, panel_row_base + j, col] = self.dtype(sB[j])
    
    @cute.jit
    def __call__(
        self,
        mData: cute.Tensor,
        mTri: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mData.element_type == self.dtype
        assert mTri.element_type == self.dtype
        tiled_copy, matvec_tiled_copy, tiler_n, threads_per_row = (
            self._get_column_tiled_copy()
        )
        num_threads = tiled_copy.size
        self.kernel(
            mData,
            mTri,
            tiler_n,
            tiled_copy,
            matvec_tiled_copy,
            threads_per_row,
        ).launch(
            grid=[mData.shape[0], 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )
        
    @staticmethod
    @jit_cache
    def compile(data_dtype, N, debug_printf: bool = True, k: int = 0, panel_size: int = 1):
        batch_sym = cute.sym_int()
        div = math.gcd(128 // data_dtype.width, N)
        data_cute = Eigh.make_fake_tensor(data_dtype, (batch_sym, N, N), div)
        tri_cute = Eigh.make_fake_tensor(data_dtype, (batch_sym, N, N), div)
        
        return cute.compile(
            Eigh(data_dtype, N, k=k, panel_size=panel_size, debug_printf=debug_printf),
            data_cute,
            tri_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

def custom_kernel_old(data: input_t) -> output_t:
    values, vectors = torch.linalg.eigh(data)
    return vectors, values

def custom_kernel(data: input_t) -> output_t:
    #values, vectors = torch.linalg.eigh(data)
    # random asserts
        
    data_dtype = torch2cute_dtype_map[data.dtype]
    tri = torch.empty_like(data)
    N = data.size(1)
    Eigh.compile(data_dtype, N)(data, tri)
    
    return torch.linalg.eigh(data)
