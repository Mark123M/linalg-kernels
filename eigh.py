import fcntl
import hashlib
import math
import os
import pickle
import sys
import tempfile
import time
from collections import namedtuple
from getpass import getuser
from pathlib import Path
from typing import Type, Tuple, Optional
from functools import lru_cache, wraps

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float16, BFloat16, Float32, const_expr
from cutlass.base_dsl.arch import Arch
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

class Eigh:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        stage: int = 1,
        reduction_dtype=Float32,
    ):
        self.dtype = dtype
        self.N = N
        self.stage = stage
        self.reduction_dtype = reduction_dtype

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (8192, 128)]:
            if N <= limit:
                return threads
        return 256

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _set_cluster_n(self):
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        # SM8x (Ampere/Ada) lacks cluster support
        if arch < Arch.sm_90:
            self.cluster_n = 1
            return
        # SM12x supports cluster up to 8
        max_cluster = 8 if arch.major == 12 else 16
        N = self.N
        if arch.major == 12 and const_expr(self.dtype.width >= 32):
            # SM12x 99 KB SMEM: fp32 bwd has 2 SMEM tensors, needs tighter clustering
            thresholds = [(8 * 1024, 1), (16 * 1024, 2), (32 * 1024, 4), (64 * 1024, 8)]
        elif const_expr(self.dtype.width == 16):
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        else:
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = max_cluster

    def _cap_cluster_n(self, vecsize: int) -> None:
        """Cap ``cluster_n`` so every peer CTA owns a distinct, non-empty N-tile.

        A clustered launch splits the row across ``cluster_n`` CTAs. If
        ``threads_per_row * cluster_n`` exceeds the number of vector blocks in the
        row (``N // vecsize``), one CTA tile already spans the whole row
        (``tiler_mn[1] >= N``); local_tile then collapses every peer onto tile 0,
        so the peers re-reduce the same columns and double-count in the cluster
        reduction. Capping to ``(N // vecsize) // threads_per_row`` guarantees
        ``tiler_mn[1] < N`` whenever the resulting ``cluster_n > 1``.
        """
        max_cluster_n = max(1, (self.N // vecsize) // self._threads_per_row())
        self.cluster_n = min(self.cluster_n, max_cluster_n)

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

    def _get_tiled_copy(self, vecsize: int = 1):
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = self.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout: cute.Layout, cluster_n: int):
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps
            if cute.rank(tv_layout.shape[0]) == 1
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(
        self, smem: cutlass.utils.SmemAllocator, tv_layout: cute.Layout, is_persistent: bool = False
    ) -> Tuple[cute.Tensor, Optional[cute.Pointer]]:
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=8,
        )
        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(
                Int64, num_elems=self.stage if not is_persistent else self.stage * 2
            )
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _initialize_cluster(
        self,
        tidx: Int32,
        mbar_ptr: cute.Pointer,
        num_warps: int
    ):
        if const_expr(self.cluster_n > 1):
            if tidx == 0:  # Initialize full barrier
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                
            cute.arch.mbarrier_init_fence()
            # Cluster arrive after barrier init
            cute.arch.cluster_arrive_relaxed()
            
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
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
    
    @cute.jit
    def __call__(
        self,
        mData: cute.Tensor,
        mTri: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mData.element_type == self.dtype
        assert mTri.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mData, mTri]))
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(
            vecsize=128 // largest_dtype_width
        )
        num_threads = tiled_copy.size
        self.kernel(mData, mTri, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[mData.shape[0], self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )
        
    @staticmethod
    @jit_cache
    def compile(data_dtype, N):
        batch_sym = cute.sym_int()
        div = math.gcd(128 // data_dtype.width, N)
        data_cute = Eigh.make_fake_tensor(data_dtype, (batch_sym, N, N), div)
        tri_cute = Eigh.make_fake_tensor(data_dtype, (batch_sym, N, N), div)
        
        return cute.compile(
            Eigh(data_dtype, N),
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
