"""Popcorn submission for the (64, 256, 256) batched Cholesky shape.

Extracted verbatim from the combined cholesky/cholesky.py fold of
cholesky/b64n256/cholesky_b64n256.py variant 18,
`cta512_rec32_scalar_tc_all_refined_pad129`.

One 512-thread CTA per matrix, recursive-32 base factors, scalar TRSM,
tcgen05 TF32 trailing updates, kLd-padded A10 block.

Note: this shape's dispatch entry is currently commented out in
cholesky/cholesky.py (temporarily routed to torch there). This file
always exercises the kernel, which is the point of keeping it separate.

Any other shape falls back to
torch.linalg.cholesky_ex(..., check_errors=False).L so non-target
leaderboard timings stay constant.
"""

import hashlib
import os
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


_BASE_CUDA_FLAGS = (
    "-O3",
    "-std=c++20",
    "--use_fast_math",
    "--extra-device-vectorization",
    "-Xptxas=-O3",
    "-gencode",
    "arch=compute_100a,code=sm_100a",
)


def _build(name, cpp_source, cuda_source, extra_cuda_flags=(),
           extra_ldflags=()):
    tag = hashlib.sha256((cpp_source + cuda_source).encode()).hexdigest()[:12]
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name=f"{name}_{tag}",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=None,
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[*extra_cuda_flags, *_BASE_CUDA_FLAGS],
            extra_ldflags=list(extra_ldflags),
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


# ---------------------------------------------------------------------------
# 64x256x256 - b64n256 variant 18 cta512_rec32_scalar_tc_all_refined_pad129
# ---------------------------------------------------------------------------

_CPP_SOURCE_B64N256 = r"""
#include <torch/extension.h>

void cholesky_b64n256_prepare();
at::Tensor cholesky_b64n256(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b64n256_prepare,
        "Configure batched 256x256 Cholesky dynamic shared memory");
  m.def("run", &cholesky_b64n256, "Batched 256x256 Cholesky");
}
"""

_CUDA_SOURCE_B64N256 = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 64;
constexpr int kN = 256;
constexpr int kTile = 128;
constexpr int kHalf = 64;
constexpr int kLd = 129;
constexpr int kThreads = 512;

// One CTA owns a whole 256x256 matrix. A00 and A11 are kLd-padded, and A10 is
// padded to kLd as well (variant 18's pad129 layout), which removes the
// shared-memory bank conflicts of the tight 128-column A10 panel.
constexpr int kA00 = 0;
constexpr int kA10 = kTile * kLd;
constexpr int kA11 = kA10 + kTile * kLd;
constexpr int kStorageFloats = kA11 + kTile * kLd;
constexpr int kTcScratchFloats = kHalf * kHalf;
constexpr int kTcBarrierFloats = 4;
constexpr int kSharedBytes =
    (kStorageFloats + kTcScratchFloats + kTcBarrierFloats) *
    static_cast<int>(sizeof(float));
constexpr uint32_t kTmemDp = 1u << 16;

__device__ __forceinline__ void root_pair(float value,
                                          float& diagonal,
                                          float& inverse) {
  // One Newton refinement of the hardware reciprocal square root.
  inverse = rsqrtf(value);
  inverse *= fmaf(-0.5f * value, inverse * inverse, 1.5f);
  diagonal = value * inverse;
}

__device__ __forceinline__ float& single_at(float* s, int row, int col) {
  if (row < kTile) {
    return s[kA00 + row * kLd + col];
  }
  if (col < kTile) {
    return s[kA10 + (row - kTile) * kLd + col];
  }
  return s[kA11 + (row - kTile) * kLd + col - kTile];
}

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void*>(pointer)));
}

__device__ __forceinline__ uint32_t to_tf32(float value) {
  uint32_t result;
  asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ int kmajor_offset(int row, int column, int rows) {
  return (row & 7) * 4 + (row >> 3) * 32 +
         (column & 3) + (column >> 2) * (rows * 4);
}

__device__ __forceinline__ uint64_t make_kmajor_descriptor(
    const void* pointer, int rows) {
  const uint64_t start =
      static_cast<uint64_t>(shared_address(pointer) >> 4) & 0x3fffull;
  const uint64_t leading = static_cast<uint64_t>(rows);
  const uint64_t stride = 8ull;
  return start | (leading << 16) | (stride << 32) | (1ull << 46);
}

template <int M, int N>
__device__ __forceinline__ constexpr uint32_t tf32_instruction_descriptor() {
  return (1u << 4) | (2u << 7) | (2u << 10) |
         (static_cast<uint32_t>(N >> 3) << 17) |
         (static_cast<uint32_t>(M >> 4) << 24);
}

__device__ __forceinline__ void proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tmem_allocate(uint32_t* destination,
                                              int columns) {
  if (static_cast<int>(threadIdx.x) < 32) {
    const uint32_t address = shared_address(destination);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
        "[%0], %1;" :: "r"(address), "r"(columns) : "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ void tmem_deallocate(uint32_t base, int columns) {
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < 32) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
        "r"(base), "r"(columns));
  }
  __syncthreads();
}

__device__ __forceinline__ void tmem_relinquish() {
  if (static_cast<int>(threadIdx.x) < 32) {
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier) {
  if (threadIdx.x == 0) {
    const uint32_t address = shared_address(barrier);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::
                 "r"(address) : "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ void tensor_commit(uint64_t* barrier) {
  if (threadIdx.x == 0) {
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
        "shared::cluster.b64 [%0];" :: "r"(address) : "memory");
  }
}

__device__ __forceinline__ void barrier_wait(uint64_t* barrier, int phase) {
  if (threadIdx.x == 0) {
    const uint32_t address = shared_address(barrier);
    const uint32_t ticks = 0x989680u;
    uint32_t complete;
    do {
      asm volatile(
          "{\n\t"
          ".reg .pred done;\n\t"
          "mbarrier.try_wait.parity.shared::cta.b64 done, "
          "[%1], %2, %3;\n\t"
          "selp.b32 %0, 1, 0, done;\n\t"
          "}\n"
          : "=r"(complete)
          : "r"(address), "r"(phase), "r"(ticks)
          : "memory");
    } while (complete == 0);
  }
  __syncthreads();
}

template <int M, int N>
__device__ __forceinline__ void issue_tf32_mma(
    uint32_t tmem_base, uint64_t a_desc, uint64_t b_desc, bool accumulate) {
  if (threadIdx.x == 0) {
    const uint32_t instruction = tf32_instruction_descriptor<M, N>();
    const uint32_t scale = accumulate ? 1u : 0u;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 "
        "[%0], %1, %2, %3, {%5,%6,%7,%8}, p;\n\t"
        "}\n" ::
        "r"(tmem_base), "l"(a_desc), "l"(b_desc), "r"(instruction),
        "r"(scale), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
  }
}

__device__ __forceinline__ float tmem_load_one(uint32_t address) {
  uint32_t value;
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
      : "=r"(value) : "r"(address));
  return __uint_as_float(value);
}

// Unblocked CTA-wide POTF2 over a diagonal block of the shared matrix.
__device__ __forceinline__ void potf2_single(float* s, int begin, int size) {
  for (int column = 0; column < size; ++column) {
    const int j = begin + column;
    if (threadIdx.x == 0) {
      float diagonal;
      float inverse;
      root_pair(single_at(s, j, j), diagonal, inverse);
      single_at(s, j, j) = diagonal;
      s[kStorageFloats - 1] = inverse;
    }
    __syncthreads();
    const float inverse = s[kStorageFloats - 1];
    for (int row = column + 1 + static_cast<int>(threadIdx.x);
         row < size; row += static_cast<int>(blockDim.x)) {
      single_at(s, begin + row, j) *= inverse;
    }
    __syncthreads();

    const int trailing = size - column - 1;
    const int pairs = trailing * (trailing + 1) / 2;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < pairs; linear += static_cast<int>(blockDim.x)) {
      int local_row = 0;
      int remainder = linear;
      for (int width = 1; remainder >= width; ++width) {
        remainder -= width;
        ++local_row;
      }
      const int row = j + 1 + local_row;
      const int col = j + 1 + remainder;
      single_at(s, row, col) =
          fmaf(-single_at(s, row, j), single_at(s, col, j),
               single_at(s, row, col));
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void trsm_single(
    float* s, int row_begin, int rows, int col_begin, int cols) {
  for (int local_row = static_cast<int>(threadIdx.x);
       local_row < rows; local_row += static_cast<int>(blockDim.x)) {
    const int row = row_begin + local_row;
    for (int local_col = 0; local_col < cols; ++local_col) {
      const int col = col_begin + local_col;
      float value = single_at(s, row, col);
      for (int k = 0; k < local_col; ++k) {
        value = fmaf(-single_at(s, row, col_begin + k),
                     single_at(s, col, col_begin + k), value);
      }
      single_at(s, row, col) = value / single_at(s, col, col);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void simt_update_single(
    float* s, int target, int size, int panel, int panel_cols) {
  constexpr int kMicro = 16;
  const int tile_count = (size + kMicro - 1) / kMicro;
  const int lower_tiles = tile_count * (tile_count + 1) / 2;
  for (int tile_linear = 0; tile_linear < lower_tiles; ++tile_linear) {
    int tile_row = 0;
    int tile_col = tile_linear;
    while (tile_col > tile_row) {
      tile_col -= tile_row + 1;
      ++tile_row;
    }
    for (int element = static_cast<int>(threadIdx.x);
         element < kMicro * kMicro;
         element += static_cast<int>(blockDim.x)) {
      const int local_row = tile_row * kMicro + element / kMicro;
      const int local_col = tile_col * kMicro + element % kMicro;
      if (local_row < size && local_col < size && local_col <= local_row) {
        float value = single_at(s, target + local_row, target + local_col);
#pragma unroll 4
        for (int k = 0; k < panel_cols; ++k) {
          value = fmaf(
              -single_at(s, target + local_row, panel + k),
              single_at(s, target + local_col, panel + k), value);
        }
        single_at(s, target + local_row, target + local_col) = value;
      }
    }
  }
  __syncthreads();
}

// Trailing SYRK through one tcgen05 TF32 MMA per eight-deep k slice.
template <int M>
__device__ __forceinline__ void tc_update_single(
    float* s, int target, int panel, float* scratch,
    uint32_t* tmem_slot, uint64_t* barrier, int& phase) {
  tmem_allocate(tmem_slot, kTile);
  const uint32_t tmem_base = *tmem_slot;
  for (int k = 0; k < M; k += 8) {
    for (int linear = static_cast<int>(threadIdx.x);
         linear < M * 8; linear += static_cast<int>(blockDim.x)) {
      const int row = linear >> 3;
      const int column = linear & 7;
      const int packed = kmajor_offset(row, column, M);
      reinterpret_cast<uint32_t*>(scratch)[packed] =
          to_tf32(single_at(s, target + row, panel + k + column));
    }
    __syncthreads();
    proxy_fence();
    __syncthreads();
    const uint64_t descriptor = make_kmajor_descriptor(scratch, M);
    issue_tf32_mma<M, M>(tmem_base, descriptor, descriptor, k != 0);
    tensor_commit(barrier);
    barrier_wait(barrier, phase);
    phase ^= 1;
  }

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if constexpr (M == 128) {
    if (warp < 4) {
      const int row = warp * 32 + lane;
      for (int col = 0; col < M; ++col) {
        const uint32_t address =
            tmem_base + static_cast<uint32_t>(warp * 32) * kTmemDp +
            static_cast<uint32_t>(col);
        const float product = tmem_load_one(address);
        if (col <= row) {
          single_at(s, target + row, target + col) -= product;
        }
      }
    }
  } else {
    if (warp < 4) {
      const int row = warp * 16 + lane;
      for (int col = 0; col < M; ++col) {
        const uint32_t address =
            tmem_base + static_cast<uint32_t>(warp * 32) * kTmemDp +
            static_cast<uint32_t>(col);
        const float product = tmem_load_one(address);
        if (lane < 16 && col <= row) {
          single_at(s, target + row, target + col) -= product;
        }
      }
    }
  }
  __syncthreads();
  tmem_deallocate(tmem_base, kTile);
}

// 128x128 diagonal block: recursive 32/32 base factors with SIMT updates and
// one TF32 tensor-core update for the 64-wide trailing block.
__device__ __forceinline__ void potrf128_single(
    float* s, int begin, float* scratch, uint32_t* tmem_slot,
    uint64_t* barrier, int& phase) {
  potf2_single(s, begin, 32);
  trsm_single(s, begin + 32, 32, begin, 32);
  simt_update_single(s, begin + 32, 32, begin, 32);
  potf2_single(s, begin + 32, 32);

  trsm_single(s, begin + 64, 64, begin, 64);
  tc_update_single<64>(
      s, begin + 64, begin, scratch, tmem_slot, barrier, phase);

  potf2_single(s, begin + 64, 32);
  trsm_single(s, begin + 96, 32, begin + 64, 32);
  simt_update_single(s, begin + 96, 32, begin + 64, 32);
  potf2_single(s, begin + 96, 32);
}

__global__ __launch_bounds__(kThreads, 1)
void single_kernel(const float* __restrict__ input,
                   float* __restrict__ output) {
  extern __shared__ __align__(16) float storage[];
  float* scratch = storage + kStorageFloats;
  uint32_t* tmem_slot =
      reinterpret_cast<uint32_t*>(scratch + kTcScratchFloats);
  uint64_t* barrier =
      reinterpret_cast<uint64_t*>(scratch + kTcScratchFloats + 2);
  int phase = 0;

  const int matrix = static_cast<int>(blockIdx.x);
  const float* matrix_input = input + static_cast<int64_t>(matrix) * kN * kN;
  float* matrix_output = output + static_cast<int64_t>(matrix) * kN * kN;

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kN * kN; linear += static_cast<int>(blockDim.x)) {
    matrix_output[linear] = 0.0f;
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      single_at(storage, row, col) = matrix_input[row * kN + col];
      single_at(storage, row + kTile, col + kTile) =
          matrix_input[(row + kTile) * kN + col + kTile];
    }
    single_at(storage, row + kTile, col) =
        matrix_input[(row + kTile) * kN + col];
  }
  __syncthreads();
  barrier_init(barrier);

  potrf128_single(storage, 0, scratch, tmem_slot, barrier, phase);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      matrix_output[row * kN + col] = single_at(storage, row, col);
    }
  }
  __syncthreads();

  trsm_single(storage, kTile, kTile, 0, kTile);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    matrix_output[(row + kTile) * kN + col] =
        single_at(storage, row + kTile, col);
  }
  __syncthreads();

  tc_update_single<128>(storage, kTile, 0, scratch, tmem_slot, barrier, phase);
  potrf128_single(storage, kTile, scratch, tmem_slot, barrier, phase);

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      matrix_output[(row + kTile) * kN + col + kTile] =
          single_at(storage, row + kTile, col + kTile);
    }
  }
  __syncthreads();
  tmem_relinquish();
}

}  // namespace

void cholesky_b64n256_prepare() {
  auto status = cudaFuncSetAttribute(
      single_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  TORCH_CHECK(status == cudaSuccess,
              "dynamic shared-memory opt-in failed: ",
              cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      single_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(status == cudaSuccess,
              "shared-memory carveout failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b64n256(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (64, 256, 256)");
  auto out = at::empty_like(data);
  single_kernel<<<kBatch, kThreads, kSharedBytes>>>(
      data.data_ptr<float>(), out.data_ptr<float>());
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return out;
}
"""


@lru_cache(maxsize=1)
def _module_b64n256():
    module = _build(
        "cholesky_b64n256", _CPP_SOURCE_B64N256, _CUDA_SOURCE_B64N256,
        extra_cuda_flags=("--restrict",))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SHAPE = (64, 256, 256)


def custom_kernel(data: input_t) -> output_t:
    if (
        data.is_cuda
        and data.dtype == torch.float32
        and data.is_contiguous()
        and tuple(data.shape) == _SHAPE
    ):
        return _module_b64n256().run(data)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
