"""Popcorn submission for the (1, 8192, 8192) batched Cholesky shape.

Extracted verbatim from the combined cholesky/cholesky.py fold of
cholesky/b1n8192/cholesky_b1n8192.py variant 8,
`ll_nb512_m64_microfused_split2_tf32`.

Left-looking 512-column panels with a fused producer/consumer 64-wide
micro block; the inverse application of each 64x64 tile is split across
two consumer CTAs.

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
# 1x8192x8192 - b1n8192 variant 8 ll_nb512_m64_microfused_split2_tf32
# ---------------------------------------------------------------------------

_CPP_SOURCE_B1N8192 = r"""
#include <torch/extension.h>

void cholesky_b1n8192_prepare();
at::Tensor cholesky_b1n8192(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b1n8192_prepare,
        "Configure the fused 8192 Cholesky kernel");
  m.def("run", &cholesky_b1n8192, "Single 8192 Cholesky");
}
"""

_CUDA_SOURCE_B1N8192 = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kN = 8192;
constexpr int kNb = 512;
constexpr int kMicro = 64;
constexpr int kTileLd = kMicro + 1;
constexpr int kPanelLd = 9;
constexpr int kThreads = 256;
constexpr int kConsumerSplit = 2;
constexpr int kFactorBytes =
    static_cast<int>(sizeof(float)) *
    (2 * kMicro * kTileLd + kMicro + kMicro * kPanelLd + 32 * 32);
static_assert(kFactorBytes == 39936);

__device__ __forceinline__ int64_t matrix_index(
    int row, int column) {
  return static_cast<int64_t>(row) * kN + column;
}

__device__ __forceinline__ float load_global(const float* pointer) {
  return __ldcg(pointer);
}

__device__ __forceinline__ void store_global(
    float* pointer, float value) {
  __stcg(pointer, value);
}

__device__ __forceinline__ float& tile_at(
    float* tile, int row, int column) {
  return tile[row * kTileLd + column];
}

// Eight-column redundant-corner factorization. Four threads share
// each row solve and split its rank-8 trailing update.
// corner_sm and inverse are in shared memory to cut register pressure.
__device__ __forceinline__ void factor_wide(
    float* __restrict__ tile,
    float* __restrict__ inverse_diagonal,
    float* __restrict__ panel,
    float* __restrict__ corner_sm) {
  constexpr int kGroup = 8;
  const int thread = static_cast<int>(threadIdx.x);
  const int row_index = thread >> 2;
  const int quarter = thread & 3;
  float* inverse_sm = corner_sm + kGroup * kGroup;
#pragma unroll 1
  for (int base = 0; base < kMicro; base += kGroup) {
    if (thread < kGroup) {
#pragma unroll
      for (int i = thread; i < kGroup; ++i) {
        corner_sm[i * kGroup + thread] =
            tile_at(tile, base + i, base + thread);
      }
    }
    __syncthreads();
    if (thread == 0) {
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        const float diagonal =
            __fsqrt_rn(corner_sm[j * kGroup + j]);
        const float inv = __fdiv_rn(1.0f, diagonal);
        corner_sm[j * kGroup + j] = diagonal;
        inverse_sm[j] = inv;
#pragma unroll
        for (int i = j + 1; i < kGroup; ++i) {
          corner_sm[i * kGroup + j] *= inv;
        }
#pragma unroll
        for (int i = j + 1; i < kGroup; ++i) {
#pragma unroll
          for (int target = j + 1; target <= i; ++target) {
            corner_sm[i * kGroup + target] = fmaf(
                -corner_sm[i * kGroup + j],
                corner_sm[target * kGroup + j],
                corner_sm[i * kGroup + target]);
          }
        }
      }
    }
    __syncthreads();
    if (thread < kGroup) {
      inverse_diagonal[base + thread] = inverse_sm[thread];
#pragma unroll
      for (int i = thread; i < kGroup; ++i) {
        tile_at(tile, base + i, base + thread) =
            corner_sm[i * kGroup + thread];
      }
    }
    const int row = base + kGroup + row_index;
    float solved[kGroup];
    if (row < kMicro) {
#pragma unroll
      for (int k = 0; k < kGroup; ++k) {
        solved[k] = tile_at(tile, row, base + k);
      }
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        float value = solved[j];
#pragma unroll
        for (int i = 0; i < j; ++i) {
          value = fmaf(
              -solved[i], corner_sm[j * kGroup + i], value);
        }
        solved[j] = value * inverse_sm[j];
      }
      if (quarter == 0) {
#pragma unroll
        for (int k = 0; k < kGroup; ++k) {
          tile_at(tile, row, base + k) = solved[k];
          panel[row * kPanelLd + k] = solved[k];
        }
      }
    }
    __syncthreads();
    if (row < kMicro) {
      const int first = base + kGroup;
      for (int target = first + quarter * 4; target <= row;
           target += 16) {
        if (target + 3 <= row) {
          float value0 = tile_at(tile, row, target);
          float value1 = tile_at(tile, row, target + 1);
          float value2 = tile_at(tile, row, target + 2);
          float value3 = tile_at(tile, row, target + 3);
#pragma unroll
          for (int k = 0; k < kGroup; ++k) {
            const float left = solved[k];
            value0 = fmaf(
                -left, panel[target * kPanelLd + k], value0);
            value1 = fmaf(
                -left, panel[(target + 1) * kPanelLd + k], value1);
            value2 = fmaf(
                -left, panel[(target + 2) * kPanelLd + k], value2);
            value3 = fmaf(
                -left, panel[(target + 3) * kPanelLd + k], value3);
          }
          tile_at(tile, row, target) = value0;
          tile_at(tile, row, target + 1) = value1;
          tile_at(tile, row, target + 2) = value2;
          tile_at(tile, row, target + 3) = value3;
        } else {
          for (int single = target; single <= row; ++single) {
            float value = tile_at(tile, row, single);
#pragma unroll
            for (int k = 0; k < kGroup; ++k) {
              value = fmaf(
                  -solved[k], panel[single * kPanelLd + k], value);
            }
            tile_at(tile, row, single) = value;
          }
        }
      }
    }
    __syncthreads();
  }
}

// Invert the two 32-wide diagonal blocks, then combine them into the
// 64-wide triangular inverse the consumers apply.
__device__ __forceinline__ void build_inverse(
    const float* tile, const float* inverse_diagonal,
    float* tinv, float* mid) {
  const int thread = static_cast<int>(threadIdx.x);
  for (int linear = thread; linear < kMicro * kTileLd;
       linear += static_cast<int>(blockDim.x)) {
    tinv[linear] = 0.0f;
  }
  __syncthreads();
  const int warp = thread >> 5;
  const int lane = thread & 31;
  if (warp < kMicro / 32) {
    const int base = warp * 32;
    const int column = base + lane;
    tinv[column * kTileLd + column] = inverse_diagonal[column];
    for (int row = lane + 1; row < 32; ++row) {
      const int target = base + row;
      float partial = 0.0f;
      for (int k = lane; k < row; ++k) {
        partial = fmaf(
            tile[target * kTileLd + base + k],
            tinv[(base + k) * kTileLd + column], partial);
      }
      tinv[target * kTileLd + column] =
          -partial * inverse_diagonal[target];
    }
  }
  __syncthreads();
  for (int linear = thread; linear < 32 * 32;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 5;
    const int column = linear & 31;
    float partial = 0.0f;
#pragma unroll 4
    for (int k = column; k < 32; ++k) {
      partial = fmaf(
          tile[(32 + row) * kTileLd + k],
          tinv[k * kTileLd + column], partial);
    }
    mid[row * 32 + column] = partial;
  }
  __syncthreads();
  for (int linear = thread; linear < 32 * 32;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 5;
    const int column = linear & 31;
    float partial = 0.0f;
#pragma unroll 4
    for (int k = 0; k <= row; ++k) {
      partial = fmaf(
          tinv[(32 + row) * kTileLd + 32 + k],
          mid[k * 32 + column], partial);
    }
    tinv[(32 + row) * kTileLd + column] = -partial;
  }
  __syncthreads();
}

__device__ __forceinline__ void publish_flag(int* flag) {
  asm volatile(
      "st.release.gpu.global.u32 [%0], %1;"
      :: "l"(flag), "r"(1) : "memory");
}

__device__ __forceinline__ int poll_flag(const int* flag) {
  int value;
  asm volatile(
      "ld.global.relaxed.gpu.L1::no_allocate.u32 %0, [%1];"
      : "=r"(value) : "l"(flag));
  return value;
}

__device__ __forceinline__ void acquire_fence() {
  asm volatile("fence.acquire.gpu;" ::: "memory");
}

__device__ __forceinline__ void load_x_tile(
    float* x_tile, const float* output, int begin, int tile_index) {
  const int row_begin = begin + kMicro + tile_index * kMicro;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kMicro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kMicro;
    const int column = linear & (kMicro - 1);
    x_tile[row * kTileLd + column] = load_global(
        output + matrix_index(row_begin + row, begin + column));
  }
}

// Each 64x64 application is split across two CTAs. Both halves use all
// eight warps: four 16-row warp bands by two 16-column stripes.
__device__ __forceinline__ void apply_tile_half(
    const float* __restrict__ x_tile,
    const float* __restrict__ t_tile,
    float* __restrict__ output, int begin, int tile_index,
    int half) {
  constexpr int kStripe = 16;
  constexpr int kThreadColumns = 2;
  const int row_begin = begin + kMicro + tile_index * kMicro;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_row = warp >> 1;
  const int local_warp_column = warp & 1;
  const int warp_column = half * 2 + local_warp_column;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;
  const int k_limit = (warp_column + 1) * kStripe;
  float value[4][kThreadColumns];
#pragma unroll
  for (int row = 0; row < 4; ++row) {
#pragma unroll
    for (int column = 0; column < kThreadColumns; ++column) {
      value[row][column] = 0.0f;
    }
  }
#pragma unroll 1
  for (int k = 0; k < k_limit; ++k) {
    float left[4];
    float right[kThreadColumns];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      left[row] = x_tile[
          (warp_row * 16 + lane_row + row * 4) * kTileLd + k];
    }
#pragma unroll
    for (int column = 0; column < kThreadColumns; ++column) {
      const int t_row =
          warp_column * kStripe + lane_column + column * 8;
      right[column] = t_tile[t_row * kTileLd + k];
    }
#pragma unroll
    for (int row = 0; row < 4; ++row) {
#pragma unroll
      for (int column = 0; column < kThreadColumns; ++column) {
        value[row][column] = fmaf(
            left[row], right[column], value[row][column]);
      }
    }
  }
#pragma unroll
  for (int row = 0; row < 4; ++row) {
#pragma unroll
    for (int column = 0; column < kThreadColumns; ++column) {
      const int output_row =
          warp_row * 16 + lane_row + row * 4;
      const int output_column =
          warp_column * kStripe + lane_column + column * 8;
      store_global(
          output +
              matrix_index(
                  row_begin + output_row,
                  begin + output_column),
          value[row][column]);
    }
  }
}

// CTA 0 factors the 64-wide micro block and publishes its inverse;
// every other CTA preloads its X tile, waits on the release flag, and
// applies the inverse to one half of one 64x64 tile.
__global__ __launch_bounds__(kThreads)
void fused_micro_kernel(
    float* __restrict__ output, int begin,
    float* __restrict__ t_inv, int* __restrict__ flags) {
  extern __shared__ __align__(16) float dynamic_floats[];
  const int tiles = (kN - begin - kMicro) / kMicro;
  int* flag = flags + begin / kMicro;
  if (blockIdx.x == 0) {
    float* tile = dynamic_floats;
    float* inverse_diagonal = tile + kMicro * kTileLd;
    float* panel = inverse_diagonal + kMicro;
    float* tinv = panel + kMicro * kPanelLd;
    float* mid = tinv + kMicro * kTileLd;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kMicro * kMicro;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear / kMicro;
      const int column = linear & (kMicro - 1);
      tile_at(tile, row, column) =
          column <= row
              ? load_global(
                    output +
                    matrix_index(begin + row, begin + column))
              : 0.0f;
    }
    __syncthreads();
    factor_wide(tile, inverse_diagonal, panel, mid);
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kMicro * kMicro;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear / kMicro;
      const int column = linear & (kMicro - 1);
      if (column <= row) {
        store_global(
            output + matrix_index(begin + row, begin + column),
            tile_at(tile, row, column));
      }
    }
    if (tiles > 0) {
      build_inverse(tile, inverse_diagonal, tinv, mid);
      for (int linear = static_cast<int>(threadIdx.x);
           linear < kMicro * kMicro;
           linear += static_cast<int>(blockDim.x)) {
        const int row = linear / kMicro;
        const int column = linear & (kMicro - 1);
        store_global(t_inv + linear, tinv[row * kTileLd + column]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        publish_flag(flag);
      }
    }
  } else {
    float* x_tile = dynamic_floats;
    float* t_tile = x_tile + kMicro * kTileLd;
    const int consumer = static_cast<int>(blockIdx.x) - 1;
    const int consumer_count = static_cast<int>(gridDim.x) - 1;
    const int part = consumer % kConsumerSplit;
    const int consumer_stride = consumer_count / kConsumerSplit;
    int tile_index = consumer / kConsumerSplit;
    load_x_tile(x_tile, output, begin, tile_index);
    if (threadIdx.x == 0) {
      while (poll_flag(flag) == 0) {
        __nanosleep(64);
      }
      acquire_fence();
    }
    __syncthreads();
    constexpr int kRowsPerConsumer = kMicro / kConsumerSplit;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kRowsPerConsumer * kMicro;
         linear += static_cast<int>(blockDim.x)) {
      const int local_row = linear / kMicro;
      const int row = part * kRowsPerConsumer + local_row;
      const int column = linear & (kMicro - 1);
      t_tile[row * kTileLd + column] =
          load_global(t_inv + row * kMicro + column);
    }
    __syncthreads();
    while (true) {
      apply_tile_half(
          x_tile, t_tile, output, begin, tile_index, part);
      tile_index += consumer_stride;
      if (tile_index >= tiles) {
        break;
      }
      __syncthreads();
      load_x_tile(x_tile, output, begin, tile_index);
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(256)
void copy_lower_kernel(
    const float* __restrict__ input,
    float* __restrict__ output) {
  constexpr int64_t quads = static_cast<int64_t>(kN) * kN / 4;
  constexpr int quads_per_row = kN / 4;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t quad = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       quad < quads; quad += stride) {
    const int row = static_cast<int>(quad / quads_per_row);
    const int column =
        static_cast<int>(quad % quads_per_row) * 4;
    const float4* source =
        reinterpret_cast<const float4*>(input) + quad;
    float4 value;
    if (column + 3 <= row) {
      value = __ldcg(source);
    } else if (column > row) {
      value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
      const float4 loaded = __ldcg(source);
      value.x = loaded.x;
      value.y = column + 1 <= row ? loaded.y : 0.0f;
      value.z = column + 2 <= row ? loaded.z : 0.0f;
      value.w = 0.0f;
    }
    __stcg(reinterpret_cast<float4*>(output) + quad, value);
  }
}

__global__ __launch_bounds__(256)
void zero_wedges_kernel(float* __restrict__ output) {
  constexpr int ctas_per_block = 8;
  constexpr int shift = 9;
  static_assert((1 << shift) == kNb);
  const int block = static_cast<int>(blockIdx.x) / ctas_per_block;
  const int rank = static_cast<int>(blockIdx.x) % ctas_per_block;
  const int base = block * kNb;
  constexpr int64_t elements = static_cast<int64_t>(kNb) * kNb;
  for (int64_t linear =
           static_cast<int64_t>(rank) * blockDim.x + threadIdx.x;
       linear < elements;
       linear +=
       static_cast<int64_t>(ctas_per_block) * blockDim.x) {
    const int row = static_cast<int>(linear >> shift);
    const int column = static_cast<int>(linear & (kNb - 1));
    if (column > row) {
      store_global(
          output + matrix_index(base + row, base + column), 0.0f);
    }
  }
}

void check_cublas(cublasStatus_t status, const char* role) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      role, " failed with cuBLAS status ", static_cast<int>(status));
}

class CublasStateGuard {
 public:
  explicit CublasStateGuard(cublasHandle_t handle)
      : handle_(handle) {
    check_cublas(
        cublasGetMathMode(handle_, &math_mode_),
        "query cuBLAS math mode");
    check_cublas(
        cublasGetAtomicsMode(handle_, &atomics_mode_),
        "query cuBLAS atomics mode");
    check_cublas(
        cublasGetPointerMode(handle_, &pointer_mode_),
        "query cuBLAS pointer mode");
    check_cublas(
        cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH),
        "select cuBLAS math mode");
    check_cublas(
        cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_ALLOWED),
        "enable cuBLAS atomic algorithms");
    check_cublas(
        cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
        "select host cuBLAS scalars");
  }

  ~CublasStateGuard() {
    cublasSetPointerMode(handle_, pointer_mode_);
    cublasSetAtomicsMode(handle_, atomics_mode_);
    cublasSetMathMode(handle_, math_mode_);
  }

  CublasStateGuard(const CublasStateGuard&) = delete;
  CublasStateGuard& operator=(const CublasStateGuard&) = delete;

 private:
  cublasHandle_t handle_;
  cublasMath_t math_mode_{};
  cublasAtomicsMode_t atomics_mode_{};
  cublasPointerMode_t pointer_mode_{};
};

void gemm_history(
    cublasHandle_t handle, float* output, int64_t panel_begin) {
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const int columns = static_cast<int>(kN - panel_begin);
  const int history = static_cast<int>(panel_begin);
  const float* panel_rows = output + panel_begin * kN;
  float* destination = output + panel_begin * kN + panel_begin;
  check_cublas(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          kNb, columns, history,
          &alpha,
          panel_rows, CUDA_R_32F, kN,
          panel_rows, CUDA_R_32F, kN,
          &beta,
          destination, CUDA_R_32F, kN,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "panel history GEMM");
}

void gemm_inner(
    cublasHandle_t handle, float* output,
    int64_t panel_begin, int64_t micro_begin) {
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const int columns = static_cast<int>(kN - micro_begin);
  const int history = static_cast<int>(micro_begin - panel_begin);
  const float* micro_rows =
      output + micro_begin * kN + panel_begin;
  float* destination = output + micro_begin * kN + micro_begin;
  check_cublas(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          kMicro, columns, history,
          &alpha,
          micro_rows, CUDA_R_32F, kN,
          micro_rows, CUDA_R_32F, kN,
          &beta,
          destination, CUDA_R_32F, kN,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "micro history GEMM");
}

void launch_copy(const float* input, float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(512, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, copy_lower_kernel, input, output);
}

int fused_micro_grid_limit() {
  static const int limit = [] {
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    TORCH_CHECK(
        status == cudaSuccess,
        "device query failed: ", cudaGetErrorString(status));
    int sm_count = 0;
    status = cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, device);
    TORCH_CHECK(
        status == cudaSuccess,
        "SM count query failed: ", cudaGetErrorString(status));
    int active = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, fused_micro_kernel, kThreads, kFactorBytes);
    TORCH_CHECK(
        status == cudaSuccess,
        "fused occupancy query failed: ", cudaGetErrorString(status));
    TORCH_CHECK(
        active >= 1 && sm_count >= 2,
        "fused micro kernel must be co-resident with consumers");
    return sm_count * active;
  }();
  return limit;
}

void launch_fused_micro(
    float* output, int begin, float* t_inv, int* flags) {
  const int tiles = (kN - begin - kMicro) / kMicro;
  const int limit = fused_micro_grid_limit();
  const int jobs = tiles * kConsumerSplit;
  int consumers = jobs < limit - 1 ? jobs : limit - 1;
  consumers -= consumers % kConsumerSplit;
  const int grid = 1 + consumers;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kFactorBytes;
  cudaLaunchKernelEx(
      &config, fused_micro_kernel, output, begin, t_inv, flags);
}

void launch_wedges(float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((kN / kNb) * 8, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, zero_wedges_kernel, output);
}

// Left-looking 512-column panels; inside each panel the trailing
// history is folded in before every fused 64-wide micro block.
void launch_staged(
    float* output, const float* input, float* t_inv, int* flags) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle);
  launch_copy(input, output);
  for (int64_t panel = 0; panel < kN; panel += kNb) {
    if (panel > 0) {
      gemm_history(handle, output, panel);
    }
    for (int64_t micro = panel; micro < panel + kNb;
         micro += kMicro) {
      if (micro > panel) {
        gemm_inner(handle, output, panel, micro);
      }
      launch_fused_micro(
          output, static_cast<int>(micro), t_inv, flags);
    }
  }
  launch_wedges(output);
}

template <typename Kernel>
void configure_dynamic(Kernel kernel, int dynamic_bytes) {
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      dynamic_bytes);
  TORCH_CHECK(
      status == cudaSuccess,
      "dynamic shared-memory opt-in failed: ",
      cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(
      status == cudaSuccess,
      "shared-memory carveout failed: ", cudaGetErrorString(status));
}

}  // namespace

void cholesky_b1n8192_prepare() {
  configure_dynamic(fused_micro_kernel, kFactorBytes);
  TORCH_CHECK(
      fused_micro_grid_limit() >= 1 + kConsumerSplit,
      "fused micro kernel needs a consumer CTA");
}

at::Tensor cholesky_b1n8192(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == 1 &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (1, 8192, 8192)");
  c10::cuda::CUDAGuard device_guard(data.device());
  auto output = at::empty_like(data);
  at::Tensor t_inv = at::empty(
      {static_cast<int64_t>(kMicro) * kMicro}, data.options());
  at::Tensor flags = at::zeros(
      {kN / kMicro}, data.options().dtype(at::kInt));
  launch_staged(
      output.data_ptr<float>(), data.data_ptr<float>(),
      t_inv.data_ptr<float>(), flags.data_ptr<int>());
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return output;
}
"""


@lru_cache(maxsize=1)
def _module_b1n8192():
    module = _build(
        "cholesky_b1n8192", _CPP_SOURCE_B1N8192, _CUDA_SOURCE_B1N8192,
        extra_cuda_flags=("-DNDEBUG", "--restrict"),
        extra_ldflags=("-lcublas",))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SHAPE = (1, 8192, 8192)


def custom_kernel(data: input_t) -> output_t:
    if (
        data.is_cuda
        and data.dtype == torch.float32
        and data.is_contiguous()
        and tuple(data.shape) == _SHAPE
    ):
        return _module_b1n8192().run(data)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
