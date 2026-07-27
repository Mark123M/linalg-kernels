"""Popcorn submission for the (16, 512, 512) batched Cholesky shape.

Extracted verbatim from the combined cholesky/cholesky.py fold of
cholesky/b16n512/cholesky_b16n512.py variant 2,
`r16_micro4x4_raw_scalar_u256`.

Staged 64x64 panel factorization across separate factor/solve/update
launches with a 4x4 micro-tiled FP32 update.

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
# 16x512x512 - b16n512 variant 2 r16_micro4x4_raw_scalar_u256
# ---------------------------------------------------------------------------

_CPP_SOURCE_B16N512 = r"""
#include <torch/extension.h>

void cholesky_b16n512_prepare();
at::Tensor cholesky_b16n512(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b16n512_prepare,
        "Configure staged 512x512 Cholesky kernels");
  m.def("run", &cholesky_b16n512, "Batched 512x512 Cholesky");
}
"""

_CUDA_SOURCE_B16N512 = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 16;
constexpr int kN = 512;
constexpr int kTile = 64;
constexpr int kTileCount = 8;
constexpr int kLd = 65;
constexpr int kFactorThreads = 128;
constexpr int kSolveThreads = 128;
constexpr int kUpdateThreads = 256;

__device__ __forceinline__ float load_global(const float* pointer) {
  return __ldcg(pointer);
}

__device__ __forceinline__ void store_global(float* pointer, float value) {
  __stcg(pointer, value);
}

__device__ __forceinline__ void root_pair(
    float value, float& diagonal, float& inverse) {
  inverse = rsqrtf(value);
  diagonal = value * inverse;
}

__device__ __forceinline__ float& tile_at(float* tile, int row, int column) {
  return tile[row * kLd + column];
}

// Unblocked 16x16 right-looking factor executed by warp 0.
__device__ __forceinline__ void factor16(
    float* tile, float* inverse_diagonal, int begin) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (warp == 0) {
#pragma unroll
    for (int local_column = 0; local_column < 16; ++local_column) {
      const int column = begin + local_column;
      float inverse = 0.0f;
      if (lane == local_column) {
        float diagonal;
        root_pair(tile_at(tile, column, column), diagonal, inverse);
        tile_at(tile, column, column) = diagonal;
        inverse_diagonal[column] = inverse;
      }
      inverse = __shfl_sync(0xffffffffu, inverse, local_column);
      if (lane > local_column && lane < 16) {
        const int row = begin + lane;
        tile_at(tile, row, column) *= inverse;
      }
      __syncwarp();
      if (lane > local_column && lane < 16) {
        const int row = begin + lane;
        const float left = tile_at(tile, row, column);
#pragma unroll
        for (int local_target = local_column + 1;
             local_target < 16; ++local_target) {
          if (local_target <= lane) {
            const int target = begin + local_target;
            tile_at(tile, row, target) = fmaf(
                -left, tile_at(tile, target, column),
                tile_at(tile, row, target));
          }
        }
      }
      __syncwarp();
    }
  }
}

// Four lanes cooperate on each row of the triangular solve.
template <int Rows, int Columns>
__device__ __forceinline__ void local_trsm_sub4(
    float* tile, const float* inverse_diagonal,
    int row_begin, int column_begin) {
  const int lane = static_cast<int>(threadIdx.x) & 3;
  const int row_index = static_cast<int>(threadIdx.x) >> 2;
  if (row_index < Rows) {
    const int row = row_begin + row_index;
#pragma unroll
    for (int local_column = 0; local_column < Columns; ++local_column) {
      const int column = column_begin + local_column;
      float partial = 0.0f;
#pragma unroll
      for (int k = lane; k < local_column; k += 4) {
        partial = fmaf(
            tile_at(tile, row, column_begin + k),
            tile_at(tile, column, column_begin + k), partial);
      }
      partial += __shfl_down_sync(0xffffffffu, partial, 2, 4);
      partial += __shfl_down_sync(0xffffffffu, partial, 1, 4);
      if (lane == 0) {
        tile_at(tile, row, column) =
            (tile_at(tile, row, column) - partial) *
            inverse_diagonal[column];
      }
      __syncwarp();
    }
  }
}

__device__ __forceinline__ void local_update16(
    float* tile, int target, int panel) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = (warp >> 1) * 8 + (lane >> 2);
  const int column0 = (warp & 1) * 8 + (lane & 3);
  const int column1 = column0 + 4;
  float product0 = 0.0f;
  float product1 = 0.0f;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const float left = tile_at(tile, target + row, panel + k);
    product0 = fmaf(
        left, tile_at(tile, target + column0, panel + k), product0);
    product1 = fmaf(
        left, tile_at(tile, target + column1, panel + k), product1);
  }
  if (column0 <= row) {
    tile_at(tile, target + row, target + column0) -= product0;
  }
  if (column1 <= row) {
    tile_at(tile, target + row, target + column1) -= product1;
  }
}

__device__ __forceinline__ void local_update32(
    float* tile, int target, int panel) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = (warp >> 1) * 16;
  const int column_base = (warp & 1) * 16;
  const int lane_row = lane >> 2;
  const int lane_column = lane & 3;
  float product[2][4] = {};
#pragma unroll
  for (int k = 0; k < 32; ++k) {
    const float left0 =
        tile_at(tile, target + row_base + lane_row, panel + k);
    const float left1 =
        tile_at(tile, target + row_base + lane_row + 8, panel + k);
    float right[4];
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      right[column] = tile_at(
          tile, target + column_base + lane_column + column * 4, panel + k);
    }
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      product[0][column] = fmaf(left0, right[column], product[0][column]);
      product[1][column] = fmaf(left1, right[column], product[1][column]);
    }
  }
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    const int output_row = row_base + lane_row + row * 8;
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_column = column_base + lane_column + column * 4;
      if (output_column <= output_row) {
        tile_at(tile, target + output_row, target + output_column) -=
            product[row][column];
      }
    }
  }
}

__device__ __forceinline__ void factor32_recursive16(
    float* tile, float* inverse_diagonal, int begin) {
  factor16(tile, inverse_diagonal, begin);
  __syncthreads();
  local_trsm_sub4<16, 16>(tile, inverse_diagonal, begin + 16, begin);
  __syncthreads();
  local_update16(tile, begin + 16, begin);
  __syncthreads();
  factor16(tile, inverse_diagonal, begin + 16);
}

// Stage 0: copy the lower triangle of the input into the output buffer, which
// the later stages then factor in place.
__global__ __launch_bounds__(256)
void copy_lower_kernel(const float* __restrict__ input,
                       float* __restrict__ output) {
  constexpr int kCtasPerMatrix = 8;
  const int matrix_index = static_cast<int>(blockIdx.x) / kCtasPerMatrix;
  const int rank = static_cast<int>(blockIdx.x) % kCtasPerMatrix;
  const int64_t base = static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear = rank * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
       linear < kN * kN;
       linear += kCtasPerMatrix * static_cast<int>(blockDim.x)) {
    const int row = linear / kN;
    const int column = linear % kN;
    store_global(output + base + linear,
                 column <= row ? input[base + linear] : 0.0f);
  }
}

// Stage 1: factor the 64x64 diagonal block of one panel.
__global__ __launch_bounds__(kFactorThreads)
void factor_kernel(float* __restrict__ output, int panel) {
  __shared__ __align__(128) float tile[kTile * kLd];
  __shared__ float inverse_diagonal[kTile];
  const int matrix_index = static_cast<int>(blockIdx.x);
  float* matrix = output + static_cast<int64_t>(matrix_index) * kN * kN;
  const int begin = panel * kTile;

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    tile_at(tile, row, column) =
        column <= row
            ? load_global(matrix + (begin + row) * kN + begin + column)
            : 0.0f;
  }
  __syncthreads();

  factor32_recursive16(tile, inverse_diagonal, 0);
  __syncthreads();

  local_trsm_sub4<32, 32>(tile, inverse_diagonal, 32, 0);
  __syncthreads();

  local_update32(tile, 32, 0);
  __syncthreads();

  factor32_recursive16(tile, inverse_diagonal, 32);
  __syncthreads();

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    if (column <= row) {
      store_global(matrix + (begin + row) * kN + begin + column,
                   tile_at(tile, row, column));
    }
  }
}

// Stage 2: triangular solve of every remaining row tile against the panel.
__global__ void solve_kernel(
    float* __restrict__ output, int panel, int remaining) {
  __shared__ __align__(128) float diagonal[kTile * kLd];
  __shared__ __align__(128) float rhs[kTile * kLd];
  __shared__ float inverse_diagonal[kTile];

  const int matrix_index = static_cast<int>(blockIdx.x) / remaining;
  const int row_tile = panel + 1 + static_cast<int>(blockIdx.x) % remaining;
  float* matrix = output + static_cast<int64_t>(matrix_index) * kN * kN;
  const int panel_begin = panel * kTile;
  const int row_begin = row_tile * kTile;

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    diagonal[row * kLd + column] =
        column <= row
            ? load_global(
                  matrix + (panel_begin + row) * kN + panel_begin + column)
            : 0.0f;
    rhs[row * kLd + column] = load_global(
        matrix + (row_begin + row) * kN + panel_begin + column);
  }
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < kTile) {
    const int column = static_cast<int>(threadIdx.x);
    inverse_diagonal[column] =
        __fdividef(1.0f, diagonal[column * kLd + column]);
  }
  __syncthreads();

  if (static_cast<int>(threadIdx.x) < kTile) {
    const int row = static_cast<int>(threadIdx.x);
#pragma unroll 1
    for (int column = 0; column < kTile; ++column) {
      float value = rhs[row * kLd + column];
#pragma unroll 4
      for (int k = 0; k < column; ++k) {
        value = fmaf(-rhs[row * kLd + k], diagonal[column * kLd + k], value);
      }
      rhs[row * kLd + column] = value * inverse_diagonal[column];
    }
  }
  __syncthreads();

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    store_global(matrix + (row_begin + row) * kN + panel_begin + column,
                 rhs[row * kLd + column]);
  }
}

__device__ __forceinline__ void decode_update_tile(
    int task, int panel, int& row_tile, int& column_tile) {
  int cursor = task;
#pragma unroll
  for (int column = 1; column < kTileCount; ++column) {
    if (column <= panel) {
      continue;
    }
    const int count = kTileCount - column;
    if (cursor < count) {
      column_tile = column;
      row_tile = column + cursor;
      return;
    }
    cursor -= count;
  }
  row_tile = -1;
  column_tile = -1;
}

// Stage 3: 4x4 micro-tiled FP32 rank-64 update of every trailing tile.
__global__ __launch_bounds__(kUpdateThreads)
void fp32_update_kernel(float* __restrict__ output, int panel, int tasks) {
  __shared__ __align__(128) float a_panel[kTile * kLd];
  __shared__ __align__(128) float b_panel[kTile * kLd];

  const int matrix_index = static_cast<int>(blockIdx.x) / tasks;
  const int task = static_cast<int>(blockIdx.x) % tasks;
  int row_tile;
  int column_tile;
  decode_update_tile(task, panel, row_tile, column_tile);
  float* matrix = output + static_cast<int64_t>(matrix_index) * kN * kN;
  const int panel_begin = panel * kTile;
  const int row_begin = row_tile * kTile;
  const int column_begin = column_tile * kTile;

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += kUpdateThreads) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    a_panel[row * kLd + column] = load_global(
        matrix + (row_begin + row) * kN + panel_begin + column);
    b_panel[row * kLd + column] = load_global(
        matrix + (column_begin + row) * kN + panel_begin + column);
  }
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = (warp >> 1) * 16;
  const int column_base = (warp & 1) * 32;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;
  float product[4][4] = {};
#pragma unroll 1
  for (int k = 0; k < kTile; ++k) {
    float left[4];
    float right[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      left[row] = a_panel[(row_base + lane_row + row * 4) * kLd + k];
    }
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      right[column] =
          b_panel[(column_base + lane_column + column * 8) * kLd + k];
    }
#pragma unroll
    for (int row = 0; row < 4; ++row) {
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        product[row][column] =
            fmaf(left[row], right[column], product[row][column]);
      }
    }
  }
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const int output_row = row_base + lane_row + row * 4;
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_column = column_base + lane_column + column * 8;
      if (row_tile != column_tile || output_column <= output_row) {
        float* destination =
            matrix + (row_begin + output_row) * kN +
            column_begin + output_column;
        store_global(destination,
                     load_global(destination) - product[row][column]);
      }
    }
  }
}

template <typename Kernel>
void prefer_shared(Kernel kernel) {
  const cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(status == cudaSuccess,
              "shared-memory carveout failed: ", cudaGetErrorString(status));
}

void launch_all(const float* input, float* output) {
  cudaLaunchConfig_t copy_config{};
  copy_config.gridDim = dim3(kBatch * 8, 1, 1);
  copy_config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&copy_config, copy_lower_kernel, input, output);

  for (int panel = 0; panel < kTileCount; ++panel) {
    cudaLaunchConfig_t factor_config{};
    factor_config.gridDim = dim3(kBatch, 1, 1);
    factor_config.blockDim = dim3(kFactorThreads, 1, 1);
    cudaLaunchKernelEx(&factor_config, factor_kernel, output, panel);

    const int remaining = kTileCount - panel - 1;
    if (remaining == 0) {
      continue;
    }

    cudaLaunchConfig_t solve_config{};
    solve_config.gridDim = dim3(kBatch * remaining, 1, 1);
    solve_config.blockDim = dim3(kSolveThreads, 1, 1);
    cudaLaunchKernelEx(&solve_config, solve_kernel, output, panel, remaining);

    const int tasks = remaining * (remaining + 1) / 2;
    cudaLaunchConfig_t update_config{};
    update_config.gridDim = dim3(kBatch * tasks, 1, 1);
    update_config.blockDim = dim3(kUpdateThreads, 1, 1);
    cudaLaunchKernelEx(
        &update_config, fp32_update_kernel, output, panel, tasks);
  }
}

}  // namespace

void cholesky_b16n512_prepare() {
  prefer_shared(factor_kernel);
  prefer_shared(solve_kernel);
  prefer_shared(fp32_update_kernel);
}

at::Tensor cholesky_b16n512(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (16, 512, 512)");
  auto output = at::empty_like(data);
  c10::cuda::CUDAGuard device_guard(data.device());
  launch_all(data.data_ptr<float>(), output.data_ptr<float>());
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return output;
}
"""


@lru_cache(maxsize=1)
def _module_b16n512():
    module = _build(
        "cholesky_b16n512", _CPP_SOURCE_B16N512, _CUDA_SOURCE_B16N512,
        extra_cuda_flags=("-DNDEBUG", "--restrict"))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SHAPE = (16, 512, 512)


def custom_kernel(data: input_t) -> output_t:
    if (
        data.is_cuda
        and data.dtype == torch.float32
        and data.is_contiguous()
        and tuple(data.shape) == _SHAPE
    ):
        return _module_b16n512().run(data)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
