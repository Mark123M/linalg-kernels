import hashlib
import os
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The autotuner may replace this exact line in retained candidate copies.
_DEFAULT_VARIANT = 0  # POPCORN_VARIANT
_VARIANT_NAMES = (
    "p64_raw_scalar_m4x4_t256",
    "p64_nr_scalar_m4x4_t256",
    "p64_precise_scalar_m4x4_t256",
    "p64_raw_sub4_m4x4_t256",
    "p32_raw_scalar_m2x4_t128",
    "p32_nr_scalar_m2x4_t128",
)
_VARIANT_COUNT = len(_VARIANT_NAMES)
_VARIANT_IDS = tuple(range(_VARIANT_COUNT))

_METADATA_COLUMNS = (
    "variant",
    "tile",
    "threads",
    "root_mode",
    "solve_mode",
    "registers",
    "shared_bytes",
    "local_bytes",
    "batch",
    "launch_count",
)

_CPP_SOURCE = r"""
#include <torch/extension.h>

void cholesky_b640n512_prepare(int64_t variant);
at::Tensor cholesky_b640n512(
    const at::Tensor& data, int64_t variant);
void cholesky_b640n512_out(
    const at::Tensor& data, at::Tensor out, int64_t variant);
at::Tensor cholesky_b640n512_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b640n512_prepare,
        "Configure one fused B200 Cholesky variant");
  m.def("run", &cholesky_b640n512,
        "Batched 640x512 Cholesky");
  m.def("run_out", &cholesky_b640n512_out,
        "Batched 640x512 Cholesky out");
  m.def("metadata", &cholesky_b640n512_metadata,
        "Fused kernel resource metadata");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 640;
constexpr int kN = 512;
constexpr int kVariantCount = 6;
constexpr int kMetadataColumns = 10;

constexpr int kPreciseRoot = 0;
constexpr int kNewtonRoot = 1;
constexpr int kRawRoot = 2;
constexpr int kScalarSolve = 0;
constexpr int kSub4Solve = 1;

template <int Id>
struct Variant;

#define SPEC(ID, TILE, THREADS, ROOT, SOLVE) \
  template <> struct Variant<ID> {           \
    static constexpr int tile = TILE;        \
    static constexpr int threads = THREADS;  \
    static constexpr int root = ROOT;        \
    static constexpr int solve = SOLVE;      \
  }

SPEC(0, 64, 256, kRawRoot, kScalarSolve);
SPEC(1, 64, 256, kNewtonRoot, kScalarSolve);
SPEC(2, 64, 256, kPreciseRoot, kScalarSolve);
SPEC(3, 64, 256, kRawRoot, kSub4Solve);
SPEC(4, 32, 128, kRawRoot, kScalarSolve);
SPEC(5, 32, 128, kNewtonRoot, kScalarSolve);

#undef SPEC

__device__ __forceinline__ float load_global(
    const float* pointer) {
  return __ldcg(pointer);
}

__device__ __forceinline__ void store_global(
    float* pointer, float value) {
  __stcg(pointer, value);
}

template <int RootMode>
__device__ __forceinline__ void root_pair(
    float value, float& diagonal, float& inverse) {
  if constexpr (RootMode == kPreciseRoot) {
    diagonal = __fsqrt_rn(value);
    inverse = __fdiv_rn(1.0f, diagonal);
  } else {
    inverse = rsqrtf(value);
    if constexpr (RootMode == kNewtonRoot) {
      inverse *= fmaf(
          -0.5f * value, inverse * inverse, 1.5f);
    }
    diagonal = value * inverse;
  }
}

template <int Tile>
__device__ __forceinline__ float& tile_at(
    float* tile, int row, int column) {
  constexpr int kLd = Tile + 1;
  return tile[row * kLd + column];
}

template <int Tile>
__device__ __forceinline__ const float& tile_at(
    const float* tile, int row, int column) {
  constexpr int kLd = Tile + 1;
  return tile[row * kLd + column];
}

template <int RootMode, int Tile>
__device__ __forceinline__ void factor16(
    float* tile, float* inverse_diagonal, int begin) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (warp == 0) {
#pragma unroll
    for (int local_column = 0;
         local_column < 16; ++local_column) {
      const int column = begin + local_column;
      float inverse = 0.0f;
      if (lane == local_column) {
        float diagonal;
        root_pair<RootMode>(
            tile_at<Tile>(tile, column, column),
            diagonal, inverse);
        tile_at<Tile>(tile, column, column) = diagonal;
        inverse_diagonal[column] = inverse;
      }
      inverse = __shfl_sync(
          0xffffffffu, inverse, local_column);
      if (lane > local_column && lane < 16) {
        const int row = begin + lane;
        tile_at<Tile>(tile, row, column) *= inverse;
      }
      __syncwarp();
      if (lane > local_column && lane < 16) {
        const int row = begin + lane;
        const float left =
            tile_at<Tile>(tile, row, column);
#pragma unroll
        for (int local_target = local_column + 1;
             local_target < 16; ++local_target) {
          if (local_target <= lane) {
            const int target = begin + local_target;
            tile_at<Tile>(tile, row, target) = fmaf(
                -left,
                tile_at<Tile>(tile, target, column),
                tile_at<Tile>(tile, row, target));
          }
        }
      }
      __syncwarp();
    }
  }
}

template <int Tile, int Rows, int Columns>
__device__ __forceinline__ void local_trsm_sub4(
    float* tile, const float* inverse_diagonal,
    int row_begin, int column_begin) {
  const int lane = static_cast<int>(threadIdx.x) & 3;
  const int row_index = static_cast<int>(threadIdx.x) >> 2;
  if (row_index < Rows) {
    const int row = row_begin + row_index;
#pragma unroll
    for (int local_column = 0;
         local_column < Columns; ++local_column) {
      const int column = column_begin + local_column;
      float partial = 0.0f;
#pragma unroll
      for (int k = lane; k < local_column; k += 4) {
        partial = fmaf(
            tile_at<Tile>(
                tile, row, column_begin + k),
            tile_at<Tile>(
                tile, column, column_begin + k),
            partial);
      }
      partial += __shfl_down_sync(
          0xffffffffu, partial, 2, 4);
      partial += __shfl_down_sync(
          0xffffffffu, partial, 1, 4);
      if (lane == 0) {
        tile_at<Tile>(tile, row, column) =
            (tile_at<Tile>(tile, row, column) - partial) *
            inverse_diagonal[column];
      }
      __syncwarp();
    }
  }
}

template <int Tile>
__device__ __forceinline__ void local_update16(
    float* tile, int target, int panel) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (warp >= 4) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = (warp >> 1) * 8 + (lane >> 2);
  const int column0 = (warp & 1) * 8 + (lane & 3);
  const int column1 = column0 + 4;
  float product0 = 0.0f;
  float product1 = 0.0f;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const float left =
        tile_at<Tile>(tile, target + row, panel + k);
    product0 = fmaf(
        left,
        tile_at<Tile>(
            tile, target + column0, panel + k),
        product0);
    product1 = fmaf(
        left,
        tile_at<Tile>(
            tile, target + column1, panel + k),
        product1);
  }
  if (column0 <= row) {
    tile_at<Tile>(
        tile, target + row, target + column0) -= product0;
  }
  if (column1 <= row) {
    tile_at<Tile>(
        tile, target + row, target + column1) -= product1;
  }
}

template <int Tile>
__device__ __forceinline__ void local_update32(
    float* tile, int target, int panel) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (warp >= 4) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = (warp >> 1) * 16;
  const int column_base = (warp & 1) * 16;
  const int lane_row = lane >> 2;
  const int lane_column = lane & 3;
  float product[2][4] = {};
#pragma unroll
  for (int k = 0; k < 32; ++k) {
    const float left0 = tile_at<Tile>(
        tile, target + row_base + lane_row, panel + k);
    const float left1 = tile_at<Tile>(
        tile, target + row_base + lane_row + 8, panel + k);
    float right[4];
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      right[column] = tile_at<Tile>(
          tile,
          target + column_base + lane_column + column * 4,
          panel + k);
    }
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      product[0][column] =
          fmaf(left0, right[column], product[0][column]);
      product[1][column] =
          fmaf(left1, right[column], product[1][column]);
    }
  }
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    const int output_row =
        row_base + lane_row + row * 8;
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_column =
          column_base + lane_column + column * 4;
      if (output_column <= output_row) {
        tile_at<Tile>(
            tile,
            target + output_row,
            target + output_column) -=
            product[row][column];
      }
    }
  }
}

template <int RootMode, int Tile>
__device__ __forceinline__ void factor32_recursive16(
    float* tile, float* inverse_diagonal, int begin) {
  factor16<RootMode, Tile>(
      tile, inverse_diagonal, begin);
  __syncthreads();
  local_trsm_sub4<Tile, 16, 16>(
      tile, inverse_diagonal, begin + 16, begin);
  __syncthreads();
  local_update16<Tile>(tile, begin + 16, begin);
  __syncthreads();
  factor16<RootMode, Tile>(
      tile, inverse_diagonal, begin + 16);
}

template <int Tile, int RootMode>
__device__ __forceinline__ void factor_tile(
    float* tile, float* inverse_diagonal) {
  factor32_recursive16<RootMode, Tile>(
      tile, inverse_diagonal, 0);
  if constexpr (Tile == 64) {
    __syncthreads();
    local_trsm_sub4<Tile, 32, 32>(
        tile, inverse_diagonal, 32, 0);
    __syncthreads();
    local_update32<Tile>(tile, 32, 0);
    __syncthreads();
    factor32_recursive16<RootMode, Tile>(
        tile, inverse_diagonal, 32);
  }
}

template <int Threads>
__device__ __forceinline__ void copy_lower(
    const float* input, float* output) {
  constexpr int kVectorsPerRow = kN / 4;
  const int vector_column = static_cast<int>(threadIdx.x);
  if (vector_column < kVectorsPerRow) {
    const int column = vector_column * 4;
    for (int row = 0; row < kN; ++row) {
      const int offset = row * kN + column;
      if (column + 3 <= row) {
        const float4 value =
            *reinterpret_cast<const float4*>(input + offset);
        *reinterpret_cast<float4*>(output + offset) = value;
      } else if (column > row) {
        *reinterpret_cast<float4*>(output + offset) =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      } else {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          output[offset + item] =
              column + item <= row
                  ? input[offset + item]
                  : 0.0f;
        }
      }
    }
  }
  __syncthreads();
}

template <int Tile, int Threads>
__device__ __forceinline__ void load_tile(
    const float* matrix, float* tile,
    int row_begin, int column_begin) {
  constexpr int kLd = Tile + 1;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Tile * Tile;
       linear += Threads) {
    const int row = linear / Tile;
    const int column = linear % Tile;
    tile[row * kLd + column] = load_global(
        matrix + (row_begin + row) * kN +
        column_begin + column);
  }
}

template <int Tile, int Threads>
__device__ __forceinline__ void load_diagonal(
    const float* matrix, float* tile, int begin) {
  constexpr int kLd = Tile + 1;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Tile * Tile;
       linear += Threads) {
    const int row = linear / Tile;
    const int column = linear % Tile;
    tile[row * kLd + column] =
        column <= row
            ? load_global(
                  matrix + (begin + row) * kN +
                  begin + column)
            : 0.0f;
  }
}

template <int Tile, int Threads>
__device__ __forceinline__ void store_tile(
    const float* tile, float* matrix,
    int row_begin, int column_begin) {
  constexpr int kLd = Tile + 1;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Tile * Tile;
       linear += Threads) {
    const int row = linear / Tile;
    const int column = linear % Tile;
    store_global(
        matrix + (row_begin + row) * kN +
        column_begin + column,
        tile[row * kLd + column]);
  }
}

template <int Tile, int Threads>
__device__ __forceinline__ void store_diagonal(
    const float* tile, float* matrix, int begin) {
  constexpr int kLd = Tile + 1;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Tile * Tile;
       linear += Threads) {
    const int row = linear / Tile;
    const int column = linear % Tile;
    if (column <= row) {
      store_global(
          matrix + (begin + row) * kN +
          begin + column,
          tile[row * kLd + column]);
    }
  }
}

template <int Tile, int SolveMode>
__device__ __forceinline__ void solve_panel(
    float* rhs, const float* diagonal,
    const float* inverse_diagonal) {
  constexpr int kLd = Tile + 1;
  if constexpr (SolveMode == kScalarSolve) {
    if (static_cast<int>(threadIdx.x) < Tile) {
      const int row = static_cast<int>(threadIdx.x);
#pragma unroll 1
      for (int column = 0; column < Tile; ++column) {
        float value = rhs[row * kLd + column];
#pragma unroll 4
        for (int k = 0; k < column; ++k) {
          value = fmaf(
              -rhs[row * kLd + k],
              diagonal[column * kLd + k],
              value);
        }
        rhs[row * kLd + column] =
            value * inverse_diagonal[column];
      }
    }
  } else {
    const int lane = static_cast<int>(threadIdx.x) & 3;
    const int row = static_cast<int>(threadIdx.x) >> 2;
    if (row < Tile) {
#pragma unroll 1
      for (int column = 0; column < Tile; ++column) {
        float partial = 0.0f;
#pragma unroll 4
        for (int k = lane; k < column; k += 4) {
          partial = fmaf(
              rhs[row * kLd + k],
              diagonal[column * kLd + k],
              partial);
        }
        partial += __shfl_down_sync(
            0xffffffffu, partial, 2, 4);
        partial += __shfl_down_sync(
            0xffffffffu, partial, 1, 4);
        if (lane == 0) {
          rhs[row * kLd + column] =
              (rhs[row * kLd + column] - partial) *
              inverse_diagonal[column];
        }
        __syncwarp();
      }
    }
  }
}

template <int Tile>
__device__ __forceinline__ void update_global(
    const float* left, const float* right,
    float* matrix, int row_begin, int column_begin,
    bool diagonal) {
  constexpr int kLd = Tile + 1;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if constexpr (Tile == 64) {
    const int row_base = (warp >> 1) * 16;
    const int column_base = (warp & 1) * 32;
    const int lane_row = lane >> 3;
    const int lane_column = lane & 7;
    float product[4][4] = {};
#pragma unroll 1
    for (int k = 0; k < Tile; ++k) {
      float left_values[4];
      float right_values[4];
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        left_values[row] = left[
            (row_base + lane_row + row * 4) * kLd + k];
      }
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        right_values[column] = right[
            (column_base + lane_column + column * 8) *
                kLd +
            k];
      }
#pragma unroll
      for (int row = 0; row < 4; ++row) {
#pragma unroll
        for (int column = 0; column < 4; ++column) {
          product[row][column] = fmaf(
              left_values[row],
              right_values[column],
              product[row][column]);
        }
      }
    }
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const int output_row =
          row_base + lane_row + row * 4;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        const int output_column =
            column_base + lane_column + column * 8;
        if (!diagonal || output_column <= output_row) {
          float* destination =
              matrix + (row_begin + output_row) * kN +
              column_begin + output_column;
          store_global(
              destination,
              load_global(destination) -
                  product[row][column]);
        }
      }
    }
  } else {
    const int row_base = (warp >> 1) * 16;
    const int column_base = (warp & 1) * 16;
    const int lane_row = lane >> 2;
    const int lane_column = lane & 3;
    float product[2][4] = {};
#pragma unroll 1
    for (int k = 0; k < Tile; ++k) {
      const float left0 =
          left[(row_base + lane_row) * kLd + k];
      const float left1 =
          left[(row_base + lane_row + 8) * kLd + k];
      float right_values[4];
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        right_values[column] = right[
            (column_base + lane_column + column * 4) *
                kLd +
            k];
      }
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        product[0][column] = fmaf(
            left0, right_values[column],
            product[0][column]);
        product[1][column] = fmaf(
            left1, right_values[column],
            product[1][column]);
      }
    }
#pragma unroll
    for (int row = 0; row < 2; ++row) {
      const int output_row =
          row_base + lane_row + row * 8;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        const int output_column =
            column_base + lane_column + column * 4;
        if (!diagonal || output_column <= output_row) {
          float* destination =
              matrix + (row_begin + output_row) * kN +
              column_begin + output_column;
          store_global(
              destination,
              load_global(destination) -
                  product[row][column]);
        }
      }
    }
  }
}

template <
    int Tile, int Threads, int RootMode, int SolveMode>
__global__ __launch_bounds__(Threads)
void fused_potrf_kernel(
    const float* __restrict__ input,
    float* __restrict__ output) {
  constexpr int kLd = Tile + 1;
  constexpr int kTileCount = kN / Tile;
  __shared__ __align__(128) float tile_a[Tile * kLd];
  __shared__ __align__(128) float tile_b[Tile * kLd];
  __shared__ float inverse_diagonal[Tile];

  const int matrix_index = static_cast<int>(blockIdx.x);
  const int64_t matrix_offset =
      static_cast<int64_t>(matrix_index) * kN * kN;
  const float* input_matrix = input + matrix_offset;
  float* matrix = output + matrix_offset;

  copy_lower<Threads>(input_matrix, matrix);

  for (int panel = 0; panel < kTileCount; ++panel) {
    const int panel_begin = panel * Tile;
    load_diagonal<Tile, Threads>(
        matrix, tile_a, panel_begin);
    __syncthreads();

    factor_tile<Tile, RootMode>(
        tile_a, inverse_diagonal);
    __syncthreads();
    store_diagonal<Tile, Threads>(
        tile_a, matrix, panel_begin);

    for (int row_tile = panel + 1;
         row_tile < kTileCount; ++row_tile) {
      const int row_begin = row_tile * Tile;
      load_tile<Tile, Threads>(
          matrix, tile_b, row_begin, panel_begin);
      __syncthreads();
      solve_panel<Tile, SolveMode>(
          tile_b, tile_a, inverse_diagonal);
      __syncthreads();
      store_tile<Tile, Threads>(
          tile_b, matrix, row_begin, panel_begin);
    }
    __syncthreads();

    for (int row_tile = panel + 1;
         row_tile < kTileCount; ++row_tile) {
      const int row_begin = row_tile * Tile;
      load_tile<Tile, Threads>(
          matrix, tile_a, row_begin, panel_begin);
      __syncthreads();

      for (int column_tile = panel + 1;
           column_tile <= row_tile; ++column_tile) {
        const int column_begin = column_tile * Tile;
        const bool is_diagonal =
            column_tile == row_tile;
        const float* right = tile_a;
        if (!is_diagonal) {
          load_tile<Tile, Threads>(
              matrix, tile_b, column_begin, panel_begin);
          __syncthreads();
          right = tile_b;
        }
        update_global<Tile>(
            tile_a, right, matrix,
            row_begin, column_begin, is_diagonal);
        __syncthreads();
      }
    }
  }
}

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be CUDA");
  TORCH_CHECK(
      data.scalar_type() == at::kFloat,
      "input must be float32");
  TORCH_CHECK(
      data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      data.dim() == 3 && data.size(0) == kBatch &&
      data.size(1) == kN && data.size(2) == kN,
      "native input must have shape (640, 512, 512)");
}

void check_output(
    const at::Tensor& data, const at::Tensor& output) {
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(
      output.scalar_type() == at::kFloat,
      "output must be float32");
  TORCH_CHECK(
      output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(
      output.sizes() == data.sizes(), "output shape mismatch");
  TORCH_CHECK(
      output.device() == data.device(),
      "output device mismatch");
}

template <typename Kernel>
void prefer_shared(Kernel kernel) {
  const cudaError_t status = cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  TORCH_CHECK(
      status == cudaSuccess,
      "shared-memory carveout failed: ",
      cudaGetErrorString(status));
}

template <typename Kernel>
cudaFuncAttributes attributes_for(Kernel kernel) {
  cudaFuncAttributes attributes{};
  const cudaError_t status =
      cudaFuncGetAttributes(&attributes, kernel);
  TORCH_CHECK(
      status == cudaSuccess,
      "kernel resource query failed: ",
      cudaGetErrorString(status));
  return attributes;
}

template <int Id>
void configure_one() {
  using V = Variant<Id>;
  auto kernel = fused_potrf_kernel<
      V::tile, V::threads, V::root, V::solve>;
  prefer_shared(kernel);
  const cudaFuncAttributes attributes =
      attributes_for(kernel);
  TORCH_CHECK(
      attributes.localSizeBytes <= 8,
      "variant ", Id, " kernel uses ",
      attributes.localSizeBytes,
      " local bytes, above the accepted 8-byte frame");
}

template <int Id>
void launch_one(const float* input, float* output) {
  using V = Variant<Id>;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch, 1, 1);
  config.blockDim = dim3(V::threads, 1, 1);
  cudaLaunchKernelEx(
      &config,
      fused_potrf_kernel<
          V::tile, V::threads, V::root, V::solve>,
      input, output);
}

void configure_variant(int variant) {
  switch (variant) {
    case 0: configure_one<0>(); break;
    case 1: configure_one<1>(); break;
    case 2: configure_one<2>(); break;
    case 3: configure_one<3>(); break;
    case 4: configure_one<4>(); break;
    case 5: configure_one<5>(); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 5]");
  }
}

void launch_variant(
    const float* input, float* output, int variant) {
  switch (variant) {
    case 0: launch_one<0>(input, output); break;
    case 1: launch_one<1>(input, output); break;
    case 2: launch_one<2>(input, output); break;
    case 3: launch_one<3>(input, output); break;
    case 4: launch_one<4>(input, output); break;
    case 5: launch_one<5>(input, output); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 5]");
  }
}

template <int Id>
void write_metadata(int64_t* rows) {
  using V = Variant<Id>;
  const cudaFuncAttributes attributes = attributes_for(
      fused_potrf_kernel<
          V::tile, V::threads, V::root, V::solve>);
  int64_t* row =
      rows + static_cast<int64_t>(Id) * kMetadataColumns;
  row[0] = Id;
  row[1] = V::tile;
  row[2] = V::threads;
  row[3] = V::root;
  row[4] = V::solve;
  row[5] = attributes.numRegs;
  row[6] = attributes.sharedSizeBytes;
  row[7] = attributes.localSizeBytes;
  row[8] = kBatch;
  row[9] = 1;
}

}  // namespace

void cholesky_b640n512_prepare(int64_t variant) {
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 5]");
  configure_variant(static_cast<int>(variant));
}

void cholesky_b640n512_out(
    const at::Tensor& data,
    at::Tensor output,
    int64_t variant) {
  check_input(data);
  check_output(data, output);
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 5]");
  c10::cuda::CUDAGuard device_guard(data.device());
  launch_variant(
      data.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(variant));
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "Cholesky launch failed: ",
      cudaGetErrorString(status));
}

at::Tensor cholesky_b640n512(
    const at::Tensor& data, int64_t variant) {
  auto output = at::empty_like(data);
  cholesky_b640n512_out(data, output, variant);
  return output;
}

at::Tensor cholesky_b640n512_metadata() {
  auto result = at::zeros(
      {kVariantCount, kMetadataColumns},
      at::TensorOptions()
          .dtype(at::kLong)
          .device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  write_metadata<0>(rows);
  write_metadata<1>(rows);
  write_metadata<2>(rows);
  write_metadata<3>(rows);
  write_metadata<4>(rows);
  write_metadata<5>(rows);
  return result;
}
"""


@lru_cache(maxsize=1)
def _native_module():
    tag = hashlib.sha256(
        (_CPP_SOURCE + _CUDA_SOURCE).encode()
    ).hexdigest()[:12]
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name=f"cholesky_b640n512_fused_{tag}",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=None,
            extra_cflags=[
                "-O3",
                "-DNDEBUG",
                "-std=c++20",
            ],
            extra_cuda_cflags=[
                "-O3",
                "-DNDEBUG",
                "-std=c++20",
                "--use_fast_math",
                "--extra-device-vectorization",
                "--restrict",
                "-lineinfo",
                "-Xptxas=-O3,-v,-warn-spills",
                "-gencode",
                "arch=compute_100a,code=sm_100a",
            ],
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


_PREPARED_VARIANTS: set[int] = set()


def _run_variant(
    data: torch.Tensor,
    variant: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if variant not in _VARIANT_IDS:
        raise ValueError(
            f"variant must be in {_VARIANT_IDS}, got {variant}"
        )
    module = _native_module()
    if variant not in _PREPARED_VARIANTS:
        module.prepare(variant)
        _PREPARED_VARIANTS.add(variant)
    if out is None:
        return module.run(data, variant)
    module.run_out(data, out, variant)
    return out


def _variant_metadata() -> torch.Tensor:
    return _native_module().metadata()


def custom_kernel(data: input_t) -> output_t:
    if (
        data.is_cuda
        and data.dtype == torch.float32
        and data.is_contiguous()
        and tuple(data.shape) == (640, 512, 512)
    ):
        return _run_variant(data, _DEFAULT_VARIANT)
    return torch.linalg.cholesky_ex(
        data, check_errors=False
    ).L
