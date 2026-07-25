import hashlib
import os
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The tuner replaces this exact line only in retained candidate copies.
_DEFAULT_VARIANT = 6  # POPCORN_VARIANT
_VARIANT_NAMES = (
    "r16_micro4x4_precise_scalar_u256",
    "r16_micro4x4_nr_scalar_u256",
    "r16_micro4x4_raw_scalar_u256",
    "r32_micro4x4_precise_scalar_u256",
    "r16_scalar_precise_scalar_u256",
    "r16_micro4x4_precise_sub4_u256",
    "r16_micro2x4_precise_scalar_u512",
)
_VARIANT_COUNT = len(_VARIANT_NAMES)
_VARIANT_IDS = tuple(range(_VARIANT_COUNT))

_METADATA_COLUMNS = (
    "variant",
    "factor_threads",
    "solve_threads",
    "update_threads",
    "factor_registers",
    "solve_registers",
    "update_registers",
    "factor_shared_bytes",
    "solve_shared_bytes",
    "update_shared_bytes",
    "factor_local_bytes",
    "solve_local_bytes",
    "update_local_bytes",
    "update_kind",
    "root_mode",
    "solve_mode",
    "factor_mode",
    "outer_block",
    "launch_count",
)

_CPP_SOURCE = r"""
#include <torch/extension.h>

void cholesky_b16n512_prepare(int64_t variant);
at::Tensor cholesky_b16n512(const at::Tensor& data, int64_t variant);
void cholesky_b16n512_out(
    const at::Tensor& data, at::Tensor out, int64_t variant);
at::Tensor cholesky_b16n512_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b16n512_prepare,
        "Configure one staged B200 Cholesky variant");
  m.def("run", &cholesky_b16n512, "Batched 16x512 Cholesky");
  m.def("run_out", &cholesky_b16n512_out,
        "Batched 16x512 Cholesky out");
  m.def("metadata", &cholesky_b16n512_metadata,
        "Staged kernel resource metadata");
}
"""

_CUDA_SOURCE = r"""
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
constexpr int kVariantCount = 7;
constexpr int kMetadataColumns = 19;

constexpr int kPreciseRoot = 0;
constexpr int kNewtonRoot = 1;
constexpr int kRawRoot = 2;
constexpr int kScalarSolve = 0;
constexpr int kSub4Solve = 1;
constexpr int kRank32Factor = 0;
constexpr int kRecursive16Factor = 1;
constexpr int kScalarUpdate = 0;
constexpr int kMicro4x4Update = 1;
constexpr int kMicro2x4Update = 2;

template <int Id>
struct Variant;

#define SPEC(ID, ROOT, SOLVE, FACTOR, UPDATE, UTHREADS) \
  template <> struct Variant<ID> {              \
    static constexpr int root = ROOT;           \
    static constexpr int solve = SOLVE;         \
    static constexpr int factor = FACTOR;       \
    static constexpr int update = UPDATE;       \
    static constexpr int update_threads = UTHREADS; \
    static constexpr int solve_threads =        \
        SOLVE == kScalarSolve ? 128 : 256;      \
  }

SPEC(0, kPreciseRoot, kScalarSolve, kRecursive16Factor,
     kMicro4x4Update, 256);
SPEC(1, kNewtonRoot, kScalarSolve, kRecursive16Factor,
     kMicro4x4Update, 256);
SPEC(2, kRawRoot, kScalarSolve, kRecursive16Factor,
     kMicro4x4Update, 256);
SPEC(3, kPreciseRoot, kScalarSolve, kRank32Factor,
     kMicro4x4Update, 256);
SPEC(4, kPreciseRoot, kScalarSolve, kRecursive16Factor,
     kScalarUpdate, 256);
SPEC(5, kPreciseRoot, kSub4Solve, kRecursive16Factor,
     kMicro4x4Update, 256);
SPEC(6, kPreciseRoot, kScalarSolve, kRecursive16Factor,
     kMicro2x4Update, 512);

#undef SPEC

__device__ __forceinline__ float load_global(const float* pointer) {
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

__device__ __forceinline__ float& tile_at(
    float* tile, int row, int column) {
  return tile[row * kLd + column];
}

template <int RootMode>
__device__ __forceinline__ void factor32(
    float* tile, float* inverse_diagonal, int begin) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (warp == 0) {
#pragma unroll 1
    for (int local_column = 0; local_column < 32; ++local_column) {
      const int column = begin + local_column;
      float inverse = 0.0f;
      if (lane == local_column) {
        float diagonal;
        root_pair<RootMode>(
            tile_at(tile, column, column), diagonal, inverse);
        tile_at(tile, column, column) = diagonal;
        inverse_diagonal[column] = inverse;
      }
      inverse = __shfl_sync(
          0xffffffffu, inverse, local_column);
      if (lane > local_column) {
        const int row = begin + lane;
        tile_at(tile, row, column) *= inverse;
      }
      __syncwarp();
      if (lane > local_column) {
        const int row = begin + lane;
        const float left = tile_at(tile, row, column);
        for (int local_target = local_column + 1;
             local_target <= lane; ++local_target) {
          const int target = begin + local_target;
          tile_at(tile, row, target) = fmaf(
              -left, tile_at(tile, target, column),
              tile_at(tile, row, target));
        }
      }
      __syncwarp();
    }
  }
}

template <int RootMode>
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
        root_pair<RootMode>(
            tile_at(tile, column, column), diagonal, inverse);
        tile_at(tile, column, column) = diagonal;
        inverse_diagonal[column] = inverse;
      }
      inverse = __shfl_sync(
          0xffffffffu, inverse, local_column);
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

template <int Rows, int Columns>
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
            tile_at(tile, row, column_begin + k),
            tile_at(tile, column, column_begin + k), partial);
      }
      partial += __shfl_down_sync(
          0xffffffffu, partial, 2, 4);
      partial += __shfl_down_sync(
          0xffffffffu, partial, 1, 4);
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
    const float left =
        tile_at(tile, target + row, panel + k);
    product0 = fmaf(
        left, tile_at(tile, target + column0, panel + k),
        product0);
    product1 = fmaf(
        left, tile_at(tile, target + column1, panel + k),
        product1);
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
          tile, target + column_base + lane_column + column * 4,
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
    const int output_row = row_base + lane_row + row * 8;
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_column =
          column_base + lane_column + column * 4;
      if (output_column <= output_row) {
        tile_at(
            tile, target + output_row, target + output_column) -=
            product[row][column];
      }
    }
  }
}

template <int RootMode>
__device__ __forceinline__ void factor32_recursive16(
    float* tile, float* inverse_diagonal, int begin) {
  factor16<RootMode>(tile, inverse_diagonal, begin);
  __syncthreads();
  local_trsm_sub4<16, 16>(
      tile, inverse_diagonal, begin + 16, begin);
  __syncthreads();
  local_update16(tile, begin + 16, begin);
  __syncthreads();
  factor16<RootMode>(tile, inverse_diagonal, begin + 16);
}

__global__ __launch_bounds__(256)
void copy_lower_kernel(
    const float* __restrict__ input,
    float* __restrict__ output) {
  constexpr int kCtasPerMatrix = 8;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / kCtasPerMatrix;
  const int rank =
      static_cast<int>(blockIdx.x) % kCtasPerMatrix;
  const int64_t base =
      static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear = rank * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
       linear < kN * kN;
       linear += kCtasPerMatrix * static_cast<int>(blockDim.x)) {
    const int row = linear / kN;
    const int column = linear % kN;
    store_global(
        output + base + linear,
        column <= row ? input[base + linear] : 0.0f);
  }
}

template <int RootMode, int FactorMode>
__global__ __launch_bounds__(128)
void factor_kernel(float* __restrict__ output, int panel) {
  __shared__ __align__(128) float tile[kTile * kLd];
  __shared__ float inverse_diagonal[kTile];
  const int matrix_index = static_cast<int>(blockIdx.x);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  const int begin = panel * kTile;

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    tile_at(tile, row, column) =
        column <= row
            ? load_global(
                  matrix + (begin + row) * kN + begin + column)
            : 0.0f;
  }
  __syncthreads();

  if constexpr (FactorMode == kRecursive16Factor) {
    factor32_recursive16<RootMode>(
        tile, inverse_diagonal, 0);
  } else {
    factor32<RootMode>(tile, inverse_diagonal, 0);
  }
  __syncthreads();

  local_trsm_sub4<32, 32>(
      tile, inverse_diagonal, 32, 0);
  __syncthreads();

  local_update32(tile, 32, 0);
  __syncthreads();

  if constexpr (FactorMode == kRecursive16Factor) {
    factor32_recursive16<RootMode>(
        tile, inverse_diagonal, 32);
  } else {
    factor32<RootMode>(tile, inverse_diagonal, 32);
  }
  __syncthreads();

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    if (column <= row) {
      store_global(
          matrix + (begin + row) * kN + begin + column,
          tile_at(tile, row, column));
    }
  }
}

template <int SolveMode, int RootMode>
__global__ void solve_kernel(
    float* __restrict__ output, int panel, int remaining) {
  __shared__ __align__(128) float diagonal[kTile * kLd];
  __shared__ __align__(128) float rhs[kTile * kLd];
  __shared__ float inverse_diagonal[kTile];

  const int matrix_index =
      static_cast<int>(blockIdx.x) / remaining;
  const int row_tile =
      panel + 1 + static_cast<int>(blockIdx.x) % remaining;
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  const int panel_begin = panel * kTile;
  const int row_begin = row_tile * kTile;

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    diagonal[row * kLd + column] =
        column <= row
            ? load_global(
                  matrix + (panel_begin + row) * kN +
                  panel_begin + column)
            : 0.0f;
    rhs[row * kLd + column] = load_global(
        matrix + (row_begin + row) * kN +
        panel_begin + column);
  }
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < kTile) {
    const int column = static_cast<int>(threadIdx.x);
    if constexpr (RootMode == kPreciseRoot) {
      inverse_diagonal[column] = __fdiv_rn(
          1.0f, diagonal[column * kLd + column]);
    } else {
      inverse_diagonal[column] = __fdividef(
          1.0f, diagonal[column * kLd + column]);
    }
  }
  __syncthreads();

  if constexpr (SolveMode == kScalarSolve) {
    if (static_cast<int>(threadIdx.x) < kTile) {
      const int row = static_cast<int>(threadIdx.x);
#pragma unroll 1
      for (int column = 0; column < kTile; ++column) {
        float value = rhs[row * kLd + column];
#pragma unroll 4
        for (int k = 0; k < column; ++k) {
          value = fmaf(
              -rhs[row * kLd + k],
              diagonal[column * kLd + k], value);
        }
        rhs[row * kLd + column] =
            value * inverse_diagonal[column];
      }
    }
  } else {
    const int lane = static_cast<int>(threadIdx.x) & 3;
    const int row = static_cast<int>(threadIdx.x) >> 2;
    if (row < kTile) {
#pragma unroll 1
      for (int column = 0; column < kTile; ++column) {
        float partial = 0.0f;
        for (int k = lane; k < column; k += 4) {
          partial = fmaf(
              rhs[row * kLd + k],
              diagonal[column * kLd + k], partial);
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
  __syncthreads();

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    store_global(
        matrix + (row_begin + row) * kN +
        panel_begin + column,
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

template <int Threads, int UpdateMode>
__global__ __launch_bounds__(Threads)
void fp32_update_kernel(
    float* __restrict__ output, int panel, int tasks) {
  __shared__ __align__(128) float a_panel[kTile * kLd];
  __shared__ __align__(128) float b_panel[kTile * kLd];

  const int matrix_index =
      static_cast<int>(blockIdx.x) / tasks;
  const int task = static_cast<int>(blockIdx.x) % tasks;
  int row_tile;
  int column_tile;
  decode_update_tile(task, panel, row_tile, column_tile);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  const int panel_begin = panel * kTile;
  const int row_begin = row_tile * kTile;
  const int column_begin = column_tile * kTile;

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile;
       linear += Threads) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    a_panel[row * kLd + column] = load_global(
        matrix + (row_begin + row) * kN +
        panel_begin + column);
    b_panel[row * kLd + column] = load_global(
        matrix + (column_begin + row) * kN +
        panel_begin + column);
  }
  __syncthreads();

  if constexpr (UpdateMode == kScalarUpdate) {
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kTile * kTile;
         linear += Threads) {
      const int row = linear / kTile;
      const int column = linear % kTile;
      if (row_tile != column_tile || column <= row) {
        float* destination =
            matrix + (row_begin + row) * kN +
            column_begin + column;
        float value = load_global(destination);
#pragma unroll 4
        for (int k = 0; k < kTile; ++k) {
          value = fmaf(
              -a_panel[row * kLd + k],
              b_panel[column * kLd + k], value);
        }
        store_global(destination, value);
      }
    }
  } else if constexpr (UpdateMode == kMicro4x4Update) {
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
        left[row] = a_panel[
            (row_base + lane_row + row * 4) * kLd + k];
      }
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        right[column] = b_panel[
            (column_base + lane_column + column * 8) * kLd + k];
      }
#pragma unroll
      for (int row = 0; row < 4; ++row) {
#pragma unroll
        for (int column = 0; column < 4; ++column) {
          product[row][column] = fmaf(
              left[row], right[column], product[row][column]);
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
        if (row_tile != column_tile ||
            output_column <= output_row) {
          float* destination =
              matrix + (row_begin + output_row) * kN +
              column_begin + output_column;
          store_global(
              destination,
              load_global(destination) - product[row][column]);
        }
      }
    }
  } else {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row_base = (warp >> 2) * 16;
    const int column_base = (warp & 3) * 16;
    const int lane_row = lane >> 2;
    const int lane_column = lane & 3;
    float product[2][4] = {};
#pragma unroll 1
    for (int k = 0; k < kTile; ++k) {
      float left[2];
      float right[4];
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        left[row] = a_panel[
            (row_base + lane_row + row * 8) * kLd + k];
      }
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        right[column] = b_panel[
            (column_base + lane_column + column * 4) * kLd + k];
      }
#pragma unroll
      for (int row = 0; row < 2; ++row) {
#pragma unroll
        for (int column = 0; column < 4; ++column) {
          product[row][column] = fmaf(
              left[row], right[column], product[row][column]);
        }
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
        if (row_tile != column_tile ||
            output_column <= output_row) {
          float* destination =
              matrix + (row_begin + output_row) * kN +
              column_begin + output_column;
          store_global(
              destination,
              load_global(destination) - product[row][column]);
        }
      }
    }
  }
}

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be CUDA");
  TORCH_CHECK(
      data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      data.dim() == 3 && data.size(0) == kBatch &&
      data.size(1) == kN && data.size(2) == kN,
      "native input must have shape (16, 512, 512)");
}

void check_output(
    const at::Tensor& data, const at::Tensor& output) {
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(
      output.scalar_type() == at::kFloat, "output must be float32");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output.sizes() == data.sizes(), "output shape mismatch");
  TORCH_CHECK(
      output.device() == data.device(), "output device mismatch");
}

template <typename Kernel>
void prefer_shared(Kernel kernel) {
  const cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(
      status == cudaSuccess,
      "shared-memory carveout failed: ", cudaGetErrorString(status));
}

template <typename Kernel>
void reject_large_local_frame(
    Kernel kernel, int variant, const char* role) {
  constexpr size_t kAcceptedLocalFrameBytes = 8;
  cudaFuncAttributes attributes{};
  const cudaError_t status =
      cudaFuncGetAttributes(&attributes, kernel);
  TORCH_CHECK(
      status == cudaSuccess,
      "kernel resource query failed: ", cudaGetErrorString(status));
  TORCH_CHECK(
      attributes.localSizeBytes <= kAcceptedLocalFrameBytes,
      "variant ", variant, " ", role, " kernel uses ",
      attributes.localSizeBytes,
      " local bytes, above the accepted 8-byte compiler frame");
}

template <int Id>
void configure_one() {
  using V = Variant<Id>;
  prefer_shared(factor_kernel<V::root, V::factor>);
  prefer_shared(solve_kernel<V::solve, V::root>);
  reject_large_local_frame(
      factor_kernel<V::root, V::factor>, Id, "factor");
  reject_large_local_frame(
      solve_kernel<V::solve, V::root>, Id, "solve");
  prefer_shared(
      fp32_update_kernel<V::update_threads, V::update>);
  reject_large_local_frame(
      fp32_update_kernel<V::update_threads, V::update>,
      Id, "update");
}

template <typename Kernel>
cudaFuncAttributes attributes_for(Kernel kernel) {
  cudaFuncAttributes attributes{};
  const cudaError_t status =
      cudaFuncGetAttributes(&attributes, kernel);
  TORCH_CHECK(
      status == cudaSuccess,
      "kernel resource query failed: ", cudaGetErrorString(status));
  return attributes;
}

void launch_copy(const float* input, float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch * 8, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &config, copy_lower_kernel, input, output);
}

template <int Id>
void launch_one(float* output, const float* input) {
  using V = Variant<Id>;
  launch_copy(input, output);
  for (int panel = 0; panel < kTileCount; ++panel) {
    cudaLaunchConfig_t factor_config{};
    factor_config.gridDim = dim3(kBatch, 1, 1);
    factor_config.blockDim = dim3(128, 1, 1);
    cudaLaunchKernelEx(
        &factor_config, factor_kernel<V::root, V::factor>,
        output, panel);

    const int remaining = kTileCount - panel - 1;
    if (remaining == 0) {
      continue;
    }

    cudaLaunchConfig_t solve_config{};
    solve_config.gridDim = dim3(kBatch * remaining, 1, 1);
    solve_config.blockDim = dim3(V::solve_threads, 1, 1);
    cudaLaunchKernelEx(
        &solve_config, solve_kernel<V::solve, V::root>,
        output, panel, remaining);

    const int tasks = remaining * (remaining + 1) / 2;
    cudaLaunchConfig_t update_config{};
    update_config.gridDim = dim3(kBatch * tasks, 1, 1);
    update_config.blockDim = dim3(V::update_threads, 1, 1);
    cudaLaunchKernelEx(
        &update_config,
        fp32_update_kernel<V::update_threads, V::update>,
        output, panel, tasks);
  }
}

void configure_variant(int variant) {
  switch (variant) {
    case 0: configure_one<0>(); break;
    case 1: configure_one<1>(); break;
    case 2: configure_one<2>(); break;
    case 3: configure_one<3>(); break;
    case 4: configure_one<4>(); break;
    case 5: configure_one<5>(); break;
    case 6: configure_one<6>(); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 6]");
  }
}

void launch_variant(
    const float* input, float* output, int variant) {
  switch (variant) {
    case 0: launch_one<0>(output, input); break;
    case 1: launch_one<1>(output, input); break;
    case 2: launch_one<2>(output, input); break;
    case 3: launch_one<3>(output, input); break;
    case 4: launch_one<4>(output, input); break;
    case 5: launch_one<5>(output, input); break;
    case 6: launch_one<6>(output, input); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 6]");
  }
}

template <int Id>
void write_metadata(int64_t* rows) {
  using V = Variant<Id>;
  const cudaFuncAttributes factor =
      attributes_for(factor_kernel<V::root, V::factor>);
  const cudaFuncAttributes solve =
      attributes_for(solve_kernel<V::solve, V::root>);
  const cudaFuncAttributes update = attributes_for(
      fp32_update_kernel<V::update_threads, V::update>);
  int64_t* row =
      rows + static_cast<int64_t>(Id) * kMetadataColumns;
  row[0] = Id;
  row[1] = 128;
  row[2] = V::solve_threads;
  row[3] = V::update_threads;
  row[4] = factor.numRegs;
  row[5] = solve.numRegs;
  row[6] = update.numRegs;
  row[7] = factor.sharedSizeBytes;
  row[8] = solve.sharedSizeBytes;
  row[9] = update.sharedSizeBytes;
  row[10] = factor.localSizeBytes;
  row[11] = solve.localSizeBytes;
  row[12] = update.localSizeBytes;
  row[13] = V::update;
  row[14] = V::root;
  row[15] = V::solve;
  row[16] = V::factor;
  row[17] = kTile;
  row[18] = 23;
}

}  // namespace

void cholesky_b16n512_prepare(int64_t variant) {
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 6]");
  configure_variant(static_cast<int>(variant));
}

void cholesky_b16n512_out(
    const at::Tensor& data, at::Tensor output, int64_t variant) {
  check_input(data);
  check_output(data, output);
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 6]");
  c10::cuda::CUDAGuard device_guard(data.device());
  launch_variant(
      data.data_ptr<float>(), output.data_ptr<float>(),
      static_cast<int>(variant));
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b16n512(
    const at::Tensor& data, int64_t variant) {
  auto output = at::empty_like(data);
  cholesky_b16n512_out(data, output, variant);
  return output;
}

at::Tensor cholesky_b16n512_metadata() {
  auto result = at::zeros(
      {kVariantCount, kMetadataColumns},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  write_metadata<0>(rows);
  write_metadata<1>(rows);
  write_metadata<2>(rows);
  write_metadata<3>(rows);
  write_metadata<4>(rows);
  write_metadata<5>(rows);
  write_metadata<6>(rows);
  return result;
}
"""


@lru_cache(maxsize=1)
def _native_module():
    tag = hashlib.sha256((_CPP_SOURCE + _CUDA_SOURCE).encode()).hexdigest()[:12]
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name=f"cholesky_b16n512_staged_{tag}",
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
        raise ValueError(f"variant must be in {_VARIANT_IDS}, got {variant}")
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
        and tuple(data.shape) == (16, 512, 512)
    ):
        return _run_variant(data, _DEFAULT_VARIANT)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
