import hashlib
import os
import re
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# Variant zero deliberately retains the measured library baseline until a
# native candidate passes the B200 promotion gate in the companion runner.
#
# Variant 14, the FP32 full-grid wavefront, is the tracked default. The
# 2026-07-29 two-round autotune measured the TF32 wavefront (15) at 1.419 ms
# against 3.272 ms for the previously promoted staged variant 11, so the
# wavefront structure wins this shape by 2.31x and only its update precision
# remained in question. Variant 14 keeps that structure with a scalar FP32
# update, avoiding the TF32 pivot exposure described in
# cholesky/b4n1024/DESIGN.md, "Precision and the secret seed". Its own time is
# not yet measured, and it has no B200 correctness or no-hang result either;
# variant 11 (3.272 ms, three-round promoted) remains the fallback.
_DEFAULT_VARIANT = 14  # POPCORN_VARIANT
_CUTLASS_BASE_VARIANT = 11
_CUTLASS_VARIANT = 13
_VARIANT_NAMES = (
    "torch_baseline",
    "rl_fixed128_custom_tf32",
    "rl_fixed512_hybrid_tf32",
    "rl_adaptive_hybrid_tf32",
    "rl_adaptive_cublas_tf32",
    "ll_fixed128_custom_tf32",
    "ll_fixed512_custom_tf32",
    "ll_adaptive_custom_tf32",
    "ll_adaptive_cublas_tf32",
    "ll_adaptive_custom_fp32",
    "rl_adaptive_hybrid_fp32",
    "ll_m128_to_m64_at_r1024_tf32",
    "ll_m128_m64_m32_at_r1024_r256_tf32",
    "ll_m128_to_m64_at_r1024_tf32_cutlass_names",
    "tilegrid64_fp32_interleaved",
    "tilegrid64_tf32_interleaved",
    "tilegrid64_tf32_batch_major",
)
_NATIVE_VARIANTS = tuple(range(1, len(_VARIANT_NAMES)))
_WAVE_PUBLIC_TO_LOCAL = {14: 0, 15: 1, 16: 2}

_METADATA_COLUMNS = (
    "variant",
    "native",
    "schedule",
    "panel_policy",
    "trsm_mode",
    "update_math",
    "factor_threads",
    "factor_registers",
    "factor_static_shared",
    "factor_local_bytes",
    "factor_dynamic_shared",
    "solve_threads",
    "solve_registers",
    "solve_static_shared",
    "solve_local_bytes",
    "solve_dynamic_shared",
    "panel_count",
    "potrf_count",
    "trsm_count",
    "gemm_count",
    "launch_count",
    "factor_active_blocks",
    "solve_active_blocks",
    "tail_policy",
    "potrf128_count",
    "potrf64_count",
    "potrf32_count",
    "trsm128_count",
    "trsm64_count",
    "trsm32_count",
)
_WAVE_METADATA_COLUMNS = (
    "wavefront_math",
    "wavefront_order",
    "wavefront_tile",
    "wavefront_threads",
    "wavefront_dynamic_shared",
    "wavefront_registers",
    "wavefront_static_shared",
    "wavefront_local_bytes",
    "wavefront_active_blocks",
    "wavefront_task_count",
    "wavefront_flag_bytes",
    "wavefront_launch_count",
)
_METADATA_COLUMNS = _METADATA_COLUMNS + _WAVE_METADATA_COLUMNS

_CPP_SOURCE = r"""
#include <torch/extension.h>

void cholesky_b8n2048_prepare(int64_t variant);
at::Tensor cholesky_b8n2048(
    const at::Tensor& data, int64_t variant);
void cholesky_b8n2048_out(
    const at::Tensor& data, at::Tensor output, int64_t variant);
at::Tensor cholesky_b8n2048_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b8n2048_prepare,
        "Configure one B200 8x2048 Cholesky variant");
  m.def("run", &cholesky_b8n2048,
        "B200 8x2048 staged Cholesky");
  m.def("run_out", &cholesky_b8n2048_out,
        "B200 8x2048 staged Cholesky out");
  m.def("metadata", &cholesky_b8n2048_metadata,
        "B200 staged-kernel resource metadata");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 8;
constexpr int kN = 2048;
constexpr int kLeaf = 128;
constexpr int kRowTile = 64;
constexpr int kVariantCount = 13;
constexpr int kMetadataColumns = 30;
constexpr int kFactorThreads = 256;
constexpr int kSolveThreads = 256;
constexpr int kFactorBytes =
    static_cast<int>(sizeof(float)) *
    (kLeaf * (kLeaf + 1) + kLeaf);
constexpr int kSolveBytes =
    static_cast<int>(sizeof(float)) *
    (32 * (kLeaf + 1) + kRowTile * (kLeaf + 4));
constexpr int kPointerWidths = 3;
constexpr int kPointerSlots = (kN / kLeaf) * kPointerWidths;
constexpr int64_t kMatrixStride =
    static_cast<int64_t>(kN) * kN;

static_assert(kFactorBytes == 66560);
static_assert(kSolveBytes == 50304);

constexpr int kRightLooking = 1;
constexpr int kLeftLooking = 2;
constexpr int kFixed128 = 0;
constexpr int kFixed512 = 1;
constexpr int kAdaptive = 2;
constexpr int kTrailingAdaptive = 3;
constexpr int kCustomSolve = 0;
constexpr int kHybridSolve = 1;
constexpr int kCublasSolve = 2;
constexpr int kFp32Math = 0;
constexpr int kTf32Math = 1;

__device__ __forceinline__ float load_global(const float* pointer) {
  return __ldcg(pointer);
}

__device__ __forceinline__ void store_global(
    float* pointer, float value) {
  __stcg(pointer, value);
}

__device__ __forceinline__ float& tile_at(
    float* tile, int row, int column) {
  return tile[row * (kLeaf + 1) + column];
}

__device__ __forceinline__ void root_pair(
    float value, float& diagonal, float& inverse) {
  diagonal = __fsqrt_rn(value);
  inverse = __fdiv_rn(1.0f, diagonal);
}

__device__ __forceinline__ void potf2_32(
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
        root_pair(
            tile_at(tile, column, column), diagonal, inverse);
        tile_at(tile, column, column) = diagonal;
        inverse_diagonal[column] = inverse;
      }
      inverse = __shfl_sync(0xffffffffu, inverse, local_column);
      if (lane > local_column) {
        const int row = begin + lane;
        tile_at(tile, row, column) *= inverse;
      }
      __syncwarp();
      if (lane > local_column) {
        const int row = begin + lane;
        const float left = tile_at(tile, row, column);
#pragma unroll 4
        for (int target_local = local_column + 1;
             target_local <= lane; ++target_local) {
          const int target = begin + target_local;
          tile_at(tile, row, target) = fmaf(
              -left, tile_at(tile, target, column),
              tile_at(tile, row, target));
        }
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

template <int Rows, int Columns, int Width>
__device__ __forceinline__ void local_trsm(
    float* tile, const float* inverse_diagonal,
    int row_begin, int column_begin) {
  const int lane = static_cast<int>(threadIdx.x) & (Width - 1);
  const int row_index = static_cast<int>(threadIdx.x) / Width;
  if (row_index < Rows) {
    const int row = row_begin + row_index;
#pragma unroll 1
    for (int local_column = 0;
         local_column < Columns; ++local_column) {
      const int column = column_begin + local_column;
      float partial = 0.0f;
#pragma unroll 4
      for (int k = lane; k < local_column; k += Width) {
        partial = fmaf(
            tile_at(tile, row, column_begin + k),
            tile_at(tile, column, column_begin + k), partial);
      }
#pragma unroll
      for (int offset = Width / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(
            0xffffffffu, partial, offset, Width);
      }
      if (lane == 0) {
        tile_at(tile, row, column) =
            (tile_at(tile, row, column) - partial) *
            inverse_diagonal[column];
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

template <int Size, int K>
__device__ __forceinline__ void local_update(
    float* tile, int target, int panel) {
  constexpr int elements = Size * Size;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < elements; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Size;
    const int column = linear % Size;
    if (column <= row) {
      float value = tile_at(tile, target + row, target + column);
#pragma unroll 4
      for (int k = 0; k < K; ++k) {
        value = fmaf(
            -tile_at(tile, target + row, panel + k),
            tile_at(tile, target + column, panel + k), value);
      }
      tile_at(tile, target + row, target + column) = value;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void factor_local(
    float* tile, float* inverse_diagonal) {
  potf2_32(tile, inverse_diagonal, 0);
  local_trsm<32, 32, 4>(tile, inverse_diagonal, 32, 0);
  local_update<32, 32>(tile, 32, 0);
  potf2_32(tile, inverse_diagonal, 32);
  local_trsm<64, 64, 4>(tile, inverse_diagonal, 64, 0);
  local_update<64, 64>(tile, 64, 0);
  potf2_32(tile, inverse_diagonal, 64);
  local_trsm<32, 32, 4>(tile, inverse_diagonal, 96, 64);
  local_update<32, 32>(tile, 96, 64);
  potf2_32(tile, inverse_diagonal, 96);
}

template <int Begin>
__global__ __launch_bounds__(kFactorThreads)
void factor_kernel(float* __restrict__ output) {
  extern __shared__ __align__(16) float work[];
  float* tile = work;
  float* inverse_diagonal = tile + kLeaf * (kLeaf + 1);
  const int matrix_index = static_cast<int>(blockIdx.x);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kMatrixStride;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kLeaf * kLeaf;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kLeaf;
    const int column = linear % kLeaf;
    tile_at(tile, row, column) =
        column <= row
            ? load_global(
                  matrix + (Begin + row) * kN + Begin + column)
            : 0.0f;
  }
  __syncthreads();
  factor_local(tile, inverse_diagonal);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kLeaf * kLeaf;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kLeaf;
    const int column = linear % kLeaf;
    if (column <= row) {
      store_global(
          matrix + (Begin + row) * kN + Begin + column,
          tile_at(tile, row, column));
    }
  }
}

template <int Block, int LocalColumn, int RegisterCount>
__device__ __forceinline__ void trsm_register_column(
    float (&values)[RegisterCount], const float* diagonal,
    const float* panel, int row, int lane) {
  constexpr int width = 4;
  constexpr int diagonal_ld = kLeaf + 1;
  constexpr int panel_ld = kLeaf + width;
  constexpr int block_begin = Block * 32;
  constexpr int column = block_begin + LocalColumn;
  constexpr int owner = LocalColumn & (width - 1);
  constexpr int owner_slot = LocalColumn / width;
  float partial = 0.0f;
#pragma unroll 4
  for (int k = lane; k < block_begin; k += width) {
    partial = fmaf(
        panel[row * panel_ld + k],
        diagonal[LocalColumn * diagonal_ld + k], partial);
  }
#pragma unroll
  for (int slot = 0; slot < RegisterCount; ++slot) {
    const int local_k = lane + slot * width;
    if (local_k < LocalColumn) {
      partial = fmaf(
          values[slot],
          diagonal[
              LocalColumn * diagonal_ld + block_begin + local_k],
          partial);
    }
  }
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(
        0xffffffffu, partial, offset, width);
  }
  const float owned_rhs = values[owner_slot];
  const float rhs = __shfl_sync(
      0xffffffffu, owned_rhs, owner, width);
  float solved = 0.0f;
  if (lane == 0) {
    solved =
        (rhs - partial) /
        diagonal[LocalColumn * diagonal_ld + column];
  }
  solved = __shfl_sync(0xffffffffu, solved, 0, width);
  if (lane == owner) {
    values[owner_slot] = solved;
  }
}

template <int Block>
__device__ __forceinline__ void trsm_register_block(
    float* matrix, int panel_begin, float* diagonal, float* panel) {
  constexpr int width = 4;
  constexpr int diagonal_ld = kLeaf + 1;
  constexpr int panel_ld = kLeaf + width;
  constexpr int block_begin = Block * 32;
  constexpr int register_count = 32 / width;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < 32 * kLeaf;
       linear += static_cast<int>(blockDim.x)) {
    const int local_row = linear / kLeaf;
    const int column = linear % kLeaf;
    const int matrix_row = block_begin + local_row;
    diagonal[local_row * diagonal_ld + column] =
        column <= matrix_row
            ? load_global(
                  matrix + (panel_begin + matrix_row) * kN +
                  panel_begin + column)
            : 0.0f;
  }
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) & (width - 1);
  const int row = static_cast<int>(threadIdx.x) / width;
  if (row < kRowTile) {
    float values[register_count];
#pragma unroll
    for (int slot = 0; slot < register_count; ++slot) {
      values[slot] =
          panel[
              row * panel_ld + block_begin +
              lane + slot * width];
    }
#define B8N2048_TRSM_COLUMN(COLUMN)                         \
    trsm_register_column<Block, COLUMN>(                    \
        values, diagonal, panel, row, lane)
    B8N2048_TRSM_COLUMN(0);
    B8N2048_TRSM_COLUMN(1);
    B8N2048_TRSM_COLUMN(2);
    B8N2048_TRSM_COLUMN(3);
    B8N2048_TRSM_COLUMN(4);
    B8N2048_TRSM_COLUMN(5);
    B8N2048_TRSM_COLUMN(6);
    B8N2048_TRSM_COLUMN(7);
    B8N2048_TRSM_COLUMN(8);
    B8N2048_TRSM_COLUMN(9);
    B8N2048_TRSM_COLUMN(10);
    B8N2048_TRSM_COLUMN(11);
    B8N2048_TRSM_COLUMN(12);
    B8N2048_TRSM_COLUMN(13);
    B8N2048_TRSM_COLUMN(14);
    B8N2048_TRSM_COLUMN(15);
    B8N2048_TRSM_COLUMN(16);
    B8N2048_TRSM_COLUMN(17);
    B8N2048_TRSM_COLUMN(18);
    B8N2048_TRSM_COLUMN(19);
    B8N2048_TRSM_COLUMN(20);
    B8N2048_TRSM_COLUMN(21);
    B8N2048_TRSM_COLUMN(22);
    B8N2048_TRSM_COLUMN(23);
    B8N2048_TRSM_COLUMN(24);
    B8N2048_TRSM_COLUMN(25);
    B8N2048_TRSM_COLUMN(26);
    B8N2048_TRSM_COLUMN(27);
    B8N2048_TRSM_COLUMN(28);
    B8N2048_TRSM_COLUMN(29);
    B8N2048_TRSM_COLUMN(30);
    B8N2048_TRSM_COLUMN(31);
#undef B8N2048_TRSM_COLUMN
#pragma unroll
    for (int slot = 0; slot < register_count; ++slot) {
      panel[
          row * panel_ld + block_begin +
          lane + slot * width] = values[slot];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void trsm_global(
    float* matrix, int row_begin, int panel_begin, float* work) {
  constexpr int width = 4;
  constexpr int diagonal_ld = kLeaf + 1;
  constexpr int panel_ld = kLeaf + width;
  float* diagonal = work;
  float* panel = diagonal + 32 * diagonal_ld;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kRowTile * kLeaf;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kLeaf;
    const int column = linear % kLeaf;
    panel[row * panel_ld + column] = load_global(
        matrix + (row_begin + row) * kN + panel_begin + column);
  }
  trsm_register_block<0>(matrix, panel_begin, diagonal, panel);
  trsm_register_block<1>(matrix, panel_begin, diagonal, panel);
  trsm_register_block<2>(matrix, panel_begin, diagonal, panel);
  trsm_register_block<3>(matrix, panel_begin, diagonal, panel);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kRowTile * kLeaf;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kLeaf;
    const int column = linear % kLeaf;
    store_global(
        matrix + (row_begin + row) * kN + panel_begin + column,
        panel[row * panel_ld + column]);
  }
}

__global__ __launch_bounds__(kSolveThreads)
void solve_kernel(
    float* __restrict__ output, int panel_begin,
    int row_begin, int row_tiles) {
  extern __shared__ __align__(16) float work[];
  const int matrix_index =
      static_cast<int>(blockIdx.x) / row_tiles;
  const int row_tile =
      static_cast<int>(blockIdx.x) % row_tiles;
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kMatrixStride;
  trsm_global(
      matrix, row_begin + row_tile * kRowTile,
      panel_begin, work);
}

template <int Width>
__global__ __launch_bounds__(Width == 32 ? 128 : 256)
void adaptive_factor_kernel(
    float* __restrict__ output, int begin) {
  static_assert(Width == 128 || Width == 64 || Width == 32);
  extern __shared__ __align__(16) float work[];
  float* tile = work;
  float* inverse_diagonal =
      tile + Width * (kLeaf + 1);
  const int matrix_index = static_cast<int>(blockIdx.x);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kMatrixStride;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Width * Width;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Width;
    const int column = linear % Width;
    tile_at(tile, row, column) =
        column <= row
            ? load_global(
                  matrix + (begin + row) * kN +
                  begin + column)
            : 0.0f;
  }
  __syncthreads();
  potf2_32(tile, inverse_diagonal, 0);
  if constexpr (Width >= 64) {
    local_trsm<32, 32, 4>(
        tile, inverse_diagonal, 32, 0);
    local_update<32, 32>(tile, 32, 0);
    potf2_32(tile, inverse_diagonal, 32);
  }
  if constexpr (Width == 128) {
    local_trsm<64, 64, 4>(
        tile, inverse_diagonal, 64, 0);
    local_update<64, 64>(tile, 64, 0);
    potf2_32(tile, inverse_diagonal, 64);
    local_trsm<32, 32, 4>(
        tile, inverse_diagonal, 96, 64);
    local_update<32, 32>(tile, 96, 64);
    potf2_32(tile, inverse_diagonal, 96);
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Width * Width;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Width;
    const int column = linear % Width;
    if (column <= row) {
      store_global(
          matrix + (begin + row) * kN + begin + column,
          tile_at(tile, row, column));
    }
  }
}

template <int RowTile, int Width>
__global__ __launch_bounds__(Width == 32 ? 128 : 256)
void adaptive_solve_kernel(
    float* __restrict__ output, int begin, int row_tiles) {
  static_assert(Width == 128 || Width == 64 || Width == 32);
  constexpr int kDiagonalLd = Width + 1;
  constexpr int kPanelLd = Width + 4;
  extern __shared__ __align__(16) float work[];
  float* diagonal = work;
  float* panel = diagonal + Width * kDiagonalLd;
  float* inverse_diagonal = panel + RowTile * kPanelLd;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / row_tiles;
  const int row_tile =
      static_cast<int>(blockIdx.x) % row_tiles;
  const int row_begin =
      begin + Width + row_tile * RowTile;
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kMatrixStride;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Width * Width;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Width;
    const int column = linear % Width;
    diagonal[row * kDiagonalLd + column] =
        column <= row
            ? load_global(
                  matrix + (begin + row) * kN +
                  begin + column)
            : 0.0f;
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < RowTile * Width;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Width;
    const int column = linear % Width;
    panel[row * kPanelLd + column] = load_global(
        matrix + (row_begin + row) * kN + begin + column);
  }
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < Width) {
    const int column = static_cast<int>(threadIdx.x);
    inverse_diagonal[column] = __fdiv_rn(
        1.0f, diagonal[column * kDiagonalLd + column]);
  }
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) & 3;
  const int row = static_cast<int>(threadIdx.x) >> 2;
  if (row < RowTile) {
#pragma unroll 1
    for (int column = 0; column < Width; ++column) {
      float partial = 0.0f;
#pragma unroll 4
      for (int k = lane; k < column; k += 4) {
        partial = fmaf(
            panel[row * kPanelLd + k],
            diagonal[column * kDiagonalLd + k], partial);
      }
      partial += __shfl_down_sync(
          0xffffffffu, partial, 2, 4);
      partial += __shfl_down_sync(
          0xffffffffu, partial, 1, 4);
      if (lane == 0) {
        panel[row * kPanelLd + column] =
            (panel[row * kPanelLd + column] - partial) *
            inverse_diagonal[column];
      }
      __syncwarp();
    }
  }
  __syncthreads();
  for (int linear = static_cast<int>(threadIdx.x);
       linear < RowTile * Width;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Width;
    const int column = linear % Width;
    store_global(
        matrix + (row_begin + row) * kN + begin + column,
        panel[row * kPanelLd + column]);
  }
}

__global__ __launch_bounds__(256)
void copy_lower_kernel(
    const float* __restrict__ input,
    float* __restrict__ output) {
  constexpr int ctas_per_matrix = 32;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / ctas_per_matrix;
  const int rank =
      static_cast<int>(blockIdx.x) % ctas_per_matrix;
  const int64_t base =
      static_cast<int64_t>(matrix_index) * kMatrixStride;
  for (int linear = rank * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
       linear < kN * kN;
       linear += ctas_per_matrix * static_cast<int>(blockDim.x)) {
    const int row = linear / kN;
    const int column = linear % kN;
    store_global(
        output + base + linear,
        column <= row ? input[base + linear] : 0.0f);
  }
}

__global__ __launch_bounds__(256)
void zero_upper_kernel(float* __restrict__ output) {
  constexpr int ctas_per_matrix = 16;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / ctas_per_matrix;
  const int rank =
      static_cast<int>(blockIdx.x) % ctas_per_matrix;
  const int64_t base =
      static_cast<int64_t>(matrix_index) * kMatrixStride;
  for (int linear = rank * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
       linear < kN * kN;
       linear += ctas_per_matrix * static_cast<int>(blockDim.x)) {
    const int row = linear / kN;
    const int column = linear % kN;
    if (column > row) {
      store_global(output + base + linear, 0.0f);
    }
  }
}

__global__ __launch_bounds__(256)
void pointer_table_kernel(
    float* output, float** a_table, float** b_table) {
  const int linear =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
      static_cast<int>(threadIdx.x);
  if (linear >= kPointerSlots * kBatch) {
    return;
  }
  const int slot = linear / kBatch;
  const int matrix_index = linear % kBatch;
  const int begin_index = slot / kPointerWidths;
  const int width_code = slot % kPointerWidths;
  const int begin = begin_index * kLeaf;
  const int width = kLeaf << width_code;
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kMatrixStride;
  if (begin + width <= kN) {
    a_table[linear] = matrix + begin * kN + begin;
    b_table[linear] = matrix + (begin + width) * kN + begin;
  } else {
    a_table[linear] = nullptr;
    b_table[linear] = nullptr;
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
        "enable cuBLAS atomics");
    check_cublas(
        cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
        "select host scalar pointers");
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

constexpr int width_code(int width) {
  return width == 128 ? 0 : (width == 256 ? 1 : 2);
}

constexpr int pointer_slot(int begin, int width) {
  return (begin / kLeaf) * kPointerWidths + width_code(width);
}

template <int Begin>
void launch_factor(float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch, 1, 1);
  config.blockDim = dim3(kFactorThreads, 1, 1);
  config.dynamicSmemBytes = kFactorBytes;
  cudaLaunchKernelEx(&config, factor_kernel<Begin>, output);
}

template <int Begin, int Width, int Rows>
void launch_custom_trsm(float* output) {
  static_assert(Width == kLeaf);
  static_assert(Rows >= 0 && Rows % kRowTile == 0);
  if constexpr (Rows > 0) {
    constexpr int row_tiles = Rows / kRowTile;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kBatch * row_tiles, 1, 1);
    config.blockDim = dim3(kSolveThreads, 1, 1);
    config.dynamicSmemBytes = kSolveBytes;
    cudaLaunchKernelEx(
        &config, solve_kernel, output, Begin,
        Begin + Width, row_tiles);
  }
}

template <int Begin, int Width, int Rows>
void launch_cublas_trsm(
    cublasHandle_t handle, float** a_table, float** b_table) {
  static_assert(
      Width == 128 || Width == 256 || Width == 512);
  static_assert(Rows >= 0);
  if constexpr (Rows > 0) {
    constexpr int slot = pointer_slot(Begin, Width);
    const float alpha = 1.0f;
    const float* const* a = reinterpret_cast<const float* const*>(
        a_table + slot * kBatch);
    float* const* b = reinterpret_cast<float* const*>(
        b_table + slot * kBatch);
    check_cublas(
        cublasStrsmBatched(
            handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_T,
            CUBLAS_DIAG_NON_UNIT,
            Width, Rows, &alpha,
            a, kN, b, kN, kBatch),
        "batched triangular solve");
  }
}

void launch_gemm_update(
    cublasHandle_t handle, float* output,
    int target_row, int target_column,
    int rows, int columns, int panel_begin, int rank,
    bool fast_tf32) {
  if (rows == 0 || columns == 0 || rank == 0) {
    return;
  }
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const float* column_panel =
      output + target_column * kN + panel_begin;
  const float* row_panel =
      output + target_row * kN + panel_begin;
  float* destination =
      output + target_row * kN + target_column;
  check_cublas(
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          columns, rows, rank,
          &alpha,
          column_panel, CUDA_R_32F, kN, kMatrixStride,
          row_panel, CUDA_R_32F, kN, kMatrixStride,
          &beta,
          destination, CUDA_R_32F, kN, kMatrixStride,
          kBatch,
          fast_tf32
              ? CUBLAS_COMPUTE_32F_FAST_TF32
              : CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT),
      "strided batched update");
}

template <int Width>
constexpr int adaptive_factor_bytes() {
  return static_cast<int>(sizeof(float)) *
      (Width * (kLeaf + 1) + Width);
}

template <int RowTile, int Width>
constexpr int adaptive_solve_bytes() {
  return static_cast<int>(sizeof(float)) *
      (Width * (Width + 1) +
       RowTile * (Width + 4) + Width);
}

template <int Width>
void launch_adaptive_left_step(
    cublasHandle_t handle, float* output, int begin) {
  constexpr int kThreads = Width == 32 ? 128 : 256;
  constexpr int kRows = Width == 128 ? 64 : Width;
  if (begin > 0) {
    launch_gemm_update(
        handle, output,
        begin, begin, kN - begin, Width, 0, begin, true);
  }
  cudaLaunchConfig_t factor_config{};
  factor_config.gridDim = dim3(kBatch, 1, 1);
  factor_config.blockDim = dim3(kThreads, 1, 1);
  factor_config.dynamicSmemBytes =
      adaptive_factor_bytes<Width>();
  cudaLaunchKernelEx(
      &factor_config, adaptive_factor_kernel<Width>,
      output, begin);
  const int trailing = kN - begin - Width;
  if (trailing == 0) {
    return;
  }
  const int row_tiles = trailing / kRows;
  cudaLaunchConfig_t solve_config{};
  solve_config.gridDim = dim3(kBatch * row_tiles, 1, 1);
  solve_config.blockDim = dim3(kThreads, 1, 1);
  solve_config.dynamicSmemBytes =
      adaptive_solve_bytes<kRows, Width>();
  cudaLaunchKernelEx(
      &solve_config, adaptive_solve_kernel<kRows, Width>,
      output, begin, row_tiles);
}

template <int Id>
void left_trailing_adaptive(
    cublasHandle_t handle, float* output) {
  int begin = 0;
  while (begin < kN) {
    const int remaining = kN - begin;
    if (remaining > 1024) {
      launch_adaptive_left_step<128>(
          handle, output, begin);
      begin += 128;
    } else if (Id == 11 || remaining > 256) {
      launch_adaptive_left_step<64>(
          handle, output, begin);
      begin += 64;
    } else {
      launch_adaptive_left_step<32>(
          handle, output, begin);
      begin += 32;
    }
  }
}

template <int Begin, int End, int SolveMode, bool FastTf32>
void factor_right_panel(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
  static_assert(Begin % kLeaf == 0 && End % kLeaf == 0);
  static_assert(Begin < End && End <= kN);
  launch_factor<Begin>(output);
  if constexpr (Begin + kLeaf < End) {
    constexpr int rows = End - Begin - kLeaf;
    if constexpr (SolveMode == kCublasSolve) {
      launch_cublas_trsm<Begin, kLeaf, rows>(
          handle, a_table, b_table);
    } else {
      launch_custom_trsm<Begin, kLeaf, rows>(output);
    }
    launch_gemm_update(
        handle, output,
        Begin + kLeaf, Begin + kLeaf,
        rows, rows, Begin, kLeaf, FastTf32);
    factor_right_panel<
        Begin + kLeaf, End, SolveMode, FastTf32>(
        handle, output, a_table, b_table);
  }
}

template <int Begin, int Width, int SolveMode, bool FastTf32>
void right_stage(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
  constexpr int end = Begin + Width;
  static_assert(end <= kN);
  factor_right_panel<
      Begin, end, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  if constexpr (end < kN) {
    constexpr int rows = kN - end;
    if constexpr (Width == kLeaf &&
                  SolveMode != kCublasSolve) {
      launch_custom_trsm<Begin, Width, rows>(output);
    } else {
      launch_cublas_trsm<Begin, Width, rows>(
          handle, a_table, b_table);
    }
    launch_gemm_update(
        handle, output,
        end, end, rows, rows, Begin, Width, FastTf32);
  }
}

template <int Begin, int End, int SolveMode, bool FastTf32>
void factor_left_panel(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
  static_assert(Begin % kLeaf == 0 && End % kLeaf == 0);
  static_assert(Begin < End && End <= kN);
  launch_factor<Begin>(output);
  if constexpr (Begin + kLeaf < kN) {
    constexpr int solve_rows = kN - Begin - kLeaf;
    if constexpr (SolveMode == kCublasSolve) {
      launch_cublas_trsm<Begin, kLeaf, solve_rows>(
          handle, a_table, b_table);
    } else {
      launch_custom_trsm<Begin, kLeaf, solve_rows>(output);
    }
  }
  if constexpr (Begin + kLeaf < End) {
    constexpr int next = Begin + kLeaf;
    constexpr int rows = kN - next;
    constexpr int columns = End - next;
    launch_gemm_update(
        handle, output,
        next, next, rows, columns, Begin, kLeaf, FastTf32);
    factor_left_panel<
        next, End, SolveMode, FastTf32>(
        handle, output, a_table, b_table);
  }
}

template <int Begin, int Width, int SolveMode, bool FastTf32>
void left_stage(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
  constexpr int end = Begin + Width;
  static_assert(end <= kN);
  if constexpr (Begin > 0) {
    launch_gemm_update(
        handle, output,
        Begin, Begin, kN - Begin, Width, 0, Begin, FastTf32);
  }
  factor_left_panel<
      Begin, end, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
}

template <int SolveMode, bool FastTf32>
void right_fixed128(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
#define B8N2048_RIGHT_128(BEGIN)                              \
  right_stage<BEGIN, 128, SolveMode, FastTf32>(               \
      handle, output, a_table, b_table)
  B8N2048_RIGHT_128(0);
  B8N2048_RIGHT_128(128);
  B8N2048_RIGHT_128(256);
  B8N2048_RIGHT_128(384);
  B8N2048_RIGHT_128(512);
  B8N2048_RIGHT_128(640);
  B8N2048_RIGHT_128(768);
  B8N2048_RIGHT_128(896);
  B8N2048_RIGHT_128(1024);
  B8N2048_RIGHT_128(1152);
  B8N2048_RIGHT_128(1280);
  B8N2048_RIGHT_128(1408);
  B8N2048_RIGHT_128(1536);
  B8N2048_RIGHT_128(1664);
  B8N2048_RIGHT_128(1792);
  B8N2048_RIGHT_128(1920);
#undef B8N2048_RIGHT_128
}

template <int SolveMode, bool FastTf32>
void right_fixed512(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
  right_stage<0, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  right_stage<512, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  right_stage<1024, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  right_stage<1536, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
}

template <int SolveMode, bool FastTf32>
void right_adaptive(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
  right_stage<0, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  right_stage<512, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  right_stage<1024, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  right_stage<1536, 256, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  right_stage<1792, 128, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  right_stage<1920, 128, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
}

template <int SolveMode, bool FastTf32>
void left_fixed128(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
#define B8N2048_LEFT_128(BEGIN)                               \
  left_stage<BEGIN, 128, SolveMode, FastTf32>(                \
      handle, output, a_table, b_table)
  B8N2048_LEFT_128(0);
  B8N2048_LEFT_128(128);
  B8N2048_LEFT_128(256);
  B8N2048_LEFT_128(384);
  B8N2048_LEFT_128(512);
  B8N2048_LEFT_128(640);
  B8N2048_LEFT_128(768);
  B8N2048_LEFT_128(896);
  B8N2048_LEFT_128(1024);
  B8N2048_LEFT_128(1152);
  B8N2048_LEFT_128(1280);
  B8N2048_LEFT_128(1408);
  B8N2048_LEFT_128(1536);
  B8N2048_LEFT_128(1664);
  B8N2048_LEFT_128(1792);
  B8N2048_LEFT_128(1920);
#undef B8N2048_LEFT_128
}

template <int SolveMode, bool FastTf32>
void left_fixed512(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
  left_stage<0, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  left_stage<512, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  left_stage<1024, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  left_stage<1536, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
}

template <int SolveMode, bool FastTf32>
void left_adaptive(
    cublasHandle_t handle, float* output,
    float** a_table, float** b_table) {
  left_stage<0, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  left_stage<512, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  left_stage<1024, 512, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  left_stage<1536, 256, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  left_stage<1792, 128, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
  left_stage<1920, 128, SolveMode, FastTf32>(
      handle, output, a_table, b_table);
}

void launch_copy(const float* input, float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch * 32, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, copy_lower_kernel, input, output);
}

void launch_zero_upper(float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch * 16, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, zero_upper_kernel, output);
}

void launch_pointer_table(
    float* output, float** a_table, float** b_table) {
  constexpr int elements = kPointerSlots * kBatch;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((elements + 255) / 256, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &config, pointer_table_kernel, output, a_table, b_table);
}

bool uses_cublas_trsm(int variant) {
  return variant == 2 || variant == 3 || variant == 4 ||
         variant == 8 || variant == 10;
}

void launch_variant(
    const float* input, float* output,
    float** a_table, float** b_table, int variant) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle);
  launch_copy(input, output);
  if (uses_cublas_trsm(variant)) {
    launch_pointer_table(output, a_table, b_table);
  }
  switch (variant) {
    case 1:
      right_fixed128<kHybridSolve, true>(
          handle, output, a_table, b_table);
      break;
    case 2:
      right_fixed512<kHybridSolve, true>(
          handle, output, a_table, b_table);
      break;
    case 3:
      right_adaptive<kHybridSolve, true>(
          handle, output, a_table, b_table);
      break;
    case 4:
      right_adaptive<kCublasSolve, true>(
          handle, output, a_table, b_table);
      break;
    case 5:
      left_fixed128<kCustomSolve, true>(
          handle, output, a_table, b_table);
      break;
    case 6:
      left_fixed512<kCustomSolve, true>(
          handle, output, a_table, b_table);
      break;
    case 7:
      left_adaptive<kCustomSolve, true>(
          handle, output, a_table, b_table);
      break;
    case 8:
      left_adaptive<kCublasSolve, true>(
          handle, output, a_table, b_table);
      break;
    case 9:
      left_adaptive<kCustomSolve, false>(
          handle, output, a_table, b_table);
      break;
    case 10:
      right_adaptive<kHybridSolve, false>(
          handle, output, a_table, b_table);
      break;
    case 11:
      left_trailing_adaptive<11>(handle, output);
      break;
    case 12:
      left_trailing_adaptive<12>(handle, output);
      break;
    default:
      TORCH_CHECK(false, "native variant must be in [1, 12]");
  }
  launch_zero_upper(output);
}

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be CUDA");
  TORCH_CHECK(
      data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      data.dim() == 3 && data.size(0) == kBatch &&
      data.size(1) == kN && data.size(2) == kN,
      "native input must have shape (8, 2048, 2048)");
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
void configure_dynamic(Kernel kernel, int dynamic_bytes) {
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      dynamic_bytes);
  TORCH_CHECK(
      status == cudaSuccess,
      "dynamic shared-memory setup failed: ",
      cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(
      status == cudaSuccess,
      "shared-memory carveout failed: ",
      cudaGetErrorString(status));
}

template <typename Kernel>
cudaFuncAttributes checked_attributes(Kernel kernel) {
  cudaFuncAttributes attributes{};
  const cudaError_t status =
      cudaFuncGetAttributes(&attributes, kernel);
  TORCH_CHECK(
      status == cudaSuccess,
      "kernel resource query failed: ",
      cudaGetErrorString(status));
  return attributes;
}

template <typename Kernel>
int active_blocks(Kernel kernel, int threads, int dynamic_bytes) {
  int active = 0;
  const cudaError_t status =
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, kernel, threads, dynamic_bytes);
  TORCH_CHECK(
      status == cudaSuccess,
      "occupancy query failed: ", cudaGetErrorString(status));
  return active;
}

void configure_all_factors() {
#define B8N2048_CONFIG_FACTOR(BEGIN)                         \
  configure_dynamic(factor_kernel<BEGIN>, kFactorBytes);     \
  checked_attributes(factor_kernel<BEGIN>)
  B8N2048_CONFIG_FACTOR(0);
  B8N2048_CONFIG_FACTOR(128);
  B8N2048_CONFIG_FACTOR(256);
  B8N2048_CONFIG_FACTOR(384);
  B8N2048_CONFIG_FACTOR(512);
  B8N2048_CONFIG_FACTOR(640);
  B8N2048_CONFIG_FACTOR(768);
  B8N2048_CONFIG_FACTOR(896);
  B8N2048_CONFIG_FACTOR(1024);
  B8N2048_CONFIG_FACTOR(1152);
  B8N2048_CONFIG_FACTOR(1280);
  B8N2048_CONFIG_FACTOR(1408);
  B8N2048_CONFIG_FACTOR(1536);
  B8N2048_CONFIG_FACTOR(1664);
  B8N2048_CONFIG_FACTOR(1792);
  B8N2048_CONFIG_FACTOR(1920);
#undef B8N2048_CONFIG_FACTOR
}

void configure_adaptive_kernels() {
  configure_dynamic(
      adaptive_factor_kernel<128>,
      adaptive_factor_bytes<128>());
  configure_dynamic(
      adaptive_factor_kernel<64>,
      adaptive_factor_bytes<64>());
  configure_dynamic(
      adaptive_factor_kernel<32>,
      adaptive_factor_bytes<32>());
  configure_dynamic(
      adaptive_solve_kernel<64, 128>,
      adaptive_solve_bytes<64, 128>());
  configure_dynamic(
      adaptive_solve_kernel<64, 64>,
      adaptive_solve_bytes<64, 64>());
  configure_dynamic(
      adaptive_solve_kernel<32, 32>,
      adaptive_solve_bytes<32, 32>());
  checked_attributes(adaptive_factor_kernel<128>);
  checked_attributes(adaptive_factor_kernel<64>);
  checked_attributes(adaptive_factor_kernel<32>);
  checked_attributes(adaptive_solve_kernel<64, 128>);
  checked_attributes(adaptive_solve_kernel<64, 64>);
  checked_attributes(adaptive_solve_kernel<32, 32>);
}

struct VariantDescription {
  int schedule;
  int panel_policy;
  int trsm_mode;
  int update_math;
  int panels;
};

VariantDescription describe_variant(int variant) {
  switch (variant) {
    case 1: return {kRightLooking, kFixed128, kCustomSolve,
                    kTf32Math, 16};
    case 2: return {kRightLooking, kFixed512, kHybridSolve,
                    kTf32Math, 4};
    case 3: return {kRightLooking, kAdaptive, kHybridSolve,
                    kTf32Math, 6};
    case 4: return {kRightLooking, kAdaptive, kCublasSolve,
                    kTf32Math, 6};
    case 5: return {kLeftLooking, kFixed128, kCustomSolve,
                    kTf32Math, 16};
    case 6: return {kLeftLooking, kFixed512, kCustomSolve,
                    kTf32Math, 4};
    case 7: return {kLeftLooking, kAdaptive, kCustomSolve,
                    kTf32Math, 6};
    case 8: return {kLeftLooking, kAdaptive, kCublasSolve,
                    kTf32Math, 6};
    case 9: return {kLeftLooking, kAdaptive, kCustomSolve,
                    kFp32Math, 6};
    case 10: return {kRightLooking, kAdaptive, kHybridSolve,
                     kFp32Math, 6};
    case 11: return {kLeftLooking, kTrailingAdaptive, kCustomSolve,
                     kTf32Math, 24};
    case 12: return {kLeftLooking, kTrailingAdaptive, kCustomSolve,
                     kTf32Math, 28};
    default: return {0, 0, 0, 0, 0};
  }
}

}  // namespace

void cholesky_b8n2048_prepare(int64_t variant) {
  TORCH_CHECK(
      variant >= 1 && variant < kVariantCount,
      "native variant must be in [1, 12]");
  configure_all_factors();
  configure_dynamic(solve_kernel, kSolveBytes);
  checked_attributes(solve_kernel);
  if (variant >= 11) {
    configure_adaptive_kernels();
  }
}

void cholesky_b8n2048_out(
    const at::Tensor& data, at::Tensor output, int64_t variant) {
  check_input(data);
  check_output(data, output);
  TORCH_CHECK(
      variant >= 1 && variant < kVariantCount,
      "native variant must be in [1, 12]");
  c10::cuda::CUDAGuard device_guard(data.device());
  at::Tensor a_pointers;
  at::Tensor b_pointers;
  float** a_table = nullptr;
  float** b_table = nullptr;
  if (uses_cublas_trsm(static_cast<int>(variant))) {
    a_pointers = at::empty(
        {kPointerSlots * kBatch},
        data.options().dtype(at::kLong));
    b_pointers = at::empty(
        {kPointerSlots * kBatch},
        data.options().dtype(at::kLong));
    a_table = reinterpret_cast<float**>(
        a_pointers.data_ptr<int64_t>());
    b_table = reinterpret_cast<float**>(
        b_pointers.data_ptr<int64_t>());
  }
  launch_variant(
      data.data_ptr<float>(), output.data_ptr<float>(),
      a_table, b_table, static_cast<int>(variant));
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b8n2048(
    const at::Tensor& data, int64_t variant) {
  auto output = at::empty_like(data);
  cholesky_b8n2048_out(data, output, variant);
  return output;
}

at::Tensor cholesky_b8n2048_metadata() {
  configure_all_factors();
  configure_dynamic(solve_kernel, kSolveBytes);
  configure_adaptive_kernels();
  auto result = at::zeros(
      {kVariantCount, kMetadataColumns},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  const cudaFuncAttributes factor =
      checked_attributes(factor_kernel<0>);
  const cudaFuncAttributes solve =
      checked_attributes(solve_kernel);
  const int factor_active = active_blocks(
      factor_kernel<0>, kFactorThreads, kFactorBytes);
  const int solve_active = active_blocks(
      solve_kernel, kSolveThreads, kSolveBytes);
  for (int variant = 0; variant < kVariantCount; ++variant) {
    int64_t* row = rows + variant * kMetadataColumns;
    row[0] = variant;
    if (variant == 0) {
      continue;
    }
    const VariantDescription description =
        describe_variant(variant);
    row[1] = 1;
    row[2] = description.schedule;
    row[3] = description.panel_policy;
    row[4] = description.trsm_mode;
    row[5] = description.update_math;
    row[6] = kFactorThreads;
    row[7] = factor.numRegs;
    row[8] = factor.sharedSizeBytes;
    row[9] = factor.localSizeBytes;
    row[10] = kFactorBytes;
    row[11] = kSolveThreads;
    row[12] = solve.numRegs;
    row[13] = solve.sharedSizeBytes;
    row[14] = solve.localSizeBytes;
    row[15] = kSolveBytes;
    row[16] = description.panels;
    const int potrf128 =
        variant >= 11 ? 8 : 16;
    const int potrf64 =
        variant == 11 ? 16 : (variant == 12 ? 12 : 0);
    const int potrf32 = variant == 12 ? 8 : 0;
    const int factors = potrf128 + potrf64 + potrf32;
    row[17] = factors;
    row[18] = factors - 1;
    row[19] = factors - 1;
    row[20] = variant >= 11
        ? 3 * factors
        : (uses_cublas_trsm(variant) ? 49 : 48);
    row[21] = factor_active;
    row[22] = solve_active;
    row[23] = variant == 11 ? 1 : (variant == 12 ? 2 : 0);
    row[24] = potrf128;
    row[25] = potrf64;
    row[26] = potrf32;
    row[27] = potrf128 > 0 && potrf64 == 0 && potrf32 == 0
        ? potrf128 - 1 : potrf128;
    row[28] = potrf64 > 0 && potrf32 == 0
        ? potrf64 - 1 : potrf64;
    row[29] = potrf32 > 0 ? potrf32 - 1 : 0;
  }
  return result;
}
"""



_CUTLASS_KERNEL_NAMES = (
    "factor_kernel",
    "solve_kernel",
    "adaptive_factor_kernel",
    "adaptive_solve_kernel",
    "copy_lower_kernel",
    "zero_upper_kernel",
    "pointer_table_kernel",
)
_CUTLASS_KERNEL_RE = re.compile(
    r"\b(" + "|".join(
        re.escape(name) for name in _CUTLASS_KERNEL_NAMES
    ) + r")\b"
)


def _cutlass_cuda_source() -> str:
    return _CUTLASS_KERNEL_RE.sub(
        lambda match: f"cutlass_{match.group(1)}", _CUDA_SOURCE
    )


@lru_cache(maxsize=1)
def _native_module():
    tag = hashlib.sha256((_CPP_SOURCE + _CUDA_SOURCE).encode()).hexdigest()[:12]
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name=f"cholesky_b8n2048_b200_{tag}",
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
                "-Xptxas=-O3,-v,-warn-spills,"
                "--allow-expensive-optimizations=true",
                "-gencode",
                "arch=compute_100a,code=sm_100a",
            ],
            extra_ldflags=["-lcublas"],
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch



@lru_cache(maxsize=1)
def _cutlass_module():
    cuda_source = _cutlass_cuda_source()
    tag = hashlib.sha256((_CPP_SOURCE + cuda_source).encode()).hexdigest()[:12]
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name=f"cholesky_b8n2048_b200_cutlass_{tag}",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=cuda_source,
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
                "-Xptxas=-O3,-v,-warn-spills,"
                "--allow-expensive-optimizations=true",
                "-gencode",
                "arch=compute_100a,code=sm_100a",
            ],
            extra_ldflags=["-lcublas"],
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


_WAVE_CPP_SOURCE = r"""
#include <torch/extension.h>

void wavefront_prepare(int64_t variant);
at::Tensor wavefront_run(const at::Tensor& data, int64_t variant);
void wavefront_run_out(
    const at::Tensor& data, at::Tensor output, int64_t variant);
at::Tensor wavefront_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &wavefront_prepare,
        "Configure one B200 full-grid Cholesky variant");
  m.def("run", &wavefront_run, "Run one B200 full-grid Cholesky");
  m.def("run_out", &wavefront_run_out,
        "Run one B200 full-grid Cholesky into an output");
  m.def("metadata", &wavefront_metadata,
        "B200 full-grid Cholesky resource metadata");
}
"""

_WAVE_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kBatch = 8;
constexpr int kN = 2048;
constexpr int kTile = 64;
constexpr int kLd = 68;
constexpr int kPanelLd = 9;
constexpr int kThreads = 256;
constexpr int kTiles = kN / kTile;
constexpr int kTasks = kTiles * (kTiles + 1) / 2;
constexpr int kDynamicBytes =
    3 * kTile * kLd * static_cast<int>(sizeof(float));
constexpr int kLocalVariants = 3;
constexpr int kMetadataColumns = 13;
constexpr int64_t kMatrixStride = static_cast<int64_t>(kN) * kN;

static_assert(kDynamicBytes == 52224);

at::Tensor gFlags;
int gFlagDevice = -1;

__device__ __forceinline__ int64_t matrix_index(
    int batch, int row, int column) {
  return static_cast<int64_t>(batch) * kMatrixStride +
      static_cast<int64_t>(row) * kN + column;
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
  return tile[row * kLd + column];
}

__device__ __forceinline__ const float& tile_at(
    const float* tile, int row, int column) {
  return tile[row * kLd + column];
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

__device__ __forceinline__ uint32_t shared_address(
    const void* pointer) {
  return static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void*>(pointer)));
}

__device__ __forceinline__ void copy_async_16(
    void* destination, const void* source) {
  const uint32_t address = shared_address(destination);
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;"
      :: "r"(address), "l"(source) : "memory");
}

__device__ __forceinline__ void commit_async_copies() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void wait_async_copies() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ int task_index(int row, int column) {
  return column * (2 * kTiles - column + 1) / 2 +
      row - column;
}

__device__ __forceinline__ void decode_task(
    int task, int& row, int& column) {
  int first = 0;
  int count = kTiles;
  column = 0;
#pragma unroll 1
  while (task >= first + count) {
    first += count;
    --count;
    ++column;
  }
  row = column + task - first;
}

__device__ __forceinline__ void wait_for_pair(
    const int* flags, int first, int second) {
  if (threadIdx.x == 0) {
    while (
        poll_flag(flags + first) == 0 ||
        (second != first && poll_flag(flags + second) == 0)) {
      __nanosleep(64);
    }
    acquire_fence();
  }
  __syncthreads();
}

__device__ __forceinline__ void stage_plain(
    float* destination, const float* source) {
  constexpr int kChunksPerRow = kTile / 4;
  constexpr int kChunks = kTile * kChunksPerRow;
  for (int chunk = static_cast<int>(threadIdx.x);
       chunk < kChunks; chunk += kThreads) {
    const int row = chunk / kChunksPerRow;
    const int column = (chunk - row * kChunksPerRow) * 4;
    copy_async_16(
        destination + row * kLd + column,
        source + static_cast<int64_t>(row) * kN + column);
  }
}

__device__ __forceinline__ void stage_swizzled(
    float* destination, const float* source) {
  constexpr int kChunksPerRow = kTile / 4;
  constexpr int kChunks = kTile * kChunksPerRow;
  for (int chunk = static_cast<int>(threadIdx.x);
       chunk < kChunks; chunk += kThreads) {
    const int row = chunk / kChunksPerRow;
    const int logical_group = chunk - row * kChunksPerRow;
    const int logical_column = logical_group * 4;
    const int physical_group = logical_group ^ (row >> 2);
    copy_async_16(
        destination + row * kLd + physical_group * 4,
        source +
            static_cast<int64_t>(row) * kN + logical_column);
  }
}

__device__ __forceinline__ float load_shared(
    uint32_t base, int index) {
  float value;
  const uint32_t address =
      base + static_cast<uint32_t>(index * sizeof(float));
  asm volatile(
      "ld.shared.f32 %0, [%1];"
      : "=f"(value) : "r"(address));
  return value;
}

__device__ __forceinline__ int swizzled_index(
    int row, int column) {
  const int physical_group =
      (column >> 2) ^ (row >> 2);
  return row * kLd + physical_group * 4 + (column & 3);
}

__device__ __forceinline__ void scalar_update(
    float value[4][4],
    const float* first, const float* second,
    int row_base, int column_base) {
  const uint32_t first_base = shared_address(first);
  const uint32_t second_base = shared_address(second);
#pragma unroll
  for (int k = 0; k < kTile; ++k) {
    float left[4];
    float right[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      left[item] = load_shared(
          first_base, swizzled_index(row_base + item, k));
      right[item] = load_shared(
          second_base, swizzled_index(column_base + item, k));
    }
#pragma unroll
    for (int local_row = 0; local_row < 4; ++local_row) {
#pragma unroll
      for (int local_column = 0;
           local_column < 4; ++local_column) {
        value[local_row][local_column] = fmaf(
            -left[local_row], right[local_column],
            value[local_row][local_column]);
      }
    }
  }
}

__device__ __forceinline__ void factor64(
    float* tile, float* inverse_diagonal, float* panel) {
  constexpr int kGroup = 8;
  constexpr unsigned kFullMask = 0xffffffffu;
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
#pragma unroll 1
  for (int base = 0; base < kTile; base += kGroup) {
    if (thread < 32) {
#pragma unroll
      for (int column = 0; column < kGroup; ++column) {
        if (lane == 0) {
          const float diagonal = __fsqrt_rn(
              tile_at(tile, base + column, base + column));
          tile_at(tile, base + column, base + column) = diagonal;
          inverse_diagonal[base + column] =
              __fdiv_rn(1.0f, diagonal);
        }
        __syncwarp(kFullMask);
        const int row = column + 1 + lane;
        if (row < kGroup) {
          tile_at(tile, base + row, base + column) *=
              inverse_diagonal[base + column];
        }
        __syncwarp(kFullMask);
        if (row < kGroup) {
          const float left =
              tile_at(tile, base + row, base + column);
#pragma unroll
          for (int target = column + 1; target <= row; ++target) {
            tile_at(tile, base + row, base + target) = fmaf(
                -left,
                tile_at(tile, base + target, base + column),
                tile_at(tile, base + row, base + target));
          }
        }
        __syncwarp(kFullMask);
      }
    }
    __syncthreads();

    const int solve_row = base + kGroup + thread;
    if (solve_row < kTile) {
      float solved[kGroup];
#pragma unroll
      for (int column = 0; column < kGroup; ++column) {
        float value =
            tile_at(tile, solve_row, base + column);
#pragma unroll
        for (int prior = 0; prior < column; ++prior) {
          value = fmaf(
              -solved[prior],
              tile_at(tile, base + column, base + prior),
              value);
        }
        solved[column] =
            value * inverse_diagonal[base + column];
      }
#pragma unroll
      for (int column = 0; column < kGroup; ++column) {
        tile_at(tile, solve_row, base + column) =
            solved[column];
        panel[solve_row * kPanelLd + column] =
            solved[column];
      }
    }
    __syncthreads();

    const int update_row = base + kGroup + (thread >> 2);
    const int quarter = thread & 3;
    if (update_row < kTile) {
      float solved[kGroup];
#pragma unroll
      for (int column = 0; column < kGroup; ++column) {
        solved[column] =
            panel[update_row * kPanelLd + column];
      }
      const int first = base + kGroup;
      for (int target = first + quarter * 4;
           target <= update_row; target += 16) {
        if (target + 3 <= update_row) {
          float value0 = tile_at(tile, update_row, target);
          float value1 = tile_at(tile, update_row, target + 1);
          float value2 = tile_at(tile, update_row, target + 2);
          float value3 = tile_at(tile, update_row, target + 3);
#pragma unroll
          for (int column = 0; column < kGroup; ++column) {
            const float left = solved[column];
            value0 = fmaf(
                -left,
                panel[target * kPanelLd + column], value0);
            value1 = fmaf(
                -left,
                panel[(target + 1) * kPanelLd + column],
                value1);
            value2 = fmaf(
                -left,
                panel[(target + 2) * kPanelLd + column],
                value2);
            value3 = fmaf(
                -left,
                panel[(target + 3) * kPanelLd + column],
                value3);
          }
          tile_at(tile, update_row, target) = value0;
          tile_at(tile, update_row, target + 1) = value1;
          tile_at(tile, update_row, target + 2) = value2;
          tile_at(tile, update_row, target + 3) = value3;
        } else {
          for (int single = target;
               single <= update_row; ++single) {
            float value = tile_at(tile, update_row, single);
#pragma unroll
            for (int column = 0; column < kGroup; ++column) {
              value = fmaf(
                  -solved[column],
                  panel[single * kPanelLd + column],
                  value);
            }
            tile_at(tile, update_row, single) = value;
          }
        }
      }
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void trsm64(
    float* tile, const float* diagonal,
    float* inverse_diagonal) {
  const int thread = static_cast<int>(threadIdx.x);
  if (thread < kTile) {
    inverse_diagonal[thread] = __fdiv_rn(
        1.0f, diagonal[thread * kLd + thread]);
  }
  __syncthreads();
#pragma unroll 1
  for (int base = 0; base < kTile; base += 8) {
    if (thread < kTile) {
      const int row = thread;
      float solved[8];
#pragma unroll
      for (int column = 0; column < 8; ++column) {
        float current =
            tile_at(tile, row, base + column);
#pragma unroll
        for (int prior = 0; prior < column; ++prior) {
          current = fmaf(
              -solved[prior],
              diagonal[
                  (base + column) * kLd + base + prior],
              current);
        }
        solved[column] =
            current * inverse_diagonal[base + column];
        tile_at(tile, row, base + column) = solved[column];
      }
    }
    __syncthreads();

    const int row = thread >> 2;
    const int lane = thread & 3;
    for (int target = base + 8 + lane;
         target < kTile; target += 4) {
      float current = tile_at(tile, row, target);
#pragma unroll
      for (int column = 0; column < 8; ++column) {
        current = fmaf(
            -tile_at(tile, row, base + column),
            diagonal[
                target * kLd + base + column],
            current);
      }
      tile_at(tile, row, target) = current;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void zero_mirror(
    float* output, int batch, int row, int column) {
  if (row == column) {
    return;
  }
  constexpr int kChunksPerRow = kTile / 4;
  constexpr int kChunks = kTile * kChunksPerRow;
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int chunk = static_cast<int>(threadIdx.x);
       chunk < kChunks; chunk += kThreads) {
    const int local_row = chunk / kChunksPerRow;
    const int local_column =
        (chunk - local_row * kChunksPerRow) * 4;
    float4* destination = reinterpret_cast<float4*>(
        output +
        matrix_index(
            batch,
            column * kTile + local_row,
            row * kTile + local_column));
    *destination = zero;
  }
}

template <bool Tf32, bool Interleaved>
__global__ __launch_bounds__(kThreads, 1)
void tilegrid64_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ all_flags) {
  const int linear = static_cast<int>(blockIdx.x);
  const int batch = Interleaved ? linear % kBatch : linear / kTasks;
  const int task = Interleaved ? linear / kBatch : linear % kTasks;
  int tile_row;
  int tile_column;
  decode_task(task, tile_row, tile_column);
  int* flags = all_flags + batch * kTasks;

  extern __shared__ __align__(16) float shared_floats[];
  float* first = shared_floats;
  float* second = first + kTile * kLd;
  float* tile = second + kTile * kLd;
  float* inverse = second;
  float* panel = second + kTile;

  zero_mirror(output, batch, tile_row, tile_column);

  if constexpr (!Tf32) {
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int row_base =
        ((warp >> 1) * 4 + (lane >> 3)) * 4;
    const int column_base =
        ((warp & 1) * 8 + (lane & 7)) * 4;
    float value[4][4];
#pragma unroll
    for (int local_row = 0; local_row < 4; ++local_row) {
#pragma unroll
      for (int local_column = 0;
           local_column < 4; ++local_column) {
        value[local_row][local_column] = load_global(
            input +
            matrix_index(
                batch,
                tile_row * kTile + row_base + local_row,
                tile_column * kTile +
                    column_base + local_column));
      }
    }

#pragma unroll 1
    for (int history = 0;
         history < tile_column; ++history) {
      wait_for_pair(
          flags,
          task_index(tile_row, history),
          task_index(tile_column, history));
      stage_swizzled(
          first,
          output +
              matrix_index(
                  batch, tile_row * kTile,
                  history * kTile));
      stage_swizzled(
          second,
          output +
              matrix_index(
                  batch, tile_column * kTile,
                  history * kTile));
      commit_async_copies();
      wait_async_copies();
      __syncthreads();
      scalar_update(
          value, first, second, row_base, column_base);
      __syncthreads();
    }

#pragma unroll
    for (int local_row = 0; local_row < 4; ++local_row) {
#pragma unroll
      for (int local_column = 0;
           local_column < 4; ++local_column) {
        tile_at(
            tile, row_base + local_row,
            column_base + local_column) =
            value[local_row][local_column];
      }
    }
    __syncthreads();
  } else {
    stage_plain(
        tile,
        input +
            matrix_index(
                batch, tile_row * kTile,
                tile_column * kTile));
    commit_async_copies();
    wait_async_copies();
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int fragment_column = warp & 3;
    const int first_fragment_row = warp >> 2;
    const int second_fragment_row = first_fragment_row + 2;
    wmma::fragment<
        wmma::accumulator, 16, 16, 8, float> accumulator0;
    wmma::fragment<
        wmma::accumulator, 16, 16, 8, float> accumulator1;
    wmma::load_matrix_sync(
        accumulator0,
        tile + first_fragment_row * 16 * kLd +
            fragment_column * 16,
        kLd, wmma::mem_row_major);
    wmma::load_matrix_sync(
        accumulator1,
        tile + second_fragment_row * 16 * kLd +
            fragment_column * 16,
        kLd, wmma::mem_row_major);

#pragma unroll 1
    for (int history = 0;
         history < tile_column; ++history) {
      wait_for_pair(
          flags,
          task_index(tile_row, history),
          task_index(tile_column, history));
      stage_plain(
          first,
          output +
              matrix_index(
                  batch, tile_row * kTile,
                  history * kTile));
      stage_plain(
          second,
          output +
              matrix_index(
                  batch, tile_column * kTile,
                  history * kTile));
      commit_async_copies();
      wait_async_copies();
      __syncthreads();

#pragma unroll
      for (int k = 0; k < kTile; k += 8) {
        wmma::fragment<
            wmma::matrix_a, 16, 16, 8,
            wmma::precision::tf32,
            wmma::row_major> left0;
        wmma::fragment<
            wmma::matrix_a, 16, 16, 8,
            wmma::precision::tf32,
            wmma::row_major> left1;
        wmma::fragment<
            wmma::matrix_b, 16, 16, 8,
            wmma::precision::tf32,
            wmma::col_major> right;
        wmma::load_matrix_sync(
            left0,
            first + first_fragment_row * 16 * kLd + k,
            kLd);
        wmma::load_matrix_sync(
            left1,
            first + second_fragment_row * 16 * kLd + k,
            kLd);
        wmma::load_matrix_sync(
            right,
            second + fragment_column * 16 * kLd + k,
            kLd);
#pragma unroll
        for (int item = 0;
             item < left0.num_elements; ++item) {
          left0.x[item] =
              -wmma::__float_to_tf32(left0.x[item]);
          left1.x[item] =
              -wmma::__float_to_tf32(left1.x[item]);
        }
#pragma unroll
        for (int item = 0;
             item < right.num_elements; ++item) {
          right.x[item] =
              wmma::__float_to_tf32(right.x[item]);
        }
        wmma::mma_sync(
            accumulator0, left0, right, accumulator0);
        wmma::mma_sync(
            accumulator1, left1, right, accumulator1);
      }
      __syncthreads();
    }

    wmma::store_matrix_sync(
        tile + first_fragment_row * 16 * kLd +
            fragment_column * 16,
        accumulator0, kLd, wmma::mem_row_major);
    wmma::store_matrix_sync(
        tile + second_fragment_row * 16 * kLd +
            fragment_column * 16,
        accumulator1, kLd, wmma::mem_row_major);
    __syncthreads();
  }

  if (tile_row == tile_column) {
    factor64(tile, inverse, panel);
  } else {
    const int diagonal_task =
        task_index(tile_column, tile_column);
    wait_for_pair(flags, diagonal_task, diagonal_task);
    stage_plain(
        first,
        output +
            matrix_index(
                batch, tile_column * kTile,
                tile_column * kTile));
    commit_async_copies();
    wait_async_copies();
    __syncthreads();
    trsm64(tile, first, inverse);
  }

  for (int linear_item = static_cast<int>(threadIdx.x);
       linear_item < kTile * kTile;
       linear_item += kThreads) {
    const int local_row = linear_item / kTile;
    const int local_column =
        linear_item - local_row * kTile;
    const float result =
        tile_row != tile_column ||
                local_column <= local_row
            ? tile_at(tile, local_row, local_column)
            : 0.0f;
    store_global(
        output +
            matrix_index(
                batch,
                tile_row * kTile + local_row,
                tile_column * kTile + local_column),
        result);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    publish_flag(flags + task);
  }
}

template <typename Kernel>
void configure_dynamic(Kernel kernel) {
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kDynamicBytes);
  TORCH_CHECK(
      status == cudaSuccess,
      "wavefront dynamic shared-memory opt-in failed: ",
      cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(
      status == cudaSuccess,
      "wavefront shared-memory carveout failed: ",
      cudaGetErrorString(status));
}

template <typename Kernel>
cudaFuncAttributes checked_attributes(Kernel kernel) {
  cudaFuncAttributes attributes{};
  const cudaError_t status =
      cudaFuncGetAttributes(&attributes, kernel);
  TORCH_CHECK(
      status == cudaSuccess,
      "wavefront resource query failed: ",
      cudaGetErrorString(status));
  return attributes;
}

template <typename Kernel>
int active_blocks(Kernel kernel) {
  int active = 0;
  const cudaError_t status =
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, kernel, kThreads, kDynamicBytes);
  TORCH_CHECK(
      status == cudaSuccess,
      "wavefront occupancy query failed: ",
      cudaGetErrorString(status));
  return active;
}

void ensure_state() {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TORCH_CHECK(
      status == cudaSuccess,
      "wavefront device query failed: ",
      cudaGetErrorString(status));
  if (gFlagDevice == device && gFlags.defined()) {
    return;
  }
  gFlags = at::empty(
      {kBatch, kTasks},
      at::TensorOptions()
          .dtype(at::kInt)
          .device(at::Device(at::kCUDA, device)));
  gFlagDevice = device;
}

void check_device() {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TORCH_CHECK(
      status == cudaSuccess,
      "wavefront device query failed: ",
      cudaGetErrorString(status));
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TORCH_CHECK(
      status == cudaSuccess,
      "wavefront property query failed: ",
      cudaGetErrorString(status));
  TORCH_CHECK(
      properties.major == 10 && properties.minor == 0,
      "wavefront variants require compute capability 10.0");
  TORCH_CHECK(
      properties.sharedMemPerBlockOptin >= kDynamicBytes,
      "wavefront variants require 52,224 dynamic shared bytes");
}

void configure_variant(int variant) {
  switch (variant) {
    case 0:
      configure_dynamic(tilegrid64_kernel<false, true>);
      TORCH_CHECK(
          active_blocks(tilegrid64_kernel<false, true>) >= 1,
          "FP32 wavefront kernel has no active CTA");
      break;
    case 1:
      configure_dynamic(tilegrid64_kernel<true, true>);
      TORCH_CHECK(
          active_blocks(tilegrid64_kernel<true, true>) >= 1,
          "TF32 wavefront kernel has no active CTA");
      break;
    case 2:
      configure_dynamic(tilegrid64_kernel<true, false>);
      TORCH_CHECK(
          active_blocks(tilegrid64_kernel<true, false>) >= 1,
          "batch-major wavefront kernel has no active CTA");
      break;
    default:
      TORCH_CHECK(false, "wavefront variant must be in [0, 2]");
  }
  check_device();
  ensure_state();
}

void launch_variant(
    const float* input, float* output, int variant) {
  ensure_state();
  int* flags = gFlags.data_ptr<int>();
  cudaError_t status = cudaMemsetAsync(
      flags, 0,
      static_cast<size_t>(kBatch) * kTasks * sizeof(int),
      nullptr);
  TORCH_CHECK(
      status == cudaSuccess,
      "wavefront flag reset failed: ",
      cudaGetErrorString(status));
  const int blocks = kBatch * kTasks;
  switch (variant) {
    case 0:
      tilegrid64_kernel<false, true>
          <<<blocks, kThreads, kDynamicBytes>>>(
              input, output, flags);
      break;
    case 1:
      tilegrid64_kernel<true, true>
          <<<blocks, kThreads, kDynamicBytes>>>(
              input, output, flags);
      break;
    case 2:
      tilegrid64_kernel<true, false>
          <<<blocks, kThreads, kDynamicBytes>>>(
              input, output, flags);
      break;
    default:
      TORCH_CHECK(false, "wavefront variant must be in [0, 2]");
  }
}

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be CUDA");
  TORCH_CHECK(
      data.scalar_type() == at::kFloat,
      "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      data.dim() == 3 && data.size(0) == kBatch &&
      data.size(1) == kN && data.size(2) == kN,
      "native input shape mismatch");
}

void check_output(
    const at::Tensor& data, const at::Tensor& output) {
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(
      output.scalar_type() == at::kFloat,
      "output must be float32");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output.sizes() == data.sizes(), "output shape mismatch");
  TORCH_CHECK(
      output.device() == data.device(), "output device mismatch");
}

template <typename Kernel>
void write_metadata_row(
    int64_t* row, int variant, int math, int order,
    Kernel kernel) {
  configure_dynamic(kernel);
  const cudaFuncAttributes attributes =
      checked_attributes(kernel);
  row[0] = variant;
  row[1] = math;
  row[2] = order;
  row[3] = kTile;
  row[4] = kThreads;
  row[5] = kDynamicBytes;
  row[6] = attributes.numRegs;
  row[7] = attributes.sharedSizeBytes;
  row[8] = attributes.localSizeBytes;
  row[9] = active_blocks(kernel);
  row[10] = static_cast<int64_t>(kBatch) * kTasks;
  row[11] =
      static_cast<int64_t>(kBatch) * kTasks * sizeof(int);
  row[12] = 2;
}

}  // namespace

void wavefront_prepare(int64_t variant) {
  TORCH_CHECK(
      variant >= 0 && variant < kLocalVariants,
      "wavefront variant must be in [0, 2]");
  configure_variant(static_cast<int>(variant));
}

void wavefront_run_out(
    const at::Tensor& data, at::Tensor output, int64_t variant) {
  check_input(data);
  check_output(data, output);
  TORCH_CHECK(
      variant >= 0 && variant < kLocalVariants,
      "wavefront variant must be in [0, 2]");
  c10::cuda::CUDAGuard device_guard(data.device());
  launch_variant(
      data.data_ptr<float>(), output.data_ptr<float>(),
      static_cast<int>(variant));
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "wavefront launch failed: ",
      cudaGetErrorString(status));
}

at::Tensor wavefront_run(
    const at::Tensor& data, int64_t variant) {
  auto output = at::empty_like(data);
  wavefront_run_out(data, output, variant);
  return output;
}

at::Tensor wavefront_metadata() {
  check_device();
  auto result = at::zeros(
      {kLocalVariants, kMetadataColumns},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  write_metadata_row(
      rows, 0, 0, 0, tilegrid64_kernel<false, true>);
  write_metadata_row(
      rows + kMetadataColumns,
      1, 1, 0, tilegrid64_kernel<true, true>);
  write_metadata_row(
      rows + 2 * kMetadataColumns,
      2, 1, 1, tilegrid64_kernel<true, false>);
  return result;
}
"""


@lru_cache(maxsize=1)
def _wavefront_module():
    tag = hashlib.sha256(
        (_WAVE_CPP_SOURCE + _WAVE_CUDA_SOURCE).encode()
    ).hexdigest()[:12]
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name=f"cholesky_b8n2048_wavefront_{tag}",
            cpp_sources=_WAVE_CPP_SOURCE,
            cuda_sources=_WAVE_CUDA_SOURCE,
            functions=None,
            extra_cflags=[
                "-O3",
                "-DNDEBUG",
                "-std=c++20",
                "-march=native",
                "-mtune=native",
                "-mfma",
                "-ffast-math",
                "-funsafe-math-optimizations",
            ],
            extra_cuda_cflags=[
                "-O3",
                "-DNDEBUG",
                "-std=c++20",
                "--use_fast_math",
                "--extra-device-vectorization",
                "--restrict",
                "-lineinfo",
                "-Xptxas=-O3,-v,-warn-spills,"
                "--allow-expensive-optimizations=true",
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


_PREPARED_VARIANTS: set[tuple[str, int]] = set()
_PREPARED_WAVE_VARIANTS: set[int] = set()


def _run_variant(
    data: torch.Tensor,
    variant: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if variant not in _NATIVE_VARIANTS:
        raise ValueError(
            f"native variant must be in {_NATIVE_VARIANTS}, got {variant}"
        )
    if variant in _WAVE_PUBLIC_TO_LOCAL:
        selected = _WAVE_PUBLIC_TO_LOCAL[variant]
        module = _wavefront_module()
        if selected not in _PREPARED_WAVE_VARIANTS:
            module.prepare(selected)
            _PREPARED_WAVE_VARIANTS.add(selected)
        if out is None:
            return module.run(data, selected)
        module.run_out(data, out, selected)
        return out
    use_cutlass = variant == _CUTLASS_VARIANT
    selected = _CUTLASS_BASE_VARIANT if use_cutlass else variant
    module_kind = "cutlass" if use_cutlass else "native"
    module = _cutlass_module() if use_cutlass else _native_module()
    prepare_key = (module_kind, selected)
    if prepare_key not in _PREPARED_VARIANTS:
        module.prepare(selected)
        _PREPARED_VARIANTS.add(prepare_key)
    if out is None:
        return module.run(data, selected)
    module.run_out(data, out, selected)
    return out


def _variant_metadata() -> torch.Tensor:
    native = _native_module().metadata()
    result = torch.zeros(
        (len(_VARIANT_NAMES), len(_METADATA_COLUMNS)),
        dtype=torch.int64,
    )
    result[: native.shape[0], : native.shape[1]] = native
    result[_CUTLASS_VARIANT, : native.shape[1]] = (
        native[_CUTLASS_BASE_VARIANT]
    )
    result[_CUTLASS_VARIANT, 0] = _CUTLASS_VARIANT
    wave = _wavefront_module().metadata()
    for public, local in _WAVE_PUBLIC_TO_LOCAL.items():
        result[public, 0] = public
        result[public, -len(_WAVE_METADATA_COLUMNS) :] = wave[local, 1:]
    return result


def custom_kernel(data: input_t) -> output_t:
    if (
        _DEFAULT_VARIANT != 0
        and data.is_cuda
        and data.dtype == torch.float32
        and data.is_contiguous()
        and tuple(data.shape) == (8, 2048, 2048)
    ):
        return _run_variant(data, _DEFAULT_VARIANT)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
