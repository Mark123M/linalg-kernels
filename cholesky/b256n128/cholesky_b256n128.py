import hashlib
import os
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The Popcorn tuner replaces this exact line in temporary, untracked copies.
_DEFAULT_VARIANT = 23  # POPCORN_VARIANT
_VARIANT_COUNT = 26
_VARIANT_IDS = tuple(range(_VARIANT_COUNT))

_CPP_SOURCE = r"""
#include <torch/extension.h>

void cholesky_256x128_prepare();
at::Tensor cholesky_256x128(const at::Tensor& data, int64_t variant);
void cholesky_256x128_out(const at::Tensor& data,
                          at::Tensor out,
                          int64_t variant);
at::Tensor cholesky_256x128_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_256x128_prepare,
        "Configure batched 128x128 Cholesky kernels");
  m.def("run", &cholesky_256x128, "Batched 128x128 Cholesky");
  m.def("run_out", &cholesky_256x128_out,
        "Batched 128x128 Cholesky out");
  m.def("metadata", &cholesky_256x128_metadata,
        "Kernel resource metadata");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 256;
constexpr int kN = 128;
constexpr int kHalf = 64;
constexpr int kLd = 129;
constexpr int kThreads = 128;
constexpr int kSmemFloats = kN * kLd;
constexpr int kSmemBytes = kSmemFloats * static_cast<int>(sizeof(float));
constexpr int kAsyncBase = 8320;
constexpr int kVariantCount = 26;
constexpr int kMetadataColumns = 18;
constexpr unsigned kFullMask = 0xffffffffu;

constexpr int kPreciseRoot = 0;
constexpr int kRefinedRoot = 1;
constexpr int kRawRoot = 2;

constexpr int kCrout64 = 0;
constexpr int kRight64 = 1;

constexpr int kPhaseUpdate = 0;
constexpr int kRank1Update = 1;
constexpr int kSimtUpdate = 2;
constexpr int kTf32Update = 3;
constexpr int kSimtBalancedUpdate = 4;

constexpr int kL11Crout = 0;
constexpr int kL11Right = 1;
constexpr int kL11Rows = 2;
constexpr int kL11WarpTail = 3;

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (256, 128, 128)");
}

void check_output(const at::Tensor& data, const at::Tensor& out) {
  TORCH_CHECK(out.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "output must be float32");
  TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(out.sizes() == data.sizes(), "output shape must match input");
  TORCH_CHECK(out.device() == data.device(), "output device must match input");
}

template <int RootMode>
__device__ __forceinline__ void root_pair(float value,
                                          float& diagonal,
                                          float& inverse) {
  if constexpr (RootMode == kPreciseRoot) {
    diagonal = __fsqrt_rn(value);
    inverse = __fdiv_rn(1.0f, diagonal);
  } else if constexpr (RootMode == kRefinedRoot) {
    inverse = rsqrtf(value);
    inverse *= fmaf(-0.5f * value, inverse * inverse, 1.5f);
    diagonal = value * inverse;
  } else {
    inverse = rsqrtf(value);
    diagonal = value * inverse;
  }
}

template <int RootMode, bool GlobalInput, int Base>
__device__ __forceinline__ void factor64_crout(
    const float* __restrict__ input,
    float* __restrict__ tile,
    bool keep_inverse) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row0 = 2 * lane;
  const int row1 = row0 + 1;
  float state[2][kHalf];

#pragma unroll
  for (int k = 0; k < kHalf; ++k) {
    state[0][k] = 0.0f;
    state[1][k] = 0.0f;
  }

#pragma unroll
  for (int j = 0; j < kHalf; ++j) {
    float value0;
    float value1;
    if constexpr (GlobalInput) {
      value0 = input[j * kN + row0];
      value1 = input[j * kN + row1];
    } else {
      value0 = row0 >= j ? tile[(Base + row0) * kLd + Base + j] : 0.0f;
      value1 = row1 >= j ? tile[(Base + row1) * kLd + Base + j] : 0.0f;
    }

    const int pivot_lane = j >> 1;
    const int pivot_slot = j & 1;
#pragma unroll
    for (int k = 0; k < j; ++k) {
      const float pivot =
          __shfl_sync(kFullMask, state[pivot_slot][k], pivot_lane);
      value0 = fmaf(-state[0][k], pivot, value0);
      value1 = fmaf(-state[1][k], pivot, value1);
    }

    float inverse = 0.0f;
    if (lane == pivot_lane) {
      const float value = pivot_slot == 0 ? value0 : value1;
      float diagonal;
      root_pair<RootMode>(value, diagonal, inverse);
      state[pivot_slot][j] = diagonal;
      if (keep_inverse) {
        tile[j * kLd + kN] = inverse;
      }
    }
    inverse = __shfl_sync(kFullMask, inverse, pivot_lane);
    if (row0 > j) {
      state[0][j] = value0 * inverse;
    }
    if (row1 > j) {
      state[1][j] = value1 * inverse;
    }
  }

#pragma unroll
  for (int owned = 0; owned < 2; ++owned) {
    const int row = 2 * lane + owned;
#pragma unroll
    for (int column = 0; column < kHalf; ++column) {
      if (column <= row) {
        tile[(Base + row) * kLd + Base + column] =
            state[owned][column];
      }
    }
  }
}

template <bool GlobalInput, int Base>
__device__ __forceinline__ void factor64_right(
    const float* __restrict__ input,
    float* __restrict__ tile,
    bool keep_inverse) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row0 = 2 * lane;
  const int row1 = row0 + 1;
  float state[2][kHalf];

#pragma unroll
  for (int j = 0; j < kHalf; ++j) {
    if constexpr (GlobalInput) {
      const float2 loaded =
          reinterpret_cast<const float2*>(input + j * kN)[lane];
      state[0][j] = loaded.x;
      state[1][j] = loaded.y;
    } else {
      state[0][j] =
          row0 >= j ? tile[(Base + row0) * kLd + Base + j] : 0.0f;
      state[1][j] =
          row1 >= j ? tile[(Base + row1) * kLd + Base + j] : 0.0f;
    }
  }

  float inverse = 0.0f;
  if (lane == 0) {
    inverse = rsqrtf(state[0][0]);
    state[0][0] *= inverse;
    if (keep_inverse) {
      tile[kN] = inverse;
    }
  }
  inverse = __shfl_sync(kFullMask, inverse, 0);
  if (row0 > 0) state[0][0] *= inverse;
  if (row1 > 0) state[1][0] *= inverse;

#pragma unroll
  for (int k = 0; k < kHalf - 1; ++k) {
    const int next = k + 1;
    const int next_lane = next >> 1;
    const int next_slot = next & 1;
    const float next_pivot = __shfl_sync(
        kFullMask, state[next_slot][k], next_lane);
    state[0][next] = fmaf(-state[0][k], next_pivot, state[0][next]);
    state[1][next] =
        fmaf(-state[1][k], next_pivot, state[1][next]);

    float next_inverse = 0.0f;
    float next_diagonal = 0.0f;
    if (lane == next_lane) {
      next_diagonal = state[next_slot][next];
      next_inverse = rsqrtf(next_diagonal);
      if (keep_inverse) {
        tile[next * kLd + kN] = next_inverse;
      }
    }

#pragma unroll
    for (int j = k + 2; j < kHalf; ++j) {
      const int pivot_lane = j >> 1;
      const int pivot_slot = j & 1;
      const float pivot = __shfl_sync(
          kFullMask, state[pivot_slot][k], pivot_lane);
      state[0][j] = fmaf(-state[0][k], pivot, state[0][j]);
      state[1][j] = fmaf(-state[1][k], pivot, state[1][j]);
    }

    if (lane == next_lane) {
      state[next_slot][next] = next_diagonal * next_inverse;
    }
    next_inverse = __shfl_sync(kFullMask, next_inverse, next_lane);
    if (row0 > next) state[0][next] *= next_inverse;
    if (row1 > next) state[1][next] *= next_inverse;
  }

#pragma unroll
  for (int owned = 0; owned < 2; ++owned) {
    const int row = 2 * lane + owned;
#pragma unroll
    for (int column = 0; column < kHalf; ++column) {
      if (column <= row) {
        tile[(Base + row) * kLd + Base + column] =
            state[owned][column];
      }
    }
  }
}

__device__ __forceinline__ void copy_async_16(float* destination,
                                               const float* source) {
  const unsigned address =
      static_cast<unsigned>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(address), "l"(source));
}

__device__ __forceinline__ void copy_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void copy_async_wait() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

template <bool FullUnroll>
__device__ __forceinline__ void solve_panel(float* __restrict__ row_values,
                                             float* __restrict__ tile,
                                             int row) {
  if constexpr (FullUnroll) {
#pragma unroll
    for (int k = 0; k < kHalf; ++k) {
      const float solved = row_values[k] * tile[k * kLd + kN];
      row_values[k] = solved;
      tile[(kHalf + row) * kLd + k] = solved;
#pragma unroll
      for (int j = k + 1; j < kHalf; ++j) {
        row_values[j] =
            fmaf(-solved, tile[j * kLd + k], row_values[j]);
      }
    }
  } else {
#pragma unroll 8
    for (int k = 0; k < kHalf; ++k) {
      const float solved = row_values[k] * tile[k * kLd + kN];
      row_values[k] = solved;
      tile[(kHalf + row) * kLd + k] = solved;
#pragma unroll 8
      for (int j = k + 1; j < kHalf; ++j) {
        row_values[j] =
            fmaf(-solved, tile[j * kLd + k], row_values[j]);
      }
    }
  }
}

template <bool FullUnroll>
__device__ __forceinline__ void solve_panel_shared(float* tile, int row) {
  float* values = tile + (kHalf + row) * kLd;
  if constexpr (FullUnroll) {
#pragma unroll
    for (int k = 0; k < kHalf; ++k) {
      const float solved = values[k] * tile[k * kLd + kN];
      values[k] = solved;
#pragma unroll
      for (int j = k + 1; j < kHalf; ++j) {
        values[j] = fmaf(-solved, tile[j * kLd + k], values[j]);
      }
    }
  } else {
#pragma unroll 8
    for (int k = 0; k < kHalf; ++k) {
      const float solved = values[k] * tile[k * kLd + kN];
      values[k] = solved;
#pragma unroll 8
      for (int j = k + 1; j < kHalf; ++j) {
        values[j] = fmaf(-solved, tile[j * kLd + k], values[j]);
      }
    }
  }
}

__device__ __forceinline__ void update_trailing_shared(float* tile, int row) {
  float* values = tile + (kHalf + row) * kLd;
#pragma unroll 8
  for (int k = 0; k < kHalf; ++k) {
    const float own = values[k];
#pragma unroll 8
    for (int j = 0; j < kHalf; ++j) {
      if (j <= row) {
        values[kHalf + j] = fmaf(
            -own, tile[(kHalf + j) * kLd + k], values[kHalf + j]);
      }
    }
  }
}

__device__ __forceinline__ void update_trailing_rows(
    const float* __restrict__ panel,
    float* __restrict__ trailing,
    const float* __restrict__ tile,
    int row) {
#pragma unroll 8
  for (int k = 0; k < kHalf; ++k) {
    const float own = panel[k];
#pragma unroll 8
    for (int j = 0; j < kHalf; ++j) {
      if (j <= row) {
        trailing[j] = fmaf(
            -own, tile[(kHalf + j) * kLd + k], trailing[j]);
      }
    }
  }
}

template <int RootMode, int Stop>
__device__ __forceinline__ void factor_l11_rows(float* __restrict__ trailing,
                                                 float* __restrict__ tile,
                                                 int row) {
#pragma unroll 1
  for (int k = 0; k < Stop; ++k) {
    if (row == k) {
      float diagonal;
      float inverse;
      root_pair<RootMode>(trailing[k], diagonal, inverse);
      trailing[k] = diagonal;
      tile[k * kLd + kN] = inverse;
    }
    __syncthreads();
    float own = 0.0f;
    if (row >= k) {
      if (row > k) trailing[k] *= tile[k * kLd + kN];
      own = trailing[k];
      tile[(kHalf + row) * kLd + kHalf + k] = own;
    }
    __syncthreads();
    if (row > k) {
#pragma unroll 4
      for (int j = k + 1; j < kHalf; ++j) {
        if (j <= row) {
          trailing[j] = fmaf(
              -own,
              tile[(kHalf + j) * kLd + kHalf + k],
              trailing[j]);
        }
      }
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void finish_l11_warp(float* __restrict__ trailing,
                                                 float* __restrict__ tile,
                                                 int row) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (row >= 32) {
#pragma unroll
    for (int k = 32; k < kHalf; ++k) {
      const int owner = k - 32;
      float inverse = 0.0f;
      if (lane == owner) {
        inverse = rsqrtf(trailing[k]);
        trailing[k] *= inverse;
      }
      inverse = __shfl_sync(kFullMask, inverse, owner);
      if (row > k) trailing[k] *= inverse;
      const float own = row >= k ? trailing[k] : 0.0f;
#pragma unroll
      for (int j = k + 1; j < kHalf; ++j) {
        const float pivot = __shfl_sync(kFullMask, trailing[k], j - 32);
        if (j <= row) trailing[j] = fmaf(-own, pivot, trailing[j]);
      }
    }
#pragma unroll
    for (int j = 32; j < kHalf; ++j) {
      if (j <= row) {
        tile[(kHalf + row) * kLd + kHalf + j] = trailing[j];
      }
    }
  }
}

__device__ __forceinline__ void simt_trailing_update(float* tile) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll 1
  for (int task = warp; task < 10; task += 4) {
    int block_row;
    int block_column;
    if (task < 4) {
      block_row = task;
      block_column = 0;
    } else if (task < 7) {
      block_row = task - 3;
      block_column = 1;
    } else if (task < 9) {
      block_row = task - 5;
      block_column = 2;
    } else {
      block_row = 3;
      block_column = 3;
    }
    float accumulators[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int element = lane + 32 * q;
      const int local_row = element >> 4;
      const int local_column = element & 15;
      const int row = 16 * block_row + local_row;
      const int column = 16 * block_column + local_column;
      accumulators[q] = row >= column
                            ? tile[(kHalf + row) * kLd + kHalf + column]
                            : 0.0f;
    }
#pragma unroll 8
    for (int k = 0; k < kHalf; ++k) {
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int element = lane + 32 * q;
        const int row = 16 * block_row + (element >> 4);
        const int column = 16 * block_column + (element & 15);
        if (row >= column) {
          accumulators[q] = fmaf(
              -tile[(kHalf + row) * kLd + k],
              tile[(kHalf + column) * kLd + k],
              accumulators[q]);
        }
      }
    }
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int element = lane + 32 * q;
      const int row = 16 * block_row + (element >> 4);
      const int column = 16 * block_column + (element & 15);
      if (row >= column) {
        tile[(kHalf + row) * kLd + kHalf + column] = accumulators[q];
      }
    }
  }
}

__device__ __forceinline__ void packed_diagonal_coordinate(
    int packed, int& row, int& column) {
  // Invert the 16x16 lower-triangle packed index with a short decision tree.
  // This runs once per accumulator, outside the 64-step dot product.
  if (packed >= 36) {
    if (packed >= 91) {
      if (packed >= 120) {
        row = 15;
      } else if (packed >= 105) {
        row = 14;
      } else {
        row = 13;
      }
    } else if (packed >= 66) {
      row = packed >= 78 ? 12 : 11;
    } else if (packed >= 55) {
      row = 10;
    } else if (packed >= 45) {
      row = 9;
    } else {
      row = 8;
    }
  } else if (packed >= 10) {
    if (packed >= 21) {
      row = packed >= 28 ? 7 : 6;
    } else {
      row = packed >= 15 ? 5 : 4;
    }
  } else if (packed >= 3) {
    row = packed >= 6 ? 3 : 2;
  } else {
    row = packed >= 1 ? 1 : 0;
  }
  column = packed - row * (row + 1) / 2;
}

__device__ __forceinline__ void simt_full_tile_update(
    float* tile, int block_row, int block_column) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  float accumulators[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int element = lane + 32 * q;
    const int row = 16 * block_row + (element >> 4);
    const int column = 16 * block_column + (element & 15);
    accumulators[q] = tile[(kHalf + row) * kLd + kHalf + column];
  }
#pragma unroll 8
  for (int k = 0; k < kHalf; ++k) {
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int element = lane + 32 * q;
      const int row = 16 * block_row + (element >> 4);
      const int column = 16 * block_column + (element & 15);
      accumulators[q] = fmaf(
          -tile[(kHalf + row) * kLd + k],
          tile[(kHalf + column) * kLd + k], accumulators[q]);
    }
  }
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int element = lane + 32 * q;
    const int row = 16 * block_row + (element >> 4);
    const int column = 16 * block_column + (element & 15);
    tile[(kHalf + row) * kLd + kHalf + column] = accumulators[q];
  }
}

__device__ __forceinline__ void simt_diagonal_tile_update(
    float* tile, int block) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  float accumulators[5];
  int rows[5];
  int columns[5];
#pragma unroll
  for (int q = 0; q < 5; ++q) {
    const int packed = lane + 32 * q;
    if (packed < 136) {
      packed_diagonal_coordinate(packed, rows[q], columns[q]);
      const int row = 16 * block + rows[q];
      const int column = 16 * block + columns[q];
      accumulators[q] = tile[(kHalf + row) * kLd + kHalf + column];
    }
  }
#pragma unroll 8
  for (int k = 0; k < kHalf; ++k) {
#pragma unroll
    for (int q = 0; q < 5; ++q) {
      const int packed = lane + 32 * q;
      if (packed < 136) {
        const int row = 16 * block + rows[q];
        const int column = 16 * block + columns[q];
        accumulators[q] = fmaf(
            -tile[(kHalf + row) * kLd + k],
            tile[(kHalf + column) * kLd + k], accumulators[q]);
      }
    }
  }
#pragma unroll
  for (int q = 0; q < 5; ++q) {
    const int packed = lane + 32 * q;
    if (packed < 136) {
      const int row = 16 * block + rows[q];
      const int column = 16 * block + columns[q];
      tile[(kHalf + row) * kLd + kHalf + column] = accumulators[q];
    }
  }
}

__device__ __forceinline__ void simt_balanced_trailing_update(float* tile) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;

  // Six full tiles and four packed diagonal tiles are assigned as
  // 512, 512, 528, and 528 useful output elements across the four warps.
  if (warp == 0) {
    simt_full_tile_update(tile, 1, 0);
    simt_full_tile_update(tile, 3, 1);
  } else if (warp == 1) {
    simt_full_tile_update(tile, 2, 0);
    simt_full_tile_update(tile, 3, 2);
  } else if (warp == 2) {
    simt_full_tile_update(tile, 2, 1);
    simt_diagonal_tile_update(tile, 0);
    simt_diagonal_tile_update(tile, 2);
  } else {
    simt_full_tile_update(tile, 3, 0);
    simt_diagonal_tile_update(tile, 1);
    simt_diagonal_tile_update(tile, 3);
  }
}

__device__ __forceinline__ void tf32_trailing_update(float* tile) {
  using namespace nvcuda;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  // A00 is no longer needed for computation after TRSM. Temporarily use the
  // top half of the tile as an aligned WMMA operand/product area. The caller
  // refactors A00 before the output epilogue.
  float* packed = tile;
  for (int index = tid; index < kHalf * kHalf; index += kThreads) {
    const int row = index >> 6;
    const int column = index & 63;
    packed[index] = tile[(kHalf + row) * kLd + column];
  }
  __syncthreads();

#pragma unroll 1
  for (int task = warp; task < 10; task += 4) {
    int block_row;
    int block_column;
    if (task < 4) {
      block_row = task;
      block_column = 0;
    } else if (task < 7) {
      block_row = task - 3;
      block_column = 1;
    } else if (task < 9) {
      block_row = task - 5;
      block_column = 2;
    } else {
      block_row = 3;
      block_column = 3;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> product;
    wmma::fill_fragment(product, 0.0f);
#pragma unroll
    for (int k = 0; k < kHalf; k += 8) {
      wmma::fragment<wmma::matrix_a, 16, 16, 8,
                     wmma::precision::tf32, wmma::row_major> left;
      wmma::fragment<wmma::matrix_b, 16, 16, 8,
                     wmma::precision::tf32, wmma::col_major> right;
      wmma::load_matrix_sync(left, packed + block_row * 16 * kHalf + k,
                             kHalf);
      wmma::load_matrix_sync(right, packed + block_column * 16 * kHalf + k,
                             kHalf);
#pragma unroll
      for (int i = 0; i < left.num_elements; ++i) {
        left.x[i] = -wmma::__float_to_tf32(left.x[i]);
      }
#pragma unroll
      for (int i = 0; i < right.num_elements; ++i) {
        right.x[i] = wmma::__float_to_tf32(right.x[i]);
      }
      wmma::mma_sync(product, left, right, product);
    }
    float* warp_product = packed + kHalf * kHalf + warp * 256;
    wmma::store_matrix_sync(warp_product, product, 16, wmma::mem_row_major);
    __syncwarp(kFullMask);
    for (int element = lane; element < 256; element += 32) {
      const int row = 16 * block_row + (element >> 4);
      const int column = 16 * block_column + (element & 15);
      if (row >= column) {
        float& destination = tile[(kHalf + row) * kLd + kHalf + column];
        destination += warp_product[element];
      }
    }
    __syncwarp(kFullMask);
  }
}

__device__ __forceinline__ void output_tile(const float* tile, float* output) {
  const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int vector_index = tid; vector_index < kN * kN / 4;
       vector_index += kThreads) {
    const int row = vector_index >> 5;
    const int column = (vector_index & 31) * 4;
    float4 values;
    values.x = column <= row ? tile[row * kLd + column] : 0.0f;
    values.y = column + 1 <= row ? tile[row * kLd + column + 1] : 0.0f;
    values.z = column + 2 <= row ? tile[row * kLd + column + 2] : 0.0f;
    values.w = column + 3 <= row ? tile[row * kLd + column + 3] : 0.0f;
    reinterpret_cast<float4*>(output)[vector_index] = values;
  }
}

__device__ __forceinline__ void store_output_vector(
    const float* tile, float* output, int row, int column) {
  float4 values;
  values.x = column <= row ? tile[row * kLd + column] : 0.0f;
  values.y = column + 1 <= row ? tile[row * kLd + column + 1] : 0.0f;
  values.z = column + 2 <= row ? tile[row * kLd + column + 2] : 0.0f;
  values.w = column + 3 <= row ? tile[row * kLd + column + 3] : 0.0f;
  reinterpret_cast<float4*>(output)[row * (kN / 4) + column / 4] = values;
}

__device__ __forceinline__ void output_upper_right_zeros(float* output) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const float4 zeros = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int vector = lane; vector < kHalf * kHalf / 4; vector += 32) {
    const int row = vector >> 4;
    const int column = kHalf + (vector & 15) * 4;
    reinterpret_cast<float4*>(output)[row * (kN / 4) + column / 4] = zeros;
  }
}

__device__ __forceinline__ void output_completed_left(
    const float* tile, float* output) {
  const int worker = static_cast<int>(threadIdx.x) - 32;
  if (worker < 0) return;
#pragma unroll
  for (int vector = worker; vector < kN * kHalf / 4; vector += 96) {
    const int row = vector >> 4;
    const int column = (vector & 15) * 4;
    store_output_vector(tile, output, row, column);
  }
}

__device__ __forceinline__ void output_l11(const float* tile, float* output) {
  const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int vector = tid; vector < kHalf * kHalf / 4;
       vector += kThreads) {
    const int row = kHalf + (vector >> 4);
    const int column = kHalf + (vector & 15) * 4;
    store_output_vector(tile, output, row, column);
  }
}

template <int FirstFactor, int RootMode, bool FullTrsm, int UpdateMode,
          int LastFactor>
__global__ __launch_bounds__(kThreads, 2)
void shared_blocked_128_kernel(const float* __restrict__ input,
                               float* __restrict__ output) {
  static_assert(UpdateMode == kPhaseUpdate ||
                UpdateMode == kSimtUpdate ||
                UpdateMode == kTf32Update);
  static_assert(LastFactor == kL11Crout || LastFactor == kL11Right);
  extern __shared__ __align__(16) float tile[];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int row = tid >= 32 && tid < 96 ? tid - 32 : -1;
  const int matrix = static_cast<int>(blockIdx.x);
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  // The two row warps stage A10 and A11 while warp 0 factors A00. Symmetry
  // turns each fixed-column input read into one coalesced 64-float load.
  if (row >= 0) {
#pragma unroll
    for (int k = 0; k < kHalf; ++k) {
      tile[(kHalf + row) * kLd + k] = a[k * kN + kHalf + row];
      tile[(kHalf + row) * kLd + kHalf + k] =
          a[(kHalf + k) * kN + kHalf + row];
    }
  }

  if (warp == 0) {
    if constexpr (FirstFactor == kCrout64) {
      factor64_crout<RootMode, true, 0>(a, tile, true);
    } else {
      factor64_right<true, 0>(a, tile, true);
    }
  }
  __syncthreads();

  if (row >= 0) solve_panel_shared<FullTrsm>(tile, row);
  __syncthreads();

  if constexpr (UpdateMode == kPhaseUpdate) {
    if (row >= 0) update_trailing_shared(tile, row);
  } else if constexpr (UpdateMode == kSimtUpdate) {
    simt_trailing_update(tile);
  } else {
    tf32_trailing_update(tile);
  }
  __syncthreads();

  // TF32 uses the top half as an aligned operand/product area. Recomputing
  // A00 shortens the 128-register factor-state lifetime to the factor itself.
  if constexpr (UpdateMode == kTf32Update) {
    if (warp == 0) {
      if constexpr (FirstFactor == kCrout64) {
        factor64_crout<RootMode, true, 0>(a, tile, true);
      } else {
        factor64_right<true, 0>(a, tile, true);
      }
    }
    __syncthreads();
  }

  if (warp == 0) {
    if constexpr (LastFactor == kL11Crout) {
      factor64_crout<RootMode, false, kHalf>(a, tile, false);
    } else {
      factor64_right<false, kHalf>(a, tile, false);
    }
  }
  __syncthreads();
  output_tile(tile, result);
}

template <int FirstFactor, int RootMode, bool FullTrsm, int UpdateMode,
          int LastFactor, bool AsyncLoad, bool OverlapOutput = false>
__global__ __launch_bounds__(kThreads, 2)
void blocked_128_kernel(const float* __restrict__ input,
                        float* __restrict__ output) {
  static_assert(!OverlapOutput ||
                (UpdateMode == kSimtUpdate ||
                 UpdateMode == kSimtBalancedUpdate));
  static_assert(!OverlapOutput || LastFactor == kL11Crout ||
                LastFactor == kL11Right);
  extern __shared__ __align__(16) float tile[];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int row = tid >= 32 && tid < 96 ? tid - 32 : -1;
  const int matrix = static_cast<int>(blockIdx.x);
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;
  float local[2 * kHalf];

  if constexpr (AsyncLoad) {
    if (tid >= 32) {
      float* packed = tile + kAsyncBase;
      const int loader = tid - 32;
      int issued = 0;
      for (int vector = loader; vector < 2 * kHalf * kHalf / 4;
           vector += 96) {
        const bool second = vector >= kHalf * kHalf / 4;
        const int local_vector =
            second ? vector - kHalf * kHalf / 4 : vector;
        const int source_row = local_vector >> 4;
        const int source_column = (local_vector & 15) * 4;
        const float* source =
            a + (kHalf + source_row) * kN +
            (second ? kHalf + source_column : source_column);
        copy_async_16(packed + vector * 4, source);
        ++issued;
        if ((issued & 7) == 0) copy_async_commit();
      }
      if ((issued & 7) != 0) copy_async_commit();
    }
  } else if (row >= 0) {
#pragma unroll
    for (int k = 0; k < kHalf; ++k) {
      local[k] = a[k * kN + kHalf + row];
      local[kHalf + k] = a[(kHalf + k) * kN + kHalf + row];
    }
  }

  // Warp 3 has no panel row to stage. On overlap variants it initializes the
  // independent upper-right output quadrant while warp 0 factors A00.
  if constexpr (OverlapOutput) {
    if (warp == 3) output_upper_right_zeros(result);
  }

  if (warp == 0) {
    if constexpr (FirstFactor == kCrout64) {
      factor64_crout<RootMode, true, 0>(a, tile, true);
    } else {
      factor64_right<true, 0>(a, tile, true);
    }
  }

  if constexpr (AsyncLoad) {
    if (tid >= 32) copy_async_wait();
  }
  __syncthreads();

  if constexpr (AsyncLoad) {
    if (row >= 0) {
      const float* packed = tile + kAsyncBase;
      const int lane = row & 31;
#pragma unroll
      for (int step = 0; step < kHalf; ++step) {
        const int k = (step + lane) & 63;
        local[k] = packed[row * kHalf + k];
        local[kHalf + k] = packed[kHalf * kHalf + row * kHalf + k];
      }
    }
    __syncthreads();
  }

  if constexpr (UpdateMode == kRank1Update) {
#pragma unroll 1
    for (int k = 0; k < kHalf; ++k) {
      float solved = 0.0f;
      if (row >= 0) {
        solved = local[k] * tile[k * kLd + kN];
        local[k] = solved;
        tile[(kHalf + row) * kLd + k] = solved;
        if constexpr (FullTrsm) {
#pragma unroll
          for (int j = k + 1; j < kHalf; ++j) {
            local[j] = fmaf(-solved, tile[j * kLd + k], local[j]);
          }
        } else {
#pragma unroll 8
          for (int j = k + 1; j < kHalf; ++j) {
            local[j] = fmaf(-solved, tile[j * kLd + k], local[j]);
          }
        }
      }
      __syncthreads();
      if (row >= 0) {
#pragma unroll 8
        for (int j = 0; j < kHalf; ++j) {
          if (j <= row) {
            local[kHalf + j] = fmaf(
                -solved, tile[(kHalf + j) * kLd + k],
                local[kHalf + j]);
          }
        }
      }
    }
    __syncthreads();
  } else {
    if (row >= 0) solve_panel<FullTrsm>(local, tile, row);
    __syncthreads();
    if constexpr (UpdateMode == kPhaseUpdate) {
      if (row >= 0) {
        update_trailing_rows(local, local + kHalf, tile, row);
      }
    } else {
      if (row >= 0) {
#pragma unroll
        for (int j = 0; j < kHalf; ++j) {
          if (j <= row) {
            tile[(kHalf + row) * kLd + kHalf + j] = local[kHalf + j];
          }
        }
      }
      __syncthreads();
      if constexpr (UpdateMode == kSimtUpdate) {
        simt_trailing_update(tile);
      } else if constexpr (UpdateMode == kSimtBalancedUpdate) {
        simt_balanced_trailing_update(tile);
      } else {
        tf32_trailing_update(tile);
      }
      __syncthreads();
      if constexpr (UpdateMode == kTf32Update) {
        if (warp == 0) {
          if constexpr (FirstFactor == kCrout64) {
            factor64_crout<RootMode, true, 0>(a, tile, true);
          } else {
            factor64_right<true, 0>(a, tile, true);
          }
        }
        __syncthreads();
      }
    }
  }

  if constexpr (LastFactor == kL11Rows) {
    factor_l11_rows<kRawRoot, kHalf>(local + kHalf, tile, row);
  } else if constexpr (LastFactor == kL11WarpTail) {
    factor_l11_rows<kRawRoot, 32>(local + kHalf, tile, row);
    if (warp == 2) finish_l11_warp(local + kHalf, tile, row);
    __syncthreads();
  } else {
    if constexpr (UpdateMode == kPhaseUpdate ||
                  UpdateMode == kRank1Update) {
      if (row >= 0) {
#pragma unroll
        for (int j = 0; j < kHalf; ++j) {
          if (j <= row) {
            tile[(kHalf + row) * kLd + kHalf + j] = local[kHalf + j];
          }
        }
      }
      __syncthreads();
    }
    if (warp == 0) {
      if constexpr (LastFactor == kL11Crout) {
        factor64_crout<RootMode, false, kHalf>(a, tile, false);
      } else {
        factor64_right<false, kHalf>(a, tile, false);
      }
    }
    if constexpr (OverlapOutput) {
      if (warp != 0) output_completed_left(tile, result);
    }
    __syncthreads();
  }

  if constexpr (OverlapOutput) {
    output_l11(tile, result);
  } else {
    output_tile(tile, result);
  }
}

template <int RootMode>
__global__ __launch_bounds__(kThreads, 2)
void unblocked_128_kernel(const float* __restrict__ input,
                          float* __restrict__ output) {
  extern __shared__ __align__(16) float tile[];
  const int row = static_cast<int>(threadIdx.x);
  const int matrix = static_cast<int>(blockIdx.x);
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;
  float state[kN];
#pragma unroll
  for (int j = 0; j < kN; ++j) state[j] = a[j * kN + row];

#pragma unroll 1
  for (int k = 0; k < kN; ++k) {
    if (row == k) {
      float diagonal;
      float inverse;
      root_pair<RootMode>(state[k], diagonal, inverse);
      state[k] = diagonal;
      tile[k * kLd + kN] = inverse;
    }
    __syncthreads();
    float own = 0.0f;
    if (row >= k) {
      if (row > k) state[k] *= tile[k * kLd + kN];
      own = state[k];
      tile[row * kLd + k] = own;
    }
    __syncthreads();
    if (row > k) {
#pragma unroll 4
      for (int j = k + 1; j < kN; ++j) {
        if (j <= row) state[j] = fmaf(-own, tile[j * kLd + k], state[j]);
      }
    }
    __syncthreads();
  }
  output_tile(tile, result);
}

template <typename Kernel>
void configure_kernel(Kernel kernel) {
  const auto status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  TORCH_CHECK(status == cudaSuccess,
              "dynamic shared-memory configuration failed: ",
              cudaGetErrorString(status));
}

void configure_all() {
  configure_kernel(blocked_128_kernel<kCrout64, kPreciseRoot, false,
                                      kPhaseUpdate, kL11Crout, false>);
  configure_kernel(blocked_128_kernel<kCrout64, kRefinedRoot, false,
                                      kPhaseUpdate, kL11Crout, false>);
  configure_kernel(blocked_128_kernel<kCrout64, kRawRoot, false,
                                      kPhaseUpdate, kL11Crout, false>);
  configure_kernel(blocked_128_kernel<kRight64, kRawRoot, true,
                                      kPhaseUpdate, kL11Right, false>);
  configure_kernel(blocked_128_kernel<kRight64, kRawRoot, false,
                                      kPhaseUpdate, kL11Right, false>);
  configure_kernel(blocked_128_kernel<kCrout64, kRawRoot, false,
                                      kPhaseUpdate, kL11Rows, false>);
  configure_kernel(blocked_128_kernel<kCrout64, kRawRoot, false,
                                      kRank1Update, kL11Rows, false>);
  configure_kernel(blocked_128_kernel<kCrout64, kRawRoot, false,
                                      kRank1Update, kL11WarpTail, false>);
  configure_kernel(blocked_128_kernel<kRight64, kRawRoot, false,
                                      kPhaseUpdate, kL11Right, true>);
  configure_kernel(blocked_128_kernel<kRight64, kRawRoot, false,
                                      kSimtUpdate, kL11Right, false>);
  configure_kernel(blocked_128_kernel<kRight64, kRawRoot, false,
                                      kTf32Update, kL11Right, false>);
  configure_kernel(unblocked_128_kernel<kRefinedRoot>);
  configure_kernel(unblocked_128_kernel<kRawRoot>);
  configure_kernel(shared_blocked_128_kernel<kCrout64, kPreciseRoot, false,
                                             kPhaseUpdate, kL11Crout>);
  configure_kernel(shared_blocked_128_kernel<kCrout64, kRefinedRoot, false,
                                             kPhaseUpdate, kL11Crout>);
  configure_kernel(shared_blocked_128_kernel<kCrout64, kRawRoot, false,
                                             kPhaseUpdate, kL11Crout>);
  configure_kernel(shared_blocked_128_kernel<kRight64, kRawRoot, true,
                                             kPhaseUpdate, kL11Right>);
  configure_kernel(shared_blocked_128_kernel<kRight64, kRawRoot, false,
                                             kPhaseUpdate, kL11Right>);
  configure_kernel(shared_blocked_128_kernel<kRight64, kRawRoot, false,
                                             kSimtUpdate, kL11Right>);
  configure_kernel(shared_blocked_128_kernel<kRight64, kRawRoot, false,
                                             kTf32Update, kL11Right>);
  configure_kernel(blocked_128_kernel<kRight64, kRawRoot, false,
                                      kSimtUpdate, kL11Right, false, true>);
  configure_kernel(blocked_128_kernel<kCrout64, kRawRoot, false,
                                      kSimtUpdate, kL11Crout, false, false>);
  configure_kernel(blocked_128_kernel<kRight64, kRawRoot, false,
                                      kSimtBalancedUpdate, kL11Right,
                                      false, false>);
  configure_kernel(blocked_128_kernel<kRight64, kRawRoot, false,
                                      kSimtBalancedUpdate, kL11Right,
                                      false, true>);
  configure_kernel(blocked_128_kernel<kCrout64, kRawRoot, false,
                                      kSimtUpdate, kL11Crout, false, true>);
  configure_kernel(blocked_128_kernel<kCrout64, kRawRoot, false,
                                      kSimtBalancedUpdate, kL11Crout,
                                      false, true>);
}

void launch_variant(const float* input, float* output, int variant) {
  switch (variant) {
    case 0:
      blocked_128_kernel<kCrout64, kPreciseRoot, false,
                         kPhaseUpdate, kL11Crout, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 1:
      blocked_128_kernel<kCrout64, kRefinedRoot, false,
                         kPhaseUpdate, kL11Crout, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 2:
      blocked_128_kernel<kCrout64, kRawRoot, false,
                         kPhaseUpdate, kL11Crout, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 3:
      blocked_128_kernel<kRight64, kRawRoot, true,
                         kPhaseUpdate, kL11Right, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 4:
      blocked_128_kernel<kRight64, kRawRoot, false,
                         kPhaseUpdate, kL11Right, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 5:
      blocked_128_kernel<kCrout64, kRawRoot, false,
                         kPhaseUpdate, kL11Rows, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 6:
      blocked_128_kernel<kCrout64, kRawRoot, false,
                         kRank1Update, kL11Rows, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 7:
      blocked_128_kernel<kCrout64, kRawRoot, false,
                         kRank1Update, kL11WarpTail, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 8:
      blocked_128_kernel<kRight64, kRawRoot, false,
                         kPhaseUpdate, kL11Right, true>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 9:
      blocked_128_kernel<kRight64, kRawRoot, false,
                         kSimtUpdate, kL11Right, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 10:
      blocked_128_kernel<kRight64, kRawRoot, false,
                         kTf32Update, kL11Right, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 11:
      unblocked_128_kernel<kRefinedRoot>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 12:
      unblocked_128_kernel<kRawRoot>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 13:
      shared_blocked_128_kernel<kCrout64, kPreciseRoot, false,
                                kPhaseUpdate, kL11Crout>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 14:
      shared_blocked_128_kernel<kCrout64, kRefinedRoot, false,
                                kPhaseUpdate, kL11Crout>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 15:
      shared_blocked_128_kernel<kCrout64, kRawRoot, false,
                                kPhaseUpdate, kL11Crout>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 16:
      shared_blocked_128_kernel<kRight64, kRawRoot, true,
                                kPhaseUpdate, kL11Right>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 17:
      shared_blocked_128_kernel<kRight64, kRawRoot, false,
                                kPhaseUpdate, kL11Right>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 18:
      shared_blocked_128_kernel<kRight64, kRawRoot, false,
                                kSimtUpdate, kL11Right>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 19:
      shared_blocked_128_kernel<kRight64, kRawRoot, false,
                                kTf32Update, kL11Right>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 20:
      blocked_128_kernel<kRight64, kRawRoot, false,
                         kSimtUpdate, kL11Right, false, true>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 21:
      blocked_128_kernel<kCrout64, kRawRoot, false,
                         kSimtUpdate, kL11Crout, false, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 22:
      blocked_128_kernel<kRight64, kRawRoot, false,
                         kSimtBalancedUpdate, kL11Right, false, false>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 23:
      blocked_128_kernel<kRight64, kRawRoot, false,
                         kSimtBalancedUpdate, kL11Right, false, true>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 24:
      blocked_128_kernel<kCrout64, kRawRoot, false,
                         kSimtUpdate, kL11Crout, false, true>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    case 25:
      blocked_128_kernel<kCrout64, kRawRoot, false,
                         kSimtBalancedUpdate, kL11Crout, false, true>
          <<<kBatch, kThreads, kSmemBytes>>>(input, output);
      break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 25]");
  }
}

template <typename Kernel>
void query_kernel(Kernel kernel,
                  cudaFuncAttributes* attributes,
                  int* active_blocks) {
  auto status = cudaFuncGetAttributes(attributes, kernel);
  TORCH_CHECK(status == cudaSuccess,
              "cudaFuncGetAttributes failed: ", cudaGetErrorString(status));
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_blocks, kernel, kThreads, kSmemBytes);
  TORCH_CHECK(status == cudaSuccess,
              "occupancy query failed: ", cudaGetErrorString(status));
}

void fill_metadata(int64_t* rows,
                   int variant,
                   const cudaFuncAttributes& attributes,
                   int active_blocks,
                   int first_factor,
                   int root_mode,
                   int update_mode,
                   int last_factor,
                   int async_load,
                   int full_trsm,
                   int unblocked) {
  int64_t* row = rows + static_cast<int64_t>(variant) * kMetadataColumns;
  row[0] = variant;
  row[1] = kThreads;
  row[2] = attributes.numRegs;
  row[3] = attributes.localSizeBytes;
  row[4] = attributes.sharedSizeBytes;
  row[5] = kSmemBytes;
  row[6] = active_blocks;
  row[7] = active_blocks * (kThreads / 32);
  row[8] = first_factor;
  row[9] = root_mode;
  row[10] = update_mode;
  row[11] = last_factor;
  row[12] = async_load;
  row[13] = full_trsm;
  row[14] = unblocked;
  row[15] = update_mode == kTf32Update ? 1 : 0;
  row[16] = 2;
  row[17] = 1;
}

template <int FirstFactor, int RootMode, bool FullTrsm, int UpdateMode,
          int LastFactor, bool AsyncLoad, bool OverlapOutput = false>
void write_blocked_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  query_kernel(blocked_128_kernel<FirstFactor, RootMode, FullTrsm,
                                  UpdateMode, LastFactor, AsyncLoad,
                                  OverlapOutput>,
               &attributes, &active_blocks);
  fill_metadata(rows, variant, attributes, active_blocks, FirstFactor,
                RootMode, UpdateMode, LastFactor, AsyncLoad ? 1 : 0,
                FullTrsm ? 1 : 0, 0);
}

template <int FirstFactor, int RootMode, bool FullTrsm, int UpdateMode,
          int LastFactor>
void write_shared_blocked_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  query_kernel(shared_blocked_128_kernel<FirstFactor, RootMode, FullTrsm,
                                         UpdateMode, LastFactor>,
               &attributes, &active_blocks);
  fill_metadata(rows, variant, attributes, active_blocks, FirstFactor,
                RootMode, UpdateMode, LastFactor, 0,
                FullTrsm ? 1 : 0, 0);
}

template <int RootMode>
void write_unblocked_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  query_kernel(unblocked_128_kernel<RootMode>, &attributes, &active_blocks);
  fill_metadata(rows, variant, attributes, active_blocks, -1, RootMode,
                -1, -1, 0, 0, 1);
}

}  // namespace

void cholesky_256x128_prepare() {
  configure_all();
}

void cholesky_256x128_out(const at::Tensor& data,
                          at::Tensor out,
                          int64_t variant) {
  check_input(data);
  check_output(data, out);
  TORCH_CHECK(variant >= 0 && variant < kVariantCount,
              "native variant must be in [0, 25]");
  launch_variant(data.data_ptr<float>(), out.data_ptr<float>(),
                 static_cast<int>(variant));
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_256x128(const at::Tensor& data, int64_t variant) {
  auto out = at::empty_like(data);
  cholesky_256x128_out(data, out, variant);
  return out;
}

at::Tensor cholesky_256x128_metadata() {
  auto result = at::zeros(
      {kVariantCount, kMetadataColumns},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  write_blocked_metadata<kCrout64, kPreciseRoot, false,
                         kPhaseUpdate, kL11Crout, false>(rows, 0);
  write_blocked_metadata<kCrout64, kRefinedRoot, false,
                         kPhaseUpdate, kL11Crout, false>(rows, 1);
  write_blocked_metadata<kCrout64, kRawRoot, false,
                         kPhaseUpdate, kL11Crout, false>(rows, 2);
  write_blocked_metadata<kRight64, kRawRoot, true,
                         kPhaseUpdate, kL11Right, false>(rows, 3);
  write_blocked_metadata<kRight64, kRawRoot, false,
                         kPhaseUpdate, kL11Right, false>(rows, 4);
  write_blocked_metadata<kCrout64, kRawRoot, false,
                         kPhaseUpdate, kL11Rows, false>(rows, 5);
  write_blocked_metadata<kCrout64, kRawRoot, false,
                         kRank1Update, kL11Rows, false>(rows, 6);
  write_blocked_metadata<kCrout64, kRawRoot, false,
                         kRank1Update, kL11WarpTail, false>(rows, 7);
  write_blocked_metadata<kRight64, kRawRoot, false,
                         kPhaseUpdate, kL11Right, true>(rows, 8);
  write_blocked_metadata<kRight64, kRawRoot, false,
                         kSimtUpdate, kL11Right, false>(rows, 9);
  write_blocked_metadata<kRight64, kRawRoot, false,
                         kTf32Update, kL11Right, false>(rows, 10);
  write_unblocked_metadata<kRefinedRoot>(rows, 11);
  write_unblocked_metadata<kRawRoot>(rows, 12);
  write_shared_blocked_metadata<kCrout64, kPreciseRoot, false,
                                kPhaseUpdate, kL11Crout>(rows, 13);
  write_shared_blocked_metadata<kCrout64, kRefinedRoot, false,
                                kPhaseUpdate, kL11Crout>(rows, 14);
  write_shared_blocked_metadata<kCrout64, kRawRoot, false,
                                kPhaseUpdate, kL11Crout>(rows, 15);
  write_shared_blocked_metadata<kRight64, kRawRoot, true,
                                kPhaseUpdate, kL11Right>(rows, 16);
  write_shared_blocked_metadata<kRight64, kRawRoot, false,
                                kPhaseUpdate, kL11Right>(rows, 17);
  write_shared_blocked_metadata<kRight64, kRawRoot, false,
                                kSimtUpdate, kL11Right>(rows, 18);
  write_shared_blocked_metadata<kRight64, kRawRoot, false,
                                kTf32Update, kL11Right>(rows, 19);
  write_blocked_metadata<kRight64, kRawRoot, false,
                         kSimtUpdate, kL11Right, false, true>(rows, 20);
  write_blocked_metadata<kCrout64, kRawRoot, false,
                         kSimtUpdate, kL11Crout, false, false>(rows, 21);
  write_blocked_metadata<kRight64, kRawRoot, false,
                         kSimtBalancedUpdate, kL11Right,
                         false, false>(rows, 22);
  write_blocked_metadata<kRight64, kRawRoot, false,
                         kSimtBalancedUpdate, kL11Right,
                         false, true>(rows, 23);
  write_blocked_metadata<kCrout64, kRawRoot, false,
                         kSimtUpdate, kL11Crout, false, true>(rows, 24);
  write_blocked_metadata<kCrout64, kRawRoot, false,
                         kSimtBalancedUpdate, kL11Crout,
                         false, true>(rows, 25);
  return result;
}
"""


@lru_cache(maxsize=1)
def _native_module():
    tag = hashlib.sha256((_CPP_SOURCE + _CUDA_SOURCE).encode()).hexdigest()[:12]
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        module = load_inline(
            name=f"cholesky_b256n128_{tag}",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=None,
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "--use_fast_math",
                "--extra-device-vectorization",
                "-lineinfo",
                "-Xptxas=-O3,-v,-warn-spills",
                "-gencode",
                "arch=compute_100a,code=sm_100a",
            ],
            verbose=False,
        )
        module.prepare()
        return module
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


def _run_variant(
    data: torch.Tensor,
    variant: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if variant not in _VARIANT_IDS:
        raise ValueError(f"variant must be in {_VARIANT_IDS}, got {variant}")
    module = _native_module()
    if out is None:
        return module.run(data, variant)
    module.run_out(data, out, variant)
    return out


def _variant_metadata() -> torch.Tensor:
    return _native_module().metadata()


def custom_kernel(data: input_t) -> output_t:
    if tuple(data.shape) != (256, 128, 128):
        return torch.linalg.cholesky_ex(data, check_errors=False).L
    return _run_variant(data, _DEFAULT_VARIANT)
