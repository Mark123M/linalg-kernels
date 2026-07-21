import hashlib
import importlib.util
import os
import sys
import tempfile
from functools import lru_cache
from getpass import getuser
from pathlib import Path

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The Popcorn tuner replaces this exact line in temporary, untracked copies.
# 8, 5, 11, 4, 7, 10
_DEFAULT_VARIANT = 6  # POPCORN_VARIANT
_VARIANT_COUNT = 24
_VARIANT_IDS = (*range(20), 22, 23)
_CUSOLVERDX_VARIANT = 19

_CPP_SOURCE = r"""
#include <torch/extension.h>

at::Tensor cholesky_1024x64(const at::Tensor& data,
                            int64_t variant);
void cholesky_1024x64_out(const at::Tensor& data,
                          at::Tensor out,
                          int64_t variant);
at::Tensor cholesky_1024x64_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &cholesky_1024x64, "Batched 64x64 Cholesky");
  m.def("run_out", &cholesky_1024x64_out, "Batched 64x64 Cholesky out");
  m.def("metadata", &cholesky_1024x64_metadata, "Kernel resource metadata");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 1024;
constexpr int kN = 64;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kPreciseRoot = 0;
constexpr int kRefinedRoot = 1;
constexpr int kRawRoot = 2;
constexpr int kMetadataColumns = 19;
constexpr int kVariantCount = 24;

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (1024, 64, 64)");
}

void check_output(const at::Tensor& data, const at::Tensor& out) {
  TORCH_CHECK(out.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "output must be float32");
  TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(out.sizes() == data.sizes(), "output shape must match input");
  TORCH_CHECK(out.device() == data.device(), "output device must match input");
}

// n = 64 exceeds the warp width, so one warp factors one matrix with two
// factor rows per lane. Every pivot broadcast therefore feeds two FMA chains,
// halving the shuffle-to-FMA ratio that limited the 32x32 kernel.
//
// RowLayout 0 (blocked): lane owns rows lane and lane + 32; the pivot for
// column j lives in lane j & 31, slot j >> 5.
// RowLayout 1 (interleaved): lane owns rows 2*lane and 2*lane + 1; the pivot
// lives in lane j >> 1, slot j & 1, and the symmetric column read collapses
// into one coalesced float2 vector load when LoadWidth == 2.
template <int Warps, int RowLayout, int LoadWidth, int RootMode,
          int DotAccumulators, int ShuffleLookahead, int MinBlocks>
__global__ __launch_bounds__(Warps * 32, MinBlocks)
void crout_64_kernel(const float* __restrict__ input,
                     float* __restrict__ output) {
  static_assert(RowLayout == 0 || RowLayout == 1);
  static_assert(LoadWidth == 1 || (LoadWidth == 2 && RowLayout == 1));
  static_assert(RootMode >= kPreciseRoot && RootMode <= kRawRoot);
  static_assert((DotAccumulators == 1 && ShuffleLookahead == 1) ||
                (DotAccumulators == 2 && ShuffleLookahead == 1) ||
                (DotAccumulators == 4 && ShuffleLookahead == 1) ||
                (DotAccumulators == 1 && ShuffleLookahead == 2));
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int matrix = static_cast<int>(blockIdx.x) * Warps + warp;
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  float row[2][kN];
#pragma unroll
  for (int owned = 0; owned < 2; ++owned) {
#pragma unroll
    for (int k = 0; k < kN; ++k) {
      row[owned][k] = 0.0f;
    }
  }

#pragma unroll
  for (int j = 0; j < kN; ++j) {
    // Reading A[j, i] instead of A[i, j] is valid for the symmetric input and
    // keeps every warp input transaction contiguous.
    float value0;
    float value1;
    if constexpr (RowLayout == 1 && LoadWidth == 2) {
      const float2 loaded =
          reinterpret_cast<const float2*>(a + j * kN)[lane];
      value0 = loaded.x;
      value1 = loaded.y;
    } else if constexpr (RowLayout == 1) {
      value0 = a[j * kN + 2 * lane];
      value1 = a[j * kN + 2 * lane + 1];
    } else {
      value0 = a[j * kN + lane];
      value1 = a[j * kN + lane + 32];
    }

    const int pivot_lane = RowLayout == 1 ? (j >> 1) : (j & 31);
    const int pivot_slot = RowLayout == 1 ? (j & 1) : (j >> 5);

    if constexpr (DotAccumulators == 1 && ShuffleLookahead == 1) {
#pragma unroll
      for (int k = 0; k < j; ++k) {
        const float pivot =
            __shfl_sync(kFullMask, row[pivot_slot][k], pivot_lane);
        value0 = fmaf(-row[0][k], pivot, value0);
        value1 = fmaf(-row[1][k], pivot, value1);
      }
    } else if constexpr (DotAccumulators == 2 && ShuffleLookahead == 1) {
      float acc0a = value0;
      float acc0b = 0.0f;
      float acc1a = value1;
      float acc1b = 0.0f;
#pragma unroll
      for (int k = 0; k < j; ++k) {
        const float pivot =
            __shfl_sync(kFullMask, row[pivot_slot][k], pivot_lane);
        if ((k & 1) == 0) {
          acc0a = fmaf(-row[0][k], pivot, acc0a);
          acc1a = fmaf(-row[1][k], pivot, acc1a);
        } else {
          acc0b = fmaf(-row[0][k], pivot, acc0b);
          acc1b = fmaf(-row[1][k], pivot, acc1b);
        }
      }
      value0 = acc0a + acc0b;
      value1 = acc1a + acc1b;
    } else if constexpr (DotAccumulators == 4 && ShuffleLookahead == 1) {
      float acc0[4] = {value0, 0.0f, 0.0f, 0.0f};
      float acc1[4] = {value1, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int k = 0; k < j; ++k) {
        const float pivot =
            __shfl_sync(kFullMask, row[pivot_slot][k], pivot_lane);
        acc0[k & 3] = fmaf(-row[0][k], pivot, acc0[k & 3]);
        acc1[k & 3] = fmaf(-row[1][k], pivot, acc1[k & 3]);
      }
      value0 = (acc0[0] + acc0[1]) + (acc0[2] + acc0[3]);
      value1 = (acc1[0] + acc1[1]) + (acc1[2] + acc1[3]);
    } else {
      int k = 0;
#pragma unroll
      for (; k + 1 < j; k += 2) {
        const float pivot0 =
            __shfl_sync(kFullMask, row[pivot_slot][k], pivot_lane);
        const float pivot1 =
            __shfl_sync(kFullMask, row[pivot_slot][k + 1], pivot_lane);
        value0 = fmaf(-row[0][k], pivot0, value0);
        value1 = fmaf(-row[1][k], pivot0, value1);
        value0 = fmaf(-row[0][k + 1], pivot1, value0);
        value1 = fmaf(-row[1][k + 1], pivot1, value1);
      }
      if (k < j) {
        const float pivot =
            __shfl_sync(kFullMask, row[pivot_slot][k], pivot_lane);
        value0 = fmaf(-row[0][k], pivot, value0);
        value1 = fmaf(-row[1][k], pivot, value1);
      }
    }

    float inverse = 0.0f;
    if (lane == pivot_lane) {
      const float diagonal_value = pivot_slot == 0 ? value0 : value1;
      float diagonal;
      if constexpr (RootMode == kRefinedRoot) {
        inverse = rsqrtf(diagonal_value);
        const float square = inverse * inverse;
        inverse *= fmaf(-0.5f * diagonal_value, square, 1.5f);
        diagonal = diagonal_value * inverse;
      } else if constexpr (RootMode == kRawRoot) {
        inverse = rsqrtf(diagonal_value);
        diagonal = diagonal_value * inverse;
      } else {
        diagonal = __fsqrt_rn(diagonal_value);
        inverse = __fdiv_rn(1.0f, diagonal);
      }
      row[pivot_slot][j] = diagonal;
    }
    inverse = __shfl_sync(kFullMask, inverse, pivot_lane);
    const int row0 = RowLayout == 1 ? 2 * lane : lane;
    const int row1 = RowLayout == 1 ? 2 * lane + 1 : lane + 32;
    if (row0 > j) {
      row[0][j] = value0 * inverse;
    }
    if (row1 > j) {
      row[1][j] = value1 * inverse;
    }
  }

  // Each output row is 256 bytes; sixteen aligned float4 writes per row zero
  // the unused upper triangle in the same kernel.
#pragma unroll
  for (int owned = 0; owned < 2; ++owned) {
    const int matrix_row =
        RowLayout == 1 ? 2 * lane + owned : lane + owned * 32;
    float* out_row = result + matrix_row * kN;
#pragma unroll
    for (int vector_index = 0; vector_index < kN / 4; ++vector_index) {
      const int column = vector_index * 4;
      float4 values;
      values.x = column <= matrix_row ? row[owned][column] : 0.0f;
      values.y = column + 1 <= matrix_row ? row[owned][column + 1] : 0.0f;
      values.z = column + 2 <= matrix_row ? row[owned][column + 2] : 0.0f;
      values.w = column + 3 <= matrix_row ? row[owned][column + 3] : 0.0f;
      reinterpret_cast<float4*>(out_row)[vector_index] = values;
    }
  }
}

// Right-looking rank-1 Cholesky with the interleaved two-rows-per-lane
// mapping. state[owned][j] begins as A[row, j], is updated once per preceding
// factor column, and becomes L[row, j] when column j is normalized. Every
// shuffle feeds updates into independent trailing-column registers instead of
// one serial dot chain.
template <int Warps, bool RootLookahead, int MinBlocks>
__global__ __launch_bounds__(Warps * 32, MinBlocks)
void right_looking_64_kernel(const float* __restrict__ input,
                             float* __restrict__ output) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int matrix = static_cast<int>(blockIdx.x) * Warps + warp;
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  float state[2][kN];
#pragma unroll
  for (int j = 0; j < kN; ++j) {
    // Symmetry converts the logical lower elements A[2*lane, j] and
    // A[2*lane + 1, j] into one coalesced row-major float2 read of A[j, .].
    const float2 loaded = reinterpret_cast<const float2*>(a + j * kN)[lane];
    state[0][j] = loaded.x;
    state[1][j] = loaded.y;
  }

  if constexpr (!RootLookahead) {
#pragma unroll
    for (int k = 0; k < kN; ++k) {
      const int owner_lane = k >> 1;
      const int owner_slot = k & 1;
      float inverse = 0.0f;
      if (lane == owner_lane) {
        const float diagonal_value = state[owner_slot][k];
        inverse = rsqrtf(diagonal_value);
        state[owner_slot][k] = diagonal_value * inverse;
      }
      inverse = __shfl_sync(kFullMask, inverse, owner_lane);
#pragma unroll
      for (int owned = 0; owned < 2; ++owned) {
        const int matrix_row = 2 * lane + owned;
        if (matrix_row > k) {
          state[owned][k] *= inverse;
        }
      }

#pragma unroll
      for (int j = k + 1; j < kN; ++j) {
        const int pivot_lane = j >> 1;
        const int pivot_slot = j & 1;
        const float pivot =
            __shfl_sync(kFullMask, state[pivot_slot][k], pivot_lane);
        state[0][j] = fmaf(-state[0][k], pivot, state[0][j]);
        state[1][j] = fmaf(-state[1][k], pivot, state[1][j]);
      }
    }
  } else {
    // Normalize column zero. Each later iteration updates the next column
    // first, issues its reciprocal square root, and fills the root latency
    // with the remaining independent rank-1 updates.
    float inverse = 0.0f;
    if (lane == 0) {
      inverse = rsqrtf(state[0][0]);
      state[0][0] *= inverse;
    }
    inverse = __shfl_sync(kFullMask, inverse, 0);
#pragma unroll
    for (int owned = 0; owned < 2; ++owned) {
      const int matrix_row = 2 * lane + owned;
      if (matrix_row > 0) {
        state[owned][0] *= inverse;
      }
    }

#pragma unroll
    for (int k = 0; k < kN - 1; ++k) {
      const int next = k + 1;
      const int next_lane = next >> 1;
      const int next_slot = next & 1;
      const float next_pivot =
          __shfl_sync(kFullMask, state[next_slot][k], next_lane);
      state[0][next] = fmaf(-state[0][k], next_pivot, state[0][next]);
      state[1][next] = fmaf(-state[1][k], next_pivot, state[1][next]);

      float next_inverse = 0.0f;
      float next_diagonal = 0.0f;
      if (lane == next_lane) {
        next_diagonal = state[next_slot][next];
        next_inverse = rsqrtf(next_diagonal);
      }

#pragma unroll
      for (int j = k + 2; j < kN; ++j) {
        const int pivot_lane = j >> 1;
        const int pivot_slot = j & 1;
        const float pivot =
            __shfl_sync(kFullMask, state[pivot_slot][k], pivot_lane);
        state[0][j] = fmaf(-state[0][k], pivot, state[0][j]);
        state[1][j] = fmaf(-state[1][k], pivot, state[1][j]);
      }

      if (lane == next_lane) {
        state[next_slot][next] = next_diagonal * next_inverse;
      }
      next_inverse = __shfl_sync(kFullMask, next_inverse, next_lane);
#pragma unroll
      for (int owned = 0; owned < 2; ++owned) {
        const int matrix_row = 2 * lane + owned;
        if (matrix_row > next) {
          state[owned][next] *= next_inverse;
        }
      }
    }
  }

#pragma unroll
  for (int owned = 0; owned < 2; ++owned) {
    const int matrix_row = 2 * lane + owned;
    float* out_row = result + matrix_row * kN;
#pragma unroll
    for (int vector_index = 0; vector_index < kN / 4; ++vector_index) {
      const int column = vector_index * 4;
      float4 values;
      values.x = column <= matrix_row ? state[owned][column] : 0.0f;
      values.y = column + 1 <= matrix_row ? state[owned][column + 1] : 0.0f;
      values.z = column + 2 <= matrix_row ? state[owned][column + 2] : 0.0f;
      values.w = column + 3 <= matrix_row ? state[owned][column + 3] : 0.0f;
      reinterpret_cast<float4*>(out_row)[vector_index] = values;
    }
  }
}

// Keep the instruction-efficient register Crout prefix, then move only the
// 16x16 late factor into shared memory. The compact tail loop replaces the
// large unrolled region where v6's instruction-fetch stalls concentrate.
template <int Warps, int MinBlocks>
__global__ __launch_bounds__(Warps * 32, MinBlocks)
void crout_hybrid_64_kernel(const float* __restrict__ input,
                            float* __restrict__ output) {
  constexpr int kTailStart = 48;
  constexpr int kTailN = kN - kTailStart;
  constexpr int kTailLd = kN + 1;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int matrix = static_cast<int>(blockIdx.x) * Warps + warp;
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  float row[2][kTailStart];
#pragma unroll
  for (int owned = 0; owned < 2; ++owned) {
#pragma unroll
    for (int k = 0; k < kTailStart; ++k) {
      row[owned][k] = 0.0f;
    }
  }

#pragma unroll
  for (int j = 0; j < kTailStart; ++j) {
    float value0 = a[j * kN + 2 * lane];
    float value1 = a[j * kN + 2 * lane + 1];
    const int pivot_lane = j >> 1;
    const int pivot_slot = j & 1;
#pragma unroll
    for (int k = 0; k < j; ++k) {
      const float pivot =
          __shfl_sync(kFullMask, row[pivot_slot][k], pivot_lane);
      value0 = fmaf(-row[0][k], pivot, value0);
      value1 = fmaf(-row[1][k], pivot, value1);
    }

    float inverse = 0.0f;
    if (lane == pivot_lane) {
      const float diagonal_value = pivot_slot == 0 ? value0 : value1;
      inverse = rsqrtf(diagonal_value);
      row[pivot_slot][j] = diagonal_value * inverse;
    }
    inverse = __shfl_sync(kFullMask, inverse, pivot_lane);
    if (2 * lane > j) {
      row[0][j] = value0 * inverse;
    }
    if (2 * lane + 1 > j) {
      row[1][j] = value1 * inverse;
    }
  }

  __shared__ float tail_tiles[Warps * kTailN * kTailLd];
  float* factor = tail_tiles + warp * kTailN * kTailLd;

  // The original owners of rows 48--63 publish their register prefix. After
  // this point lanes 0--15 each own one tail row in the compact loop.
  if (lane >= kTailStart / 2) {
    const int first_tail_row = 2 * (lane - kTailStart / 2);
#pragma unroll
    for (int owned = 0; owned < 2; ++owned) {
      const int tail_row = first_tail_row + owned;
#pragma unroll
      for (int k = 0; k < kTailStart; ++k) {
        factor[tail_row * kTailLd + k] = row[owned][k];
      }
    }
  }
  __syncwarp(kFullMask);

#pragma unroll 1
  for (int j = kTailStart; j < kN; ++j) {
    const int pivot_lane = j - kTailStart;
    float value = 0.0f;
    if (lane < kTailN) {
      value = a[j * kN + kTailStart + lane];
    }
#pragma unroll 1
    for (int k = 0; k < j; ++k) {
      if (lane < kTailN) {
        value = fmaf(-factor[lane * kTailLd + k],
                     factor[pivot_lane * kTailLd + k], value);
      }
    }

    float inverse = 0.0f;
    if (lane == pivot_lane) {
      inverse = rsqrtf(value);
      factor[lane * kTailLd + j] = value * inverse;
    }
    inverse = __shfl_sync(kFullMask, inverse, pivot_lane);
    if (lane > pivot_lane && lane < kTailN) {
      factor[lane * kTailLd + j] = value * inverse;
    }
    __syncwarp(kFullMask);
  }

  // Prefix columns remain in the original two-row register layout.
#pragma unroll
  for (int owned = 0; owned < 2; ++owned) {
    const int matrix_row = 2 * lane + owned;
    float* out_row = result + matrix_row * kN;
#pragma unroll
    for (int vector_index = 0; vector_index < kTailStart / 4; ++vector_index) {
      const int column = vector_index * 4;
      float4 values;
      values.x = column <= matrix_row ? row[owned][column] : 0.0f;
      values.y = column + 1 <= matrix_row ? row[owned][column + 1] : 0.0f;
      values.z = column + 2 <= matrix_row ? row[owned][column + 2] : 0.0f;
      values.w = column + 3 <= matrix_row ? row[owned][column + 3] : 0.0f;
      reinterpret_cast<float4*>(out_row)[vector_index] = values;
    }
#pragma unroll
    for (int vector_index = kTailStart / 4; vector_index < kN / 4;
         ++vector_index) {
      const int column = vector_index * 4;
      float4 values{0.0f, 0.0f, 0.0f, 0.0f};
      if (matrix_row >= kTailStart) {
        const int tail_row = matrix_row - kTailStart;
        values.x = column <= matrix_row
                       ? factor[tail_row * kTailLd + column]
                       : 0.0f;
        values.y = column + 1 <= matrix_row
                       ? factor[tail_row * kTailLd + column + 1]
                       : 0.0f;
        values.z = column + 2 <= matrix_row
                       ? factor[tail_row * kTailLd + column + 2]
                       : 0.0f;
        values.w = column + 3 <= matrix_row
                       ? factor[tail_row * kTailLd + column + 3]
                       : 0.0f;
      }
      reinterpret_cast<float4*>(out_row)[vector_index] = values;
    }
  }
}

// The right-looking prefix maintains the trailing Schur complement directly.
// At column 32 it can therefore publish the 32x32 remainder and reuse one
// compact shared-memory update loop instead of emitting the large SASS tail.
template <int Warps, int MinBlocks>
__global__ __launch_bounds__(Warps * 32, MinBlocks)
void right_hybrid_64_kernel(const float* __restrict__ input,
                            float* __restrict__ output) {
  constexpr int kTailStart = 32;
  constexpr int kTailN = kN - kTailStart;
  constexpr int kTailLd = kTailN + 1;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int matrix = static_cast<int>(blockIdx.x) * Warps + warp;
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  float state[2][kN];
#pragma unroll
  for (int j = 0; j < kN; ++j) {
    const float2 loaded = reinterpret_cast<const float2*>(a + j * kN)[lane];
    state[0][j] = loaded.x;
    state[1][j] = loaded.y;
  }

  float inverse = 0.0f;
  if (lane == 0) {
    inverse = rsqrtf(state[0][0]);
    state[0][0] *= inverse;
  }
  inverse = __shfl_sync(kFullMask, inverse, 0);
#pragma unroll
  for (int owned = 0; owned < 2; ++owned) {
    if (2 * lane + owned > 0) {
      state[owned][0] *= inverse;
    }
  }

  // Processing k=31 applies the last prefix update and normalizes column 32.
#pragma unroll
  for (int k = 0; k < kTailStart; ++k) {
    const int next = k + 1;
    const int next_lane = next >> 1;
    const int next_slot = next & 1;
    const float next_pivot =
        __shfl_sync(kFullMask, state[next_slot][k], next_lane);
    state[0][next] = fmaf(-state[0][k], next_pivot, state[0][next]);
    state[1][next] = fmaf(-state[1][k], next_pivot, state[1][next]);

    float next_inverse = 0.0f;
    float next_diagonal = 0.0f;
    if (lane == next_lane) {
      next_diagonal = state[next_slot][next];
      next_inverse = rsqrtf(next_diagonal);
    }
#pragma unroll
    for (int j = k + 2; j < kN; ++j) {
      const int pivot_lane = j >> 1;
      const int pivot_slot = j & 1;
      const float pivot =
          __shfl_sync(kFullMask, state[pivot_slot][k], pivot_lane);
      state[0][j] = fmaf(-state[0][k], pivot, state[0][j]);
      state[1][j] = fmaf(-state[1][k], pivot, state[1][j]);
    }
    if (lane == next_lane) {
      state[next_slot][next] = next_diagonal * next_inverse;
    }
    next_inverse = __shfl_sync(kFullMask, next_inverse, next_lane);
#pragma unroll
    for (int owned = 0; owned < 2; ++owned) {
      if (2 * lane + owned > next) {
        state[owned][next] *= next_inverse;
      }
    }
  }

  __shared__ float tail_tiles[Warps * kTailN * kTailLd];
  float* factor = tail_tiles + warp * kTailN * kTailLd;
  if (lane >= kTailStart / 2) {
    const int first_tail_row = 2 * (lane - kTailStart / 2);
#pragma unroll
    for (int owned = 0; owned < 2; ++owned) {
      const int tail_row = first_tail_row + owned;
#pragma unroll
      for (int column = 0; column < kTailN; ++column) {
        if (column <= tail_row) {
          factor[tail_row * kTailLd + column] =
              state[owned][kTailStart + column];
        }
      }
    }
  }
  __syncwarp(kFullMask);

  // Column zero of the tile (global column 32) was normalized by the prefix.
#pragma unroll 1
  for (int k = 0; k < kTailN - 1; ++k) {
    const int next = k + 1;
    float own_factor = 0.0f;
    if (lane >= k) {
      own_factor = factor[lane * kTailLd + k];
    }
    const float next_pivot = factor[next * kTailLd + k];
    if (lane >= next) {
      factor[lane * kTailLd + next] =
          fmaf(-own_factor, next_pivot,
               factor[lane * kTailLd + next]);
    }

    float next_inverse = 0.0f;
    float next_diagonal = 0.0f;
    if (lane == next) {
      next_diagonal = factor[next * kTailLd + next];
      next_inverse = rsqrtf(next_diagonal);
    }
#pragma unroll 1
    for (int j = k + 2; j < kTailN; ++j) {
      const float pivot = factor[j * kTailLd + k];
      if (lane >= j) {
        factor[lane * kTailLd + j] =
            fmaf(-own_factor, pivot, factor[lane * kTailLd + j]);
      }
    }
    if (lane == next) {
      factor[next * kTailLd + next] = next_diagonal * next_inverse;
    }
    next_inverse = __shfl_sync(kFullMask, next_inverse, next);
    if (lane > next) {
      factor[lane * kTailLd + next] *= next_inverse;
    }
    __syncwarp(kFullMask);
  }

#pragma unroll
  for (int owned = 0; owned < 2; ++owned) {
    const int matrix_row = 2 * lane + owned;
    float* out_row = result + matrix_row * kN;
#pragma unroll
    for (int vector_index = 0; vector_index < kTailStart / 4; ++vector_index) {
      const int column = vector_index * 4;
      float4 values;
      values.x = column <= matrix_row ? state[owned][column] : 0.0f;
      values.y = column + 1 <= matrix_row ? state[owned][column + 1] : 0.0f;
      values.z = column + 2 <= matrix_row ? state[owned][column + 2] : 0.0f;
      values.w = column + 3 <= matrix_row ? state[owned][column + 3] : 0.0f;
      reinterpret_cast<float4*>(out_row)[vector_index] = values;
    }
#pragma unroll
    for (int vector_index = kTailStart / 4; vector_index < kN / 4;
         ++vector_index) {
      const int column = vector_index * 4;
      float4 values{0.0f, 0.0f, 0.0f, 0.0f};
      if (matrix_row >= kTailStart) {
        const int tail_row = matrix_row - kTailStart;
        const int tail_column = column - kTailStart;
        values.x = tail_column <= tail_row
                       ? factor[tail_row * kTailLd + tail_column]
                       : 0.0f;
        values.y = tail_column + 1 <= tail_row
                       ? factor[tail_row * kTailLd + tail_column + 1]
                       : 0.0f;
        values.z = tail_column + 2 <= tail_row
                       ? factor[tail_row * kTailLd + tail_column + 2]
                       : 0.0f;
        values.w = tail_column + 3 <= tail_row
                       ? factor[tail_row * kTailLd + tail_column + 3]
                       : 0.0f;
      }
      reinterpret_cast<float4*>(out_row)[vector_index] = values;
    }
  }
}

// Block-per-matrix factorization in shared memory: 64 threads own one row
// each and cooperate through barriers instead of shuffles. The loop body
// stays tiny (dynamic column index into shared memory), so this family is
// the instruction-cache and occupancy hedge against the fully unrolled
// register kernels above.
constexpr int kSmemLd = kN + 1;  // padded leading dimension, conflict-free

template <int MatricesPerCta, int RootMode, int MinBlocks>
__global__ __launch_bounds__(MatricesPerCta * kN, MinBlocks)
void smem_64_kernel(const float* __restrict__ input,
                    float* __restrict__ output) {
  static_assert(MatricesPerCta == 1 || MatricesPerCta == 2);
  static_assert(RootMode >= kPreciseRoot && RootMode <= kRawRoot);
  const int tx = static_cast<int>(threadIdx.x) & (kN - 1);
  const int local = static_cast<int>(threadIdx.x) >> 6;
  const int matrix = static_cast<int>(blockIdx.x) * MatricesPerCta + local;
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  __shared__ float tiles[MatricesPerCta * kN * kSmemLd];
  __shared__ float inverses[MatricesPerCta];
  float* factor = tiles + local * kN * kSmemLd;

  for (int j = 0; j < kN; ++j) {
    // Thread tx computes column j of its own row: the symmetric read of
    // A[j, tx] is one coalesced transaction per matrix.
    float value = a[j * kN + tx];
#pragma unroll 8
    for (int k = 0; k < j; ++k) {
      value = fmaf(-factor[tx * kSmemLd + k], factor[j * kSmemLd + k], value);
    }
    if (tx == j) {
      float inverse;
      float diagonal;
      if constexpr (RootMode == kRefinedRoot) {
        inverse = rsqrtf(value);
        const float square = inverse * inverse;
        inverse *= fmaf(-0.5f * value, square, 1.5f);
        diagonal = value * inverse;
      } else if constexpr (RootMode == kRawRoot) {
        inverse = rsqrtf(value);
        diagonal = value * inverse;
      } else {
        diagonal = __fsqrt_rn(value);
        inverse = __fdiv_rn(1.0f, diagonal);
      }
      factor[tx * kSmemLd + j] = diagonal;
      inverses[local] = inverse;
    }
    __syncthreads();
    if (tx > j) {
      factor[tx * kSmemLd + j] = value * inverses[local];
    }
    __syncthreads();
  }

  float* out_row = result + tx * kN;
#pragma unroll
  for (int vector_index = 0; vector_index < kN / 4; ++vector_index) {
    const int column = vector_index * 4;
    float4 values;
    values.x = column <= tx ? factor[tx * kSmemLd + column] : 0.0f;
    values.y = column + 1 <= tx ? factor[tx * kSmemLd + column + 1] : 0.0f;
    values.z = column + 2 <= tx ? factor[tx * kSmemLd + column + 2] : 0.0f;
    values.w = column + 3 <= tx ? factor[tx * kSmemLd + column + 3] : 0.0f;
    reinterpret_cast<float4*>(out_row)[vector_index] = values;
  }
}

// Register-row hybrid: each thread keeps its own factor row in registers
// (halving the shared-memory traffic of the dot product) while pivot rows
// are still published through shared memory. The register array forces a
// fully unrolled column loop.
template <int RootMode, int MinBlocks>
__global__ __launch_bounds__(kN, MinBlocks)
void smem_reg_64_kernel(const float* __restrict__ input,
                        float* __restrict__ output) {
  static_assert(RootMode >= kPreciseRoot && RootMode <= kRawRoot);
  const int tx = static_cast<int>(threadIdx.x);
  const int matrix = static_cast<int>(blockIdx.x);
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  __shared__ float pivot_rows[kN * kSmemLd];
  __shared__ float inverse_shared;

  float row[kN];
#pragma unroll
  for (int k = 0; k < kN; ++k) {
    row[k] = 0.0f;
  }

#pragma unroll
  for (int j = 0; j < kN; ++j) {
    float value = a[j * kN + tx];
#pragma unroll
    for (int k = 0; k < j; ++k) {
      value = fmaf(-row[k], pivot_rows[j * kSmemLd + k], value);
    }
    if (tx == j) {
      float inverse;
      float diagonal;
      if constexpr (RootMode == kRefinedRoot) {
        inverse = rsqrtf(value);
        const float square = inverse * inverse;
        inverse *= fmaf(-0.5f * value, square, 1.5f);
        diagonal = value * inverse;
      } else if constexpr (RootMode == kRawRoot) {
        inverse = rsqrtf(value);
        diagonal = value * inverse;
      } else {
        diagonal = __fsqrt_rn(value);
        inverse = __fdiv_rn(1.0f, diagonal);
      }
      row[j] = diagonal;
      pivot_rows[tx * kSmemLd + j] = diagonal;
      inverse_shared = inverse;
    }
    __syncthreads();
    if (tx > j) {
      row[j] = value * inverse_shared;
      pivot_rows[tx * kSmemLd + j] = row[j];
    }
    __syncthreads();
  }

  float* out_row = result + tx * kN;
#pragma unroll
  for (int vector_index = 0; vector_index < kN / 4; ++vector_index) {
    const int column = vector_index * 4;
    float4 values;
    values.x = column <= tx ? row[column] : 0.0f;
    values.y = column + 1 <= tx ? row[column + 1] : 0.0f;
    values.z = column + 2 <= tx ? row[column + 2] : 0.0f;
    values.w = column + 3 <= tx ? row[column + 3] : 0.0f;
    reinterpret_cast<float4*>(out_row)[vector_index] = values;
  }
}

template <int Warps, int RowLayout, int LoadWidth, int RootMode,
          int DotAccumulators, int ShuffleLookahead, int MinBlocks>
void launch_crout(const float* input, float* output) {
  constexpr int threads = Warps * 32;
  constexpr int blocks = kBatch / Warps;
  static_assert(kBatch % Warps == 0);
  crout_64_kernel<Warps, RowLayout, LoadWidth, RootMode, DotAccumulators,
                  ShuffleLookahead, MinBlocks>
      <<<blocks, threads>>>(input, output);
}

template <int Warps, bool RootLookahead, int MinBlocks>
void launch_right_looking(const float* input, float* output) {
  constexpr int threads = Warps * 32;
  constexpr int blocks = kBatch / Warps;
  static_assert(kBatch % Warps == 0);
  right_looking_64_kernel<Warps, RootLookahead, MinBlocks>
      <<<blocks, threads>>>(input, output);
}

template <int Warps, int MinBlocks>
void launch_crout_hybrid(const float* input, float* output) {
  constexpr int threads = Warps * 32;
  constexpr int blocks = kBatch / Warps;
  static_assert(kBatch % Warps == 0);
  crout_hybrid_64_kernel<Warps, MinBlocks>
      <<<blocks, threads>>>(input, output);
}

template <int Warps, int MinBlocks>
void launch_right_hybrid(const float* input, float* output) {
  constexpr int threads = Warps * 32;
  constexpr int blocks = kBatch / Warps;
  static_assert(kBatch % Warps == 0);
  right_hybrid_64_kernel<Warps, MinBlocks>
      <<<blocks, threads>>>(input, output);
}

template <int MatricesPerCta, int RootMode, int MinBlocks>
void launch_smem(const float* input, float* output) {
  constexpr int threads = MatricesPerCta * kN;
  constexpr int blocks = kBatch / MatricesPerCta;
  static_assert(kBatch % MatricesPerCta == 0);
  smem_64_kernel<MatricesPerCta, RootMode, MinBlocks>
      <<<blocks, threads>>>(input, output);
}

template <int RootMode, int MinBlocks>
void launch_smem_reg(const float* input, float* output) {
  smem_reg_64_kernel<RootMode, MinBlocks><<<kBatch, kN>>>(input, output);
}

void launch_variant(const float* input,
                    float* output,
                    int variant) {
  switch (variant) {
    case  0: launch_crout<4, 0, 1, kPreciseRoot, 1, 1, 2>(input, output); break;
    case  1: launch_crout<4, 0, 1, kRefinedRoot, 1, 1, 2>(input, output); break;
    case  2: launch_crout<4, 0, 1, kRawRoot,     1, 1, 2>(input, output); break;
    case  3: launch_crout<4, 1, 2, kPreciseRoot, 1, 1, 2>(input, output); break;
    case  4: launch_crout<4, 1, 2, kRefinedRoot, 1, 1, 2>(input, output); break;
    case  5: launch_crout<4, 1, 2, kRawRoot,     1, 1, 2>(input, output); break;
    case  6: launch_crout<4, 1, 1, kRawRoot,     1, 1, 2>(input, output); break;
    case  7: launch_crout<2, 1, 2, kRawRoot,     1, 1, 4>(input, output); break;
    case  8: launch_crout<8, 1, 2, kRawRoot,     1, 1, 1>(input, output); break;
    case  9: launch_crout<4, 1, 2, kRawRoot,     2, 1, 2>(input, output); break;
    case 10: launch_crout<4, 1, 2, kRawRoot,     4, 1, 2>(input, output); break;
    case 11: launch_crout<4, 1, 2, kRawRoot,     1, 2, 2>(input, output); break;
    case 12: launch_right_looking<4, false, 2>(input, output); break;
    case 13: launch_right_looking<4, true,  2>(input, output); break;
    case 14: launch_right_looking<2, false, 4>(input, output); break;
    case 15: launch_smem<1, kPreciseRoot, 7>(input, output); break;
    case 16: launch_smem<1, kRawRoot,     7>(input, output); break;
    case 17: launch_smem<2, kRawRoot,     4>(input, output); break;
    case 18: launch_smem_reg<kRawRoot,    7>(input, output); break;
    case 22: launch_crout_hybrid<4, 2>(input, output); break;
    case 23: launch_right_hybrid<4, 2>(input, output); break;
    default: TORCH_CHECK(false, "native variant must be 0--18, 22, or 23");
  }
}

void fill_metadata_row(int64_t* rows,
                       int variant,
                       const cudaFuncAttributes& attributes,
                       int active_blocks,
                       int warps_per_cta,
                       int root_mode,
                       int threads_per_cta,
                       int row_layout,
                       int min_blocks,
                       int load_width,
                       int dot_accumulators,
                       int shuffle_lookahead,
                       int right_looking,
                       int root_lookahead,
                       int threads_per_matrix,
                       int matrices_per_cta) {
  int64_t* row = rows + static_cast<int64_t>(variant) * kMetadataColumns;
  row[0] = variant;
  row[1] = warps_per_cta;
  row[2] = root_mode == kRefinedRoot ? 1 : 0;
  row[3] = root_mode == kRawRoot ? 1 : 0;
  row[4] = threads_per_cta;
  row[5] = attributes.numRegs;
  row[6] = attributes.localSizeBytes;
  row[7] = attributes.sharedSizeBytes;
  row[8] = active_blocks;
  row[9] = active_blocks * (threads_per_cta / 32);
  row[10] = row_layout;
  row[11] = min_blocks;
  row[12] = load_width;
  row[13] = dot_accumulators;
  row[14] = shuffle_lookahead;
  row[15] = right_looking;
  row[16] = root_lookahead;
  row[17] = threads_per_matrix;
  row[18] = matrices_per_cta;
}

template <typename Kernel>
void query_kernel(Kernel kernel,
                  int threads,
                  cudaFuncAttributes* attributes,
                  int* active_blocks) {
  auto status = cudaFuncGetAttributes(attributes, kernel);
  TORCH_CHECK(status == cudaSuccess,
              "cudaFuncGetAttributes failed: ", cudaGetErrorString(status));
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_blocks, kernel, threads, 0);
  TORCH_CHECK(status == cudaSuccess,
              "occupancy query failed: ", cudaGetErrorString(status));
}

template <int Warps, int RowLayout, int LoadWidth, int RootMode,
          int DotAccumulators, int ShuffleLookahead, int MinBlocks>
void write_crout_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  query_kernel(crout_64_kernel<Warps, RowLayout, LoadWidth, RootMode,
                               DotAccumulators, ShuffleLookahead, MinBlocks>,
               Warps * 32, &attributes, &active_blocks);
  fill_metadata_row(rows, variant, attributes, active_blocks, Warps, RootMode,
                    Warps * 32, RowLayout, MinBlocks, LoadWidth,
                    DotAccumulators, ShuffleLookahead, 0, 0, 32, 0);
}

template <int Warps, bool RootLookahead, int MinBlocks>
void write_right_looking_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  query_kernel(right_looking_64_kernel<Warps, RootLookahead, MinBlocks>,
               Warps * 32, &attributes, &active_blocks);
  fill_metadata_row(rows, variant, attributes, active_blocks, Warps, kRawRoot,
                    Warps * 32, 1, MinBlocks, 2, 1, 1, 1,
                    RootLookahead ? 1 : 0, 32, 0);
}

template <int Warps, int MinBlocks>
void write_crout_hybrid_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  query_kernel(crout_hybrid_64_kernel<Warps, MinBlocks>, Warps * 32,
               &attributes, &active_blocks);
  fill_metadata_row(rows, variant, attributes, active_blocks, Warps, kRawRoot,
                    Warps * 32, 1, MinBlocks, 1, 1, 1, 0, 0, 32, 0);
}

template <int Warps, int MinBlocks>
void write_right_hybrid_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  query_kernel(right_hybrid_64_kernel<Warps, MinBlocks>, Warps * 32,
               &attributes, &active_blocks);
  fill_metadata_row(rows, variant, attributes, active_blocks, Warps, kRawRoot,
                    Warps * 32, 1, MinBlocks, 2, 1, 1, 1, 1, 32, 0);
}

template <int MatricesPerCta, int RootMode, int MinBlocks>
void write_smem_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  query_kernel(smem_64_kernel<MatricesPerCta, RootMode, MinBlocks>,
               MatricesPerCta * kN, &attributes, &active_blocks);
  fill_metadata_row(rows, variant, attributes, active_blocks,
                    MatricesPerCta * 2, RootMode, MatricesPerCta * kN, 0,
                    MinBlocks, 1, 1, 1, 0, 0, kN, MatricesPerCta);
}

template <int RootMode, int MinBlocks>
void write_smem_reg_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  query_kernel(smem_reg_64_kernel<RootMode, MinBlocks>, kN, &attributes,
               &active_blocks);
  fill_metadata_row(rows, variant, attributes, active_blocks, 2, RootMode, kN,
                    0, MinBlocks, 1, 1, 1, 0, 0, kN, 1);
}

}  // namespace

void cholesky_1024x64_out(const at::Tensor& data,
                          at::Tensor out,
                          int64_t variant) {
  check_input(data);
  check_output(data, out);
  TORCH_CHECK((variant >= 0 && variant <= 18) ||
                  (variant >= 22 && variant < kVariantCount),
              "native variant must be 0--18, 22, or 23");
  launch_variant(data.data_ptr<float>(), out.data_ptr<float>(),
                 static_cast<int>(variant));
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_1024x64(const at::Tensor& data,
                            int64_t variant) {
  auto out = at::empty_like(data);
  cholesky_1024x64_out(data, out, variant);
  return out;
}

at::Tensor cholesky_1024x64_metadata() {
  auto result = at::zeros({kVariantCount, kMetadataColumns},
                          at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  write_crout_metadata<4, 0, 1, kPreciseRoot, 1, 1, 2>(rows, 0);
  write_crout_metadata<4, 0, 1, kRefinedRoot, 1, 1, 2>(rows, 1);
  write_crout_metadata<4, 0, 1, kRawRoot,     1, 1, 2>(rows, 2);
  write_crout_metadata<4, 1, 2, kPreciseRoot, 1, 1, 2>(rows, 3);
  write_crout_metadata<4, 1, 2, kRefinedRoot, 1, 1, 2>(rows, 4);
  write_crout_metadata<4, 1, 2, kRawRoot,     1, 1, 2>(rows, 5);
  write_crout_metadata<4, 1, 1, kRawRoot,     1, 1, 2>(rows, 6);
  write_crout_metadata<2, 1, 2, kRawRoot,     1, 1, 4>(rows, 7);
  write_crout_metadata<8, 1, 2, kRawRoot,     1, 1, 1>(rows, 8);
  write_crout_metadata<4, 1, 2, kRawRoot,     2, 1, 2>(rows, 9);
  write_crout_metadata<4, 1, 2, kRawRoot,     4, 1, 2>(rows, 10);
  write_crout_metadata<4, 1, 2, kRawRoot,     1, 2, 2>(rows, 11);
  write_right_looking_metadata<4, false, 2>(rows, 12);
  write_right_looking_metadata<4, true,  2>(rows, 13);
  write_right_looking_metadata<2, false, 4>(rows, 14);
  write_smem_metadata<1, kPreciseRoot, 7>(rows, 15);
  write_smem_metadata<1, kRawRoot,     7>(rows, 16);
  write_smem_metadata<2, kRawRoot,     4>(rows, 17);
  write_smem_reg_metadata<kRawRoot,    7>(rows, 18);
  // Variant 19 lives in the separate cuSolverDx extension; only its identity
  // and launch shape are recorded here.
  int64_t* dx_row = rows + 19 * kMetadataColumns;
  dx_row[0] = 19;
  dx_row[4] = 128;
  dx_row[17] = kN;
  dx_row[18] = 1;
  rows[20 * kMetadataColumns] = 20;
  rows[21 * kMetadataColumns] = 21;
  write_crout_hybrid_metadata<4, 2>(rows, 22);
  write_right_hybrid_metadata<4, 2>(rows, 23);
  return result;
}
"""


# --------------------------------------------------------------------------
# Variant 19: cuSolverDx POTRF baseline. It requires relocatable device code,
# device LTO, and device-linking the packaged libcusolverdx.fatbin, so it is
# built as a separate setuptools CUDAExtension and only when requested.
# --------------------------------------------------------------------------

_DX_CPP_SOURCE = r"""
#include <torch/extension.h>

void chol_dx_prepare();
void chol_dx_run_out(const at::Tensor& data, at::Tensor out, at::Tensor info);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &chol_dx_prepare, "Configure dynamic shared memory");
  m.def("run_out", &chol_dx_run_out, "cuSolverDx batched 64x64 POTRF");
}
"""

_DX_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <cusolverdx.hpp>

#include <cstdint>

namespace {

constexpr int kBatch = 1024;
constexpr int kN = 64;
constexpr int kArch = 1000;
constexpr int kThreads = 128;

using namespace cusolverdx;

using Potrf = decltype(Size<kN, kN>() + Precision<float>() +
                       Type<type::real>() + Function<potrf>() +
                       FillMode<fill_mode::lower>() +
                       Arrangement<arrangement::row_major>() + SM<kArch>() +
                       Block() + BlockDim<kThreads>() + BatchesPerBlock<1>());

static_assert(Potrf::lda >= kN, "unexpected shared leading dimension");
static_assert(sizeof(Potrf::status_type) == sizeof(int),
              "POTRF status type must be int32");

constexpr unsigned kSharedBytes = Potrf::shared_memory_size;

__global__ __launch_bounds__(Potrf::max_threads_per_block)
void potrf_dx_kernel(const float* __restrict__ input,
                     float* __restrict__ output,
                     Potrf::status_type* info) {
  CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(Potrf);
  const int matrix = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t base = static_cast<int64_t>(matrix) * kN * kN;

  extern __shared__ __align__(16) unsigned char storage[];
  float* a_shared = reinterpret_cast<float*>(storage);
  constexpr int lda = Potrf::lda;

  for (int idx = tid; idx < kN * kN; idx += kThreads) {
    const int r = idx >> 6;
    const int c = idx & (kN - 1);
    a_shared[r * lda + c] = input[base + idx];
  }
  __syncthreads();

  Potrf().execute(a_shared, info + matrix);
  __syncthreads();

  // The checker requires exact zeros in the strict upper triangle; POTRF
  // leaves the input there, so the writeback masks it.
  for (int idx = tid; idx < kN * kN; idx += kThreads) {
    const int r = idx >> 6;
    const int c = idx & (kN - 1);
    output[base + idx] = c <= r ? a_shared[r * lda + c] : 0.0f;
  }
}

void check_pair(const at::Tensor& data, const at::Tensor& out,
                const at::Tensor& info) {
  TORCH_CHECK(data.is_cuda() && out.is_cuda() && info.is_cuda(),
              "tensors must be CUDA tensors");
  TORCH_CHECK(data.scalar_type() == at::kFloat &&
                  out.scalar_type() == at::kFloat,
              "data and out must be float32");
  TORCH_CHECK(info.scalar_type() == at::kInt, "info must be int32");
  TORCH_CHECK(data.is_contiguous() && out.is_contiguous() &&
                  info.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "cuSolverDx path requires shape (1024, 64, 64)");
  TORCH_CHECK(out.sizes() == data.sizes(), "output shape must match input");
  TORCH_CHECK(info.dim() == 1 && info.size(0) == kBatch,
              "info must have shape (1024,)");
}

}  // namespace

void chol_dx_prepare() {
  auto status = cudaFuncSetAttribute(
      potrf_dx_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  TORCH_CHECK(status == cudaSuccess,
              "POTRF shared-memory setup failed: ", cudaGetErrorString(status));
}

void chol_dx_run_out(const at::Tensor& data, at::Tensor out, at::Tensor info) {
  check_pair(data, out, info);
  potrf_dx_kernel<<<kBatch, Potrf::block_dim, kSharedBytes>>>(
      data.data_ptr<float>(), out.data_ptr<float>(),
      reinterpret_cast<Potrf::status_type*>(info.data_ptr<int>()));
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "POTRF launch failed: ", cudaGetErrorString(status));
}
"""

_DX_BUILD_ROOT = Path(tempfile.gettempdir()) / getuser() / "cholesky_b1024n64_dx"

# cuSolverDx's packaged fatbin is built with FTZ disabled and precise
# division; RDC device linking requires every input to agree on those
# options, so this translation unit omits --use_fast_math. The -U flags
# cancel the half/bfloat16 operator suppressions that torch's CUDAExtension
# injects by default — the CUTLASS headers bundled with MathDx need those
# operators to compile.
_DX_NVCC_FLAGS = (
    "-O3",
    "-DNDEBUG",
    "-std=c++20",
    "--extra-device-vectorization",
    "--restrict",
    "-Xptxas=-O3",
    "-Xptxas=--allow-expensive-optimizations=true",
    "-lineinfo",
    "-rdc=true",
    "-gencode=arch=compute_100a,code=lto_100a",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
)


def _cusolverdx_root_assets(root: Path):
    include_dir = root / "include"
    cutlass_dir = root / "external" / "cutlass" / "include"
    fatbin = root / "lib" / "libcusolverdx.fatbin"
    if (
        (include_dir / "cusolverdx.hpp").is_file()
        and cutlass_dir.is_dir()
        and fatbin.is_file()
    ):
        return (
            [str(include_dir.resolve()), str(cutlass_dir.resolve())],
            str(fatbin.resolve()),
        )
    return None


@lru_cache(maxsize=1)
def _cusolverdx_assets():
    configured = os.environ.get("MATHDX_ROOT")
    if configured:
        root = Path(configured).expanduser()
        assets = _cusolverdx_root_assets(root)
        if assets is None:
            raise RuntimeError(
                f"MATHDX_ROOT={root} does not contain cusolverdx.hpp, the bundled "
                "CUTLASS headers, and lib/libcusolverdx.fatbin"
            )
        return assets

    site_mathdx = Path(torch.__file__).resolve().parent.parent / "nvidia" / "mathdx"
    if site_mathdx.is_dir():
        for root in (site_mathdx, *sorted(site_mathdx.iterdir(), reverse=True)):
            assets = _cusolverdx_root_assets(root)
            if assets is not None:
                return assets

    roots = []
    for variable in ("CPATH", "CPLUS_INCLUDE_PATH"):
        for entry in os.environ.get(variable, "").split(os.pathsep):
            if entry:
                include_dir = Path(entry).expanduser()
                roots.append(
                    include_dir.parent if include_dir.name == "include" else include_dir
                )
    roots.extend((Path("/usr/local"), Path("/usr")))
    for root in roots:
        assets = _cusolverdx_root_assets(root)
        if assets is not None:
            return assets
    raise RuntimeError(
        "A complete cuSolverDx installation is required for variant 19. Set "
        "MATHDX_ROOT to a MathDx 26.06 root containing include/, "
        "external/cutlass/include/, and lib/libcusolverdx.fatbin."
    )


def _load_extension_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _build_cusolverdx_extension(name: str):
    from setuptools import Distribution
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    include_paths, fatbin = _cusolverdx_assets()
    build_root = _DX_BUILD_ROOT / name
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "binding.cpp"
    cuda_path = build_root / "potrf_dx.cu"
    cpp_path.write_text(_DX_CPP_SOURCE)
    cuda_path.write_text(_DX_CUDA_SOURCE)

    extension = CUDAExtension(
        name=name,
        sources=[str(cpp_path), str(cuda_path)],
        include_dirs=include_paths,
        extra_compile_args={
            "cxx": ["-O3", "-DNDEBUG", "-std=c++20"],
            "nvcc": list(_DX_NVCC_FLAGS),
            # BuildExtension uses this key to emit the required device-link
            # edge carrying the packaged fatbin.
            "nvcc_dlink": [fatbin, "-dlink", "-dlto"],
        },
    )
    distribution = Distribution({"name": name, "ext_modules": [extension]})
    distribution.cmdclass = {
        "build_ext": BuildExtension.with_options(
            use_ninja=True, no_python_abi_suffix=False
        )
    }
    command = distribution.get_command_obj("build_ext")
    command.ensure_finalized()
    command.build_lib = str(build_root)
    command.build_temp = str(build_root / "objects")
    command.inplace = False
    command.force = False
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    previous_cxx = os.environ.get("CXX")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    os.environ["CXX"] = "c++"
    try:
        command.run()
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
        if previous_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = previous_cxx
    output_path = Path(command.get_ext_fullpath(name))
    if not output_path.is_file():
        matches = sorted(build_root.glob(f"{name}*.so"))
        if not matches:
            raise RuntimeError(
                f"cuSolverDx extension build produced no module under {build_root}"
            )
        output_path = matches[-1]
    return _load_extension_file(name, output_path)


@lru_cache(maxsize=1)
def _cusolverdx_module():
    tag = hashlib.sha256(
        "\0".join((_DX_CPP_SOURCE, _DX_CUDA_SOURCE, *_DX_NVCC_FLAGS)).encode()
    ).hexdigest()[:10]
    name = f"cholesky_b1024n64_dx_{tag}"
    build_root = _DX_BUILD_ROOT / name
    build_root.mkdir(parents=True, exist_ok=True)
    if name in sys.modules:
        module = sys.modules[name]
    else:
        matches = sorted(build_root.glob(f"{name}*.so"))
        if matches:
            module = _load_extension_file(name, matches[-1])
        else:
            module = _build_cusolverdx_extension(name)
    module.prepare()
    return module


@lru_cache(maxsize=None)
def _dx_info_buffer(device: str) -> torch.Tensor:
    return torch.zeros(1024, dtype=torch.int32, device=device)


@lru_cache(maxsize=1)
def _native_module():
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name="cholesky_b1024n64_algorithms_v4",
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
        raise ValueError(
            f"variant must be one of {_VARIANT_IDS}, got {variant}"
        )
    if variant == _CUSOLVERDX_VARIANT:
        module = _cusolverdx_module()
        if out is None:
            out = torch.empty_like(data)
        module.run_out(data, out, _dx_info_buffer(str(data.device)))
        return out
    module = _native_module()
    if out is None:
        return module.run(data, variant)
    module.run_out(data, out, variant)
    return out


def _variant_metadata() -> torch.Tensor:
    return _native_module().metadata()


def custom_kernel(data: input_t) -> output_t:
    if tuple(data.shape) != (1024, 64, 64):
        return torch.linalg.cholesky_ex(data, check_errors=False).L
    return _run_variant(data, _DEFAULT_VARIANT)
