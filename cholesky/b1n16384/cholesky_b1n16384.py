import hashlib
import os
import re
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The tuner replaces this exact line in retained candidate copies.
_DEFAULT_VARIANT = 0  # POPCORN_VARIANT
_CUTLASS_BASE_VARIANT = 0
_CUTLASS_VARIANT = 7
_VARIANT_NAMES = (
    "ll_nb1024_invgemm_tf32",
    "ll_nb2048_invgemm_tf32",
    "ll_nb512_invgemm_tf32",
    "ll_nb1024_inv_tf32",
    "ll_nb1024_m128_to_m64_at_r2048_tf32",
    "ll_nb1024_m128_to_m64_at_r4096_tf32",
    "ll_nb1024_m128_m64_m32_at_r4096_r1024_tf32",
    "ll_nb1024_strsm_fp32_all_cutlass_names",
    "tilegrid64_fp32",
    "tilegrid64_tf32",
)
_VARIANT_COUNT = len(_VARIANT_NAMES)
_VARIANT_IDS = tuple(range(_VARIANT_COUNT))
_WAVE_PUBLIC_TO_LOCAL = {8: 0, 9: 1}

_METADATA_COLUMNS = (
    "variant",
    "schedule",
    "outer_block",
    "trsm_mode",
    "math_big",
    "math_inner",
    "factor_threads",
    "factor_registers",
    "factor_shared_bytes",
    "factor_local_bytes",
    "factor_dynamic_bytes",
    "apply_registers",
    "apply_shared_bytes",
    "apply_local_bytes",
    "apply_dynamic_bytes",
    "copy_registers",
    "wedge_registers",
    "active_factor_blocks",
    "active_apply_blocks",
    "launch_count",
    "emulation_available",
    "microtile",
    "guard_mode",
    "implemented",
    "factor_mode",
    "apply_mode",
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

void cholesky_b1n16384_prepare(int64_t variant);
at::Tensor cholesky_b1n16384(const at::Tensor& data, int64_t variant);
void cholesky_b1n16384_out(
    const at::Tensor& data, at::Tensor out, int64_t variant);
at::Tensor cholesky_b1n16384_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b1n16384_prepare,
        "Configure one B200 1x16384 Cholesky variant");
  m.def("run", &cholesky_b1n16384, "Single 16384 Cholesky");
  m.def("run_out", &cholesky_b1n16384_out,
        "Single 16384 Cholesky out");
  m.def("metadata", &cholesky_b1n16384_metadata,
        "B200 kernel resource metadata");
}
"""

# Port of the b1n32768 phase-2h winner (`ll_nb1024_invgemm_tf32`,
# 42.86 ms = 5.15x there): left-looking outer panels with TF32 cuBLAS
# history/inner GEMMs, a single-CTA 512-thread redundant-corner factor
# kernel that also builds T = inv(L11), and the triangular apply run
# as a dense TF32 GEMM into a scratch column plus a float4 copy-back.
# Only the shape constant differs; every kernel is micro=128-tile
# local and ports unchanged.
_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <utility>

namespace {

constexpr int kN = 16384;
constexpr int kMicro = 128;
constexpr int kTileLd = kMicro + 1;
constexpr int kPanelLd = 9;
constexpr int kVariantCount = 7;
constexpr int kMetadataColumns = 33;

// Shared-memory layout of the factor kernel:
//   [tile kMicro x kTileLd][inverse_diagonal kMicro]
//   [panel kMicro x kPanelLd][tinv kMicro x kTileLd][mid 64 x 64]
constexpr int kFactorBytes =
    static_cast<int>(sizeof(float)) *
    (kMicro * kTileLd + kMicro + kMicro * kPanelLd +
     kMicro * kTileLd + 64 * 64);
constexpr int kApplyBytes =
    static_cast<int>(sizeof(float)) * 2 * kMicro * kTileLd;

static_assert(kFactorBytes == 153600);
static_assert(kApplyBytes == 132096);

// Numeric codes match the b1n32768 metadata conventions.
constexpr int kLeftLook = 0;
constexpr int kTrsmInverse = 1;
constexpr int kTrsmGemm = 4;
constexpr int kMathTf32 = 1;
constexpr int kFactorWide = 3;

template <int Id>
struct Variant;

#define SPEC(ID, NB, TRSM)                                           \
  template <> struct Variant<ID> {                                   \
    static constexpr int nb = NB;                                    \
    static constexpr int trsm_mode = TRSM;                           \
    static constexpr bool gemm_apply = TRSM == kTrsmGemm;            \
    static constexpr bool adaptive = false;                          \
    static constexpr int tail_policy = 0;                            \
  }

SPEC(0, 1024, kTrsmGemm);
SPEC(1, 2048, kTrsmGemm);
SPEC(2, 512, kTrsmGemm);
SPEC(3, 1024, kTrsmInverse);

#undef SPEC

#define ADAPTIVE_SPEC(ID, POLICY)                                    \
  template <> struct Variant<ID> {                                   \
    static constexpr int nb = 1024;                                  \
    static constexpr int trsm_mode = kTrsmGemm;                      \
    static constexpr bool gemm_apply = true;                         \
    static constexpr bool adaptive = true;                           \
    static constexpr int tail_policy = POLICY;                       \
  }

ADAPTIVE_SPEC(4, 1);
ADAPTIVE_SPEC(5, 2);
ADAPTIVE_SPEC(6, 3);

#undef ADAPTIVE_SPEC

template <int... Ids>
constexpr std::array<bool, sizeof...(Ids)> scratch_usage_of(
    std::integer_sequence<int, Ids...>) {
  return {(Variant<Ids>::trsm_mode == kTrsmGemm)...};
}

// Which variants apply through the out-of-place cuBLAS GEMM and need
// the scratch column workspace.
constexpr auto kVariantUsesScratch =
    scratch_usage_of(std::make_integer_sequence<int, kVariantCount>{});

__device__ __forceinline__ int64_t matrix_index(int row, int column) {
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

// Wide redundant-corner factorization, 512 threads (b1n32768 phase-2e
// design). Every thread factors the 8x8 group corner redundantly in
// registers - cheaper than warp-serial communication at this size -
// four threads share each sub-panel row (redundant register solve,
// quarter-split rank-8 update), and each group needs two CTA barriers,
// with sixteen warps to hide latency.
__device__ __forceinline__ void factor_wide(
    float* __restrict__ tile,
    float* __restrict__ inverse_diagonal,
    float* __restrict__ panel) {
  constexpr int kGroup = 8;
  const int thread = static_cast<int>(threadIdx.x);
  const int row_index = thread >> 2;
  const int quarter = thread & 3;
#pragma unroll 1
  for (int base = 0; base < kMicro; base += kGroup) {
    float corner[kGroup][kGroup];
    float inverse[kGroup];
#pragma unroll
    for (int i = 0; i < kGroup; ++i) {
#pragma unroll
      for (int j = 0; j <= i; ++j) {
        corner[i][j] = tile_at(tile, base + i, base + j);
      }
    }
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      const float diagonal = __fsqrt_rn(corner[j][j]);
      const float inv = __fdiv_rn(1.0f, diagonal);
      corner[j][j] = diagonal;
      inverse[j] = inv;
#pragma unroll
      for (int i = j + 1; i < kGroup; ++i) {
        corner[i][j] *= inv;
      }
#pragma unroll
      for (int i = j + 1; i < kGroup; ++i) {
#pragma unroll
        for (int target = j + 1; target <= i; ++target) {
          corner[i][target] = fmaf(
              -corner[i][j], corner[target][j], corner[i][target]);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      if (thread == j) {
        inverse_diagonal[base + j] = inverse[j];
#pragma unroll
        for (int i = j; i < kGroup; ++i) {
          tile_at(tile, base + i, base + j) = corner[i][j];
        }
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
          value = fmaf(-solved[i], corner[j][i], value);
        }
        solved[j] = value * inverse[j];
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

// Builds tinv = inverse of the lower-triangular 128x128 factor held
// in the shared tile. Diagonal 32-blocks are inverted by forward
// substitution (one warp per block, one lane per column), then
// combined 32 -> 64 -> 128 through the block identity
//   inv([[A, 0], [B, C]]) = [[inv(A), 0],
//                            [-inv(C) B inv(A), inv(C)]].
// The strict upper triangle of tinv is written as exact zeros, so
// later dense products over it stay exact.
__device__ __forceinline__ void build_inverse_128(
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
  if (warp < 4) {
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
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    const int base = pair * 64;
    for (int linear = thread; linear < 32 * 32;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear >> 5;
      const int column = linear & 31;
      float partial = 0.0f;
#pragma unroll 4
      for (int k = column; k < 32; ++k) {
        partial = fmaf(
            tile[(base + 32 + row) * kTileLd + base + k],
            tinv[(base + k) * kTileLd + base + column], partial);
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
            tinv[(base + 32 + row) * kTileLd + base + 32 + k],
            mid[k * 32 + column], partial);
      }
      tinv[(base + 32 + row) * kTileLd + base + column] = -partial;
    }
    __syncthreads();
  }
  for (int linear = thread; linear < 64 * 64;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 6;
    const int column = linear & 63;
    float partial = 0.0f;
#pragma unroll 4
    for (int k = column; k < 64; ++k) {
      partial = fmaf(
          tile[(64 + row) * kTileLd + k],
          tinv[k * kTileLd + column], partial);
    }
    mid[row * 64 + column] = partial;
  }
  __syncthreads();
  for (int linear = thread; linear < 64 * 64;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 6;
    const int column = linear & 63;
    float partial = 0.0f;
#pragma unroll 4
    for (int k = 0; k <= row; ++k) {
      partial = fmaf(
          tinv[(64 + row) * kTileLd + 64 + k],
          mid[k * 64 + column], partial);
    }
    tinv[(64 + row) * kTileLd + column] = -partial;
  }
  __syncthreads();
}

// One CTA factors the 128x128 diagonal block at (begin, begin) in
// place and writes the dense inverse of the resulting lower factor to
// the t_inv workspace. The final micro step has no trailing rows, so
// its inverse build is skipped.
__global__ __launch_bounds__(512)
void factor128_kernel(
    float* __restrict__ output, int begin,
    float* __restrict__ t_inv) {
  extern __shared__ __align__(16) float dynamic_floats[];
  float* tile = dynamic_floats;
  float* inverse_diagonal = tile + kMicro * kTileLd;
  float* panel = inverse_diagonal + kMicro;
  float* tinv = panel + kMicro * kPanelLd;
  float* mid = tinv + kMicro * kTileLd;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kMicro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 7;
    const int column = linear & (kMicro - 1);
    tile_at(tile, row, column) =
        column <= row
            ? load_global(
                  output + matrix_index(begin + row, begin + column))
            : 0.0f;
  }
  __syncthreads();
  factor_wide(tile, inverse_diagonal, panel);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kMicro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 7;
    const int column = linear & (kMicro - 1);
    if (column <= row) {
      store_global(
          output + matrix_index(begin + row, begin + column),
          tile_at(tile, row, column));
    }
  }
  if (begin + kMicro < kN) {
    build_inverse_128(tile, inverse_diagonal, tinv, mid);
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kMicro * kMicro;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear >> 7;
      const int column = linear & (kMicro - 1);
      store_global(
          t_inv + linear, tinv[row * kTileLd + column]);
    }
  }
  __syncthreads();
}

template <int Width>
struct AdaptiveTile {
  static_assert(Width == 64 || Width == 32);
  static constexpr int ld = Width + 1;
  static constexpr int threads = Width == 64 ? 256 : 128;
  static constexpr int mid = Width == 64 ? 32 * 32 : 0;
  static constexpr int factor_bytes =
      static_cast<int>(sizeof(float)) *
      (2 * Width * ld + Width + mid);
};

template <int Width>
__device__ __forceinline__ float& adaptive_at(
    float* tile, int row, int column) {
  return tile[row * AdaptiveTile<Width>::ld + column];
}

template <int Width>
__device__ __forceinline__ void adaptive_potf2_32(
    float* tile, float* inverse_diagonal, int begin) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (static_cast<int>(threadIdx.x) < 32) {
#pragma unroll 1
    for (int local_column = 0;
         local_column < 32; ++local_column) {
      const int column = begin + local_column;
      float inverse = 0.0f;
      if (lane == local_column) {
        const float diagonal = __fsqrt_rn(
            adaptive_at<Width>(tile, column, column));
        adaptive_at<Width>(tile, column, column) = diagonal;
        inverse = __fdiv_rn(1.0f, diagonal);
        inverse_diagonal[column] = inverse;
      }
      inverse = __shfl_sync(
          0xffffffffu, inverse, local_column);
      if (lane > local_column) {
        const int row = begin + lane;
        adaptive_at<Width>(tile, row, column) *= inverse;
      }
      __syncwarp();
      if (lane > local_column) {
        const int row = begin + lane;
        const float left =
            adaptive_at<Width>(tile, row, column);
        for (int target_local = local_column + 1;
             target_local <= lane; ++target_local) {
          const int target = begin + target_local;
          adaptive_at<Width>(tile, row, target) = fmaf(
              -left,
              adaptive_at<Width>(tile, target, column),
              adaptive_at<Width>(tile, row, target));
        }
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

template <int Width>
__device__ __forceinline__ void adaptive_factor_local(
    float* tile, float* inverse_diagonal) {
  adaptive_potf2_32<Width>(
      tile, inverse_diagonal, 0);
  if constexpr (Width == 64) {
    const int lane = static_cast<int>(threadIdx.x) & 3;
    const int row_index = static_cast<int>(threadIdx.x) >> 2;
    if (row_index < 32) {
      const int row = 32 + row_index;
#pragma unroll 1
      for (int column = 0; column < 32; ++column) {
        float partial = 0.0f;
#pragma unroll 4
        for (int k = lane; k < column; k += 4) {
          partial = fmaf(
              adaptive_at<Width>(tile, row, k),
              adaptive_at<Width>(tile, column, k), partial);
        }
        partial += __shfl_down_sync(
            0xffffffffu, partial, 2, 4);
        partial += __shfl_down_sync(
            0xffffffffu, partial, 1, 4);
        if (lane == 0) {
          adaptive_at<Width>(tile, row, column) =
              (adaptive_at<Width>(tile, row, column) - partial) *
              inverse_diagonal[column];
        }
        __syncwarp();
      }
    }
    __syncthreads();
    for (int linear = static_cast<int>(threadIdx.x);
         linear < 32 * 32;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear >> 5;
      const int column = linear & 31;
      if (column <= row) {
        float value =
            adaptive_at<Width>(tile, 32 + row, 32 + column);
#pragma unroll 4
        for (int k = 0; k < 32; ++k) {
          value = fmaf(
              -adaptive_at<Width>(tile, 32 + row, k),
              adaptive_at<Width>(tile, 32 + column, k),
              value);
        }
        adaptive_at<Width>(
            tile, 32 + row, 32 + column) = value;
      }
    }
    __syncthreads();
    adaptive_potf2_32<Width>(
        tile, inverse_diagonal, 32);
  }
}

template <int Width>
__device__ __forceinline__ void adaptive_build_inverse(
    const float* tile, const float* inverse_diagonal,
    float* tinv, float* mid) {
  constexpr int kLd = AdaptiveTile<Width>::ld;
  const int thread = static_cast<int>(threadIdx.x);
  for (int linear = thread; linear < Width * kLd;
       linear += static_cast<int>(blockDim.x)) {
    tinv[linear] = 0.0f;
  }
  __syncthreads();
  const int warp = thread >> 5;
  const int lane = thread & 31;
  if (warp < Width / 32) {
    const int base = warp * 32;
    const int column = base + lane;
    tinv[column * kLd + column] = inverse_diagonal[column];
    for (int row = lane + 1; row < 32; ++row) {
      const int target = base + row;
      float partial = 0.0f;
      for (int k = lane; k < row; ++k) {
        partial = fmaf(
            tile[target * kLd + base + k],
            tinv[(base + k) * kLd + column], partial);
      }
      tinv[target * kLd + column] =
          -partial * inverse_diagonal[target];
    }
  }
  __syncthreads();
  if constexpr (Width == 64) {
    for (int linear = thread; linear < 32 * 32;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear >> 5;
      const int column = linear & 31;
      float partial = 0.0f;
#pragma unroll 4
      for (int k = column; k < 32; ++k) {
        partial = fmaf(
            tile[(32 + row) * kLd + k],
            tinv[k * kLd + column], partial);
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
            tinv[(32 + row) * kLd + 32 + k],
            mid[k * 32 + column], partial);
      }
      tinv[(32 + row) * kLd + column] = -partial;
    }
    __syncthreads();
  }
}

template <int Width>
__global__ __launch_bounds__(AdaptiveTile<Width>::threads)
void adaptive_factor_kernel(
    float* __restrict__ output, int begin,
    float* __restrict__ t_inv) {
  constexpr int kLd = AdaptiveTile<Width>::ld;
  extern __shared__ __align__(16) float dynamic_floats[];
  float* tile = dynamic_floats;
  float* inverse_diagonal = tile + Width * kLd;
  float* tinv = inverse_diagonal + Width;
  float* mid = tinv + Width * kLd;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Width * Width;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Width;
    const int column = linear & (Width - 1);
    adaptive_at<Width>(tile, row, column) =
        column <= row
            ? load_global(
                  output + matrix_index(begin + row, begin + column))
            : 0.0f;
  }
  __syncthreads();
  adaptive_factor_local<Width>(tile, inverse_diagonal);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Width * Width;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Width;
    const int column = linear & (Width - 1);
    if (column <= row) {
      store_global(
          output + matrix_index(begin + row, begin + column),
          adaptive_at<Width>(tile, row, column));
    }
  }
  if (begin + Width < kN) {
    adaptive_build_inverse<Width>(
        tile, inverse_diagonal, tinv, mid);
    for (int linear = static_cast<int>(threadIdx.x);
         linear < Width * Width;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear / Width;
      const int column = linear & (Width - 1);
      store_global(
          t_inv + linear, tinv[row * kLd + column]);
    }
  }
}

// Applies X := X * T^T for one 128x128 tile of the sub-column below
// the diagonal block at `begin` (control variant; the production path
// runs this product through cuBLAS). Each thread accumulates an 8x8
// register tile; the k loop is clipped per 64-column half because
// T[column][k] vanishes for k > column.
__global__ __launch_bounds__(256)
void trsm_apply_kernel(
    float* __restrict__ output, int begin,
    const float* __restrict__ t_inv) {
  extern __shared__ __align__(16) float dynamic_floats[];
  float* x_tile = dynamic_floats;
  float* t_tile = x_tile + kMicro * kTileLd;
  const int row_begin =
      begin + kMicro + static_cast<int>(blockIdx.x) * kMicro;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kMicro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 7;
    const int column = linear & (kMicro - 1);
    x_tile[row * kTileLd + column] = load_global(
        output + matrix_index(row_begin + row, begin + column));
    t_tile[row * kTileLd + column] = load_global(t_inv + linear);
  }
  __syncthreads();
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_row = warp >> 1;
  const int warp_column = warp & 1;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;
  const int k_limit = warp_column * 64 + 64;
  float value[8][8];
#pragma unroll
  for (int row = 0; row < 8; ++row) {
#pragma unroll
    for (int column = 0; column < 8; ++column) {
      value[row][column] = 0.0f;
    }
  }
#pragma unroll 1
  for (int k = 0; k < k_limit; ++k) {
    float left[8];
    float right[8];
#pragma unroll
    for (int row = 0; row < 8; ++row) {
      left[row] = x_tile[
          (warp_row * 32 + lane_row + row * 4) * kTileLd + k];
    }
#pragma unroll
    for (int column = 0; column < 8; ++column) {
      const int t_row =
          warp_column * 64 + lane_column + column * 8;
      right[column] = t_tile[t_row * kTileLd + k];
    }
#pragma unroll
    for (int row = 0; row < 8; ++row) {
#pragma unroll
      for (int column = 0; column < 8; ++column) {
        value[row][column] = fmaf(
            left[row], right[column], value[row][column]);
      }
    }
  }
#pragma unroll
  for (int row = 0; row < 8; ++row) {
#pragma unroll
    for (int column = 0; column < 8; ++column) {
      const int output_row = warp_row * 32 + lane_row + row * 4;
      const int output_column =
          warp_column * 64 + lane_column + column * 8;
      store_global(
          output +
              matrix_index(
                  row_begin + output_row, begin + output_column),
          value[row][column]);
    }
  }
}

// Moves the out-of-place apply result (scratch, row-major ld 128)
// back into the panel sub-column it belongs to. Both sides are
// float4-aligned: 128 columns = 32 quads per row.
__global__ __launch_bounds__(256)
void copy_back_kernel(
    float* __restrict__ output,
    const float* __restrict__ scratch, int begin) {
  constexpr int quads_per_row = kMicro / 4;
  const int rows = kN - begin - kMicro;
  const int64_t quads =
      static_cast<int64_t>(rows) * quads_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t quad = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       quad < quads; quad += stride) {
    const int row = static_cast<int>(quad / quads_per_row);
    const int column =
        static_cast<int>(quad % quads_per_row) * 4;
    const float4 value = __ldcg(
        reinterpret_cast<const float4*>(
            scratch + static_cast<int64_t>(row) * kMicro + column));
    __stcg(
        reinterpret_cast<float4*>(
            output +
            matrix_index(begin + kMicro + row, begin + column)),
        value);
  }
}

template <int Width>
__global__ __launch_bounds__(256)
void adaptive_copy_back_kernel(
    float* __restrict__ output,
    const float* __restrict__ scratch, int begin) {
  constexpr int kQuadsPerRow = Width / 4;
  const int rows = kN - begin - Width;
  const int64_t quads =
      static_cast<int64_t>(rows) * kQuadsPerRow;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t quad = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       quad < quads; quad += stride) {
    const int row = static_cast<int>(quad / kQuadsPerRow);
    const int column =
        static_cast<int>(quad % kQuadsPerRow) * 4;
    const float4 value = __ldcg(
        reinterpret_cast<const float4*>(
            scratch + static_cast<int64_t>(row) * Width + column));
    __stcg(
        reinterpret_cast<float4*>(
            output +
            matrix_index(begin + Width + row, begin + column)),
        value);
  }
}

// Vectorized lower-triangle copy: the strict upper triangle of the
// output is written as exact zeros and is never touched again except
// inside NB-block diagonal wedges, which zero_wedges_kernel restores.
__global__ __launch_bounds__(256)
void copy_lower_kernel(
    const float* __restrict__ input, float* __restrict__ output) {
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

// Restores exact zeros in the strict upper wedge of each nb x nb
// diagonal block (the only region library GEMMs overwrite).
__global__ __launch_bounds__(256)
void zero_wedges_kernel(float* __restrict__ output, int nb) {
  constexpr int ctas_per_block = 8;
  const int block = static_cast<int>(blockIdx.x) / ctas_per_block;
  const int rank = static_cast<int>(blockIdx.x) % ctas_per_block;
  const int shift = __ffs(nb) - 1;
  const int base = block * nb;
  const int64_t elements = static_cast<int64_t>(nb) * nb;
  for (int64_t linear =
           static_cast<int64_t>(rank) * blockDim.x + threadIdx.x;
       linear < elements;
       linear +=
       static_cast<int64_t>(ctas_per_block) * blockDim.x) {
    const int row = static_cast<int>(linear >> shift);
    const int column = static_cast<int>(linear & (nb - 1));
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

// Saves and restores handle state around the launch sequence; the
// handle already targets the caller's execution queue, so no queue
// is ever named here.
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

// Row-major storage is presented to cuBLAS as its column-major
// transpose: OP_T on the first operand and OP_N on the second give
// C[j + n_idx][j + m_idx] -= sum_k L[j + m_idx][k] L[j + n_idx][k],
// which by symmetry of L L^T is the wanted history update.
void gemm_history(
    cublasHandle_t handle, float* output, int64_t panel_begin,
    int nb) {
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const int columns = static_cast<int>(kN - panel_begin);
  const int history = static_cast<int>(panel_begin);
  const float* panel_rows = output + panel_begin * kN;
  float* destination = output + panel_begin * kN + panel_begin;
  check_cublas(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          nb, columns, history,
          &alpha,
          panel_rows, CUDA_R_32F, kN,
          panel_rows, CUDA_R_32F, kN,
          &beta,
          destination, CUDA_R_32F, kN,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "panel history GEMM");
}

void gemm_inner(
    cublasHandle_t handle, float* output, int64_t panel_begin,
    int64_t micro_begin) {
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const int columns = static_cast<int>(kN - micro_begin);
  const int history = static_cast<int>(micro_begin - panel_begin);
  const float* micro_rows = output + micro_begin * kN + panel_begin;
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

// Out-of-place apply scratch = X T^T as a dense GEMM: T is the dense
// inverse with exact strict-upper zeros, so the dense product equals
// the triangular one while cuBLAS runs it on TF32 tensor cores with
// deep pipelining. Col-major mapping: scratch_cm(128 x rows) =
// Tc^T(128 x 128) * Xc(128 x rows) with Xc the transposed view of the
// row-major sub-column (ld kN) and Tc that of t_inv (ld 128).
void gemm_apply_column(
    cublasHandle_t handle, float* output, const float* t_inv,
    float* scratch, int64_t micro_begin) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int rows = static_cast<int>(kN - micro_begin - kMicro);
  const float* x_rows =
      output + (micro_begin + kMicro) * kN + micro_begin;
  check_cublas(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          kMicro, rows, kMicro,
          &alpha,
          t_inv, CUDA_R_32F, kMicro,
          x_rows, CUDA_R_32F, kN,
          &beta,
          scratch, CUDA_R_32F, kMicro,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "apply GEMM");
}

template <int Width>
void adaptive_gemm_inner(
    cublasHandle_t handle, float* output, int64_t panel_begin,
    int64_t micro_begin) {
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
          Width, columns, history,
          &alpha,
          micro_rows, CUDA_R_32F, kN,
          micro_rows, CUDA_R_32F, kN,
          &beta,
          destination, CUDA_R_32F, kN,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "adaptive micro history GEMM");
}

template <int Width>
void adaptive_gemm_apply_column(
    cublasHandle_t handle, float* output, const float* t_inv,
    float* scratch, int64_t micro_begin) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int rows = static_cast<int>(kN - micro_begin - Width);
  const float* x_rows =
      output + (micro_begin + Width) * kN + micro_begin;
  check_cublas(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          Width, rows, Width,
          &alpha,
          t_inv, CUDA_R_32F, Width,
          x_rows, CUDA_R_32F, kN,
          &beta,
          scratch, CUDA_R_32F, Width,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "adaptive apply GEMM");
}

void launch_copy(const float* input, float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(2048, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, copy_lower_kernel, input, output);
}

void launch_factor(float* output, int begin, float* t_inv) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(512, 1, 1);
  config.dynamicSmemBytes = kFactorBytes;
  cudaLaunchKernelEx(
      &config, factor128_kernel, output, begin, t_inv);
}

template <int Width>
void launch_adaptive_factor(
    float* output, int begin, float* t_inv) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim =
      dim3(AdaptiveTile<Width>::threads, 1, 1);
  config.dynamicSmemBytes =
      AdaptiveTile<Width>::factor_bytes;
  cudaLaunchKernelEx(
      &config, adaptive_factor_kernel<Width>,
      output, begin, t_inv);
}

void launch_apply(float* output, int begin, const float* t_inv) {
  const int tiles = (kN - begin - kMicro) / kMicro;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(tiles, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  config.dynamicSmemBytes = kApplyBytes;
  cudaLaunchKernelEx(
      &config, trsm_apply_kernel, output, begin, t_inv);
}

void launch_copy_back(float* output, const float* scratch, int begin) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(256, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &config, copy_back_kernel, output, scratch, begin);
}

template <int Width>
void launch_adaptive_copy_back(
    float* output, const float* scratch, int begin) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(256, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &config, adaptive_copy_back_kernel<Width>,
      output, scratch, begin);
}

void launch_wedges(float* output, int nb) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((kN / nb) * 8, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, zero_wedges_kernel, output, nb);
}

template <int Id>
void launch_staged(
    float* output, const float* input, float* t_inv,
    float* scratch) {
  using V = Variant<Id>;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle);
  launch_copy(input, output);
  for (int64_t panel = 0; panel < kN; panel += V::nb) {
    if (panel > 0) {
      gemm_history(handle, output, panel, V::nb);
    }
    for (int64_t micro = panel; micro < panel + V::nb;
         micro += kMicro) {
      if (micro > panel) {
        gemm_inner(handle, output, panel, micro);
      }
      launch_factor(output, static_cast<int>(micro), t_inv);
      if (micro + kMicro < kN) {
        if constexpr (V::gemm_apply) {
          gemm_apply_column(handle, output, t_inv, scratch, micro);
          launch_copy_back(
              output, scratch, static_cast<int>(micro));
        } else {
          launch_apply(output, static_cast<int>(micro), t_inv);
        }
      }
    }
  }
  launch_wedges(output, V::nb);
}

template <int Id>
void launch_adaptive_staged(
    float* output, const float* input, float* t_inv,
    float* scratch) {
  using V = Variant<Id>;
  static_assert(V::adaptive);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle);
  launch_copy(input, output);
  for (int64_t panel = 0; panel < kN; panel += V::nb) {
    if (panel > 0) {
      gemm_history(handle, output, panel, V::nb);
    }
    int64_t micro = panel;
    while (micro < panel + V::nb) {
      const int remaining = static_cast<int>(kN - micro);
      int width = 128;
      if (Id == 4 && remaining <= 2048) {
        width = 64;
      } else if ((Id == 5 || Id == 6) && remaining <= 4096) {
        width = Id == 6 && remaining <= 1024 ? 32 : 64;
      }
      if (width == 128) {
        if (micro > panel) {
          gemm_inner(handle, output, panel, micro);
        }
        launch_factor(output, static_cast<int>(micro), t_inv);
        if (micro + 128 < kN) {
          gemm_apply_column(handle, output, t_inv, scratch, micro);
          launch_copy_back(
              output, scratch, static_cast<int>(micro));
        }
      } else if (width == 64) {
        if (micro > panel) {
          adaptive_gemm_inner<64>(
              handle, output, panel, micro);
        }
        launch_adaptive_factor<64>(
            output, static_cast<int>(micro), t_inv);
        if (micro + 64 < kN) {
          adaptive_gemm_apply_column<64>(
              handle, output, t_inv, scratch, micro);
          launch_adaptive_copy_back<64>(
              output, scratch, static_cast<int>(micro));
        }
      } else {
        if (micro > panel) {
          adaptive_gemm_inner<32>(
              handle, output, panel, micro);
        }
        launch_adaptive_factor<32>(
            output, static_cast<int>(micro), t_inv);
        if (micro + 32 < kN) {
          adaptive_gemm_apply_column<32>(
              handle, output, t_inv, scratch, micro);
          launch_adaptive_copy_back<32>(
              output, scratch, static_cast<int>(micro));
        }
      }
      micro += width;
    }
  }
  launch_wedges(output, V::nb);
}

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be CUDA");
  TORCH_CHECK(
      data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      data.dim() == 3 && data.size(0) == 1 &&
      data.size(1) == kN && data.size(2) == kN,
      "native input must have shape (1, 16384, 16384)");
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
      "dynamic shared-memory opt-in failed: ",
      cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(
      status == cudaSuccess,
      "shared-memory carveout failed: ", cudaGetErrorString(status));
}

template <typename Kernel>
cudaFuncAttributes checked_attributes(Kernel kernel) {
  cudaFuncAttributes attributes{};
  const cudaError_t status =
      cudaFuncGetAttributes(&attributes, kernel);
  TORCH_CHECK(
      status == cudaSuccess,
      "kernel resource query failed: ", cudaGetErrorString(status));
  return attributes;
}

void configure_kernels() {
  configure_dynamic(factor128_kernel, kFactorBytes);
  configure_dynamic(trsm_apply_kernel, kApplyBytes);
  configure_dynamic(
      adaptive_factor_kernel<64>,
      AdaptiveTile<64>::factor_bytes);
  configure_dynamic(
      adaptive_factor_kernel<32>,
      AdaptiveTile<32>::factor_bytes);
  checked_attributes(factor128_kernel);
  checked_attributes(trsm_apply_kernel);
  checked_attributes(adaptive_factor_kernel<64>);
  checked_attributes(adaptive_factor_kernel<32>);
}

void launch_variant(
    const float* input, float* output, float* t_inv,
    float* scratch, int variant) {
  switch (variant) {
    case 0: launch_staged<0>(output, input, t_inv, scratch); break;
    case 1: launch_staged<1>(output, input, t_inv, scratch); break;
    case 2: launch_staged<2>(output, input, t_inv, scratch); break;
    case 3: launch_staged<3>(output, input, t_inv, scratch); break;
    case 4: launch_adaptive_staged<4>(
        output, input, t_inv, scratch); break;
    case 5: launch_adaptive_staged<5>(
        output, input, t_inv, scratch); break;
    case 6: launch_adaptive_staged<6>(
        output, input, t_inv, scratch); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 6]");
  }
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

template <int Id>
void write_metadata(int64_t* rows) {
  using V = Variant<Id>;
  int64_t* row =
      rows + static_cast<int64_t>(Id) * kMetadataColumns;
  configure_kernels();
  const cudaFuncAttributes factor =
      checked_attributes(factor128_kernel);
  const cudaFuncAttributes apply =
      checked_attributes(trsm_apply_kernel);
  const cudaFuncAttributes copy =
      checked_attributes(copy_lower_kernel);
  const cudaFuncAttributes wedges =
      checked_attributes(zero_wedges_kernel);
  const int panels = kN / V::nb;
  const int potrf128 =
      !V::adaptive ? kN / 128 : (Id == 4 ? 112 : 96);
  const int potrf64 =
      !V::adaptive ? 0 : (Id == 4 ? 32 : (Id == 5 ? 64 : 48));
  const int potrf32 = V::adaptive && Id == 6 ? 32 : 0;
  const int micros = potrf128 + potrf64 + potrf32;
  const int inner = micros - panels;
  const int big = panels - 1;
  const int applies =
      V::gemm_apply ? 2 * (micros - 1) : micros - 1;
  row[0] = Id;
  row[1] = kLeftLook;
  row[2] = V::nb;
  row[3] = V::trsm_mode;
  row[4] = kMathTf32;
  row[5] = kMathTf32;
  row[6] = 512;
  row[7] = factor.numRegs;
  row[8] = factor.sharedSizeBytes;
  row[9] = factor.localSizeBytes;
  row[10] = kFactorBytes;
  row[11] = apply.numRegs;
  row[12] = apply.sharedSizeBytes;
  row[13] = apply.localSizeBytes;
  row[14] = V::gemm_apply ? 0 : kApplyBytes;
  row[15] = copy.numRegs;
  row[16] = wedges.numRegs;
  row[17] = active_blocks(factor128_kernel, 512, kFactorBytes);
  row[18] = active_blocks(trsm_apply_kernel, 256, kApplyBytes);
  row[19] = 2 + big + inner + micros + applies;
  row[20] = 0;
  row[21] = potrf32 > 0 ? 32 : (potrf64 > 0 ? 64 : kMicro);
  row[22] = 0;
  row[23] = 1;
  row[24] = kFactorWide;
  row[25] = 0;
  row[26] = V::tail_policy;
  row[27] = potrf128;
  row[28] = potrf64;
  row[29] = potrf32;
  row[30] = potrf128 > 0 && potrf64 == 0 && potrf32 == 0
      ? potrf128 - 1 : potrf128;
  row[31] = potrf64 > 0 && potrf32 == 0
      ? potrf64 - 1 : potrf64;
  row[32] = potrf32 > 0 ? potrf32 - 1 : 0;
}

}  // namespace

void cholesky_b1n16384_prepare(int64_t variant) {
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 6]");
  configure_kernels();
}

void cholesky_b1n16384_out(
    const at::Tensor& data, at::Tensor output, int64_t variant) {
  check_input(data);
  check_output(data, output);
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 6]");
  c10::cuda::CUDAGuard device_guard(data.device());
  at::Tensor t_inv = at::empty(
      {kMicro, kMicro}, data.options());
  at::Tensor scratch;
  float* scratch_pointer = nullptr;
  if (kVariantUsesScratch[static_cast<int>(variant)]) {
    scratch = at::empty({kN - kMicro, kMicro}, data.options());
    scratch_pointer = scratch.data_ptr<float>();
  }
  launch_variant(
      data.data_ptr<float>(), output.data_ptr<float>(),
      t_inv.data_ptr<float>(), scratch_pointer,
      static_cast<int>(variant));
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b1n16384(
    const at::Tensor& data, int64_t variant) {
  auto output = at::empty_like(data);
  cholesky_b1n16384_out(data, output, variant);
  return output;
}

at::Tensor cholesky_b1n16384_metadata() {
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



_CUTLASS_KERNEL_NAMES = (
    "factor128_kernel",
    "adaptive_factor_kernel",
    "trsm_apply_kernel",
    "copy_back_kernel",
    "adaptive_copy_back_kernel",
    "copy_lower_kernel",
    "zero_wedges_kernel",
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
            name=f"cholesky_b1n16384_b200_{tag}",
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
            name=f"cholesky_b1n16384_b200_cutlass_{tag}",
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
                "-Xptxas=-O3,-v,-warn-spills",
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

constexpr int kBatch = 1;
constexpr int kN = 16384;
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
            name=f"cholesky_b1n16384_wavefront_{tag}",
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
    if variant not in _VARIANT_IDS:
        raise ValueError(f"variant must be in {_VARIANT_IDS}, got {variant}")
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
        data.is_cuda
        and data.dtype == torch.float32
        and data.is_contiguous()
        and tuple(data.shape) == (1, 16384, 16384)
    ):
        return _run_variant(data, _DEFAULT_VARIANT)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
