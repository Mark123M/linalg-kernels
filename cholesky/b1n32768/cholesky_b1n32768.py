import hashlib
import os
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The tuner replaces this exact line in retained candidate copies.
_DEFAULT_VARIANT = 14  # POPCORN_VARIANT
_VARIANT_NAMES = (
    "ll_nb1024_strsm_fp32_all",
    "ll_nb1024_strsm_tf32_big",
    "ll_nb1024_inv_tf32_all",
    "ll_nb2048_inv_tf32_all",
    "ll_nb1024_inv_bf16x9_big",
    "rl_nb1024_ssyrk_tf32",
    "ll_nb1024_fused_tf32",
    "ll_nb2048_fused_tf32",
    "ll_nb1024_flatf_tf32",
    "ll_nb1024_flatf_lightapply_tf32",
    "ll_nb1024_rank8f_tf32",
    "ll_nb2048_rank8f_tf32",
    "ll_nb1024_rank8w_tf32",
    "ll_nb1024_microfused_tf32",
    "ll_nb1024_invgemm_tf32",
    "ll_nb1024_m128_to_m64_at_r4096_tf32",
    "ll_nb1024_m128_to_m64_at_r8192_tf32",
    "ll_nb1024_m128_m64_m32_at_r8192_r1024_tf32",
)
_VARIANT_COUNT = len(_VARIANT_NAMES)
_VARIANT_IDS = tuple(range(_VARIANT_COUNT))

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

_CPP_SOURCE = r"""
#include <torch/extension.h>

void cholesky_b1n32768_prepare(int64_t variant);
at::Tensor cholesky_b1n32768(const at::Tensor& data, int64_t variant);
void cholesky_b1n32768_out(
    const at::Tensor& data, at::Tensor out, int64_t variant);
at::Tensor cholesky_b1n32768_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b1n32768_prepare,
        "Configure one B200 1x32768 Cholesky variant");
  m.def("run", &cholesky_b1n32768, "Single 32768 Cholesky");
  m.def("run_out", &cholesky_b1n32768_out,
        "Single 32768 Cholesky out");
  m.def("metadata", &cholesky_b1n32768_metadata,
        "B200 kernel resource metadata");
}
"""

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

constexpr int kN = 32768;
constexpr int kMicro = 128;
constexpr int kTileLd = kMicro + 1;
constexpr int kPanelLd = 9;
constexpr int kVariantCount = 18;
constexpr int kMetadataColumns = 33;

// Shared-memory layout of the factor kernel:
//   [tile kMicro x kTileLd][inverse_diagonal kMicro]
//   [tinv kMicro x kTileLd][mid 64 x 64]        (inverse build only)
constexpr int kFactorPlainBytes =
    static_cast<int>(sizeof(float)) * (kMicro * kTileLd + kMicro);
constexpr int kFactorInverseBytes =
    kFactorPlainBytes +
    static_cast<int>(sizeof(float)) * (kMicro * kTileLd + 64 * 64);
constexpr int kApplyBytes =
    static_cast<int>(sizeof(float)) * 2 * kMicro * kTileLd;
constexpr int kApplyLightBytes =
    static_cast<int>(sizeof(float)) * kMicro * kTileLd;
constexpr int kFactorPanelBytes =
    static_cast<int>(sizeof(float)) * kMicro * kPanelLd;

// The fused per-micro kernel serves both roles from one dynamic
// allocation: CTA 0 uses the factor + inverse layout, consumer CTAs
// the [x_tile][t_tile] apply layout.
constexpr int kFusedBytes =
    kFactorInverseBytes + kFactorPanelBytes > kApplyBytes
        ? kFactorInverseBytes + kFactorPanelBytes
        : kApplyBytes;

static_assert(kFactorPlainBytes == 66560);
static_assert(kFactorInverseBytes == 148992);
static_assert(kApplyBytes == 132096);
static_assert(kApplyLightBytes == 66048);
static_assert(kFactorPanelBytes == 4608);
static_assert(kFusedBytes == 153600);

constexpr int kLeftLook = 0;
constexpr int kRightLook = 1;
constexpr int kTrsmLibrary = 0;
constexpr int kTrsmInverse = 1;
constexpr int kTrsmFused = 2;
constexpr int kTrsmFusedMicro = 3;
constexpr int kTrsmGemm = 4;
constexpr int kMathFp32 = 0;
constexpr int kMathTf32 = 1;
constexpr int kMathBf16x9 = 2;
constexpr int kGuardDefault = 0;
constexpr int kGuardTf32Math = 1;
constexpr int kGuardEmulated = 2;

#if defined(CUBLAS_VER_MAJOR) && (CUBLAS_VER_MAJOR >= 13)
constexpr bool kHasEmulation = true;
#define B1N32768_EMULATED_COMPUTE CUBLAS_COMPUTE_32F_EMULATED_16BFX9
#define B1N32768_HAS_EMULATION_API 1
#else
constexpr bool kHasEmulation = false;
#define B1N32768_EMULATED_COMPUTE CUBLAS_COMPUTE_32F
#define B1N32768_HAS_EMULATION_API 0
#endif

template <int Id>
struct Variant;

constexpr int kFactorBlocked = 0;
constexpr int kFactorFlat = 1;
constexpr int kFactorRank8 = 2;
constexpr int kFactorWide = 3;
constexpr int kApplyShared = 0;
constexpr int kApplyLight = 1;

constexpr int factor_bytes_of(bool build_inverse, int factor_mode) {
  return (build_inverse ? kFactorInverseBytes : kFactorPlainBytes) +
         (factor_mode == kFactorRank8 || factor_mode == kFactorWide
              ? kFactorPanelBytes
              : 0);
}

constexpr int factor_threads_of(int factor_mode) {
  return factor_mode == kFactorWide ? 512 : 256;
}

#define SPEC(ID, NB, SCHED, TRSM, MATH_BIG, MATH_INNER, GUARD,       \
             FACTOR, APPLY, IMPL)                                    \
  template <> struct Variant<ID> {                                   \
    static constexpr int nb = NB;                                    \
    static constexpr int schedule = SCHED;                           \
    static constexpr int trsm_mode = TRSM;                           \
    static constexpr int math_big = MATH_BIG;                        \
    static constexpr int math_inner = MATH_INNER;                    \
    static constexpr int guard_mode = GUARD;                         \
    static constexpr int factor_mode = FACTOR;                       \
    static constexpr int apply_mode = APPLY;                         \
    static constexpr bool implemented = IMPL;                        \
    static constexpr bool build_inverse =                            \
        TRSM == kTrsmInverse || TRSM == kTrsmFusedMicro ||           \
        TRSM == kTrsmGemm;                                           \
    static constexpr bool flat_factor = FACTOR == kFactorFlat;       \
    static constexpr bool light_apply = APPLY == kApplyLight;        \
    static constexpr bool fused_micro = TRSM == kTrsmFusedMicro;     \
    static constexpr bool gemm_apply = TRSM == kTrsmGemm;            \
    static constexpr bool adaptive = false;                          \
    static constexpr int tail_policy = 0;                            \
  }

SPEC(0, 1024, kLeftLook, kTrsmLibrary, kMathFp32, kMathFp32,
     kGuardDefault, kFactorBlocked, kApplyShared, true);
SPEC(1, 1024, kLeftLook, kTrsmLibrary, kMathTf32, kMathFp32,
     kGuardDefault, kFactorBlocked, kApplyShared, true);
SPEC(2, 1024, kLeftLook, kTrsmInverse, kMathTf32, kMathTf32,
     kGuardDefault, kFactorBlocked, kApplyShared, true);
SPEC(3, 2048, kLeftLook, kTrsmInverse, kMathTf32, kMathTf32,
     kGuardDefault, kFactorBlocked, kApplyShared, true);
SPEC(4, 1024, kLeftLook, kTrsmInverse, kMathBf16x9, kMathFp32,
     kGuardEmulated, kFactorBlocked, kApplyShared, true);
SPEC(5, 1024, kRightLook, kTrsmInverse, kMathTf32, kMathTf32,
     kGuardTf32Math, kFactorBlocked, kApplyShared, true);
SPEC(6, 1024, kLeftLook, kTrsmFused, kMathTf32, kMathTf32,
     kGuardDefault, kFactorFlat, kApplyShared, false);
SPEC(7, 2048, kLeftLook, kTrsmFused, kMathTf32, kMathTf32,
     kGuardDefault, kFactorFlat, kApplyShared, false);
SPEC(8, 1024, kLeftLook, kTrsmInverse, kMathTf32, kMathTf32,
     kGuardDefault, kFactorFlat, kApplyShared, true);
SPEC(9, 1024, kLeftLook, kTrsmInverse, kMathTf32, kMathTf32,
     kGuardDefault, kFactorFlat, kApplyLight, true);
SPEC(10, 1024, kLeftLook, kTrsmInverse, kMathTf32, kMathTf32,
     kGuardDefault, kFactorRank8, kApplyShared, true);
SPEC(11, 2048, kLeftLook, kTrsmInverse, kMathTf32, kMathTf32,
     kGuardDefault, kFactorRank8, kApplyShared, true);
SPEC(12, 1024, kLeftLook, kTrsmInverse, kMathTf32, kMathTf32,
     kGuardDefault, kFactorWide, kApplyShared, true);
SPEC(13, 1024, kLeftLook, kTrsmFusedMicro, kMathTf32, kMathTf32,
     kGuardDefault, kFactorWide, kApplyShared, true);
SPEC(14, 1024, kLeftLook, kTrsmGemm, kMathTf32, kMathTf32,
     kGuardDefault, kFactorWide, kApplyShared, true);

#undef SPEC

#define ADAPTIVE_SPEC(ID, POLICY)                                    \
  template <> struct Variant<ID> {                                   \
    static constexpr int nb = 1024;                                  \
    static constexpr int schedule = kLeftLook;                       \
    static constexpr int trsm_mode = kTrsmGemm;                      \
    static constexpr int math_big = kMathTf32;                       \
    static constexpr int math_inner = kMathTf32;                     \
    static constexpr int guard_mode = kGuardDefault;                 \
    static constexpr int factor_mode = kFactorWide;                  \
    static constexpr int apply_mode = kApplyShared;                  \
    static constexpr bool implemented = true;                        \
    static constexpr bool build_inverse = true;                      \
    static constexpr bool flat_factor = false;                       \
    static constexpr bool light_apply = false;                       \
    static constexpr bool fused_micro = false;                       \
    static constexpr bool gemm_apply = true;                         \
    static constexpr bool adaptive = true;                           \
    static constexpr int tail_policy = POLICY;                       \
  }

ADAPTIVE_SPEC(15, 1);
ADAPTIVE_SPEC(16, 2);
ADAPTIVE_SPEC(17, 3);

#undef ADAPTIVE_SPEC

template <int... Ids>
constexpr std::array<bool, sizeof...(Ids)> flags_usage_of(
    std::integer_sequence<int, Ids...>) {
  return {(Variant<Ids>::trsm_mode == kTrsmFusedMicro)...};
}

// Which variants consume the per-micro flag workspace; derived from
// the SPEC table so new fused variants cannot drift out of step.
constexpr auto kVariantUsesFlags =
    flags_usage_of(std::make_integer_sequence<int, kVariantCount>{});

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

__device__ __forceinline__ void root_pair(
    float value, float& diagonal, float& inverse) {
  diagonal = __fsqrt_rn(value);
  inverse = __fdiv_rn(1.0f, diagonal);
}

// Right-looking factorization of a 32-column strip of the shared
// tile, single warp; mirrors LAPACK spotf2 on rows [begin, 128).
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
  const int elements = Size * Size;
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

// Flat right-looking factorization of the whole 128x128 shared tile:
// per column, a single root, a CTA-parallel scale, and a row-parallel
// rank-1 trailing update. Replaces the warp-serial blocked chain for
// batch=1, where NCU measured it 47.6% barrier-stalled with nothing
// concurrent to hide behind.
__device__ __forceinline__ void factor_flat(
    float* tile, float* inverse_diagonal) {
  const int thread = static_cast<int>(threadIdx.x);
#pragma unroll 1
  for (int column = 0; column < kMicro; ++column) {
    if (thread == 0) {
      float diagonal;
      float inverse;
      root_pair(
          tile_at(tile, column, column), diagonal, inverse);
      tile_at(tile, column, column) = diagonal;
      inverse_diagonal[column] = inverse;
    }
    __syncthreads();
    const float inverse = inverse_diagonal[column];
    for (int row = column + 1 + thread; row < kMicro;
         row += static_cast<int>(blockDim.x)) {
      tile_at(tile, row, column) *= inverse;
    }
    __syncthreads();
    for (int row = column + 1 + thread; row < kMicro;
         row += static_cast<int>(blockDim.x)) {
      const float left = tile_at(tile, row, column);
#pragma unroll 4
      for (int target = column + 1; target <= row; ++target) {
        tile_at(tile, row, target) = fmaf(
            -left, tile_at(tile, target, column),
            tile_at(tile, row, target));
      }
    }
    __syncthreads();
  }
}

// Register-blocked flat factorization: columns advance in groups of
// eight (MAGMA POTF2_NB precedent). Each group factors its 8x8 corner
// with one warp, solves each sub-panel row's eight entries entirely in
// registers, publishes the solved columns into the separate
// __restrict__ panel buffer, then applies a register-accumulated
// rank-8 trailing update with four concurrent accumulators. The
// separate buffer removes the tile store/load alias that serialized
// the naive flat scheme (variant 8), and one store per eight FMAs
// keeps the update throughput-bound.
__device__ __forceinline__ void factor_rank8(
    float* __restrict__ tile,
    float* __restrict__ inverse_diagonal,
    float* __restrict__ panel) {
  constexpr int kGroup = 8;
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
#pragma unroll 1
  for (int base = 0; base < kMicro; base += kGroup) {
    if (thread < 32) {
#pragma unroll 1
      for (int j = 0; j < kGroup; ++j) {
        const int column = base + j;
        float inverse = 0.0f;
        if (lane == j) {
          float diagonal;
          root_pair(
              tile_at(tile, column, column), diagonal, inverse);
          tile_at(tile, column, column) = diagonal;
          inverse_diagonal[column] = inverse;
        }
        inverse = __shfl_sync(0xffffffffu, inverse, j);
        if (lane > j && lane < kGroup) {
          tile_at(tile, base + lane, column) *= inverse;
        }
        __syncwarp();
        if (lane > j && lane < kGroup) {
          const int row = base + lane;
          const float left = tile_at(tile, row, column);
#pragma unroll
          for (int target = j + 1; target <= lane; ++target) {
            tile_at(tile, row, base + target) = fmaf(
                -left, tile_at(tile, base + target, column),
                tile_at(tile, row, base + target));
          }
        }
        __syncwarp();
      }
    }
    __syncthreads();
    const int row = base + kGroup + thread;
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
              -solved[i], tile_at(tile, base + j, base + i), value);
        }
        solved[j] = value * inverse_diagonal[base + j];
      }
#pragma unroll
      for (int k = 0; k < kGroup; ++k) {
        tile_at(tile, row, base + k) = solved[k];
        panel[row * kPanelLd + k] = solved[k];
      }
    }
    __syncthreads();
    if (row < kMicro) {
      int target = base + kGroup;
      for (; target + 3 <= row; target += 4) {
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
      }
      for (; target <= row; ++target) {
        float value = tile_at(tile, row, target);
#pragma unroll
        for (int k = 0; k < kGroup; ++k) {
          value = fmaf(
              -solved[k], panel[target * kPanelLd + k], value);
        }
        tile_at(tile, row, target) = value;
      }
    }
    __syncthreads();
  }
}

// Wide redundant-corner factorization, 512 threads. NCU measured the
// rank-8 kernel at 12.5% occupancy (8 warps on one SM) with 11 warp
// cycles per issued instruction and the corner/solve phases barrier-
// dominated. Here every thread factors the 8x8 group corner
// redundantly in registers (a ~150-cycle chain all warps run in
// parallel - cheaper than warp-serial communication at this size),
// four threads share each sub-panel row (redundant register solve,
// quarter-split rank-8 update), and each group needs two CTA barriers
// instead of three, with sixteen warps to hide latency.
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
// place and, when BuildInverse is set, writes the dense inverse of
// the resulting lower factor to the t_inv workspace.
template <bool BuildInverse, int FactorMode>
__global__ __launch_bounds__(factor_threads_of(FactorMode))
void factor128_kernel(
    float* __restrict__ output, int begin,
    float* __restrict__ t_inv) {
  extern __shared__ __align__(16) float dynamic_floats[];
  float* tile = dynamic_floats;
  float* inverse_diagonal = tile + kMicro * kTileLd;
  float* panel = inverse_diagonal + kMicro;
  float* inverse_base =
      FactorMode == kFactorRank8
          ? panel + kMicro * kPanelLd
          : panel;
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
  if constexpr (FactorMode == kFactorWide) {
    factor_wide(tile, inverse_diagonal, panel);
  } else if constexpr (FactorMode == kFactorRank8) {
    factor_rank8(tile, inverse_diagonal, panel);
  } else if constexpr (FactorMode == kFactorFlat) {
    factor_flat(tile, inverse_diagonal);
  } else {
    factor_local(tile, inverse_diagonal);
  }
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
  if constexpr (BuildInverse) {
    float* tinv = inverse_base;
    float* mid = tinv + kMicro * kTileLd;
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
// the diagonal block at `begin`, where T is the dense inverse of the
// lower factor (strict upper exactly zero). Each thread accumulates
// an 8x8 register tile; the k loop is clipped per 64-column half
// because T[column][k] vanishes for k > column. LightT skips the
// shared copy of T and reads it through the read-only cache, halving
// dynamic shared memory so two CTAs can share an SM.
template <bool LightT>
__global__ __launch_bounds__(256)
void trsm_apply_kernel(
    float* __restrict__ output, int begin,
    const float* __restrict__ t_inv) {
  extern __shared__ __align__(16) float dynamic_floats[];
  float* x_tile = dynamic_floats;
  float* t_tile = LightT ? nullptr : x_tile + kMicro * kTileLd;
  const int row_begin =
      begin + kMicro + static_cast<int>(blockIdx.x) * kMicro;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kMicro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 7;
    const int column = linear & (kMicro - 1);
    x_tile[row * kTileLd + column] = load_global(
        output + matrix_index(row_begin + row, begin + column));
    if constexpr (!LightT) {
      t_tile[row * kTileLd + column] = load_global(t_inv + linear);
    }
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
      if constexpr (LightT) {
        right[column] = __ldg(t_inv + t_row * kMicro + k);
      } else {
        right[column] = t_tile[t_row * kTileLd + k];
      }
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

// Release/acquire flag protocol between co-resident CTAs of one
// launch (gaunernst QR precedent). The publishing CTA runs
// __syncthreads() first so the release by thread 0 orders every
// thread's prior global writes; pollers use a relaxed no-allocate
// load so the spin never caches the flag line, then acquire.
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
    const int row = linear >> 7;
    const int column = linear & (kMicro - 1);
    x_tile[row * kTileLd + column] = load_global(
        output + matrix_index(row_begin + row, begin + column));
  }
}

// 512-thread in-place X := X T^T for one 128x128 tile: sixteen warps
// in a 4x4 grid, each warp owning 32 rows x 32 columns, each thread
// an 8x4 register tile. The k loop clips per 32-column warp stripe
// because T[column][k] vanishes for k > column.
__device__ __forceinline__ void apply_tile_wide(
    const float* __restrict__ x_tile,
    const float* __restrict__ t_tile,
    float* __restrict__ output, int begin, int tile_index) {
  const int row_begin = begin + kMicro + tile_index * kMicro;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_row = warp >> 2;
  const int warp_column = warp & 3;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;
  const int k_limit = warp_column * 32 + 32;
  float value[8][4];
#pragma unroll
  for (int row = 0; row < 8; ++row) {
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      value[row][column] = 0.0f;
    }
  }
#pragma unroll 1
  for (int k = 0; k < k_limit; ++k) {
    float left[8];
    float right[4];
#pragma unroll
    for (int row = 0; row < 8; ++row) {
      left[row] = x_tile[
          (warp_row * 32 + lane_row + row * 4) * kTileLd + k];
    }
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int t_row =
          warp_column * 32 + lane_column + column * 8;
      right[column] = t_tile[t_row * kTileLd + k];
    }
#pragma unroll
    for (int row = 0; row < 8; ++row) {
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        value[row][column] = fmaf(
            left[row], right[column], value[row][column]);
      }
    }
  }
#pragma unroll
  for (int row = 0; row < 8; ++row) {
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_row = warp_row * 32 + lane_row + row * 4;
      const int output_column =
          warp_column * 32 + lane_column + column * 8;
      store_global(
          output +
              matrix_index(
                  row_begin + output_row, begin + output_column),
          value[row][column]);
    }
  }
}

// One launch per micro step replacing the factor + apply pair. CTA 0
// factors the diagonal block and builds its inverse while every other
// CTA prefetches its first X tile into shared memory, hiding the tile
// loads under the factor's serial critical path. CTA 0 publishes the
// inverse through a per-micro release flag; consumers spin on one
// thread, acquire, broadcast through a barrier, then run in-place
// X := X T^T applies over a static strided tile assignment. The grid
// is clamped to co-resident capacity (checked in prepare), so every
// spinning CTA shares the GPU with CTA 0 and cannot deadlock. All
// tile data the consumers read was written by earlier kernels on the
// same queue; the inverse is the only intra-launch dependency.
__global__ __launch_bounds__(factor_threads_of(kFactorWide))
void fused_micro_kernel(
    float* __restrict__ output, int begin,
    float* __restrict__ t_inv, int* __restrict__ flags) {
  extern __shared__ __align__(16) float dynamic_floats[];
  const int tiles = (kN - begin - kMicro) / kMicro;
  int* flag = flags + (begin >> 7);
  if (blockIdx.x == 0) {
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
                    output +
                    matrix_index(begin + row, begin + column))
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
    if (tiles > 0) {
      build_inverse_128(tile, inverse_diagonal, tinv, mid);
      for (int linear = static_cast<int>(threadIdx.x);
           linear < kMicro * kMicro;
           linear += static_cast<int>(blockDim.x)) {
        const int row = linear >> 7;
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
    const int consumer_stride = static_cast<int>(gridDim.x) - 1;
    int tile_index = static_cast<int>(blockIdx.x) - 1;
    load_x_tile(x_tile, output, begin, tile_index);
    if (threadIdx.x == 0) {
      while (poll_flag(flag) == 0) {
        __nanosleep(64);
      }
      acquire_fence();
    }
    __syncthreads();
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kMicro * kMicro;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear >> 7;
      const int column = linear & (kMicro - 1);
      t_tile[row * kTileLd + column] = load_global(t_inv + linear);
    }
    __syncthreads();
    while (true) {
      apply_tile_wide(x_tile, t_tile, output, begin, tile_index);
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
  CublasStateGuard(cublasHandle_t handle, int mode)
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
        cublasSetMathMode(
            handle_,
            mode == kGuardTf32Math
                ? CUBLAS_TF32_TENSOR_OP_MATH
                : CUBLAS_DEFAULT_MATH),
        "select cuBLAS math mode");
    check_cublas(
        cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_ALLOWED),
        "enable cuBLAS atomic algorithms");
    check_cublas(
        cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
        "select host cuBLAS scalars");
#if B1N32768_HAS_EMULATION_API
    check_cublas(
        cublasGetEmulationStrategy(handle_, &emulation_),
        "query cuBLAS emulation strategy");
    if (mode == kGuardEmulated) {
      check_cublas(
          cublasSetEmulationStrategy(
              handle_, CUBLAS_EMULATION_STRATEGY_EAGER),
          "select eager cuBLAS emulation");
    }
#else
    TORCH_CHECK(
        mode != kGuardEmulated,
        "BF16x9 emulation requires cuBLAS 13 headers");
#endif
  }

  ~CublasStateGuard() {
#if B1N32768_HAS_EMULATION_API
    cublasSetEmulationStrategy(handle_, emulation_);
#endif
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
#if B1N32768_HAS_EMULATION_API
  cublasEmulationStrategy_t emulation_{};
#endif
};

cublasComputeType_t compute_of(int math) {
  switch (math) {
    case kMathFp32:
      return CUBLAS_COMPUTE_32F;
    case kMathTf32:
      return CUBLAS_COMPUTE_32F_FAST_TF32;
    default:
      TORCH_CHECK(
          kHasEmulation,
          "BF16x9 emulation requires cuBLAS 13 headers");
      return B1N32768_EMULATED_COMPUTE;
  }
}

// Row-major storage is presented to cuBLAS as its column-major
// transpose: OP_T on the first operand and OP_N on the second give
// C[j + n_idx][j + m_idx] -= sum_k L[j + m_idx][k] L[j + n_idx][k],
// which by symmetry of L L^T is the wanted history update.
void gemm_history(
    cublasHandle_t handle, float* output, int64_t panel_begin,
    int nb, int math) {
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
          compute_of(math), CUBLAS_GEMM_DEFAULT),
      "panel history GEMM");
}

void gemm_inner(
    cublasHandle_t handle, float* output, int64_t panel_begin,
    int64_t micro_begin, int math) {
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
          compute_of(math), CUBLAS_GEMM_DEFAULT),
      "micro history GEMM");
}

// Out-of-place apply scratch = X T^T as a dense GEMM: T is the dense
// inverse with exact strict-upper zeros, so the dense product equals
// the triangular one while cuBLAS runs it on TF32 tensor cores with
// deep pipelining - the in-kernel apply measured latency-bound at one
// CTA per SM. Col-major mapping: scratch_cm(128 x rows) =
// Tc^T(128 x 128) * Xc(128 x rows) with Xc the transposed view of the
// row-major sub-column (ld kN) and Tc that of t_inv (ld 128).
void gemm_apply_column(
    cublasHandle_t handle, float* output, const float* t_inv,
    float* scratch, int64_t micro_begin, int math) {
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
          compute_of(math), CUBLAS_GEMM_DEFAULT),
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

// Row-major right-side solve X := X inv(L11)^T maps to a col-major
// left-side solve with the transposed upper view of L11.
void strsm_column(
    cublasHandle_t handle, float* output, int64_t micro_begin) {
  const float one = 1.0f;
  const int rows = static_cast<int>(kN - micro_begin - kMicro);
  check_cublas(
      cublasStrsm(
          handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
          CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
          kMicro, rows, &one,
          output + micro_begin * kN + micro_begin, kN,
          output + (micro_begin + kMicro) * kN + micro_begin, kN),
      "column TRSM");
}

// Row-major lower SYRK maps to a col-major upper SYRK on the
// transposed view.
void ssyrk_trailing(
    cublasHandle_t handle, float* output, int64_t panel_begin,
    int nb) {
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const int64_t trailing_begin = panel_begin + nb;
  const int rows = static_cast<int>(kN - trailing_begin);
  check_cublas(
      cublasSsyrk(
          handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
          rows, nb, &alpha,
          output + trailing_begin * kN + panel_begin, kN,
          &beta,
          output + trailing_begin * kN + trailing_begin, kN),
      "trailing SYRK");
}

void launch_copy(const float* input, float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(2048, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, copy_lower_kernel, input, output);
}

template <bool BuildInverse, int FactorMode>
void launch_factor(float* output, int begin, float* t_inv) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(factor_threads_of(FactorMode), 1, 1);
  config.dynamicSmemBytes =
      factor_bytes_of(BuildInverse, FactorMode);
  cudaLaunchKernelEx(
      &config, factor128_kernel<BuildInverse, FactorMode>,
      output, begin, t_inv);
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

template <bool LightT>
void launch_apply(float* output, int begin, const float* t_inv) {
  const int tiles = (kN - begin - kMicro) / kMicro;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(tiles, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  config.dynamicSmemBytes =
      LightT ? kApplyLightBytes : kApplyBytes;
  cudaLaunchKernelEx(
      &config, trsm_apply_kernel<LightT>, output, begin, t_inv);
}

// Co-resident CTA capacity of the fused kernel, latched once after
// prepare() has opted the kernel into its dynamic shared memory.
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
        &active, fused_micro_kernel,
        factor_threads_of(kFactorWide), kFusedBytes);
    TORCH_CHECK(
        status == cudaSuccess,
        "fused occupancy query failed: ", cudaGetErrorString(status));
    TORCH_CHECK(
        active >= 1 && sm_count >= 2,
        "fused micro kernel must be co-resident with its consumers");
    return sm_count * active;
  }();
  return limit;
}

void launch_fused_micro(
    float* output, int begin, float* t_inv, int* flags) {
  const int tiles = (kN - begin - kMicro) / kMicro;
  const int limit = fused_micro_grid_limit();
  const int grid = 1 + tiles < limit ? 1 + tiles : limit;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid, 1, 1);
  config.blockDim = dim3(factor_threads_of(kFactorWide), 1, 1);
  config.dynamicSmemBytes = kFusedBytes;
  cudaLaunchKernelEx(
      &config, fused_micro_kernel, output, begin, t_inv, flags);
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
    float* output, const float* input, float* t_inv, int* flags,
    float* scratch) {
  using V = Variant<Id>;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle, V::guard_mode);
  launch_copy(input, output);
  for (int64_t panel = 0; panel < kN; panel += V::nb) {
    if (V::schedule == kLeftLook && panel > 0) {
      gemm_history(handle, output, panel, V::nb, V::math_big);
    }
    for (int64_t micro = panel; micro < panel + V::nb;
         micro += kMicro) {
      if (micro > panel) {
        gemm_inner(handle, output, panel, micro, V::math_inner);
      }
      if constexpr (V::fused_micro) {
        launch_fused_micro(
            output, static_cast<int>(micro), t_inv, flags);
      } else {
        launch_factor<V::build_inverse, V::factor_mode>(
            output, static_cast<int>(micro), t_inv);
        if (micro + kMicro < kN) {
          if constexpr (V::gemm_apply) {
            gemm_apply_column(
                handle, output, t_inv, scratch, micro,
                V::math_inner);
            launch_copy_back(
                output, scratch, static_cast<int>(micro));
          } else if (V::trsm_mode == kTrsmInverse) {
            launch_apply<V::light_apply>(
                output, static_cast<int>(micro), t_inv);
          } else {
            strsm_column(handle, output, micro);
          }
        }
      }
    }
    if (V::schedule == kRightLook && panel + V::nb < kN) {
      ssyrk_trailing(handle, output, panel, V::nb);
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
  CublasStateGuard guard(handle, V::guard_mode);
  launch_copy(input, output);
  for (int64_t panel = 0; panel < kN; panel += V::nb) {
    if (panel > 0) {
      gemm_history(
          handle, output, panel, V::nb, kMathTf32);
    }
    int64_t micro = panel;
    while (micro < panel + V::nb) {
      const int remaining = static_cast<int>(kN - micro);
      int width = 128;
      if (Id == 15 && remaining <= 4096) {
        width = 64;
      } else if ((Id == 16 || Id == 17) && remaining <= 8192) {
        width = Id == 17 && remaining <= 1024 ? 32 : 64;
      }
      if (width == 128) {
        if (micro > panel) {
          gemm_inner(
              handle, output, panel, micro, kMathTf32);
        }
        launch_factor<true, kFactorWide>(
            output, static_cast<int>(micro), t_inv);
        if (micro + 128 < kN) {
          gemm_apply_column(
              handle, output, t_inv, scratch, micro, kMathTf32);
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
      "native input must have shape (1, 32768, 32768)");
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

template <int Id>
void configure_one() {
  using V = Variant<Id>;
  TORCH_CHECK(
      V::implemented,
      "variant ", Id, " (fused panel) is not implemented yet");
  if constexpr (V::fused_micro) {
    configure_dynamic(fused_micro_kernel, kFusedBytes);
    checked_attributes(fused_micro_kernel);
    TORCH_CHECK(
        fused_micro_grid_limit() >= 2,
        "fused micro kernel needs a co-resident consumer CTA");
  } else if constexpr (V::build_inverse) {
    configure_dynamic(
        factor128_kernel<true, V::factor_mode>,
        factor_bytes_of(true, V::factor_mode));
    configure_dynamic(
        trsm_apply_kernel<V::light_apply>,
        V::light_apply ? kApplyLightBytes : kApplyBytes);
    checked_attributes(factor128_kernel<true, V::factor_mode>);
    checked_attributes(trsm_apply_kernel<V::light_apply>);
    if constexpr (V::adaptive) {
      configure_dynamic(
          adaptive_factor_kernel<64>,
          AdaptiveTile<64>::factor_bytes);
      configure_dynamic(
          adaptive_factor_kernel<32>,
          AdaptiveTile<32>::factor_bytes);
      checked_attributes(adaptive_factor_kernel<64>);
      checked_attributes(adaptive_factor_kernel<32>);
    }
  } else {
    configure_dynamic(
        factor128_kernel<false, V::factor_mode>,
        factor_bytes_of(false, V::factor_mode));
    checked_attributes(factor128_kernel<false, V::factor_mode>);
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
    case 7: configure_one<7>(); break;
    case 8: configure_one<8>(); break;
    case 9: configure_one<9>(); break;
    case 10: configure_one<10>(); break;
    case 11: configure_one<11>(); break;
    case 12: configure_one<12>(); break;
    case 13: configure_one<13>(); break;
    case 14: configure_one<14>(); break;
    case 15: configure_one<15>(); break;
    case 16: configure_one<16>(); break;
    case 17: configure_one<17>(); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 17]");
  }
}

void launch_variant(
    const float* input, float* output, float* t_inv, int* flags,
    float* scratch, int variant) {
  switch (variant) {
    case 0: launch_staged<0>(output, input, t_inv, flags, scratch); break;
    case 1: launch_staged<1>(output, input, t_inv, flags, scratch); break;
    case 2: launch_staged<2>(output, input, t_inv, flags, scratch); break;
    case 3: launch_staged<3>(output, input, t_inv, flags, scratch); break;
    case 4: launch_staged<4>(output, input, t_inv, flags, scratch); break;
    case 5: launch_staged<5>(output, input, t_inv, flags, scratch); break;
    case 6:
    case 7:
      TORCH_CHECK(
          false, "fused panel variants are not implemented yet");
      break;
    case 8: launch_staged<8>(output, input, t_inv, flags, scratch); break;
    case 9: launch_staged<9>(output, input, t_inv, flags, scratch); break;
    case 10: launch_staged<10>(output, input, t_inv, flags, scratch); break;
    case 11: launch_staged<11>(output, input, t_inv, flags, scratch); break;
    case 12: launch_staged<12>(output, input, t_inv, flags, scratch); break;
    case 13: launch_staged<13>(output, input, t_inv, flags, scratch); break;
    case 14: launch_staged<14>(output, input, t_inv, flags, scratch); break;
    case 15: launch_adaptive_staged<15>(
        output, input, t_inv, scratch); break;
    case 16: launch_adaptive_staged<16>(
        output, input, t_inv, scratch); break;
    case 17: launch_adaptive_staged<17>(
        output, input, t_inv, scratch); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 17]");
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
  row[0] = Id;
  row[1] = V::schedule;
  row[2] = V::nb;
  row[3] = V::trsm_mode;
  row[4] = V::math_big;
  row[5] = V::math_inner;
  row[20] = kHasEmulation ? 1 : 0;
  row[21] = V::adaptive
      ? (Id == 17 ? 32 : 64) : kMicro;
  row[22] = V::guard_mode;
  row[23] = V::implemented ? 1 : 0;
  row[24] = V::factor_mode;
  row[25] = V::apply_mode;
  if constexpr (!V::implemented) {
    return;
  }
  constexpr int factor_bytes =
      V::fused_micro
          ? kFusedBytes
          : factor_bytes_of(V::build_inverse, V::factor_mode);
  constexpr int apply_bytes =
      V::fused_micro
          ? kFusedBytes
          : (V::light_apply ? kApplyLightBytes : kApplyBytes);
  cudaFuncAttributes factor{};
  cudaFuncAttributes apply{};
  int factor_active = 0;
  int apply_active = 0;
  if constexpr (V::fused_micro) {
    configure_dynamic(fused_micro_kernel, kFusedBytes);
    factor = checked_attributes(fused_micro_kernel);
    apply = factor;
    factor_active = active_blocks(
        fused_micro_kernel, factor_threads_of(V::factor_mode),
        kFusedBytes);
    apply_active = factor_active;
  } else if constexpr (V::build_inverse) {
    configure_dynamic(
        factor128_kernel<true, V::factor_mode>, factor_bytes);
    configure_dynamic(
        trsm_apply_kernel<V::light_apply>, apply_bytes);
    factor = checked_attributes(
        factor128_kernel<true, V::factor_mode>);
    apply = checked_attributes(trsm_apply_kernel<V::light_apply>);
    factor_active = active_blocks(
        factor128_kernel<true, V::factor_mode>,
        factor_threads_of(V::factor_mode), factor_bytes);
    apply_active = active_blocks(
        trsm_apply_kernel<V::light_apply>, 256, apply_bytes);
  } else {
    configure_dynamic(
        factor128_kernel<false, V::factor_mode>, factor_bytes);
    factor = checked_attributes(
        factor128_kernel<false, V::factor_mode>);
    factor_active = active_blocks(
        factor128_kernel<false, V::factor_mode>,
        factor_threads_of(V::factor_mode), factor_bytes);
  }
  const cudaFuncAttributes copy =
      checked_attributes(copy_lower_kernel);
  const cudaFuncAttributes wedges =
      checked_attributes(zero_wedges_kernel);
  const int panels = kN / V::nb;
  const int potrf128 =
      !V::adaptive ? kN / 128 : (Id == 15 ? 224 : 192);
  const int potrf64 =
      !V::adaptive ? 0 : (Id == 15 ? 64 : (Id == 16 ? 128 : 112));
  const int potrf32 = V::adaptive && Id == 17 ? 32 : 0;
  const int micros = potrf128 + potrf64 + potrf32;
  const int inner = micros - panels;
  const int big =
      V::schedule == kLeftLook ? panels - 1 : 0;
  const int syrk =
      V::schedule == kRightLook ? panels - 1 : 0;
  row[6] = factor_threads_of(V::factor_mode);
  row[7] = factor.numRegs;
  row[8] = factor.sharedSizeBytes;
  row[9] = factor.localSizeBytes;
  row[10] = factor_bytes;
  row[11] = apply.numRegs;
  row[12] = apply.sharedSizeBytes;
  row[13] = apply.localSizeBytes;
  row[14] = V::build_inverse ? apply_bytes : 0;
  row[15] = copy.numRegs;
  row[16] = wedges.numRegs;
  const int applies =
      V::fused_micro
          ? 0
          : (V::gemm_apply ? 2 * (micros - 1) : micros - 1);
  const int flag_fill = V::fused_micro ? 1 : 0;
  row[17] = factor_active;
  row[18] = apply_active;
  row[19] = 2 + big + syrk + inner + micros + applies + flag_fill;
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

void cholesky_b1n32768_prepare(int64_t variant) {
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 17]");
  configure_variant(static_cast<int>(variant));
}

void cholesky_b1n32768_out(
    const at::Tensor& data, at::Tensor output, int64_t variant) {
  check_input(data);
  check_output(data, output);
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 17]");
  c10::cuda::CUDAGuard device_guard(data.device());
  at::Tensor t_inv = at::empty(
      {kMicro, kMicro}, data.options());
  at::Tensor flags;
  int* flags_pointer = nullptr;
  if (kVariantUsesFlags[static_cast<int>(variant)]) {
    flags = at::zeros(
        {kN / kMicro}, data.options().dtype(at::kInt));
    flags_pointer = flags.data_ptr<int>();
  }
  at::Tensor scratch;
  float* scratch_pointer = nullptr;
  if (kVariantUsesScratch[static_cast<int>(variant)]) {
    scratch = at::empty({kN - kMicro, kMicro}, data.options());
    scratch_pointer = scratch.data_ptr<float>();
  }
  launch_variant(
      data.data_ptr<float>(), output.data_ptr<float>(),
      t_inv.data_ptr<float>(), flags_pointer, scratch_pointer,
      static_cast<int>(variant));
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b1n32768(
    const at::Tensor& data, int64_t variant) {
  auto output = at::empty_like(data);
  cholesky_b1n32768_out(data, output, variant);
  return output;
}

at::Tensor cholesky_b1n32768_metadata() {
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
  write_metadata<7>(rows);
  write_metadata<8>(rows);
  write_metadata<9>(rows);
  write_metadata<10>(rows);
  write_metadata<11>(rows);
  write_metadata<12>(rows);
  write_metadata<13>(rows);
  write_metadata<14>(rows);
  write_metadata<15>(rows);
  write_metadata<16>(rows);
  write_metadata<17>(rows);
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
            name=f"cholesky_b1n32768_b200_{tag}",
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
        and tuple(data.shape) == (1, 32768, 32768)
    ):
        return _run_variant(data, _DEFAULT_VARIANT)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
