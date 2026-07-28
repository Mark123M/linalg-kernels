import hashlib
import os
import re
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The autotuner may replace this exact line in retained candidate copies.
_DEFAULT_VARIANT = 21  # POPCORN_VARIANT
_CUTLASS_BASE_VARIANT = 21
_CUTLASS_VARIANT = 23
_VARIANT_NAMES = (
    "p64_raw_scalar_m4x4_t256",
    "p64_nr_scalar_m4x4_t256",
    "p64_precise_scalar_m4x4_t256",
    "p64_raw_sub4_m4x4_t256",
    "p32_raw_scalar_m2x4_t128",
    "p32_nr_scalar_m2x4_t128",
    "p64_raw_scalar_m4x4_t256_occ5",
    "p64_raw_tcgen05_tf32_t128_occ6",
    "p64_raw_scalar_preload_m4x4_t256_occ5",
    "p64_raw_scalar_left_m4x4_t256_occ5",
    "p64_raw_scalar_left_warp2_m4x4_t256_occ5",
    "p64_raw_block16_left_warp2_m4x4_t256_occ4",
    "p64_raw_block16_left_warp2_m4x4_t256_occ5",
    "p64_raw_scalar_left_product_m2x4_t256_occ4",
    "p64_raw_scalar_left_product_warp2_m2x4_t256_occ4",
    "p64_raw_block8_left_product_warp2_m2x4_t256_occ4",
    "p64_precise_scalar_left_product_warp2_m2x4_t256_occ4",
    "p64_raw_scalar_left_shared_warp2_m4x4_t256_occ4",
    "p64_raw_scalar_preload_warp2_m4x4_t256_occ5",
    "staged_p128_precise_sub4_cublas_fp32_t256",
    "staged_p128_precise_sub4_cublas_tf32_t256",
    "staged_p128_to_p64_at_r256_tf32",
    "staged_p128_p64_p32_at_r256_r64_tf32",
    "fused_stage3_regrow_cutlass_names",
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
    "update_mode",
    "minimum_blocks",
    "tmem_columns",
    "schedule_mode",
    "factor_mode",
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
#include <ATen/cuda/CUDAContextLight.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 640;
constexpr int kN = 512;
constexpr int kVariantCount = 23;
constexpr int kMetadataColumns = 22;
constexpr int kTmemDp = 1 << 16;
constexpr int kStageOuter = 128;
constexpr int kStageMicro = 64;
constexpr int kStagePanelCount = 4;
constexpr int kStageMicroCount = 8;
constexpr int kStageWidth = 4;
constexpr int kStageFactorBytes =
    static_cast<int>(sizeof(float)) *
    (kStageOuter * (kStageOuter + 1) + kStageOuter);
constexpr int kStageSolveBytes =
    static_cast<int>(sizeof(float)) *
    (32 * (kStageOuter + 1) +
     kStageMicro * (kStageOuter + kStageWidth));
constexpr int kStageLaunchCount = 12;
static_assert(kStageFactorBytes == 66560);
static_assert(kStageSolveBytes == 50304);

constexpr int kPreciseRoot = 0;
constexpr int kNewtonRoot = 1;
constexpr int kRawRoot = 2;
constexpr int kScalarSolve = 0;
constexpr int kSub4Solve = 1;
constexpr int kBlock16Solve = 2;
constexpr int kBlock8Solve = 3;
constexpr int kFp32Update = 0;
constexpr int kTensorUpdate = 1;
constexpr int kFp32PreloadUpdate = 2;
constexpr int kBlasFp32Update = 3;
constexpr int kBlasTf32Update = 4;
constexpr int kRightSchedule = 0;
constexpr int kLeftSchedule = 1;
constexpr int kProductLeftSchedule = 2;
constexpr int kSharedLeftSchedule = 3;
constexpr int kStagedSchedule = 4;
constexpr int kAdaptiveStagedSchedule = 5;
constexpr int kRecursiveFactor = 0;
constexpr int kWarp2Factor = 1;

template <int Id>
struct Variant;

#define SPEC(                                                           \
    ID, TILE, THREADS, ROOT, SOLVE, UPDATE, MIN_BLOCKS, TMEM,           \
    SCHEDULE, FACTOR)                                                   \
  template <> struct Variant<ID> {                                     \
    static constexpr int tile = TILE;                                  \
    static constexpr int threads = THREADS;                            \
    static constexpr int root = ROOT;                                  \
    static constexpr int solve = SOLVE;                                \
    static constexpr int update = UPDATE;                              \
    static constexpr int minimum_blocks = MIN_BLOCKS;                  \
    static constexpr int tmem_columns = TMEM;                          \
    static constexpr int schedule = SCHEDULE;                          \
    static constexpr int factor = FACTOR;                              \
    static constexpr int tail_policy = 0;                              \
  }

SPEC(0, 64, 256, kRawRoot, kScalarSolve, kFp32Update, 1, 0,
     kRightSchedule, kRecursiveFactor);
SPEC(1, 64, 256, kNewtonRoot, kScalarSolve, kFp32Update, 1, 0,
     kRightSchedule, kRecursiveFactor);
SPEC(2, 64, 256, kPreciseRoot, kScalarSolve, kFp32Update, 1, 0,
     kRightSchedule, kRecursiveFactor);
SPEC(3, 64, 256, kRawRoot, kSub4Solve, kFp32Update, 1, 0,
     kRightSchedule, kRecursiveFactor);
SPEC(4, 32, 128, kRawRoot, kScalarSolve, kFp32Update, 1, 0,
     kRightSchedule, kRecursiveFactor);
SPEC(5, 32, 128, kNewtonRoot, kScalarSolve, kFp32Update, 1, 0,
     kRightSchedule, kRecursiveFactor);
SPEC(6, 64, 256, kRawRoot, kScalarSolve, kFp32Update, 5, 0,
     kRightSchedule, kRecursiveFactor);
SPEC(7, 64, 128, kRawRoot, kScalarSolve, kTensorUpdate, 6, 64,
     kRightSchedule, kRecursiveFactor);
SPEC(8, 64, 256, kRawRoot, kScalarSolve, kFp32PreloadUpdate, 5, 0,
     kRightSchedule, kRecursiveFactor);
SPEC(9, 64, 256, kRawRoot, kScalarSolve, kFp32Update, 5, 0,
     kLeftSchedule, kRecursiveFactor);
SPEC(10, 64, 256, kRawRoot, kScalarSolve, kFp32Update, 5, 0,
     kLeftSchedule, kWarp2Factor);
SPEC(11, 64, 256, kRawRoot, kBlock16Solve, kFp32Update, 4, 0,
     kLeftSchedule, kWarp2Factor);
SPEC(12, 64, 256, kRawRoot, kBlock16Solve, kFp32Update, 5, 0,
     kLeftSchedule, kWarp2Factor);
SPEC(13, 64, 256, kRawRoot, kScalarSolve, kFp32Update, 4, 0,
     kProductLeftSchedule, kRecursiveFactor);
SPEC(14, 64, 256, kRawRoot, kScalarSolve, kFp32Update, 4, 0,
     kProductLeftSchedule, kWarp2Factor);
SPEC(15, 64, 256, kRawRoot, kBlock8Solve, kFp32Update, 4, 0,
     kProductLeftSchedule, kWarp2Factor);
SPEC(16, 64, 256, kPreciseRoot, kScalarSolve, kFp32Update, 4, 0,
     kProductLeftSchedule, kWarp2Factor);
SPEC(17, 64, 256, kRawRoot, kScalarSolve, kFp32Update, 4, 0,
     kSharedLeftSchedule, kWarp2Factor);
SPEC(18, 64, 256, kRawRoot, kScalarSolve, kFp32PreloadUpdate, 5, 0,
     kRightSchedule, kWarp2Factor);
SPEC(19, 128, 256, kPreciseRoot, kSub4Solve, kBlasFp32Update, 1, 0,
     kStagedSchedule, kRecursiveFactor);
SPEC(20, 128, 256, kPreciseRoot, kSub4Solve, kBlasTf32Update, 1, 0,
     kStagedSchedule, kRecursiveFactor);

#undef SPEC

#define ADAPTIVE_SPEC(ID, POLICY)                                     \
  template <> struct Variant<ID> {                                    \
    static constexpr int tile = 128;                                  \
    static constexpr int threads = 256;                               \
    static constexpr int root = kPreciseRoot;                         \
    static constexpr int solve = kSub4Solve;                          \
    static constexpr int update = kBlasTf32Update;                    \
    static constexpr int minimum_blocks = 1;                          \
    static constexpr int tmem_columns = 0;                            \
    static constexpr int schedule = kAdaptiveStagedSchedule;          \
    static constexpr int factor = kRecursiveFactor;                   \
    static constexpr int tail_policy = POLICY;                        \
  }

ADAPTIVE_SPEC(21, 1);
ADAPTIVE_SPEC(22, 2);

#undef ADAPTIVE_SPEC

__device__ __forceinline__ float load_global(
    const float* pointer) {
  return __ldcg(pointer);
}

__device__ __forceinline__ void store_global(
    float* pointer, float value) {
  __stcg(pointer, value);
}

__device__ __forceinline__ uint32_t shared_address(
    const void* pointer) {
  return static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void*>(pointer)));
}

__device__ __forceinline__ uint32_t to_tf32(float value) {
  uint32_t result;
  asm volatile(
      "cvt.rna.tf32.f32 %0, %1;"
      : "=r"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ int kmajor_offset(
    int row, int column) {
  return (row & 7) * 4 + (row >> 3) * 32 +
         (column & 3) + (column >> 2) * (64 * 4);
}

__device__ __forceinline__ uint64_t
make_kmajor_descriptor(const void* pointer) {
  const uint64_t start =
      static_cast<uint64_t>(shared_address(pointer) >> 4) &
      0x3fffull;
  const uint64_t leading = 64ull;
  const uint64_t stride = 8ull;
  return start | (leading << 16) | (stride << 32) |
         (1ull << 46);
}

__device__ __forceinline__ constexpr uint32_t
tf32_mma_descriptor() {
  return (1u << 4) | (2u << 7) | (2u << 10) |
         (8u << 17) | (4u << 24);
}

__device__ __forceinline__ void proxy_fence() {
  asm volatile(
      "fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tensor_after_sync_fence() {
  asm volatile(
      "tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tensor_barrier_init(
    uint64_t* barrier) {
  if (threadIdx.x == 0) {
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;" ::
        "r"(address) : "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ void tmem_allocate(
    uint32_t* destination, int columns) {
  if (static_cast<int>(threadIdx.x) < 32) {
    const uint32_t address = shared_address(destination);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned."
        "shared::cta.b32 [%0], %1;" ::
        "r"(address), "r"(columns) : "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ void tmem_deallocate(
    uint32_t base, int columns) {
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < 32) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 "
        "%0, %1;" :: "r"(base), "r"(columns));
  }
  __syncthreads();
}

__device__ __forceinline__ void tmem_relinquish() {
  if (static_cast<int>(threadIdx.x) < 32) {
    asm volatile(
        "tcgen05.relinquish_alloc_permit."
        "cta_group::1.sync.aligned;");
  }
  __syncthreads();
}

__device__ __forceinline__ void issue_tf32_mma(
    uint32_t tmem_base, uint64_t a_descriptor,
    uint64_t b_descriptor, bool accumulate) {
  if (threadIdx.x == 0) {
    const uint32_t instruction = tf32_mma_descriptor();
    const uint32_t scale = accumulate ? 1u : 0u;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 "
        "[%0], %1, %2, %3, {%5,%6,%7,%8}, p;\n\t"
        "}\n" ::
        "r"(tmem_base), "l"(a_descriptor),
        "l"(b_descriptor), "r"(instruction), "r"(scale),
        "r"(0u), "r"(0u), "r"(0u), "r"(0u));
  }
}

__device__ __forceinline__ void tensor_commit(
    uint64_t* barrier) {
  if (threadIdx.x == 0) {
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
        "shared::cluster.b64 [%0];" ::
        "r"(address) : "memory");
  }
}

__device__ __forceinline__ void tensor_wait_warp(
    uint64_t* barrier, int phase) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (lane == 0) {
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
  __syncwarp();
}

__device__ __forceinline__ float tmem_load_one(
    uint32_t address) {
  uint32_t value;
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x1.b32 "
      "{%0}, [%1];" : "=r"(value) : "r"(address));
  return __uint_as_float(value);
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

template <int RootMode>
__device__ __forceinline__ void factor_tile_warp2(
    float* tile, float* inverse_diagonal) {
  constexpr int kTile = 64;
  constexpr int kLd = kTile + 1;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (warp != 0) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row0 = lane * 2;
  const int row1 = row0 + 1;

#pragma unroll 1
  for (int column = 0; column < kTile; ++column) {
    const int owner = column >> 1;
    float inverse = 0.0f;
    if (lane == owner) {
      float diagonal;
      root_pair<RootMode>(
          tile[column * kLd + column], diagonal, inverse);
      tile[column * kLd + column] = diagonal;
      inverse_diagonal[column] = inverse;
    }
    inverse = __shfl_sync(0xffffffffu, inverse, owner);

    float value0 = 0.0f;
    float value1 = 0.0f;
    if (row0 > column) {
      value0 = tile[row0 * kLd + column] * inverse;
      tile[row0 * kLd + column] = value0;
    }
    if (row1 > column) {
      value1 = tile[row1 * kLd + column] * inverse;
      tile[row1 * kLd + column] = value1;
    }

#pragma unroll 4
    for (int target = column + 1; target < kTile; ++target) {
      const int source = target >> 1;
      const float pivot = (target & 1)
          ? __shfl_sync(0xffffffffu, value1, source)
          : __shfl_sync(0xffffffffu, value0, source);
      if (row0 >= target) {
        tile[row0 * kLd + target] = fmaf(
            -value0, pivot, tile[row0 * kLd + target]);
      }
      if (row1 >= target) {
        tile[row1 * kLd + target] = fmaf(
            -value1, pivot, tile[row1 * kLd + target]);
      }
    }
    __syncwarp();
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

template <int Tile, int Threads>
__device__ __forceinline__ void store_factor_tile(
    const float* tile, float* matrix, int begin) {
  constexpr int kLd = Tile + 1;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Tile * Tile; linear += Threads) {
    const int row = linear / Tile;
    const int column = linear % Tile;
    store_global(
        matrix + (begin + row) * kN + begin + column,
        column <= row ? tile[row * kLd + column] : 0.0f);
  }
}

template <int Tile, int Threads>
__device__ __forceinline__ void zero_outer_upper(
    float* matrix) {
  constexpr int kTileCount = kN / Tile;
  constexpr int kVectorsPerTileRow = Tile / 4;
  const float4 zeros =
      make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int row_tile = 0; row_tile < kTileCount; ++row_tile) {
    for (int column_tile = row_tile + 1;
         column_tile < kTileCount; ++column_tile) {
      for (int vector = static_cast<int>(threadIdx.x);
           vector < Tile * kVectorsPerTileRow;
           vector += Threads) {
        const int row = vector / kVectorsPerTileRow;
        const int vector_column =
            vector % kVectorsPerTileRow;
        float* destination =
            matrix + (row_tile * Tile + row) * kN +
            column_tile * Tile + vector_column * 4;
        *reinterpret_cast<float4*>(destination) = zeros;
      }
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
  } else if constexpr (SolveMode == kSub4Solve) {
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
  } else {
    static_assert(
        SolveMode == kBlock16Solve ||
        SolveMode == kBlock8Solve);
    static_assert(Tile == 64);
    constexpr int kSolveBlock =
        SolveMode == kBlock16Solve ? 16 : 8;
    const int row = static_cast<int>(threadIdx.x);
    if (row < Tile) {
#pragma unroll
      for (int block = 0;
           block < Tile; block += kSolveBlock) {
        float values[kSolveBlock];
#pragma unroll
        for (int item = 0; item < kSolveBlock; ++item) {
          values[item] =
              rhs[row * kLd + block + item];
        }

#pragma unroll 4
        for (int k = 0; k < block; ++k) {
          const float solved = rhs[row * kLd + k];
#pragma unroll
          for (int item = 0; item < kSolveBlock; ++item) {
            values[item] = fmaf(
                -solved,
                diagonal[(block + item) * kLd + k],
                values[item]);
          }
        }

#pragma unroll
        for (int item = 0; item < kSolveBlock; ++item) {
          const int column = block + item;
          const float solved =
              values[item] * inverse_diagonal[column];
          values[item] = solved;
          rhs[row * kLd + column] = solved;
#pragma unroll
          for (int target = item + 1;
               target < kSolveBlock; ++target) {
            values[target] = fmaf(
                -solved,
                diagonal[(block + target) * kLd + column],
                values[target]);
          }
        }
      }
    }
  }
}

template <int Tile, bool PreloadDestination>
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
    float product[4][4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const int output_row =
          row_base + lane_row + row * 4;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        const int output_column =
            column_base + lane_column + column * 8;
        if constexpr (PreloadDestination) {
          product[row][column] =
              !diagonal || output_column <= output_row
                  ? load_global(
                        matrix + (row_begin + output_row) * kN +
                        column_begin + output_column)
                  : 0.0f;
        } else {
          product[row][column] = 0.0f;
        }
      }
    }
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
              PreloadDestination
                  ? -left_values[row] : left_values[row],
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
              PreloadDestination
                  ? product[row][column]
                  : load_global(destination) -
                        product[row][column]);
        }
      }
    }
  } else {
    const int row_base = (warp >> 1) * 16;
    const int column_base = (warp & 1) * 16;
    const int lane_row = lane >> 2;
    const int lane_column = lane & 3;
    float product[2][4];
#pragma unroll
    for (int row = 0; row < 2; ++row) {
      const int output_row =
          row_base + lane_row + row * 8;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        const int output_column =
            column_base + lane_column + column * 4;
        if constexpr (PreloadDestination) {
          product[row][column] =
              !diagonal || output_column <= output_row
                  ? load_global(
                        matrix + (row_begin + output_row) * kN +
                        column_begin + output_column)
                  : 0.0f;
        } else {
          product[row][column] = 0.0f;
        }
      }
    }
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
            PreloadDestination ? -left0 : left0,
            right_values[column],
            product[0][column]);
        product[1][column] = fmaf(
            PreloadDestination ? -left1 : left1,
            right_values[column],
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
              PreloadDestination
                  ? product[row][column]
                  : load_global(destination) -
                        product[row][column]);
        }
      }
    }
  }
}

template <bool Diagonal, int Threads>
__device__ __forceinline__ void left_current_tile(
    const float* input, const float* matrix,
    float* current, float* operand,
    int row_begin, int column_begin, int column_tile) {
  constexpr int kTile = 64;
  constexpr int kLd = kTile + 1;
  static_assert(Threads == 256);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = (warp >> 1) * 16;
  const int column_base = (warp & 1) * 32;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;
  float accumulator[4][4];

#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const int output_row =
        row_base + lane_row + row * 4;
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_column =
          column_base + lane_column + column * 8;
      accumulator[row][column] = load_global(
          input + (row_begin + output_row) * kN +
          column_begin + output_column);
    }
  }

#pragma unroll 1
  for (int previous_tile = 0;
       previous_tile < column_tile; ++previous_tile) {
    const int previous_begin = previous_tile * kTile;
    load_tile<kTile, Threads>(
        matrix, current, row_begin, previous_begin);
    if constexpr (!Diagonal) {
      load_tile<kTile, Threads>(
          matrix, operand, column_begin, previous_begin);
    }
    __syncthreads();
    const float* right = Diagonal ? current : operand;

#pragma unroll 1
    for (int k = 0; k < kTile; ++k) {
      float left_values[4];
      float right_values[4];
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        left_values[row] = current[
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
          accumulator[row][column] = fmaf(
              -left_values[row], right_values[column],
              accumulator[row][column]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const int output_row =
        row_base + lane_row + row * 4;
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_column =
          column_base + lane_column + column * 8;
      current[output_row * kLd + output_column] =
          accumulator[row][column];
    }
  }
  if constexpr (!Diagonal) {
    load_tile<kTile, Threads>(
        matrix, operand, column_begin, column_begin);
  }
  __syncthreads();
}

template <bool Diagonal, int Threads>
__device__ __forceinline__ void left_product_current_tile(
    const float* input, const float* matrix,
    float* left, float* right, float* result,
    int row_begin, int column_begin, int column_tile) {
  constexpr int kTile = 64;
  constexpr int kLd = kTile + 1;
  static_assert(Threads == 256);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = (warp >> 1) * 16;
  const int column_base = (warp & 1) * 32;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;

#pragma unroll
  for (int row_phase = 0; row_phase < 2; ++row_phase) {
    float accumulator[2][4];
#pragma unroll
    for (int row = 0; row < 2; ++row) {
      const int output_row =
          row_base + lane_row + row_phase * 8 + row * 4;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        const int output_column =
            column_base + lane_column + column * 8;
        accumulator[row][column] = load_global(
            input + (row_begin + output_row) * kN +
            column_begin + output_column);
      }
    }

#pragma unroll 1
    for (int previous_tile = 0;
         previous_tile < column_tile; ++previous_tile) {
      const int previous_begin = previous_tile * kTile;
      load_tile<kTile, Threads>(
          matrix, left, row_begin, previous_begin);
      if constexpr (!Diagonal) {
        load_tile<kTile, Threads>(
            matrix, right, column_begin, previous_begin);
      }
      __syncthreads();
      const float* right_values_tile =
          Diagonal ? left : right;
      float product[2][4] = {};

#pragma unroll 1
      for (int k = 0; k < kTile; ++k) {
        float left_values[2];
        float right_values[4];
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          left_values[row] = left[
              (row_base + lane_row +
               row_phase * 8 + row * 4) *
                  kLd +
              k];
        }
#pragma unroll
        for (int column = 0; column < 4; ++column) {
          right_values[column] = right_values_tile[
              (column_base + lane_column + column * 8) *
                  kLd +
              k];
        }
#pragma unroll
        for (int row = 0; row < 2; ++row) {
#pragma unroll
          for (int column = 0; column < 4; ++column) {
            product[row][column] = fmaf(
                left_values[row], right_values[column],
                product[row][column]);
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (int row = 0; row < 2; ++row) {
#pragma unroll
        for (int column = 0; column < 4; ++column) {
          accumulator[row][column] -= product[row][column];
        }
      }
    }

#pragma unroll
    for (int row = 0; row < 2; ++row) {
      const int output_row =
          row_base + lane_row + row_phase * 8 + row * 4;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        const int output_column =
            column_base + lane_column + column * 8;
        result[output_row * kLd + output_column] =
            accumulator[row][column];
      }
    }
  }
  if constexpr (!Diagonal) {
    load_tile<kTile, Threads>(
        matrix, right, column_begin, column_begin);
  }
  __syncthreads();
}

template <bool Diagonal, int Threads>
__device__ __forceinline__ void left_shared_current_tile(
    const float* input, const float* matrix,
    float* left, float* right, float* result,
    int row_begin, int column_begin, int column_tile) {
  constexpr int kTile = 64;
  constexpr int kLd = kTile + 1;
  static_assert(Threads == 256);
  load_tile<kTile, Threads>(
      input, result, row_begin, column_begin);
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = (warp >> 1) * 16;
  const int column_base = (warp & 1) * 32;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;

#pragma unroll 1
  for (int previous_tile = 0;
       previous_tile < column_tile; ++previous_tile) {
    const int previous_begin = previous_tile * kTile;
    load_tile<kTile, Threads>(
        matrix, left, row_begin, previous_begin);
    if constexpr (!Diagonal) {
      load_tile<kTile, Threads>(
          matrix, right, column_begin, previous_begin);
    }
    __syncthreads();
    const float* right_values_tile =
        Diagonal ? left : right;
    float product[4][4] = {};

#pragma unroll 1
    for (int k = 0; k < kTile; ++k) {
      float left_values[4];
      float right_values[4];
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        left_values[row] = left[
            (row_base + lane_row + row * 4) * kLd + k];
      }
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        right_values[column] = right_values_tile[
            (column_base + lane_column + column * 8) *
                kLd +
            k];
      }
#pragma unroll
      for (int row = 0; row < 4; ++row) {
#pragma unroll
        for (int column = 0; column < 4; ++column) {
          product[row][column] = fmaf(
              left_values[row], right_values[column],
              product[row][column]);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const int output_row =
          row_base + lane_row + row * 4;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        const int output_column =
            column_base + lane_column + column * 8;
        result[output_row * kLd + output_column] -=
            product[row][column];
      }
    }
  }
  if constexpr (!Diagonal) {
    load_tile<kTile, Threads>(
        matrix, right, column_begin, column_begin);
  }
  __syncthreads();
}

__device__ __forceinline__ float& stage_tile_at(
    float* tile, int row, int column) {
  return tile[row * (kStageOuter + 1) + column];
}

template <int RootMode>
__device__ __forceinline__ void stage_potf2_32(
    float* tile, float* inverse_diagonal, int begin) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (warp == 0) {
#pragma unroll 1
    for (int local_column = 0;
         local_column < 32; ++local_column) {
      const int column = begin + local_column;
      float inverse = 0.0f;
      if (lane == local_column) {
        float diagonal;
        root_pair<RootMode>(
            stage_tile_at(tile, column, column),
            diagonal, inverse);
        stage_tile_at(tile, column, column) = diagonal;
        inverse_diagonal[column] = inverse;
      }
      inverse = __shfl_sync(
          0xffffffffu, inverse, local_column);
      if (lane > local_column) {
        const int row = begin + lane;
        stage_tile_at(tile, row, column) *= inverse;
      }
      __syncwarp();
      if (lane > local_column) {
        const int row = begin + lane;
        const float left =
            stage_tile_at(tile, row, column);
#pragma unroll 4
        for (int target_local = local_column + 1;
             target_local <= lane; ++target_local) {
          const int target = begin + target_local;
          stage_tile_at(tile, row, target) = fmaf(
              -left,
              stage_tile_at(tile, target, column),
              stage_tile_at(tile, row, target));
        }
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

template <int Rows, int Columns>
__device__ __forceinline__ void stage_local_trsm(
    float* tile, const float* inverse_diagonal,
    int row_begin, int column_begin) {
  const int lane =
      static_cast<int>(threadIdx.x) & (kStageWidth - 1);
  const int row_index =
      static_cast<int>(threadIdx.x) / kStageWidth;
  if (row_index < Rows) {
    const int row = row_begin + row_index;
#pragma unroll 1
    for (int local_column = 0;
         local_column < Columns; ++local_column) {
      const int column = column_begin + local_column;
      float partial = 0.0f;
#pragma unroll 4
      for (int k = lane;
           k < local_column; k += kStageWidth) {
        partial = fmaf(
            stage_tile_at(
                tile, row, column_begin + k),
            stage_tile_at(
                tile, column, column_begin + k),
            partial);
      }
#pragma unroll
      for (int offset = kStageWidth / 2;
           offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(
            0xffffffffu, partial, offset, kStageWidth);
      }
      if (lane == 0) {
        stage_tile_at(tile, row, column) =
            (stage_tile_at(tile, row, column) - partial) *
            inverse_diagonal[column];
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

template <int Size, int K>
__device__ __forceinline__ void stage_local_update(
    float* tile, int target, int panel) {
  constexpr int kElements = Size * Size;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kElements;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Size;
    const int column = linear % Size;
    if (column <= row) {
      float value =
          stage_tile_at(tile, target + row, target + column);
#pragma unroll 4
      for (int k = 0; k < K; ++k) {
        value = fmaf(
            -stage_tile_at(tile, target + row, panel + k),
            stage_tile_at(tile, target + column, panel + k),
            value);
      }
      stage_tile_at(
          tile, target + row, target + column) = value;
    }
  }
  __syncthreads();
}

template <int RootMode>
__device__ __forceinline__ void stage_factor_local(
    float* tile, float* inverse_diagonal) {
  stage_potf2_32<RootMode>(
      tile, inverse_diagonal, 0);
  stage_local_trsm<32, 32>(
      tile, inverse_diagonal, 32, 0);
  stage_local_update<32, 32>(tile, 32, 0);
  stage_potf2_32<RootMode>(
      tile, inverse_diagonal, 32);
  stage_local_trsm<64, 64>(
      tile, inverse_diagonal, 64, 0);
  stage_local_update<64, 64>(tile, 64, 0);
  stage_potf2_32<RootMode>(
      tile, inverse_diagonal, 64);
  stage_local_trsm<32, 32>(
      tile, inverse_diagonal, 96, 64);
  stage_local_update<32, 32>(tile, 96, 64);
  stage_potf2_32<RootMode>(
      tile, inverse_diagonal, 96);
}

template <int RootMode>
__device__ __forceinline__ void stage_factor_global(
    float* matrix, int begin, float* work) {
  float* tile = work;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kStageOuter * kStageOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kStageOuter;
    const int column = linear % kStageOuter;
    stage_tile_at(tile, row, column) =
        column <= row
            ? load_global(
                  matrix + (begin + row) * kN +
                  begin + column)
            : 0.0f;
  }
  __syncthreads();
  float* inverse_diagonal =
      tile + kStageOuter * (kStageOuter + 1);
  stage_factor_local<RootMode>(
      tile, inverse_diagonal);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kStageOuter * kStageOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kStageOuter;
    const int column = linear % kStageOuter;
    if (column <= row) {
      store_global(
          matrix + (begin + row) * kN +
          begin + column,
          stage_tile_at(tile, row, column));
    }
  }
}

template <int Block, int LocalColumn, int RegisterCount>
__device__ __forceinline__ void stage_trsm_column(
    float (&values)[RegisterCount],
    const float* diagonal, const float* panel,
    int row, int lane) {
  constexpr int kDiagonalLd = kStageOuter + 1;
  constexpr int kPanelLd = kStageOuter + kStageWidth;
  constexpr int kBlockBegin = Block * 32;
  constexpr int kColumn = kBlockBegin + LocalColumn;
  constexpr int kOwner =
      LocalColumn & (kStageWidth - 1);
  constexpr int kOwnerSlot =
      LocalColumn / kStageWidth;
  static_assert(RegisterCount == 32 / kStageWidth);
  float partial = 0.0f;
#pragma unroll 4
  for (int k = lane; k < kBlockBegin;
       k += kStageWidth) {
    partial = fmaf(
        panel[row * kPanelLd + k],
        diagonal[LocalColumn * kDiagonalLd + k],
        partial);
  }
#pragma unroll
  for (int slot = 0; slot < RegisterCount; ++slot) {
    const int local_k =
        lane + slot * kStageWidth;
    if (local_k < LocalColumn) {
      partial = fmaf(
          values[slot],
          diagonal[
              LocalColumn * kDiagonalLd +
              kBlockBegin + local_k],
          partial);
    }
  }
#pragma unroll
  for (int offset = kStageWidth / 2;
       offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(
        0xffffffffu, partial, offset, kStageWidth);
  }
  const float owned_rhs = values[kOwnerSlot];
  const float rhs = __shfl_sync(
      0xffffffffu, owned_rhs, kOwner, kStageWidth);
  float solved = 0.0f;
  if (lane == 0) {
    solved =
        (rhs - partial) /
        diagonal[LocalColumn * kDiagonalLd + kColumn];
  }
  solved = __shfl_sync(
      0xffffffffu, solved, 0, kStageWidth);
  if (lane == kOwner) {
    values[kOwnerSlot] = solved;
  }
}

template <int Block>
__device__ __forceinline__ void stage_trsm_block(
    float* matrix, int panel_begin,
    float* diagonal, float* panel) {
  constexpr int kDiagonalLd = kStageOuter + 1;
  constexpr int kPanelLd = kStageOuter + kStageWidth;
  constexpr int kBlockBegin = Block * 32;
  constexpr int kRegisterCount = 32 / kStageWidth;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < 32 * kStageOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int local_row = linear / kStageOuter;
    const int column = linear % kStageOuter;
    const int matrix_row = kBlockBegin + local_row;
    diagonal[local_row * kDiagonalLd + column] =
        column <= matrix_row
            ? load_global(
                  matrix +
                  (panel_begin + matrix_row) * kN +
                  panel_begin + column)
            : 0.0f;
  }
  __syncthreads();
  const int lane =
      static_cast<int>(threadIdx.x) & (kStageWidth - 1);
  const int row =
      static_cast<int>(threadIdx.x) / kStageWidth;
  if (row < kStageMicro) {
    float values[kRegisterCount];
#pragma unroll
    for (int slot = 0;
         slot < kRegisterCount; ++slot) {
      values[slot] = panel[
          row * kPanelLd + kBlockBegin +
          lane + slot * kStageWidth];
    }
#define STAGE_TRSM_COLUMN(COLUMN)                              \
    stage_trsm_column<Block, COLUMN>(                          \
        values, diagonal, panel, row, lane)
    STAGE_TRSM_COLUMN(0);
    STAGE_TRSM_COLUMN(1);
    STAGE_TRSM_COLUMN(2);
    STAGE_TRSM_COLUMN(3);
    STAGE_TRSM_COLUMN(4);
    STAGE_TRSM_COLUMN(5);
    STAGE_TRSM_COLUMN(6);
    STAGE_TRSM_COLUMN(7);
    STAGE_TRSM_COLUMN(8);
    STAGE_TRSM_COLUMN(9);
    STAGE_TRSM_COLUMN(10);
    STAGE_TRSM_COLUMN(11);
    STAGE_TRSM_COLUMN(12);
    STAGE_TRSM_COLUMN(13);
    STAGE_TRSM_COLUMN(14);
    STAGE_TRSM_COLUMN(15);
    STAGE_TRSM_COLUMN(16);
    STAGE_TRSM_COLUMN(17);
    STAGE_TRSM_COLUMN(18);
    STAGE_TRSM_COLUMN(19);
    STAGE_TRSM_COLUMN(20);
    STAGE_TRSM_COLUMN(21);
    STAGE_TRSM_COLUMN(22);
    STAGE_TRSM_COLUMN(23);
    STAGE_TRSM_COLUMN(24);
    STAGE_TRSM_COLUMN(25);
    STAGE_TRSM_COLUMN(26);
    STAGE_TRSM_COLUMN(27);
    STAGE_TRSM_COLUMN(28);
    STAGE_TRSM_COLUMN(29);
    STAGE_TRSM_COLUMN(30);
    STAGE_TRSM_COLUMN(31);
#undef STAGE_TRSM_COLUMN
#pragma unroll
    for (int slot = 0;
         slot < kRegisterCount; ++slot) {
      panel[
          row * kPanelLd + kBlockBegin +
          lane + slot * kStageWidth] = values[slot];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void stage_trsm_global(
    float* matrix, int row_begin,
    int panel_begin, float* work) {
  constexpr int kDiagonalLd = kStageOuter + 1;
  constexpr int kPanelLd = kStageOuter + kStageWidth;
  float* diagonal = work;
  float* panel = diagonal + 32 * kDiagonalLd;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kStageMicro * kStageOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kStageOuter;
    const int column = linear % kStageOuter;
    panel[row * kPanelLd + column] = load_global(
        matrix + (row_begin + row) * kN +
        panel_begin + column);
  }
  stage_trsm_block<0>(
      matrix, panel_begin, diagonal, panel);
  stage_trsm_block<1>(
      matrix, panel_begin, diagonal, panel);
  stage_trsm_block<2>(
      matrix, panel_begin, diagonal, panel);
  stage_trsm_block<3>(
      matrix, panel_begin, diagonal, panel);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kStageMicro * kStageOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kStageOuter;
    const int column = linear % kStageOuter;
    store_global(
        matrix + (row_begin + row) * kN +
        panel_begin + column,
        panel[row * kPanelLd + column]);
  }
}

__global__ __launch_bounds__(256)
void stage_copy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output) {
  constexpr int kCtasPerMatrix = 4;
  constexpr int kVectors = kN * kN / 4;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / kCtasPerMatrix;
  const int rank =
      static_cast<int>(blockIdx.x) % kCtasPerMatrix;
  const int64_t base =
      static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear =
           rank * static_cast<int>(blockDim.x) +
           static_cast<int>(threadIdx.x);
       linear < kVectors;
       linear +=
           kCtasPerMatrix * static_cast<int>(blockDim.x)) {
    const int64_t offset = base + linear * 4;
    *reinterpret_cast<float4*>(output + offset) =
        *reinterpret_cast<const float4*>(input + offset);
  }
}

__global__ __launch_bounds__(256)
void stage_zero_upper_kernel(
    float* __restrict__ output) {
  constexpr int kCtasPerMatrix = 2;
  constexpr int kVectorsPerRow = kN / 4;
  constexpr int kVectors = kN * kVectorsPerRow;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / kCtasPerMatrix;
  const int rank =
      static_cast<int>(blockIdx.x) % kCtasPerMatrix;
  const int64_t base =
      static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear =
           rank * static_cast<int>(blockDim.x) +
           static_cast<int>(threadIdx.x);
       linear < kVectors;
       linear +=
           kCtasPerMatrix * static_cast<int>(blockDim.x)) {
    const int row = linear / kVectorsPerRow;
    const int column =
        (linear % kVectorsPerRow) * 4;
    float* destination =
        output + base + row * kN + column;
    if (column > row) {
      *reinterpret_cast<float4*>(destination) =
          make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else if (column + 3 > row) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        if (column + item > row) {
          destination[item] = 0.0f;
        }
      }
    }
  }
}

template <int RootMode>
__global__ __launch_bounds__(256)
void stage_factor_kernel(
    float* __restrict__ output, int panel) {
  extern __shared__ __align__(128)
      unsigned char dynamic_bytes[];
  const int matrix_index = static_cast<int>(blockIdx.x);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  stage_factor_global<RootMode>(
      matrix, panel * kStageOuter,
      reinterpret_cast<float*>(dynamic_bytes));
}

__global__ __launch_bounds__(256)
void stage_solve_kernel(
    float* __restrict__ output,
    int panel, int remaining) {
  extern __shared__ __align__(128)
      unsigned char dynamic_bytes[];
  const int matrix_index =
      static_cast<int>(blockIdx.x) / remaining;
  const int row_index =
      static_cast<int>(blockIdx.x) % remaining;
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  stage_trsm_global(
      matrix,
      (panel * 2 + 2 + row_index) * kStageMicro,
      panel * kStageOuter,
      reinterpret_cast<float*>(dynamic_bytes));
}

template <int Width, int RootMode>
__global__ __launch_bounds__(Width == 64 ? 256 : 128)
void adaptive_stage_factor_kernel(
    float* __restrict__ output, int begin) {
  static_assert(Width == 64 || Width == 32);
  extern __shared__ __align__(128)
      unsigned char dynamic_bytes[];
  float* tile = reinterpret_cast<float*>(dynamic_bytes);
  float* inverse_diagonal =
      tile + Width * (kStageOuter + 1);
  const int matrix_index = static_cast<int>(blockIdx.x);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Width * Width;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Width;
    const int column = linear % Width;
    stage_tile_at(tile, row, column) =
        column <= row
            ? load_global(
                  matrix + (begin + row) * kN +
                  begin + column)
            : 0.0f;
  }
  __syncthreads();
  stage_potf2_32<RootMode>(
      tile, inverse_diagonal, 0);
  if constexpr (Width == 64) {
    stage_local_trsm<32, 32>(
        tile, inverse_diagonal, 32, 0);
    stage_local_update<32, 32>(tile, 32, 0);
    stage_potf2_32<RootMode>(
        tile, inverse_diagonal, 32);
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Width * Width;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Width;
    const int column = linear % Width;
    if (column <= row) {
      store_global(
          matrix + (begin + row) * kN + begin + column,
          stage_tile_at(tile, row, column));
    }
  }
}

template <int RowTile, int Width>
__global__ __launch_bounds__(Width == 64 ? 256 : 128)
void adaptive_stage_solve_kernel(
    float* __restrict__ output, int begin, int remaining) {
  static_assert(RowTile == Width);
  static_assert(Width == 64 || Width == 32);
  constexpr int kDiagonalLd = Width + 1;
  constexpr int kPanelLd = Width + kStageWidth;
  extern __shared__ __align__(128)
      unsigned char dynamic_bytes[];
  float* diagonal = reinterpret_cast<float*>(dynamic_bytes);
  float* panel = diagonal + Width * kDiagonalLd;
  float* inverse_diagonal = panel + RowTile * kPanelLd;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / remaining;
  const int row_index =
      static_cast<int>(blockIdx.x) % remaining;
  const int row_begin = begin + Width + row_index * RowTile;
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
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
  const int lane =
      static_cast<int>(threadIdx.x) & (kStageWidth - 1);
  const int row =
      static_cast<int>(threadIdx.x) / kStageWidth;
  if (row < RowTile) {
#pragma unroll 1
    for (int column = 0; column < Width; ++column) {
      float partial = 0.0f;
#pragma unroll 4
      for (int k = lane; k < column; k += kStageWidth) {
        partial = fmaf(
            panel[row * kPanelLd + k],
            diagonal[column * kDiagonalLd + k], partial);
      }
#pragma unroll
      for (int offset = kStageWidth / 2;
           offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(
            0xffffffffu, partial, offset, kStageWidth);
      }
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

__device__ __forceinline__ void decode_adaptive_update_tile(
    int task, int tile_count, int& row_tile, int& column_tile) {
  int cursor = task;
  for (int column = 0; column < tile_count; ++column) {
    const int count = tile_count - column;
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

template <int Rank, int Tile>
__global__ __launch_bounds__(Rank == 64 ? 256 : 128)
void adaptive_lower_update_kernel(
    float* __restrict__ output, int begin,
    int tile_count, int tasks) {
  static_assert(Rank == Tile);
  constexpr int kPanelLd = Rank + 1;
  __shared__ __align__(128) float left[Tile * kPanelLd];
  __shared__ __align__(128) float right[Tile * kPanelLd];
  const int matrix_index =
      static_cast<int>(blockIdx.x) / tasks;
  const int task = static_cast<int>(blockIdx.x) % tasks;
  int row_tile;
  int column_tile;
  decode_adaptive_update_tile(
      task, tile_count, row_tile, column_tile);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  const int trailing_begin = begin + Rank;
  const int row_begin = trailing_begin + row_tile * Tile;
  const int column_begin =
      trailing_begin + column_tile * Tile;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Tile * Rank;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Rank;
    const int column = linear % Rank;
    left[row * kPanelLd + column] = load_global(
        matrix + (row_begin + row) * kN + begin + column);
    right[row * kPanelLd + column] = load_global(
        matrix + (column_begin + row) * kN + begin + column);
  }
  __syncthreads();
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Tile * Tile;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Tile;
    const int column = linear % Tile;
    if (row_tile != column_tile || column <= row) {
      float* destination =
          matrix + (row_begin + row) * kN +
          column_begin + column;
      float value = load_global(destination);
#pragma unroll 4
      for (int k = 0; k < Rank; ++k) {
        value = fmaf(
            -left[row * kPanelLd + k],
            right[column * kPanelLd + k], value);
      }
      store_global(destination, value);
    }
  }
}

template <int Threads>
__device__ __forceinline__ void pack_panel_tf32(
    uint32_t* packed, const float* matrix,
    int row_begin, int panel_begin) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < 64 * 64; linear += Threads) {
    const int row = linear >> 6;
    const int column = linear & 63;
    packed[kmajor_offset(row, column)] = to_tf32(
        load_global(
            matrix + (row_begin + row) * kN +
            panel_begin + column));
  }
}

__device__ __forceinline__ int tensor_update_global(
    const uint32_t* left, const uint32_t* right,
    float* matrix, int row_begin, int column_begin,
    bool diagonal, uint32_t tmem_base,
    uint64_t* barrier, int phase) {
  proxy_fence();
  __syncthreads();

#pragma unroll
  for (int k = 0; k < 64; k += 8) {
    issue_tf32_mma(
        tmem_base,
        make_kmajor_descriptor(left + k * 64),
        make_kmajor_descriptor(right + k * 64),
        k != 0);
  }
  tensor_commit(barrier);
  tensor_wait_warp(barrier, phase);

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int output_row = warp * 16 + lane;
#pragma unroll 1
  for (int column = 0; column < 64; ++column) {
    const uint32_t address =
        tmem_base + static_cast<uint32_t>(warp * 32) *
                        kTmemDp +
        static_cast<uint32_t>(column);
    const float product = tmem_load_one(address);
    if (lane < 16 &&
        (!diagonal || column <= output_row)) {
      float* destination =
          matrix + (row_begin + output_row) * kN +
          column_begin + column;
      store_global(
          destination,
          load_global(destination) - product);
    }
  }
  __syncthreads();
  return phase ^ 1;
}

template <
    int Tile, int Threads, int RootMode, int SolveMode,
    int UpdateMode, int MinimumBlocks, int ScheduleMode,
    int FactorMode>
__global__ __launch_bounds__(Threads, MinimumBlocks)
void fused_potrf_kernel(
    const float* __restrict__ input,
    float* __restrict__ output) {
  constexpr int kLd = Tile + 1;
  constexpr int kTileCount = kN / Tile;
  __shared__ __align__(128) float tile_a[Tile * kLd];
  __shared__ __align__(128) float tile_b[Tile * kLd];
  extern __shared__ __align__(128)
      unsigned char dynamic_shared[];
  float* tile_c =
      reinterpret_cast<float*>(dynamic_shared);
  __shared__ float inverse_diagonal[Tile];
  __shared__ __align__(8) uint64_t tensor_barrier;
  __shared__ uint32_t tmem_base_word;

  const int matrix_index = static_cast<int>(blockIdx.x);
  const int64_t matrix_offset =
      static_cast<int64_t>(matrix_index) * kN * kN;
  const float* input_matrix = input + matrix_offset;
  float* matrix = output + matrix_offset;

  uint32_t tmem_base = 0;
  int tensor_phase = 0;
  if constexpr (UpdateMode == kTensorUpdate) {
    static_assert(Tile == 64);
    static_assert(Threads == 128);
    tensor_barrier_init(&tensor_barrier);
    tmem_allocate(&tmem_base_word, 64);
    tmem_base = tmem_base_word;
    tensor_after_sync_fence();
  }

  if constexpr (ScheduleMode == kRightSchedule) {
    copy_lower<Threads>(input_matrix, matrix);

    for (int panel = 0; panel < kTileCount; ++panel) {
      const int panel_begin = panel * Tile;
      load_diagonal<Tile, Threads>(
          matrix, tile_a, panel_begin);
      __syncthreads();

      if constexpr (FactorMode == kWarp2Factor) {
        static_assert(Tile == 64);
        factor_tile_warp2<RootMode>(
            tile_a, inverse_diagonal);
      } else {
        static_assert(FactorMode == kRecursiveFactor);
        factor_tile<Tile, RootMode>(
            tile_a, inverse_diagonal);
      }
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

      if constexpr (
          UpdateMode == kFp32Update ||
          UpdateMode == kFp32PreloadUpdate) {
        constexpr bool kPreload =
            UpdateMode == kFp32PreloadUpdate;
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
            update_global<Tile, kPreload>(
                tile_a, right, matrix,
                row_begin, column_begin, is_diagonal);
            __syncthreads();
          }
        }
      } else {
        static_assert(UpdateMode == kTensorUpdate);
        uint32_t* left =
            reinterpret_cast<uint32_t*>(tile_a);
        uint32_t* right =
            reinterpret_cast<uint32_t*>(tile_b);
        for (int row_tile = panel + 1;
             row_tile < kTileCount; ++row_tile) {
          const int row_begin = row_tile * Tile;
          pack_panel_tf32<Threads>(
              left, matrix, row_begin, panel_begin);

          for (int column_tile = panel + 1;
               column_tile <= row_tile; ++column_tile) {
            const int column_begin = column_tile * Tile;
            const bool is_diagonal =
                column_tile == row_tile;
            const uint32_t* right_operand = left;
            if (!is_diagonal) {
              pack_panel_tf32<Threads>(
                  right, matrix, column_begin, panel_begin);
              right_operand = right;
            }
            __syncthreads();
            tensor_phase = tensor_update_global(
                left, right_operand, matrix,
                row_begin, column_begin, is_diagonal,
                tmem_base, &tensor_barrier, tensor_phase);
          }
        }
      }
    }
  } else if constexpr (
      ScheduleMode == kProductLeftSchedule ||
      ScheduleMode == kSharedLeftSchedule) {
    static_assert(Tile == 64);
    static_assert(Threads == 256);
    static_assert(UpdateMode == kFp32Update);

    for (int panel = 0; panel < kTileCount; ++panel) {
      const int panel_begin = panel * Tile;
      if constexpr (
          ScheduleMode == kSharedLeftSchedule) {
        left_shared_current_tile<true, Threads>(
            input_matrix, matrix, tile_a, tile_b, tile_c,
            panel_begin, panel_begin, panel);
      } else {
        left_product_current_tile<true, Threads>(
            input_matrix, matrix, tile_a, tile_b, tile_c,
            panel_begin, panel_begin, panel);
      }

      if constexpr (FactorMode == kWarp2Factor) {
        factor_tile_warp2<RootMode>(
            tile_c, inverse_diagonal);
      } else {
        static_assert(FactorMode == kRecursiveFactor);
        factor_tile<Tile, RootMode>(
            tile_c, inverse_diagonal);
      }
      __syncthreads();
      store_factor_tile<Tile, Threads>(
          tile_c, matrix, panel_begin);
      __syncthreads();

      for (int row_tile = panel + 1;
           row_tile < kTileCount; ++row_tile) {
        const int row_begin = row_tile * Tile;
        if constexpr (
            ScheduleMode == kSharedLeftSchedule) {
          left_shared_current_tile<false, Threads>(
              input_matrix, matrix, tile_a, tile_b, tile_c,
              row_begin, panel_begin, panel);
        } else {
          left_product_current_tile<false, Threads>(
              input_matrix, matrix, tile_a, tile_b, tile_c,
              row_begin, panel_begin, panel);
        }
        solve_panel<Tile, SolveMode>(
            tile_c, tile_b, inverse_diagonal);
        __syncthreads();
        store_tile<Tile, Threads>(
            tile_c, matrix, row_begin, panel_begin);
      }
      __syncthreads();
    }
    zero_outer_upper<Tile, Threads>(matrix);
  } else {
    static_assert(ScheduleMode == kLeftSchedule);
    static_assert(Tile == 64);
    static_assert(Threads == 256);
    static_assert(UpdateMode == kFp32Update);

    for (int panel = 0; panel < kTileCount; ++panel) {
      const int panel_begin = panel * Tile;
      left_current_tile<true, Threads>(
          input_matrix, matrix, tile_a, tile_b,
          panel_begin, panel_begin, panel);

      if constexpr (FactorMode == kWarp2Factor) {
        factor_tile_warp2<RootMode>(
            tile_a, inverse_diagonal);
      } else {
        static_assert(FactorMode == kRecursiveFactor);
        factor_tile<Tile, RootMode>(
            tile_a, inverse_diagonal);
      }
      __syncthreads();
      store_factor_tile<Tile, Threads>(
          tile_a, matrix, panel_begin);

      for (int row_tile = panel + 1;
           row_tile < kTileCount; ++row_tile) {
        const int row_begin = row_tile * Tile;
        left_current_tile<false, Threads>(
            input_matrix, matrix, tile_a, tile_b,
            row_begin, panel_begin, panel);
        solve_panel<Tile, SolveMode>(
            tile_a, tile_b, inverse_diagonal);
        __syncthreads();
        store_tile<Tile, Threads>(
            tile_a, matrix, row_begin, panel_begin);
      }
    }
    zero_outer_upper<Tile, Threads>(matrix);
  }

  if constexpr (UpdateMode == kTensorUpdate) {
    tmem_deallocate(tmem_base, 64);
    tmem_relinquish();
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

template <int Id>
constexpr int dynamic_shared_bytes() {
  using V = Variant<Id>;
  if constexpr (
      V::schedule == kProductLeftSchedule ||
      V::schedule == kSharedLeftSchedule) {
    return V::tile * (V::tile + 1) *
        static_cast<int>(sizeof(float));
  }
  return 0;
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

void check_cublas(
    cublasStatus_t status, const char* role) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      role, " failed with cuBLAS status ",
      static_cast<int>(status));
}

class CublasFastState {
 public:
  explicit CublasFastState(cublasHandle_t handle)
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
        "enable default high-performance cuBLAS math");
    check_cublas(
        cublasSetAtomicsMode(
            handle_, CUBLAS_ATOMICS_ALLOWED),
        "enable cuBLAS atomic algorithms");
    check_cublas(
        cublasSetPointerMode(
            handle_, CUBLAS_POINTER_MODE_HOST),
        "select host cuBLAS scalars");
  }

  ~CublasFastState() {
    cublasSetPointerMode(handle_, pointer_mode_);
    cublasSetAtomicsMode(handle_, atomics_mode_);
    cublasSetMathMode(handle_, math_mode_);
  }

  CublasFastState(const CublasFastState&) = delete;
  CublasFastState& operator=(
      const CublasFastState&) = delete;

 private:
  cublasHandle_t handle_;
  cublasMath_t math_mode_{};
  cublasAtomicsMode_t atomics_mode_{};
  cublasPointerMode_t pointer_mode_{};
};

void launch_stage_copy(
    const float* input, float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch * 4, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &config, stage_copy_kernel, input, output);
}

void launch_stage_zero(float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch * 2, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &config, stage_zero_upper_kernel, output);
}

void launch_stage_blas_update(
    cublasHandle_t handle, float* output,
    int panel, bool fast_tf32) {
  const int begin = (panel + 1) * kStageOuter;
  const int remaining = kN - begin;
  const int panel_begin = panel * kStageOuter;
  float* panel_pointer =
      output + begin * kN + panel_begin;
  float* destination =
      output + begin * kN + begin;
  const float alpha = -1.0f;
  const float beta = 1.0f;
  constexpr long long kStride =
      static_cast<long long>(kN) * kN;
  check_cublas(
      cublasGemmStridedBatchedEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          remaining, remaining, kStageOuter,
          &alpha,
          panel_pointer, CUDA_R_32F, kN, kStride,
          panel_pointer, CUDA_R_32F, kN, kStride,
          &beta,
          destination, CUDA_R_32F, kN, kStride,
          kBatch,
          fast_tf32
              ? CUBLAS_COMPUTE_32F_FAST_TF32
              : CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT),
      "staged batched trailing GEMM");
}

template <int Rank>
void launch_adaptive_blas_update(
    cublasHandle_t handle, float* output,
    int begin, bool fast_tf32) {
  const int trailing_begin = begin + Rank;
  const int remaining = kN - trailing_begin;
  float* panel_pointer =
      output + trailing_begin * kN + begin;
  float* destination =
      output + trailing_begin * kN + trailing_begin;
  const float alpha = -1.0f;
  const float beta = 1.0f;
  constexpr long long kStride =
      static_cast<long long>(kN) * kN;
  check_cublas(
      cublasGemmStridedBatchedEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          remaining, remaining, Rank,
          &alpha,
          panel_pointer, CUDA_R_32F, kN, kStride,
          panel_pointer, CUDA_R_32F, kN, kStride,
          &beta,
          destination, CUDA_R_32F, kN, kStride,
          kBatch,
          fast_tf32
              ? CUBLAS_COMPUTE_32F_FAST_TF32
              : CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT),
      "adaptive batched trailing GEMM");
}

template <int Width>
constexpr int adaptive_factor_bytes() {
  return static_cast<int>(sizeof(float)) *
      (Width * (kStageOuter + 1) + Width);
}

template <int Width>
constexpr int adaptive_solve_bytes() {
  return static_cast<int>(sizeof(float)) *
      (Width * (Width + 1) +
       Width * (Width + kStageWidth) + Width);
}

template <int Width>
void launch_adaptive_small_step(
    cublasHandle_t handle, float* output,
    int begin, bool fast_tf32) {
  constexpr int kThreads = Width == 64 ? 256 : 128;
  cudaLaunchConfig_t factor_config{};
  factor_config.gridDim = dim3(kBatch, 1, 1);
  factor_config.blockDim = dim3(kThreads, 1, 1);
  factor_config.dynamicSmemBytes =
      adaptive_factor_bytes<Width>();
  cudaLaunchKernelEx(
      &factor_config,
      adaptive_stage_factor_kernel<Width, kPreciseRoot>,
      output, begin);
  const int remaining = (kN - begin - Width) / Width;
  if (remaining == 0) {
    return;
  }
  cudaLaunchConfig_t solve_config{};
  solve_config.gridDim = dim3(kBatch * remaining, 1, 1);
  solve_config.blockDim = dim3(kThreads, 1, 1);
  solve_config.dynamicSmemBytes =
      adaptive_solve_bytes<Width>();
  cudaLaunchKernelEx(
      &solve_config,
      adaptive_stage_solve_kernel<Width, Width>,
      output, begin, remaining);
  const int trailing = kN - begin - Width;
  if (trailing <= 128) {
    const int tile_count = trailing / Width;
    const int tasks = tile_count * (tile_count + 1) / 2;
    cudaLaunchConfig_t update_config{};
    update_config.gridDim = dim3(kBatch * tasks, 1, 1);
    update_config.blockDim = dim3(kThreads, 1, 1);
    cudaLaunchKernelEx(
        &update_config,
        adaptive_lower_update_kernel<Width, Width>,
        output, begin, tile_count, tasks);
  } else {
    launch_adaptive_blas_update<Width>(
        handle, output, begin, fast_tf32);
  }
}

template <int Id>
void launch_adaptive_staged(
    const float* input, float* output) {
  using V = Variant<Id>;
  static_assert(V::schedule == kAdaptiveStagedSchedule);
  cublasHandle_t handle =
      at::cuda::getCurrentCUDABlasHandle();
  CublasFastState fast_state(handle);
  launch_stage_copy(input, output);
  int begin = 0;
  while (begin < kN) {
    const int remaining = kN - begin;
    if (remaining > 256) {
      const int panel = begin / kStageOuter;
      cudaLaunchConfig_t factor_config{};
      factor_config.gridDim = dim3(kBatch, 1, 1);
      factor_config.blockDim = dim3(256, 1, 1);
      factor_config.dynamicSmemBytes = kStageFactorBytes;
      cudaLaunchKernelEx(
          &factor_config,
          stage_factor_kernel<kPreciseRoot>,
          output, panel);
      const int row_tiles =
          (kN - begin - kStageOuter) / kStageMicro;
      cudaLaunchConfig_t solve_config{};
      solve_config.gridDim =
          dim3(kBatch * row_tiles, 1, 1);
      solve_config.blockDim = dim3(256, 1, 1);
      solve_config.dynamicSmemBytes = kStageSolveBytes;
      cudaLaunchKernelEx(
          &solve_config, stage_solve_kernel,
          output, panel, row_tiles);
      launch_adaptive_blas_update<128>(
          handle, output, begin, true);
      begin += 128;
    } else if (Id == 21 || remaining > 64) {
      launch_adaptive_small_step<64>(
          handle, output, begin, true);
      begin += 64;
    } else {
      launch_adaptive_small_step<32>(
          handle, output, begin, true);
      begin += 32;
    }
  }
  launch_stage_zero(output);
}

template <int Id>
void launch_staged(
    const float* input, float* output) {
  using V = Variant<Id>;
  static_assert(V::schedule == kStagedSchedule);
  static_assert(
      V::update == kBlasFp32Update ||
      V::update == kBlasTf32Update);
  cublasHandle_t handle =
      at::cuda::getCurrentCUDABlasHandle();
  CublasFastState fast_state(handle);
  launch_stage_copy(input, output);
  for (int panel = 0;
       panel < kStagePanelCount; ++panel) {
    cudaLaunchConfig_t factor_config{};
    factor_config.gridDim = dim3(kBatch, 1, 1);
    factor_config.blockDim = dim3(256, 1, 1);
    factor_config.dynamicSmemBytes =
        kStageFactorBytes;
    cudaLaunchKernelEx(
        &factor_config,
        stage_factor_kernel<V::root>,
        output, panel);

    const int remaining =
        kStageMicroCount - panel * 2 - 2;
    if (remaining == 0) {
      continue;
    }
    cudaLaunchConfig_t solve_config{};
    solve_config.gridDim =
        dim3(kBatch * remaining, 1, 1);
    solve_config.blockDim = dim3(256, 1, 1);
    solve_config.dynamicSmemBytes =
        kStageSolveBytes;
    cudaLaunchKernelEx(
        &solve_config, stage_solve_kernel,
        output, panel, remaining);
    launch_stage_blas_update(
        handle, output, panel,
        V::update == kBlasTf32Update);
  }
  launch_stage_zero(output);
}

template <typename Kernel>
cudaFuncAttributes configure_stage_kernel(
    Kernel kernel, int dynamic_bytes,
    int variant, const char* role) {
  prefer_shared(kernel);
  const cudaError_t status = cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      dynamic_bytes);
  TORCH_CHECK(
      status == cudaSuccess,
      "dynamic shared-memory opt-in failed for variant ",
      variant, " ", role, ": ",
      cudaGetErrorString(status));
  const cudaFuncAttributes attributes =
      attributes_for(kernel);
  TORCH_CHECK(
      attributes.localSizeBytes <= 8,
      "variant ", variant, " ", role, " uses ",
      attributes.localSizeBytes,
      " local bytes, above the accepted 8-byte frame");
  return attributes;
}

template <int Id>
void configure_one() {
  using V = Variant<Id>;
  if constexpr (V::schedule == kAdaptiveStagedSchedule) {
    configure_stage_kernel(
        stage_factor_kernel<V::root>,
        kStageFactorBytes, Id, "factor128");
    configure_stage_kernel(
        stage_solve_kernel,
        kStageSolveBytes, Id, "solve128");
    configure_stage_kernel(
        adaptive_stage_factor_kernel<64, V::root>,
        adaptive_factor_bytes<64>(), Id, "factor64");
    configure_stage_kernel(
        adaptive_stage_solve_kernel<64, 64>,
        adaptive_solve_bytes<64>(), Id, "solve64");
    configure_stage_kernel(
        adaptive_stage_factor_kernel<32, V::root>,
        adaptive_factor_bytes<32>(), Id, "factor32");
    configure_stage_kernel(
        adaptive_stage_solve_kernel<32, 32>,
        adaptive_solve_bytes<32>(), Id, "solve32");
    configure_stage_kernel(
        adaptive_lower_update_kernel<64, 64>,
        0, Id, "update64");
    configure_stage_kernel(
        adaptive_lower_update_kernel<32, 32>,
        0, Id, "update32");
  } else if constexpr (V::schedule == kStagedSchedule) {
    configure_stage_kernel(
        stage_factor_kernel<V::root>,
        kStageFactorBytes, Id, "factor");
    configure_stage_kernel(
        stage_solve_kernel,
        kStageSolveBytes, Id, "solve");
  } else {
    auto kernel = fused_potrf_kernel<
        V::tile, V::threads, V::root, V::solve,
        V::update, V::minimum_blocks,
        V::schedule, V::factor>;
    prefer_shared(kernel);
    constexpr int kDynamicSharedBytes =
        dynamic_shared_bytes<Id>();
    if constexpr (kDynamicSharedBytes != 0) {
      const cudaError_t status = cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          kDynamicSharedBytes);
      TORCH_CHECK(
          status == cudaSuccess,
          "dynamic shared-memory opt-in failed for variant ",
          Id, ": ", cudaGetErrorString(status));
    }
    const cudaFuncAttributes attributes =
        attributes_for(kernel);
    TORCH_CHECK(
        attributes.localSizeBytes <= 8,
        "variant ", Id, " kernel uses ",
        attributes.localSizeBytes,
        " local bytes, above the accepted 8-byte frame");
  }
}

template <int Id>
void launch_one(const float* input, float* output) {
  using V = Variant<Id>;
  if constexpr (V::schedule == kAdaptiveStagedSchedule) {
    launch_adaptive_staged<Id>(input, output);
  } else if constexpr (V::schedule == kStagedSchedule) {
    launch_staged<Id>(input, output);
  } else {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kBatch, 1, 1);
    config.blockDim = dim3(V::threads, 1, 1);
    config.dynamicSmemBytes =
        dynamic_shared_bytes<Id>();
    cudaLaunchKernelEx(
        &config,
        fused_potrf_kernel<
            V::tile, V::threads, V::root, V::solve,
            V::update, V::minimum_blocks,
            V::schedule, V::factor>,
        input, output);
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
    case 18: configure_one<18>(); break;
    case 19: configure_one<19>(); break;
    case 20: configure_one<20>(); break;
    case 21: configure_one<21>(); break;
    case 22: configure_one<22>(); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 22]");
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
    case 6: launch_one<6>(input, output); break;
    case 7: launch_one<7>(input, output); break;
    case 8: launch_one<8>(input, output); break;
    case 9: launch_one<9>(input, output); break;
    case 10: launch_one<10>(input, output); break;
    case 11: launch_one<11>(input, output); break;
    case 12: launch_one<12>(input, output); break;
    case 13: launch_one<13>(input, output); break;
    case 14: launch_one<14>(input, output); break;
    case 15: launch_one<15>(input, output); break;
    case 16: launch_one<16>(input, output); break;
    case 17: launch_one<17>(input, output); break;
    case 18: launch_one<18>(input, output); break;
    case 19: launch_one<19>(input, output); break;
    case 20: launch_one<20>(input, output); break;
    case 21: launch_one<21>(input, output); break;
    case 22: launch_one<22>(input, output); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 22]");
  }
}

template <int Id>
void write_metadata(int64_t* rows) {
  using V = Variant<Id>;
  cudaFuncAttributes attributes{};
  int shared_bytes = 0;
  int launch_count = 1;
  if constexpr (V::schedule == kAdaptiveStagedSchedule) {
    const cudaFuncAttributes factor = attributes_for(
        stage_factor_kernel<V::root>);
    const cudaFuncAttributes solve = attributes_for(
        stage_solve_kernel);
    attributes = factor.numRegs >= solve.numRegs
        ? factor : solve;
    attributes.localSizeBytes =
        factor.localSizeBytes >= solve.localSizeBytes
            ? factor.localSizeBytes
            : solve.localSizeBytes;
    shared_bytes =
        kStageFactorBytes >= kStageSolveBytes
            ? kStageFactorBytes
            : kStageSolveBytes;
    launch_count = Id == 21 ? 18 : 21;
  } else if constexpr (V::schedule == kStagedSchedule) {
    const cudaFuncAttributes factor = attributes_for(
        stage_factor_kernel<V::root>);
    const cudaFuncAttributes solve = attributes_for(
        stage_solve_kernel);
    attributes = factor.numRegs >= solve.numRegs
        ? factor : solve;
    attributes.localSizeBytes =
        factor.localSizeBytes >= solve.localSizeBytes
            ? factor.localSizeBytes
            : solve.localSizeBytes;
    shared_bytes =
        kStageFactorBytes >= kStageSolveBytes
            ? kStageFactorBytes
            : kStageSolveBytes;
    launch_count = kStageLaunchCount;
  } else {
    attributes = attributes_for(
        fused_potrf_kernel<
            V::tile, V::threads, V::root, V::solve,
            V::update, V::minimum_blocks,
            V::schedule, V::factor>);
    shared_bytes =
        attributes.sharedSizeBytes +
        dynamic_shared_bytes<Id>();
  }
  int64_t* row =
      rows + static_cast<int64_t>(Id) * kMetadataColumns;
  row[0] = Id;
  row[1] = V::tile;
  row[2] = V::threads;
  row[3] = V::root;
  row[4] = V::solve;
  row[5] = attributes.numRegs;
  row[6] = shared_bytes;
  row[7] = attributes.localSizeBytes;
  row[8] = kBatch;
  row[9] = launch_count;
  row[10] = V::update;
  row[11] = V::minimum_blocks;
  row[12] = V::tmem_columns;
  row[13] = V::schedule;
  row[14] = V::factor;
  const int potrf128 =
      V::schedule == kAdaptiveStagedSchedule
          ? 2 : (V::tile == 128 ? kN / 128 : 0);
  const int potrf64 =
      V::schedule == kAdaptiveStagedSchedule
          ? (Id == 21 ? 4 : 3)
          : (V::tile == 64 ? kN / 64 : 0);
  const int potrf32 =
      V::schedule == kAdaptiveStagedSchedule
          ? (Id == 22 ? 2 : 0)
          : (V::tile == 32 ? kN / 32 : 0);
  row[15] = V::tail_policy;
  row[16] = potrf128;
  row[17] = potrf64;
  row[18] = potrf32;
  row[19] = potrf128 > 0 && potrf64 == 0 && potrf32 == 0
      ? potrf128 - 1 : potrf128;
  row[20] = potrf64 > 0 && potrf32 == 0
      ? potrf64 - 1 : potrf64;
  row[21] = potrf32 > 0 ? potrf32 - 1 : 0;
}

}  // namespace

void cholesky_b640n512_prepare(int64_t variant) {
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 22]");
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
      "native variant must be in [0, 22]");
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
  write_metadata<18>(rows);
  write_metadata<19>(rows);
  write_metadata<20>(rows);
  write_metadata<21>(rows);
  write_metadata<22>(rows);
  return result;
}
"""



_CUTLASS_KERNEL_NAMES = (
    "stage_copy_kernel",
    "stage_zero_upper_kernel",
    "stage_factor_kernel",
    "stage_solve_kernel",
    "adaptive_stage_factor_kernel",
    "adaptive_stage_solve_kernel",
    "adaptive_lower_update_kernel",
    "fused_potrf_kernel",
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
    tag = hashlib.sha256(
        (_CPP_SOURCE + cuda_source).encode()
    ).hexdigest()[:12]
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name=f"cholesky_b640n512_fused_cutlass_{tag}",
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


_PREPARED_VARIANTS: set[tuple[str, int]] = set()


def _run_variant(
    data: torch.Tensor,
    variant: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if variant not in _VARIANT_IDS:
        raise ValueError(f"variant must be in {_VARIANT_IDS}, got {variant}")
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
    metadata = _native_module().metadata()
    cutlass = metadata[_CUTLASS_BASE_VARIANT].clone().unsqueeze(0)
    cutlass[0, 0] = _CUTLASS_VARIANT
    return torch.cat((metadata, cutlass), dim=0)


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
