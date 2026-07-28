import hashlib
import os
import re
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The tuner replaces this exact line in retained candidate copies. Variant
# zero remains the tracked default until the B200 promotion gate passes.
_DEFAULT_VARIANT = 0  # POPCORN_VARIANT
_VARIANT_NAMES = (
    "torch_cusolver",
    "ll_nb512_m64_fused_split2_tf32",
    "ll_nb512_m64_inverse_gemm_tf32",
    "ll_nb256_m64_fused_tf32",
    "direct_m128_to_m64_at_r1024_tf32",
    "direct_m128_m64_m32_at_r1024_r256_tf32",
    "direct_fixed_m64_tf32",
    "direct_adaptive_m128_m64_m32_fp32",
    "hybrid_cpu_potrf64_aten",
    "hybrid_cpu_potrf64_compile",
    "hybrid_cpu_potrf128_aten",
)
_VARIANT_COUNT = len(_VARIANT_NAMES)
_VARIANT_IDS = tuple(range(_VARIANT_COUNT))
_NATIVE_VARIANTS = (1, 2, 3, 4, 5, 6, 7, 8, 10)

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

void cholesky_b1n4096_prepare(int64_t variant);
at::Tensor cholesky_b1n4096(const at::Tensor& data, int64_t variant);
void cholesky_b1n4096_out(
    const at::Tensor& data, at::Tensor out, int64_t variant);
at::Tensor cholesky_b1n4096_metadata();
void cholesky_b1n4096_profile(bool enabled);
void cholesky_b1n4096_hybrid_copy(
    const at::Tensor& data, at::Tensor out);
void cholesky_b1n4096_hybrid_stage(
    at::Tensor out, at::Tensor host_panel,
    int64_t begin, int64_t width);
void cholesky_b1n4096_hybrid_finish(
    at::Tensor out, const at::Tensor& host_factor,
    int64_t begin, int64_t width);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b1n4096_prepare,
        "Configure one B200 1x4096 Cholesky variant");
  m.def("run", &cholesky_b1n4096, "Single 4096 Cholesky");
  m.def("run_out", &cholesky_b1n4096_out,
        "Single 4096 Cholesky out");
  m.def("metadata", &cholesky_b1n4096_metadata,
        "B200 kernel resource metadata");
  m.def("profile", &cholesky_b1n4096_profile,
        "Toggle hybrid NVTX ranges");
  m.def("hybrid_copy", &cholesky_b1n4096_hybrid_copy,
        "Initialize the hybrid output");
  m.def("hybrid_stage", &cholesky_b1n4096_hybrid_stage,
        "Run a hybrid diagonal and history phase");
  m.def("hybrid_finish", &cholesky_b1n4096_hybrid_finish,
        "Return a CPU panel and solve it on the GPU");
}
"""

# Shape-specialized descendants of b1n32768 variants 13 and 14.
# Both retain the left-looking TF32 update schedule and the wide
# redundant-corner factor. The 64-wide family shortens the serial
# factor chain for this four-times-smaller dimension; the 128-wide
# family is retained as the exact measured control.
_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/ops/linalg_cholesky_ex.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <utility>

namespace {

constexpr int kN = 4096;
constexpr int kPanelLd = 9;
constexpr int kVariantCount = 11;
constexpr int kMetadataColumns = 33;

constexpr int kLeftLook = 0;
constexpr int kTrsmFusedMicro = 3;
constexpr int kTrsmGemm = 4;
constexpr int kMathTf32 = 1;
constexpr int kMathFp32 = 0;
constexpr int kFactorWide = 3;
constexpr int kApplyWhole = 0;
constexpr int kApplySplit2 = 1;
constexpr int kDirectLook = 1;
constexpr int kHybridLook = 2;

bool gProfileRanges = false;
cudaEvent_t gPanelReady = nullptr;
at::Tensor gHostPanel64;
at::Tensor gHostFactor64;
at::Tensor gHostInfo64;
at::Tensor gHostPanel128;
at::Tensor gHostFactor128;
at::Tensor gHostInfo128;

class ProfileRange {
 public:
  explicit ProfileRange(const char* name) : active_(gProfileRanges) {
    if (active_) {
      nvtxRangePushA(name);
    }
  }
  ~ProfileRange() {
    if (active_) {
      nvtxRangePop();
    }
  }
 private:
  bool active_;
};

template <int Micro>
struct Tile;

template <>
struct Tile<32> {
  static constexpr int micro = 32;
  static constexpr int ld = 33;
  static constexpr int threads = 128;
  static constexpr int half = 16;
  static constexpr int factor_bytes =
      static_cast<int>(sizeof(float)) *
      (2 * micro * ld + micro + micro * kPanelLd + half * half);
  static constexpr int apply_bytes =
      static_cast<int>(sizeof(float)) * 2 * micro * ld;
};

template <>
struct Tile<64> {
  static constexpr int micro = 64;
  static constexpr int ld = 65;
  static constexpr int threads = 256;
  static constexpr int half = 32;
  static constexpr int factor_bytes =
      static_cast<int>(sizeof(float)) *
      (2 * micro * ld + micro + micro * kPanelLd + half * half);
  static constexpr int apply_bytes =
      static_cast<int>(sizeof(float)) * 2 * micro * ld;
};

template <>
struct Tile<128> {
  static constexpr int micro = 128;
  static constexpr int ld = 129;
  static constexpr int threads = 512;
  static constexpr int half = 64;
  static constexpr int factor_bytes =
      static_cast<int>(sizeof(float)) *
      (2 * micro * ld + micro + micro * kPanelLd + half * half);
  static constexpr int apply_bytes =
      static_cast<int>(sizeof(float)) * 2 * micro * ld;
};

static_assert(Tile<32>::factor_bytes == 10752);
static_assert(Tile<32>::apply_bytes == 8448);
static_assert(Tile<64>::factor_bytes == 39936);
static_assert(Tile<64>::apply_bytes == 33280);
static_assert(Tile<128>::factor_bytes == 153600);
static_assert(Tile<128>::apply_bytes == 132096);

template <int Id>
struct Variant;

#define SPEC(ID, NB, MICRO, TRSM, SPLIT)                             \
  template <> struct Variant<ID> {                                   \
    static constexpr int nb = NB;                                    \
    static constexpr int micro = MICRO;                              \
    static constexpr int trsm_mode = TRSM;                           \
    static constexpr int factor_mode = kFactorWide;                  \
    static constexpr int consumer_split = SPLIT;                     \
    static constexpr int min_micro = MICRO;                          \
    static constexpr int tail_policy = 0;                            \
    static constexpr int tail_cutoff = 0;                            \
    static constexpr bool adaptive = false;                          \
    static constexpr bool fused = TRSM == kTrsmFusedMicro;           \
    static constexpr bool gemm = TRSM == kTrsmGemm;                  \
    static constexpr bool compact = false;                           \
  }

SPEC(0, 64, 64, kTrsmFusedMicro, 1);
SPEC(1, 512, 64, kTrsmFusedMicro, 2);
SPEC(2, 512, 64, kTrsmGemm, 1);
SPEC(3, 256, 64, kTrsmFusedMicro, 1);
SPEC(4, 128, 128, kTrsmFusedMicro, 1);
SPEC(5, 128, 128, kTrsmFusedMicro, 1);
SPEC(6, 64, 64, kTrsmFusedMicro, 1);
SPEC(7, 128, 128, kTrsmFusedMicro, 1);
SPEC(8, 64, 64, kTrsmGemm, 1);
SPEC(9, 64, 64, kTrsmGemm, 1);
SPEC(10, 128, 128, kTrsmGemm, 1);

#undef SPEC

template <int... Ids>
constexpr std::array<bool, sizeof...(Ids)> flags_usage_of(
    std::integer_sequence<int, Ids...>) {
  return {Variant<Ids>::fused...};
}

template <int... Ids>
constexpr std::array<bool, sizeof...(Ids)> scratch_usage_of(
    std::integer_sequence<int, Ids...>) {
  return {Variant<Ids>::gemm...};
}

template <int... Ids>
constexpr std::array<int, sizeof...(Ids)> micro_of(
    std::integer_sequence<int, Ids...>) {
  return {Variant<Ids>::micro...};
}

template <int... Ids>
constexpr std::array<int, sizeof...(Ids)> min_micro_of(
    std::integer_sequence<int, Ids...>) {
  return {Variant<Ids>::min_micro...};
}

constexpr auto kVariantUsesFlags =
    flags_usage_of(std::make_integer_sequence<int, kVariantCount>{});
constexpr auto kVariantUsesScratch =
    scratch_usage_of(std::make_integer_sequence<int, kVariantCount>{});
constexpr auto kVariantMicro =
    micro_of(std::make_integer_sequence<int, kVariantCount>{});
constexpr auto kVariantMinMicro =
    min_micro_of(std::make_integer_sequence<int, kVariantCount>{});

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

template <int Micro>
__device__ __forceinline__ float& tile_at(
    float* tile, int row, int column) {
  return tile[row * Tile<Micro>::ld + column];
}

template <int Micro>
__device__ __forceinline__ const float& tile_at(
    const float* tile, int row, int column) {
  return tile[row * Tile<Micro>::ld + column];
}

// Eight-column redundant-corner factorization. Four threads share
// each row solve and split its rank-8 trailing update. The corner and
// inverse use the later inverse-build scratch to avoid per-thread copies.
template <int Micro>
__device__ __forceinline__ void factor_wide(
    float* __restrict__ tile,
    float* __restrict__ inverse_diagonal,
    float* __restrict__ panel,
    float* __restrict__ corner_scratch) {
  constexpr int kGroup = 8;
  const int thread = static_cast<int>(threadIdx.x);
  const int row_index = thread >> 2;
  const int quarter = thread & 3;
  float* inverse_scratch = corner_scratch + kGroup * kGroup;
#pragma unroll 1
  for (int base = 0; base < Micro; base += kGroup) {
    if (thread < kGroup) {
#pragma unroll
      for (int i = thread; i < kGroup; ++i) {
        corner_scratch[i * kGroup + thread] =
            tile_at<Micro>(tile, base + i, base + thread);
      }
    }
    __syncthreads();
    if (thread == 0) {
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        const float diagonal =
            __fsqrt_rn(corner_scratch[j * kGroup + j]);
        const float inv = __fdiv_rn(1.0f, diagonal);
        corner_scratch[j * kGroup + j] = diagonal;
        inverse_scratch[j] = inv;
#pragma unroll
        for (int i = j + 1; i < kGroup; ++i) {
          corner_scratch[i * kGroup + j] *= inv;
        }
#pragma unroll
        for (int i = j + 1; i < kGroup; ++i) {
#pragma unroll
          for (int target = j + 1; target <= i; ++target) {
            corner_scratch[i * kGroup + target] = fmaf(
                -corner_scratch[i * kGroup + j],
                corner_scratch[target * kGroup + j],
                corner_scratch[i * kGroup + target]);
          }
        }
      }
    }
    __syncthreads();
    if (thread < kGroup) {
      inverse_diagonal[base + thread] = inverse_scratch[thread];
#pragma unroll
      for (int i = thread; i < kGroup; ++i) {
        tile_at<Micro>(tile, base + i, base + thread) =
            corner_scratch[i * kGroup + thread];
      }
    }
    const int row = base + kGroup + row_index;
    float solved[kGroup];
    if (row < Micro) {
#pragma unroll
      for (int k = 0; k < kGroup; ++k) {
        solved[k] = tile_at<Micro>(tile, row, base + k);
      }
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        float value = solved[j];
#pragma unroll
        for (int i = 0; i < j; ++i) {
          value = fmaf(
              -solved[i], corner_scratch[j * kGroup + i], value);
        }
        solved[j] = value * inverse_scratch[j];
      }
      if (quarter == 0) {
#pragma unroll
        for (int k = 0; k < kGroup; ++k) {
          tile_at<Micro>(tile, row, base + k) = solved[k];
          panel[row * kPanelLd + k] = solved[k];
        }
      }
    }
    __syncthreads();
    if (row < Micro) {
      const int first = base + kGroup;
      for (int target = first + quarter * 4; target <= row;
           target += 16) {
        if (target + 3 <= row) {
          float value0 = tile_at<Micro>(tile, row, target);
          float value1 = tile_at<Micro>(tile, row, target + 1);
          float value2 = tile_at<Micro>(tile, row, target + 2);
          float value3 = tile_at<Micro>(tile, row, target + 3);
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
          tile_at<Micro>(tile, row, target) = value0;
          tile_at<Micro>(tile, row, target + 1) = value1;
          tile_at<Micro>(tile, row, target + 2) = value2;
          tile_at<Micro>(tile, row, target + 3) = value3;
        } else {
          for (int single = target; single <= row; ++single) {
            float value = tile_at<Micro>(tile, row, single);
#pragma unroll
            for (int k = 0; k < kGroup; ++k) {
              value = fmaf(
                  -solved[k], panel[single * kPanelLd + k], value);
            }
            tile_at<Micro>(tile, row, single) = value;
          }
        }
      }
    }
    __syncthreads();
  }
}

// The profiled 64-wide producer made every thread repeat the serial
// 8x8 corner factor and repeated each row solve four times. This
// candidate assigns the corner to one warp, assigns one row solve to
// one thread, then returns all 256 threads to the rank-8 update.
template <int Micro>
__device__ __forceinline__ void factor_compact(
    float* __restrict__ tile,
    float* __restrict__ inverse_diagonal,
    float* __restrict__ panel) {
  static_assert(Micro == 64);
  constexpr int kGroup = 8;
  constexpr unsigned kFullMask = 0xffffffffu;
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
#pragma unroll 1
  for (int base = 0; base < Micro; base += kGroup) {
    if (thread < 32) {
#pragma unroll
      for (int column = 0; column < kGroup; ++column) {
        if (lane == 0) {
          const float diagonal = __fsqrt_rn(
              tile_at<Micro>(
                  tile, base + column, base + column));
          tile_at<Micro>(
              tile, base + column, base + column) = diagonal;
          inverse_diagonal[base + column] =
              __fdiv_rn(1.0f, diagonal);
        }
        __syncwarp(kFullMask);
        const int row = column + 1 + lane;
        if (row < kGroup) {
          tile_at<Micro>(tile, base + row, base + column) *=
              inverse_diagonal[base + column];
        }
        __syncwarp(kFullMask);
        if (row < kGroup) {
          const float left =
              tile_at<Micro>(tile, base + row, base + column);
#pragma unroll
          for (int target = column + 1; target <= row; ++target) {
            tile_at<Micro>(tile, base + row, base + target) =
                fmaf(
                    -left,
                    tile_at<Micro>(
                        tile, base + target, base + column),
                    tile_at<Micro>(
                        tile, base + row, base + target));
          }
        }
        __syncwarp(kFullMask);
      }
    }
    __syncthreads();

    const int solve_row = base + kGroup + thread;
    if (solve_row < Micro) {
      float solved[kGroup];
#pragma unroll
      for (int column = 0; column < kGroup; ++column) {
        float value =
            tile_at<Micro>(tile, solve_row, base + column);
#pragma unroll
        for (int prior = 0; prior < column; ++prior) {
          value = fmaf(
              -solved[prior],
              tile_at<Micro>(
                  tile, base + column, base + prior),
              value);
        }
        solved[column] =
            value * inverse_diagonal[base + column];
      }
#pragma unroll
      for (int column = 0; column < kGroup; ++column) {
        tile_at<Micro>(tile, solve_row, base + column) =
            solved[column];
        panel[solve_row * kPanelLd + column] = solved[column];
      }
    }
    __syncthreads();

    const int update_row = base + kGroup + (thread >> 2);
    const int quarter = thread & 3;
    if (update_row < Micro) {
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
          float value0 =
              tile_at<Micro>(tile, update_row, target);
          float value1 =
              tile_at<Micro>(tile, update_row, target + 1);
          float value2 =
              tile_at<Micro>(tile, update_row, target + 2);
          float value3 =
              tile_at<Micro>(tile, update_row, target + 3);
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
          tile_at<Micro>(tile, update_row, target) = value0;
          tile_at<Micro>(tile, update_row, target + 1) = value1;
          tile_at<Micro>(tile, update_row, target + 2) = value2;
          tile_at<Micro>(tile, update_row, target + 3) = value3;
        } else {
          for (int single = target;
               single <= update_row; ++single) {
            float value =
                tile_at<Micro>(tile, update_row, single);
#pragma unroll
            for (int column = 0; column < kGroup; ++column) {
              value = fmaf(
                  -solved[column],
                  panel[single * kPanelLd + column],
                  value);
            }
            tile_at<Micro>(tile, update_row, single) = value;
          }
        }
      }
    }
    __syncthreads();
  }
}

// Invert 32-wide diagonal blocks, combine adjacent pairs into 64,
// and, for the 128-wide control, combine the two halves once more.
template <int Micro>
__device__ __forceinline__ void build_inverse(
    const float* tile, const float* inverse_diagonal,
    float* tinv, float* mid) {
  constexpr int kLd = Tile<Micro>::ld;
  const int thread = static_cast<int>(threadIdx.x);
  for (int linear = thread; linear < Micro * kLd;
       linear += static_cast<int>(blockDim.x)) {
    tinv[linear] = 0.0f;
  }
  __syncthreads();
  const int warp = thread >> 5;
  const int lane = thread & 31;
  if (warp < Micro / 32) {
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
#pragma unroll
  for (int pair = 0; pair < Micro / 64; ++pair) {
    const int base = pair * 64;
    for (int linear = thread; linear < 32 * 32;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear >> 5;
      const int column = linear & 31;
      float partial = 0.0f;
#pragma unroll 4
      for (int k = column; k < 32; ++k) {
        partial = fmaf(
            tile[(base + 32 + row) * kLd + base + k],
            tinv[(base + k) * kLd + base + column], partial);
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
            tinv[(base + 32 + row) * kLd + base + 32 + k],
            mid[k * 32 + column], partial);
      }
      tinv[(base + 32 + row) * kLd + base + column] = -partial;
    }
    __syncthreads();
  }
  if constexpr (Micro == 128) {
    for (int linear = thread; linear < 64 * 64;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear >> 6;
      const int column = linear & 63;
      float partial = 0.0f;
#pragma unroll 4
      for (int k = column; k < 64; ++k) {
        partial = fmaf(
            tile[(64 + row) * kLd + k],
            tinv[k * kLd + column], partial);
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
            tinv[(64 + row) * kLd + 64 + k],
            mid[k * 64 + column], partial);
      }
      tinv[(64 + row) * kLd + column] = -partial;
    }
    __syncthreads();
  }
}

template <int Micro>
__global__ __launch_bounds__(Tile<Micro>::threads)
void factor_kernel(
    float* __restrict__ output, int begin,
    float* __restrict__ t_inv) {
  constexpr int kLd = Tile<Micro>::ld;
  extern __shared__ __align__(16) float dynamic_floats[];
  float* tile = dynamic_floats;
  float* inverse_diagonal = tile + Micro * kLd;
  float* panel = inverse_diagonal + Micro;
  float* tinv = panel + Micro * kPanelLd;
  float* mid = tinv + Micro * kLd;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Micro * Micro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Micro;
    const int column = linear & (Micro - 1);
    tile_at<Micro>(tile, row, column) =
        column <= row
            ? load_global(
                  output + matrix_index(begin + row, begin + column))
            : 0.0f;
  }
  __syncthreads();
  factor_wide<Micro>(tile, inverse_diagonal, panel, mid);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Micro * Micro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Micro;
    const int column = linear & (Micro - 1);
    if (column <= row) {
      store_global(
          output + matrix_index(begin + row, begin + column),
          tile_at<Micro>(tile, row, column));
    }
  }
  if (begin + Micro < kN) {
    build_inverse<Micro>(tile, inverse_diagonal, tinv, mid);
    for (int linear = static_cast<int>(threadIdx.x);
         linear < Micro * Micro;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear / Micro;
      const int column = linear & (Micro - 1);
      store_global(
          t_inv + linear, tinv[row * kLd + column]);
    }
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

template <int Micro>
__device__ __forceinline__ void load_x_tile(
    float* x_tile, const float* output, int begin, int tile_index) {
  constexpr int kLd = Tile<Micro>::ld;
  const int row_begin = begin + Micro + tile_index * Micro;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < Micro * Micro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / Micro;
    const int column = linear & (Micro - 1);
    x_tile[row * kLd + column] = load_global(
        output + matrix_index(row_begin + row, begin + column));
  }
}

// Four warp-column stripes are used at both widths. A 128-wide CTA
// has four warp rows and an 8x4 thread tile; a 64-wide CTA has two
// warp rows and an 8x2 thread tile.
template <int Micro>
__device__ __forceinline__ void apply_tile_wide(
    const float* __restrict__ x_tile,
    const float* __restrict__ t_tile,
    float* __restrict__ output, int begin, int tile_index) {
  constexpr int kLd = Tile<Micro>::ld;
  constexpr int kStripe = Micro / 4;
  constexpr int kThreadColumns = Micro / 32;
  const int row_begin = begin + Micro + tile_index * Micro;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_row = warp >> 2;
  const int warp_column = warp & 3;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;
  const int k_limit = (warp_column + 1) * kStripe;
  float value[8][4];
#pragma unroll
  for (int row = 0; row < 8; ++row) {
#pragma unroll
    for (int column = 0; column < kThreadColumns; ++column) {
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
          (warp_row * 32 + lane_row + row * 4) * kLd + k];
    }
#pragma unroll
    for (int column = 0; column < kThreadColumns; ++column) {
      const int t_row =
          warp_column * kStripe + lane_column + column * 8;
      right[column] = t_tile[t_row * kLd + k];
    }
#pragma unroll
    for (int row = 0; row < 8; ++row) {
#pragma unroll
      for (int column = 0; column < kThreadColumns; ++column) {
        value[row][column] = fmaf(
            left[row], right[column], value[row][column]);
      }
    }
  }
#pragma unroll
  for (int row = 0; row < 8; ++row) {
#pragma unroll
    for (int column = 0; column < kThreadColumns; ++column) {
      const int output_row = warp_row * 32 + lane_row + row * 4;
      const int output_column =
          warp_column * kStripe + lane_column + column * 8;
      store_global(
          output +
              matrix_index(
                  row_begin + output_row, begin + output_column),
          value[row][column]);
    }
  }
}

// Split one 64x64 application across two CTAs. Both halves use all
// eight warps: four 16-row warp bands by two 16-column stripes.
template <int Micro>
__device__ __forceinline__ void apply_tile_half(
    const float* __restrict__ x_tile,
    const float* __restrict__ t_tile,
    float* __restrict__ output, int begin, int tile_index,
    int half) {
  static_assert(Micro == 64);
  constexpr int kLd = Tile<Micro>::ld;
  constexpr int kStripe = 16;
  constexpr int kThreadColumns = 2;
  const int row_begin = begin + Micro + tile_index * Micro;
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
          (warp_row * 16 + lane_row + row * 4) * kLd + k];
    }
#pragma unroll
    for (int column = 0; column < kThreadColumns; ++column) {
      const int t_row =
          warp_column * kStripe + lane_column + column * 8;
      right[column] = t_tile[t_row * kLd + k];
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

template <int Micro, bool Compact, int ConsumerSplit>
__global__ __launch_bounds__(Tile<Micro>::threads)
void fused_micro_kernel(
    float* __restrict__ output, int begin,
    float* __restrict__ t_inv, int* __restrict__ flags) {
  static_assert(ConsumerSplit == 1 || ConsumerSplit == 2);
  static_assert(ConsumerSplit == 1 || Micro == 64);
  constexpr int kLd = Tile<Micro>::ld;
  extern __shared__ __align__(16) float dynamic_floats[];
  const int tiles = (kN - begin - Micro) / Micro;
  int* flag = flags + begin / Micro;
  if (blockIdx.x == 0) {
    float* tile = dynamic_floats;
    float* inverse_diagonal = tile + Micro * kLd;
    float* panel = inverse_diagonal + Micro;
    float* tinv = panel + Micro * kPanelLd;
    float* mid = tinv + Micro * kLd;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < Micro * Micro;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear / Micro;
      const int column = linear & (Micro - 1);
      tile_at<Micro>(tile, row, column) =
          column <= row
              ? load_global(
                    output +
                    matrix_index(begin + row, begin + column))
              : 0.0f;
    }
    __syncthreads();
    if constexpr (Compact) {
      factor_compact<Micro>(tile, inverse_diagonal, panel);
    } else {
      factor_wide<Micro>(tile, inverse_diagonal, panel, mid);
    }
    for (int linear = static_cast<int>(threadIdx.x);
         linear < Micro * Micro;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear / Micro;
      const int column = linear & (Micro - 1);
      if (column <= row) {
        store_global(
            output + matrix_index(begin + row, begin + column),
            tile_at<Micro>(tile, row, column));
      }
    }
    if (tiles > 0) {
      build_inverse<Micro>(tile, inverse_diagonal, tinv, mid);
      for (int linear = static_cast<int>(threadIdx.x);
           linear < Micro * Micro;
           linear += static_cast<int>(blockDim.x)) {
        const int row = linear / Micro;
        const int column = linear & (Micro - 1);
        store_global(t_inv + linear, tinv[row * kLd + column]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        publish_flag(flag);
      }
    }
  } else {
    float* x_tile = dynamic_floats;
    float* t_tile = x_tile + Micro * kLd;
    const int consumer = static_cast<int>(blockIdx.x) - 1;
    const int consumer_count = static_cast<int>(gridDim.x) - 1;
    const int part = consumer % ConsumerSplit;
    const int consumer_stride = consumer_count / ConsumerSplit;
    int tile_index = consumer / ConsumerSplit;
    load_x_tile<Micro>(x_tile, output, begin, tile_index);
    if (threadIdx.x == 0) {
      while (poll_flag(flag) == 0) {
        __nanosleep(64);
      }
      acquire_fence();
    }
    __syncthreads();
    constexpr int kRowsPerConsumer = Micro / ConsumerSplit;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kRowsPerConsumer * Micro;
         linear += static_cast<int>(blockDim.x)) {
      const int local_row = linear / Micro;
      const int row = part * kRowsPerConsumer + local_row;
      const int column = linear & (Micro - 1);
      t_tile[row * kLd + column] =
          load_global(t_inv + row * Micro + column);
    }
    __syncthreads();
    while (true) {
      if constexpr (ConsumerSplit == 1) {
        apply_tile_wide<Micro>(
            x_tile, t_tile, output, begin, tile_index);
      } else {
        apply_tile_half<Micro>(
            x_tile, t_tile, output, begin, tile_index, part);
      }
      tile_index += consumer_stride;
      if (tile_index >= tiles) {
        break;
      }
      __syncthreads();
      load_x_tile<Micro>(x_tile, output, begin, tile_index);
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

template <int Micro>
__global__ __launch_bounds__(256)
void copy_back_kernel(
    float* __restrict__ output,
    const float* __restrict__ scratch, int begin) {
  constexpr int quads_per_row = Micro / 4;
  const int rows = kN - begin - Micro;
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
            scratch + static_cast<int64_t>(row) * Micro + column));
    __stcg(
        reinterpret_cast<float4*>(
            output +
            matrix_index(begin + Micro + row, begin + column)),
        value);
  }
}

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

void gemm_update(
    cublasHandle_t handle, float* output,
    int target_row, int target_column,
    int rows, int columns, int history, bool fast_tf32) {
  if (rows == 0 || columns == 0 || history == 0) {
    return;
  }
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const float* column_panel =
      output + static_cast<int64_t>(target_column) * kN;
  const float* row_panel =
      output + static_cast<int64_t>(target_row) * kN;
  float* destination =
      output + static_cast<int64_t>(target_row) * kN +
      target_column;
  check_cublas(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          columns, rows, history,
          &alpha,
          column_panel, CUDA_R_32F, kN,
          row_panel, CUDA_R_32F, kN,
          &beta,
          destination, CUDA_R_32F, kN,
          fast_tf32
              ? CUBLAS_COMPUTE_32F_FAST_TF32
              : CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT),
      "panel history GEMM");
}

void gemm_history(
    cublasHandle_t handle, float* output,
    int panel_begin, int nb, bool fast_tf32) {
  gemm_update(
      handle, output, panel_begin, panel_begin,
      kN - panel_begin, nb, panel_begin, fast_tf32);
}

template <int Micro>
void gemm_inner(
    cublasHandle_t handle, float* output,
    int64_t panel_begin, int64_t micro_begin, bool fast_tf32) {
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
          Micro, columns, history,
          &alpha,
          micro_rows, CUDA_R_32F, kN,
          micro_rows, CUDA_R_32F, kN,
          &beta,
          destination, CUDA_R_32F, kN,
          fast_tf32
              ? CUBLAS_COMPUTE_32F_FAST_TF32
              : CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT),
      "micro history GEMM");
}

template <int Micro>
void gemm_apply_column(
    cublasHandle_t handle, float* output, const float* t_inv,
    float* scratch, int64_t micro_begin) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int rows = static_cast<int>(kN - micro_begin - Micro);
  const float* x_rows =
      output + (micro_begin + Micro) * kN + micro_begin;
  check_cublas(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          Micro, rows, Micro,
          &alpha,
          t_inv, CUDA_R_32F, Micro,
          x_rows, CUDA_R_32F, kN,
          &beta,
          scratch, CUDA_R_32F, Micro,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "apply GEMM");
}

void launch_copy(const float* input, float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(512, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, copy_lower_kernel, input, output);
}

template <int Micro>
void launch_factor(float* output, int begin, float* t_inv) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(Tile<Micro>::threads, 1, 1);
  config.dynamicSmemBytes = Tile<Micro>::factor_bytes;
  cudaLaunchKernelEx(
      &config, factor_kernel<Micro>, output, begin, t_inv);
}

template <int Micro, bool Compact, int ConsumerSplit>
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
        &active,
        fused_micro_kernel<Micro, Compact, ConsumerSplit>,
        Tile<Micro>::threads, Tile<Micro>::factor_bytes);
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

template <int Micro, bool Compact, int ConsumerSplit>
void launch_fused_micro(
    float* output, int begin, float* t_inv, int* flags) {
  const int tiles = (kN - begin - Micro) / Micro;
  const int limit =
      fused_micro_grid_limit<Micro, Compact, ConsumerSplit>();
  const int jobs = tiles * ConsumerSplit;
  int consumers = jobs < limit - 1 ? jobs : limit - 1;
  consumers -= consumers % ConsumerSplit;
  const int grid = 1 + consumers;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid, 1, 1);
  config.blockDim = dim3(Tile<Micro>::threads, 1, 1);
  config.dynamicSmemBytes = Tile<Micro>::factor_bytes;
  cudaLaunchKernelEx(
      &config,
      fused_micro_kernel<Micro, Compact, ConsumerSplit>,
      output, begin, t_inv, flags);
}

template <int Micro>
void launch_copy_back(
    float* output, const float* scratch, int begin) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(128, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &config, copy_back_kernel<Micro>, output, scratch, begin);
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
    int* flags, float* scratch) {
  using V = Variant<Id>;
  constexpr int Micro = V::micro;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle);
  launch_copy(input, output);
  for (int64_t panel = 0; panel < kN; panel += V::nb) {
    if (panel > 0) {
      gemm_history(handle, output, panel, V::nb, true);
    }
    for (int64_t micro = panel; micro < panel + V::nb;
         micro += Micro) {
      if (micro > panel) {
        gemm_inner<Micro>(handle, output, panel, micro, true);
      }
      if constexpr (V::fused) {
        launch_fused_micro<
            Micro, V::compact, V::consumer_split>(
            output, static_cast<int>(micro), t_inv, flags);
      } else {
        launch_factor<Micro>(
            output, static_cast<int>(micro), t_inv);
        if (micro + Micro < kN) {
          gemm_apply_column<Micro>(
              handle, output, t_inv, scratch, micro);
          launch_copy_back<Micro>(
              output, scratch, static_cast<int>(micro));
        }
      }
    }
  }
  launch_wedges(output, V::nb);
}

template <int Id>
void launch_direct(
    float* output, const float* input, float* t_inv,
    int* flags) {
  static_assert(Id >= 4 && Id <= 7);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle);
  launch_copy(input, output);
  int begin = 0;
  while (begin < kN) {
    const int remaining = kN - begin;
    int width = 64;
    if constexpr (Id != 6) {
      if (remaining > 1024) {
        width = 128;
      } else if constexpr (Id == 5 || Id == 7) {
        if (remaining <= 256) {
          width = 32;
        }
      }
    }
    if (begin > 0) {
      gemm_history(
          handle, output, begin, width, Id != 7);
    }
    if (width == 128) {
      launch_fused_micro<128, false, 1>(
          output, begin, t_inv, flags);
    } else if (width == 64) {
      launch_fused_micro<64, false, 1>(
          output, begin, t_inv, flags);
    } else {
      launch_fused_micro<32, false, 1>(
          output, begin, t_inv, flags);
    }
    begin += width;
  }
  launch_wedges(output, 128);
}

void check_cuda(cudaError_t status, const char* role) {
  TORCH_CHECK(
      status == cudaSuccess, role, " failed: ",
      cudaGetErrorString(status));
}

void ensure_hybrid_state() {
  if (gPanelReady == nullptr) {
    check_cuda(
        cudaEventCreateWithFlags(
            &gPanelReady, cudaEventDisableTiming),
        "hybrid event creation");
  }
  const auto host_float = at::TensorOptions()
      .dtype(at::kFloat)
      .device(at::kCPU)
      .pinned_memory(true);
  const auto host_int = at::TensorOptions()
      .dtype(at::kInt)
      .device(at::kCPU);
  if (!gHostPanel64.defined()) {
    gHostPanel64 = at::empty({1, 64, 64}, host_float);
    gHostFactor64 = at::empty({1, 64, 64}, host_float);
    gHostInfo64 = at::empty({1}, host_int);
  }
  if (!gHostPanel128.defined()) {
    gHostPanel128 = at::empty({1, 128, 128}, host_float);
    gHostFactor128 = at::empty({1, 128, 128}, host_float);
    gHostInfo128 = at::empty({1}, host_int);
  }
}

void check_host_panel(const at::Tensor& panel, int width) {
  TORCH_CHECK(!panel.is_cuda(), "hybrid panel must be on CPU");
  TORCH_CHECK(panel.is_pinned(), "hybrid panel must be pinned");
  TORCH_CHECK(
      panel.scalar_type() == at::kFloat,
      "hybrid panel must be float32");
  TORCH_CHECK(panel.is_contiguous(), "hybrid panel must be contiguous");
  TORCH_CHECK(
      panel.dim() == 3 && panel.size(0) == 1 &&
      panel.size(1) == width && panel.size(2) == width,
      "hybrid panel shape mismatch");
}

void hybrid_stage_impl(
    cublasHandle_t handle, float* output, float* host_panel,
    int begin, int width) {
  if (begin > 0) {
    ProfileRange range("gpu diagonal update");
    gemm_update(
        handle, output, begin, begin,
        width, width, begin, true);
    check_cuda(
        cudaDeviceSynchronize(),
        "hybrid diagonal dependency");
  }
  {
    ProfileRange range("panel transfer device to host");
    check_cuda(
        cudaMemcpy2DAsync(
            host_panel, static_cast<size_t>(width) * sizeof(float),
            output + static_cast<int64_t>(begin) * kN + begin,
            static_cast<size_t>(kN) * sizeof(float),
            static_cast<size_t>(width) * sizeof(float),
            width, cudaMemcpyDeviceToHost, nullptr),
        "hybrid panel download");
    check_cuda(
        cudaEventRecord(gPanelReady, nullptr),
        "hybrid readiness record");
  }
  if (begin + width < kN && begin > 0) {
    ProfileRange range("gpu history update");
    gemm_update(
        handle, output, begin + width, begin,
        kN - begin - width, width, begin, true);
  }
  {
    ProfileRange range("panel wait");
    check_cuda(
        cudaEventSynchronize(gPanelReady),
        "hybrid readiness wait");
  }
}

void hybrid_finish_impl(
    cublasHandle_t handle, float* output,
    const float* host_factor, int begin, int width) {
  {
    ProfileRange range("panel transfer host to device");
    check_cuda(
        cudaMemcpy2DAsync(
            output + static_cast<int64_t>(begin) * kN + begin,
            static_cast<size_t>(kN) * sizeof(float),
            host_factor,
            static_cast<size_t>(width) * sizeof(float),
            static_cast<size_t>(width) * sizeof(float),
            width, cudaMemcpyHostToDevice, nullptr),
        "hybrid panel upload");
    check_cuda(
        cudaEventRecord(gPanelReady, nullptr),
        "hybrid upload record");
    check_cuda(
        cudaEventSynchronize(gPanelReady),
        "hybrid upload wait");
  }
  const int rows = kN - begin - width;
  if (rows > 0) {
    ProfileRange range("gpu panel solve");
    const float alpha = 1.0f;
    check_cublas(
        cublasStrsm(
            handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            width, rows, &alpha,
            output + static_cast<int64_t>(begin) * kN + begin,
            kN,
            output + static_cast<int64_t>(begin + width) * kN +
                begin,
            kN),
        "hybrid triangular solve");
  }
}

void launch_hybrid(
    const float* input, float* output, int width) {
  ensure_hybrid_state();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle);
  at::Tensor& host_panel =
      width == 64 ? gHostPanel64 : gHostPanel128;
  at::Tensor& host_factor =
      width == 64 ? gHostFactor64 : gHostFactor128;
  at::Tensor& host_info =
      width == 64 ? gHostInfo64 : gHostInfo128;
  launch_copy(input, output);
  for (int begin = 0; begin < kN; begin += width) {
    hybrid_stage_impl(
        handle, output, host_panel.data_ptr<float>(),
        begin, width);
    {
      ProfileRange range("cpu potrf");
      at::linalg_cholesky_ex_out(
          host_factor, host_info, host_panel, false, false);
    }
    TORCH_CHECK(
        host_info.data_ptr<int>()[0] == 0,
        "CPU panel factorization failed at ", begin);
    hybrid_finish_impl(
        handle, output, host_factor.data_ptr<float>(),
        begin, width);
  }
}

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be CUDA");
  TORCH_CHECK(
      data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      data.dim() == 3 && data.size(0) == 1 &&
      data.size(1) == kN && data.size(2) == kN,
      "native input must have shape (1, 4096, 4096)");
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
void configure_one() {
  using V = Variant<Id>;
  constexpr int Micro = V::micro;
  if constexpr (V::adaptive) {
    configure_dynamic(
        fused_micro_kernel<64, false, 1>,
        Tile<64>::factor_bytes);
    configure_dynamic(
        fused_micro_kernel<32, false, 1>,
        Tile<32>::factor_bytes);
    checked_attributes(fused_micro_kernel<64, false, 1>);
    checked_attributes(fused_micro_kernel<32, false, 1>);
    TORCH_CHECK(
        (fused_micro_grid_limit<64, false, 1>() >= 2 &&
         fused_micro_grid_limit<32, false, 1>() >= 2),
        "adaptive fused kernels need a consumer CTA");
  } else if constexpr (V::fused) {
    configure_dynamic(
        fused_micro_kernel<
            Micro, V::compact, V::consumer_split>,
        Tile<Micro>::factor_bytes);
    checked_attributes(
        fused_micro_kernel<
            Micro, V::compact, V::consumer_split>);
    TORCH_CHECK(
        (fused_micro_grid_limit<
             Micro, V::compact, V::consumer_split>() >=
         1 + V::consumer_split),
        "fused micro kernel needs a consumer CTA");
  } else {
    configure_dynamic(
        factor_kernel<Micro>, Tile<Micro>::factor_bytes);
    checked_attributes(factor_kernel<Micro>);
  }
}

void configure_direct() {
  configure_dynamic(
      fused_micro_kernel<128, false, 1>,
      Tile<128>::factor_bytes);
  configure_dynamic(
      fused_micro_kernel<64, false, 1>,
      Tile<64>::factor_bytes);
  configure_dynamic(
      fused_micro_kernel<32, false, 1>,
      Tile<32>::factor_bytes);
  checked_attributes(fused_micro_kernel<128, false, 1>);
  checked_attributes(fused_micro_kernel<64, false, 1>);
  checked_attributes(fused_micro_kernel<32, false, 1>);
  TORCH_CHECK(
      (fused_micro_grid_limit<128, false, 1>() >= 2 &&
       fused_micro_grid_limit<64, false, 1>() >= 2 &&
       fused_micro_grid_limit<32, false, 1>() >= 2),
      "direct fused kernels need a consumer CTA");
}

void configure_variant(int variant) {
  switch (variant) {
    case 1: configure_one<1>(); break;
    case 2: configure_one<2>(); break;
    case 3: configure_one<3>(); break;
    case 4:
    case 5:
    case 6:
    case 7:
      configure_direct();
      break;
    case 8:
    case 9:
    case 10:
      break;
    default:
      TORCH_CHECK(false, "native variant must be in [1, 10]");
  }
}

void launch_variant(
    const float* input, float* output, float* t_inv,
    int* flags, float* scratch, int variant) {
  switch (variant) {
    case 1: launch_staged<1>(
        output, input, t_inv, flags, scratch); break;
    case 2: launch_staged<2>(
        output, input, t_inv, flags, scratch); break;
    case 3: launch_staged<3>(
        output, input, t_inv, flags, scratch); break;
    case 4: launch_direct<4>(
        output, input, t_inv, flags); break;
    case 5: launch_direct<5>(
        output, input, t_inv, flags); break;
    case 6: launch_direct<6>(
        output, input, t_inv, flags); break;
    case 7: launch_direct<7>(
        output, input, t_inv, flags); break;
    default:
      TORCH_CHECK(false, "GPU variant must be in [1, 7]");
  }
}

template <int Id>
void write_metadata(int64_t* rows) {
  using V = Variant<Id>;
  constexpr int Micro = V::micro;
  constexpr bool Direct = Id >= 4 && Id <= 7;
  int64_t* row =
      rows + static_cast<int64_t>(Id) * kMetadataColumns;
  if constexpr (Direct) {
    configure_direct();
  } else {
    configure_one<Id>();
  }
  cudaFuncAttributes factor{};
  cudaFuncAttributes apply{};
  int factor_active = 0;
  int apply_active = 0;
  if constexpr (V::fused) {
    factor = checked_attributes(
        fused_micro_kernel<
            Micro, V::compact, V::consumer_split>);
    apply = factor;
    factor_active = active_blocks(
        fused_micro_kernel<
            Micro, V::compact, V::consumer_split>,
        Tile<Micro>::threads,
        Tile<Micro>::factor_bytes);
    apply_active = factor_active;
  } else {
    factor = checked_attributes(factor_kernel<Micro>);
    factor_active = active_blocks(
        factor_kernel<Micro>, Tile<Micro>::threads,
        Tile<Micro>::factor_bytes);
  }
  const cudaFuncAttributes copy =
      checked_attributes(copy_lower_kernel);
  const cudaFuncAttributes wedges =
      checked_attributes(zero_wedges_kernel);
  const int panels = Direct
      ? 1
      : kN / V::nb;
  const int potrf128 = Direct
      ? (Id == 6 ? 0 : 24)
      : (Micro == 128 ? kN / 128 : 0);
  const int potrf64 = Direct
      ? (Id == 6 ? 64 : (Id == 4 ? 16 : 12))
      : (Micro == 64 ? kN / 64 : 0);
  const int potrf32 =
      Direct && (Id == 5 || Id == 7) ? 8 : 0;
  const int micros = potrf128 + potrf64 + potrf32;
  const int inner = micros - panels;
  const int big = panels - 1;
  const int applies = V::gemm ? 2 * (micros - 1) : 0;
  const int flag_fill = V::fused ? 1 : 0;
  row[0] = Id;
  row[1] = Direct ? kDirectLook : kLeftLook;
  row[2] = Direct ? 0 : V::nb;
  row[3] = V::trsm_mode;
  row[4] = Id == 7 ? kMathFp32 : kMathTf32;
  row[5] = Id == 7 ? kMathFp32 : kMathTf32;
  row[6] = Tile<Micro>::threads;
  row[7] = factor.numRegs;
  row[8] = factor.sharedSizeBytes;
  row[9] = factor.localSizeBytes;
  row[10] = Tile<Micro>::factor_bytes;
  row[11] = apply.numRegs;
  row[12] = apply.sharedSizeBytes;
  row[13] = apply.localSizeBytes;
  row[14] = V::fused ? Tile<Micro>::factor_bytes : 0;
  row[15] = copy.numRegs;
  row[16] = wedges.numRegs;
  row[17] = factor_active;
  row[18] = apply_active;
  row[19] =
      2 + big + inner + micros + applies + flag_fill;
  row[20] = 0;
  row[21] = Micro;
  row[22] = 0;
  row[23] = 1;
  row[24] = V::factor_mode;
  row[25] =
      V::consumer_split == 2 ? kApplySplit2 : kApplyWhole;
  row[26] =
      Id == 4 ? 1 : ((Id == 5 || Id == 7) ? 2 : 0);
  row[27] = potrf128;
  row[28] = potrf64;
  row[29] = potrf32;
  row[30] = potrf128 > 0 && potrf64 == 0 && potrf32 == 0
      ? potrf128 - 1 : potrf128;
  row[31] = potrf64 > 0 && potrf32 == 0
      ? potrf64 - 1 : potrf64;
  row[32] = potrf32 > 0 ? potrf32 - 1 : 0;
}

void write_hybrid_metadata(int64_t* rows, int id, int width) {
  int64_t* row =
      rows + static_cast<int64_t>(id) * kMetadataColumns;
  row[0] = id;
  row[1] = kHybridLook;
  row[2] = width;
  row[3] = kTrsmGemm;
  row[4] = kMathTf32;
  row[5] = kMathTf32;
  row[19] = 1 + 5 * (kN / width);
  row[20] = 1;
  row[21] = width;
  row[23] = 1;
  row[24] = 5;
  row[25] = id == 9 ? 3 : 2;
  row[27] = width == 128 ? kN / 128 : 0;
  row[28] = width == 64 ? kN / 64 : 0;
  row[30] = row[27] > 0 ? row[27] - 1 : 0;
  row[31] = row[28] > 0 ? row[28] - 1 : 0;
}

}  // namespace

void cholesky_b1n4096_profile(bool enabled) {
  gProfileRanges = enabled;
}

void cholesky_b1n4096_hybrid_copy(
    const at::Tensor& data, at::Tensor output) {
  check_input(data);
  check_output(data, output);
  c10::cuda::CUDAGuard device_guard(data.device());
  ensure_hybrid_state();
  launch_copy(data.data_ptr<float>(), output.data_ptr<float>());
}

void cholesky_b1n4096_hybrid_stage(
    at::Tensor output, at::Tensor host_panel,
    int64_t begin, int64_t width) {
  check_input(output);
  TORCH_CHECK(
      width == 64, "compiled hybrid phase requires width 64");
  TORCH_CHECK(
      begin >= 0 && begin + width <= kN && begin % width == 0,
      "invalid hybrid panel offset");
  check_host_panel(host_panel, static_cast<int>(width));
  c10::cuda::CUDAGuard device_guard(output.device());
  ensure_hybrid_state();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  hybrid_stage_impl(
      handle, output.data_ptr<float>(), host_panel.data_ptr<float>(),
      static_cast<int>(begin), static_cast<int>(width));
}

void cholesky_b1n4096_hybrid_finish(
    at::Tensor output, const at::Tensor& host_factor,
    int64_t begin, int64_t width) {
  check_input(output);
  TORCH_CHECK(
      width == 64, "compiled hybrid phase requires width 64");
  TORCH_CHECK(
      begin >= 0 && begin + width <= kN && begin % width == 0,
      "invalid hybrid panel offset");
  check_host_panel(host_factor, static_cast<int>(width));
  c10::cuda::CUDAGuard device_guard(output.device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  hybrid_finish_impl(
      handle, output.data_ptr<float>(),
      host_factor.data_ptr<float>(),
      static_cast<int>(begin), static_cast<int>(width));
}

void cholesky_b1n4096_prepare(int64_t variant) {
  TORCH_CHECK(
      variant >= 1 && variant < kVariantCount,
      "native variant must be in [1, 10]");
  if (variant >= 8) {
    ensure_hybrid_state();
  }
  configure_variant(static_cast<int>(variant));
}

void cholesky_b1n4096_out(
    const at::Tensor& data, at::Tensor output, int64_t variant) {
  check_input(data);
  check_output(data, output);
  TORCH_CHECK(
      variant >= 1 && variant < kVariantCount,
      "native variant must be in [1, 10]");
  c10::cuda::CUDAGuard device_guard(data.device());
  const int selected = static_cast<int>(variant);
  if (selected == 8 || selected == 10) {
    launch_hybrid(
        data.data_ptr<float>(), output.data_ptr<float>(),
        selected == 8 ? 64 : 128);
    return;
  }
  TORCH_CHECK(
      selected != 9,
      "variant 9 is driven by the compiled CPU phase helpers");
  const int micro =
      selected >= 4 ? 128 : kVariantMicro[selected];
  const int min_micro =
      selected >= 4 ? 32 : kVariantMinMicro[selected];
  at::Tensor t_inv = at::empty(
      {static_cast<int64_t>(micro) * micro}, data.options());
  at::Tensor flags;
  int* flags_pointer = nullptr;
  if (kVariantUsesFlags[selected]) {
    flags = at::zeros(
        {kN / min_micro}, data.options().dtype(at::kInt));
    flags_pointer = flags.data_ptr<int>();
  }
  at::Tensor scratch;
  float* scratch_pointer = nullptr;
  if (kVariantUsesScratch[selected]) {
    scratch = at::empty(
        {static_cast<int64_t>(kN - micro) * micro},
        data.options());
    scratch_pointer = scratch.data_ptr<float>();
  }
  launch_variant(
      data.data_ptr<float>(), output.data_ptr<float>(),
      t_inv.data_ptr<float>(), flags_pointer, scratch_pointer,
      selected);
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b1n4096(
    const at::Tensor& data, int64_t variant) {
  auto output = at::empty_like(data);
  cholesky_b1n4096_out(data, output, variant);
  return output;
}

at::Tensor cholesky_b1n4096_metadata() {
  auto result = at::zeros(
      {kVariantCount, kMetadataColumns},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  write_metadata<1>(rows);
  write_metadata<2>(rows);
  write_metadata<3>(rows);
  write_metadata<4>(rows);
  write_metadata<5>(rows);
  write_metadata<6>(rows);
  write_metadata<7>(rows);
  write_hybrid_metadata(rows, 8, 64);
  write_hybrid_metadata(rows, 9, 64);
  write_hybrid_metadata(rows, 10, 128);
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
            name=f"cholesky_b1n4096_b200_{tag}",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
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
            extra_ldflags=["-lcublas"],
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch

_PREPARED_VARIANTS: set[int] = set()


@lru_cache(maxsize=1)
def _compiled_cpu_potrf():
    import torch._inductor.config as inductor_config

    def cpu_potrf(panel: torch.Tensor):
        return torch.linalg.cholesky_ex(panel, check_errors=False)

    with inductor_config.patch(
        {
            "cpp.enable_unsafe_math_opt_flag": True,
            "cpp.enable_floating_point_contract_flag": "fast",
        }
    ):
        compiled = torch.compile(
            cpu_potrf,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            mode="max-autotune",
        )
        warm = torch.eye(64, dtype=torch.float32).unsqueeze(0)
        compiled(warm)
    return compiled


@lru_cache(maxsize=1)
def _compiled_hybrid_buffers() -> tuple[torch.Tensor, torch.Tensor]:
    panel = torch.empty(
        (1, 64, 64), dtype=torch.float32, pin_memory=True
    )
    factor = torch.empty_like(panel, pin_memory=True)
    return panel, factor


def _run_compiled_hybrid(
    data: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    module = _native_module()
    if 9 not in _PREPARED_VARIANTS:
        module.prepare(9)
        _compiled_cpu_potrf()
        _compiled_hybrid_buffers()
        _PREPARED_VARIANTS.add(9)
    profiling = os.environ.get("CHOLESKY_PROFILE_NVTX") == "1"
    module.profile(profiling)
    output = torch.empty_like(data) if out is None else out
    panel, pinned_factor = _compiled_hybrid_buffers()
    compiled = _compiled_cpu_potrf()
    module.hybrid_copy(data, output)
    for begin in range(0, 4096, 64):
        module.hybrid_stage(output, panel, begin, 64)
        if profiling:
            torch.cuda.nvtx.range_push("cpu potrf compiled")
        factor, info = compiled(panel)
        pinned_factor.copy_(factor)
        if profiling:
            torch.cuda.nvtx.range_pop()
        if int(info[0]) != 0:
            raise RuntimeError(f"CPU panel factorization failed at {begin}")
        module.hybrid_finish(output, pinned_factor, begin, 64)
    return output


def _run_variant(
    data: torch.Tensor,
    variant: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if variant not in _VARIANT_IDS:
        raise ValueError(f"variant must be in {_VARIANT_IDS}, got {variant}")
    if variant == 0:
        factor = torch.linalg.cholesky_ex(data, check_errors=False).L
        if out is not None:
            out.copy_(factor)
            return out
        return factor
    if variant == 9:
        return _run_compiled_hybrid(data, out)
    module = _native_module()
    if variant not in _PREPARED_VARIANTS:
        module.prepare(variant)
        _PREPARED_VARIANTS.add(variant)
    module.profile(os.environ.get("CHOLESKY_PROFILE_NVTX") == "1")
    if out is None:
        return module.run(data, variant)
    module.run_out(data, out, variant)
    return out


def _variant_metadata() -> torch.Tensor:
    return _native_module().metadata()


def custom_kernel(data: input_t) -> output_t:
    if (
        _DEFAULT_VARIANT != 0
        and data.is_cuda
        and data.dtype == torch.float32
        and data.is_contiguous()
        and tuple(data.shape) == (1, 4096, 4096)
    ):
        return _run_variant(data, _DEFAULT_VARIANT)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
