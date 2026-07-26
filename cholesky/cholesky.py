"""Batched dense Cholesky factorization for the NVIDIA B200 (sm_100a).

Each benchmark shape that has a tuned specialization routes to its own custom
CUDA extension; every other shape falls back to torch.

Implemented shapes and the production variant folded in from each per-shape
submission:

  (4096, 32, 32)   b4096n32  variant 32  subwarp_left_32
                             left-looking, four matrices per warp in width-8
                             warp subdivisions, 2 warps/CTA.
  (1024, 64, 64)   b1024n64  variant 13  w4_int_f2_raw_right_rootlook
                             right-looking rank-1 with the interleaved
                             two-rows-per-lane mapping and next-diagonal root
                             lookahead, 4 warps/CTA.
  (256, 128, 128)  b256n128  variant 23  simt_balanced_v13_raw_overlap
                             blocked 64/64 in shared memory, right-looking
                             64x64 factors, warp-balanced SIMT SYRK,
                             overlapped output epilogue.
  (64, 256, 256)   b64n256   variant 18  cta512_rec32_scalar_tc_all_refined_pad129
                             one 512-thread CTA per matrix, recursive-32 base
                             factors, scalar TRSM, tcgen05 TF32 trailing
                             updates, kLd-padded A10 block.
  (16, 512, 512)   b16n512   variant 2   r16_micro4x4_raw_scalar_u256
                             staged 64x64 panel factorization across separate
                             factor/solve/update launches with a 4x4
                             micro-tiled FP32 update.
  (640, 512, 512)  b640n512  variant 8   p64_raw_scalar_preload_m4x4_t256_occ5
                             one 256-thread CTA per matrix running the whole
                             right-looking 64-panel factorization, with a 4x4
                             micro-tiled FP32 update that preloads its
                             destination, at five blocks per SM.

Every other benchmark and test shape uses
torch.linalg.cholesky_ex(..., check_errors=False).L.
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


def _build(name, cpp_source, cuda_source, extra_cuda_flags=()):
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
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


# ---------------------------------------------------------------------------
# (4096, 32, 32) - b4096n32 variant 32
# ---------------------------------------------------------------------------

_CPP_SOURCE_B4096N32 = r"""
#include <torch/extension.h>

at::Tensor cholesky_b4096n32(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &cholesky_b4096n32, "Batched 32x32 Cholesky");
}
"""

_CUDA_SOURCE_B4096N32 = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 4096;
constexpr int kN = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// Pack independent matrices into power-of-two warp subdivisions. A width-8
// shuffle performs four unrelated matrix broadcasts in one hardware
// instruction, so each thread owns four rows of its subdivision's matrix.
constexpr int kWarps = 2;
constexpr int kGroupWidth = 8;
constexpr int kMinBlocks = 4;
constexpr int kMatricesPerWarp = 32 / kGroupWidth;
constexpr int kRowsPerLane = kN / kGroupWidth;

__global__ __launch_bounds__(kWarps * 32, kMinBlocks)
void subwarp_left_32_kernel(const float* __restrict__ input,
                            float* __restrict__ output) {
  const int physical_lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int group = physical_lane / kGroupWidth;
  const int lane = physical_lane & (kGroupWidth - 1);
  const int physical_warp = static_cast<int>(blockIdx.x) * kWarps + warp;
  const int matrix = physical_warp * kMatricesPerWarp + group;
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  float row[kRowsPerLane][kN];
#pragma unroll
  for (int owned = 0; owned < kRowsPerLane; ++owned) {
#pragma unroll
    for (int k = 0; k < kN; ++k) {
      row[owned][k] = 0.0f;
    }
  }

#pragma unroll
  for (int j = 0; j < kN; ++j) {
    // Reading A[j, i] instead of A[i, j] is valid for the symmetric input and
    // makes every warp input transaction contiguous.
    float value[kRowsPerLane];
#pragma unroll
    for (int owned = 0; owned < kRowsPerLane; ++owned) {
      const int matrix_row = lane + owned * kGroupWidth;
      value[owned] = a[j * kN + matrix_row];
    }

#pragma unroll
    for (int k = 0; k < j; ++k) {
      const int pivot_owner = j / kGroupWidth;
      const int pivot_lane = j & (kGroupWidth - 1);
      const float pivot = __shfl_sync(
          kFullMask, row[pivot_owner][k], pivot_lane, kGroupWidth);
#pragma unroll
      for (int owned = 0; owned < kRowsPerLane; ++owned) {
        value[owned] = fmaf(-row[owned][k], pivot, value[owned]);
      }
    }

    const int diagonal_owner = j / kGroupWidth;
    const int diagonal_lane = j & (kGroupWidth - 1);
    float inverse = 0.0f;
    if (lane == diagonal_lane) {
      const float diagonal_value = value[diagonal_owner];
      inverse = rsqrtf(diagonal_value);
      row[diagonal_owner][j] = diagonal_value * inverse;
    }
    inverse = __shfl_sync(kFullMask, inverse, diagonal_lane, kGroupWidth);
#pragma unroll
    for (int owned = 0; owned < kRowsPerLane; ++owned) {
      const int matrix_row = lane + owned * kGroupWidth;
      if (matrix_row > j) {
        row[owned][j] = value[owned] * inverse;
      }
    }
  }

  // Each output row is 128-byte aligned. Eight explicit float4 writes retain
  // alignment while zeroing the unused upper triangle in the same kernel.
#pragma unroll
  for (int owned = 0; owned < kRowsPerLane; ++owned) {
    const int matrix_row = lane + owned * kGroupWidth;
    float* out_row = result + matrix_row * kN;
#pragma unroll
    for (int vector_index = 0; vector_index < 8; ++vector_index) {
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

}  // namespace

at::Tensor cholesky_b4096n32(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (4096, 32, 32)");
  auto out = at::empty_like(data);
  constexpr int threads = kWarps * 32;
  constexpr int blocks = kBatch / (kWarps * kMatricesPerWarp);
  static_assert(kBatch % (kWarps * kMatricesPerWarp) == 0);
  subwarp_left_32_kernel<<<blocks, threads>>>(
      data.data_ptr<float>(), out.data_ptr<float>());
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return out;
}
"""


@lru_cache(maxsize=1)
def _module_b4096n32():
    return _build(
        "cholesky_b4096n32", _CPP_SOURCE_B4096N32, _CUDA_SOURCE_B4096N32)


# ---------------------------------------------------------------------------
# (1024, 64, 64) - b1024n64 variant 13
# ---------------------------------------------------------------------------

_CPP_SOURCE_B1024N64 = r"""
#include <torch/extension.h>

at::Tensor cholesky_b1024n64(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &cholesky_b1024n64, "Batched 64x64 Cholesky");
}
"""

_CUDA_SOURCE_B1024N64 = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 1024;
constexpr int kN = 64;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kWarps = 4;
constexpr int kMinBlocks = 2;

// Right-looking rank-1 Cholesky with the interleaved two-rows-per-lane
// mapping: lane owns rows 2*lane and 2*lane + 1, so the symmetric column read
// collapses into one coalesced float2 vector load. state[owned][j] begins as
// A[row, j], is updated once per preceding factor column, and becomes
// L[row, j] when column j is normalized. Every shuffle feeds updates into
// independent trailing-column registers instead of one serial dot chain.
__global__ __launch_bounds__(kWarps * 32, kMinBlocks)
void right_looking_64_kernel(const float* __restrict__ input,
                             float* __restrict__ output) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int matrix = static_cast<int>(blockIdx.x) * kWarps + warp;
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

  // Normalize column zero. Each later iteration updates the next column
  // first, issues its reciprocal square root, and fills the root latency with
  // the remaining independent rank-1 updates.
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

  // Each output row is 256 bytes; sixteen aligned float4 writes per row zero
  // the unused upper triangle in the same kernel.
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

}  // namespace

at::Tensor cholesky_b1024n64(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (1024, 64, 64)");
  auto out = at::empty_like(data);
  constexpr int threads = kWarps * 32;
  constexpr int blocks = kBatch / kWarps;
  static_assert(kBatch % kWarps == 0);
  right_looking_64_kernel<<<blocks, threads>>>(
      data.data_ptr<float>(), out.data_ptr<float>());
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return out;
}
"""


@lru_cache(maxsize=1)
def _module_b1024n64():
    return _build(
        "cholesky_b1024n64", _CPP_SOURCE_B1024N64, _CUDA_SOURCE_B1024N64)


# ---------------------------------------------------------------------------
# (256, 128, 128) - b256n128 variant 23
# ---------------------------------------------------------------------------

_CPP_SOURCE_B256N128 = r"""
#include <torch/extension.h>

void cholesky_b256n128_prepare();
at::Tensor cholesky_b256n128(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b256n128_prepare,
        "Configure batched 128x128 Cholesky dynamic shared memory");
  m.def("run", &cholesky_b256n128, "Batched 128x128 Cholesky");
}
"""

_CUDA_SOURCE_B256N128 = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
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
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ void store_float4_cg(float* addr, float4 val) {
  asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};"
               :: "l"(addr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));
}

// Right-looking 64x64 factor executed by a single warp with two interleaved
// rows per lane. GlobalInput reads the leading block straight from global
// memory; otherwise the block is already staged in the shared tile at Base.
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
          __ldg(reinterpret_cast<const float2*>(input + j * kN) + lane);
      state[0][j] = loaded.x;
      state[1][j] = loaded.y;
    } else {
      state[0][j] = row0 >= j ? tile[(Base + row0) * kLd + Base + j] : 0.0f;
      state[1][j] = row1 >= j ? tile[(Base + row1) * kLd + Base + j] : 0.0f;
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
    const float next_pivot =
        __shfl_sync(kFullMask, state[next_slot][k], next_lane);
    state[0][next] = fmaf(-state[0][k], next_pivot, state[0][next]);
    state[1][next] = fmaf(-state[1][k], next_pivot, state[1][next]);

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
      const float pivot =
          __shfl_sync(kFullMask, state[pivot_slot][k], pivot_lane);
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
        tile[(Base + row) * kLd + Base + column] = state[owned][column];
      }
    }
  }
}

__device__ __forceinline__ void solve_panel(float* __restrict__ row_values,
                                            float* __restrict__ tile,
                                            int row) {
#pragma unroll 8
  for (int k = 0; k < kHalf; ++k) {
    const float solved = row_values[k] * tile[k * kLd + kN];
    row_values[k] = solved;
    tile[(kHalf + row) * kLd + k] = solved;
#pragma unroll 8
    for (int j = k + 1; j < kHalf; ++j) {
      row_values[j] = fmaf(-solved, tile[j * kLd + k], row_values[j]);
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

__device__ __forceinline__ void store_output_vector(
    const float* tile, float* output, int row, int column) {
  float4 values;
  values.x = column <= row ? tile[row * kLd + column] : 0.0f;
  values.y = column + 1 <= row ? tile[row * kLd + column + 1] : 0.0f;
  values.z = column + 2 <= row ? tile[row * kLd + column + 2] : 0.0f;
  values.w = column + 3 <= row ? tile[row * kLd + column + 3] : 0.0f;
  store_float4_cg(output + row * kN + column, values);
}

__device__ __forceinline__ void output_upper_right_zeros(float* output) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const float4 zeros = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int vector = lane; vector < kHalf * kHalf / 4; vector += 32) {
    const int row = vector >> 4;
    const int column = kHalf + (vector & 15) * 4;
    store_float4_cg(output + row * kN + column, zeros);
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
  for (int vector = tid; vector < kHalf * kHalf / 4; vector += kThreads) {
    const int row = kHalf + (vector >> 4);
    const int column = kHalf + (vector & 15) * 4;
    store_output_vector(tile, output, row, column);
  }
}

// Blocked 64/64 factorization. Warp 0 factors A00 while the two row warps
// stage A10 and A11 in registers and warp 3 pre-zeroes the upper-right
// output quadrant; the trailing SYRK is a warp-balanced SIMT update over
// 16x16 tiles, and the completed left half is written out while warp 0
// factors L11.
__global__ __launch_bounds__(kThreads, 2)
void blocked_128_kernel(const float* __restrict__ input,
                        float* __restrict__ output) {
  extern __shared__ __align__(16) float tile[];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int row = tid >= 32 && tid < 96 ? tid - 32 : -1;
  const int matrix = static_cast<int>(blockIdx.x);
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;
  float local[2 * kHalf];

  if (row >= 0) {
#pragma unroll
    for (int k = 0; k < kHalf; ++k) {
      local[k] = __ldg(a + k * kN + kHalf + row);
      local[kHalf + k] = __ldg(a + (kHalf + k) * kN + kHalf + row);
    }
  }

  if (warp == 3) output_upper_right_zeros(result);

  if (warp == 0) {
    factor64_right<true, 0>(a, tile, true);
  }
  __syncthreads();

  if (row >= 0) solve_panel(local, tile, row);
  __syncthreads();

  if (row >= 0) {
#pragma unroll
    for (int j = 0; j < kHalf; ++j) {
      if (j <= row) {
        tile[(kHalf + row) * kLd + kHalf + j] = local[kHalf + j];
      }
    }
  }
  __syncthreads();

  simt_balanced_trailing_update(tile);
  __syncthreads();

  if (warp == 0) {
    factor64_right<false, kHalf>(a, tile, false);
  }
  if (warp != 0) output_completed_left(tile, result);
  __syncthreads();

  output_l11(tile, result);
}

}  // namespace

void cholesky_b256n128_prepare() {
  const auto status = cudaFuncSetAttribute(
      blocked_128_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmemBytes);
  TORCH_CHECK(status == cudaSuccess,
              "dynamic shared-memory configuration failed: ",
              cudaGetErrorString(status));
}

at::Tensor cholesky_b256n128(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (256, 128, 128)");
  auto out = at::empty_like(data);
  blocked_128_kernel<<<kBatch, kThreads, kSmemBytes>>>(
      data.data_ptr<float>(), out.data_ptr<float>());
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return out;
}
"""


@lru_cache(maxsize=1)
def _module_b256n128():
    module = _build(
        "cholesky_b256n128", _CPP_SOURCE_B256N128, _CUDA_SOURCE_B256N128)
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# (64, 256, 256) - b64n256 variant 18
# ---------------------------------------------------------------------------

_CPP_SOURCE_B64N256 = r"""
#include <torch/extension.h>

void cholesky_b64n256_prepare();
at::Tensor cholesky_b64n256(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b64n256_prepare,
        "Configure batched 256x256 Cholesky dynamic shared memory");
  m.def("run", &cholesky_b64n256, "Batched 256x256 Cholesky");
}
"""

_CUDA_SOURCE_B64N256 = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 64;
constexpr int kN = 256;
constexpr int kTile = 128;
constexpr int kHalf = 64;
constexpr int kLd = 129;
constexpr int kThreads = 512;

// One CTA owns a whole 256x256 matrix. A00 and A11 are kLd-padded, and A10 is
// padded to kLd as well (variant 18's pad129 layout), which removes the
// shared-memory bank conflicts of the tight 128-column A10 panel.
constexpr int kA00 = 0;
constexpr int kA10 = kTile * kLd;
constexpr int kA11 = kA10 + kTile * kLd;
constexpr int kStorageFloats = kA11 + kTile * kLd;
constexpr int kTcScratchFloats = kHalf * kHalf;
constexpr int kTcBarrierFloats = 4;
constexpr int kSharedBytes =
    (kStorageFloats + kTcScratchFloats + kTcBarrierFloats) *
    static_cast<int>(sizeof(float));
constexpr uint32_t kTmemDp = 1u << 16;

__device__ __forceinline__ void root_pair(float value,
                                          float& diagonal,
                                          float& inverse) {
  // One Newton refinement of the hardware reciprocal square root.
  inverse = rsqrtf(value);
  inverse *= fmaf(-0.5f * value, inverse * inverse, 1.5f);
  diagonal = value * inverse;
}

__device__ __forceinline__ float& single_at(float* s, int row, int col) {
  if (row < kTile) {
    return s[kA00 + row * kLd + col];
  }
  if (col < kTile) {
    return s[kA10 + (row - kTile) * kLd + col];
  }
  return s[kA11 + (row - kTile) * kLd + col - kTile];
}

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void*>(pointer)));
}

__device__ __forceinline__ uint32_t to_tf32(float value) {
  uint32_t result;
  asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ int kmajor_offset(int row, int column, int rows) {
  return (row & 7) * 4 + (row >> 3) * 32 +
         (column & 3) + (column >> 2) * (rows * 4);
}

__device__ __forceinline__ uint64_t make_kmajor_descriptor(
    const void* pointer, int rows) {
  const uint64_t start =
      static_cast<uint64_t>(shared_address(pointer) >> 4) & 0x3fffull;
  const uint64_t leading = static_cast<uint64_t>(rows);
  const uint64_t stride = 8ull;
  return start | (leading << 16) | (stride << 32) | (1ull << 46);
}

template <int M, int N>
__device__ __forceinline__ constexpr uint32_t tf32_instruction_descriptor() {
  return (1u << 4) | (2u << 7) | (2u << 10) |
         (static_cast<uint32_t>(N >> 3) << 17) |
         (static_cast<uint32_t>(M >> 4) << 24);
}

__device__ __forceinline__ void proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tmem_allocate(uint32_t* destination,
                                              int columns) {
  if (static_cast<int>(threadIdx.x) < 32) {
    const uint32_t address = shared_address(destination);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
        "[%0], %1;" :: "r"(address), "r"(columns) : "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ void tmem_deallocate(uint32_t base, int columns) {
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < 32) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
        "r"(base), "r"(columns));
  }
  __syncthreads();
}

__device__ __forceinline__ void tmem_relinquish() {
  if (static_cast<int>(threadIdx.x) < 32) {
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier) {
  if (threadIdx.x == 0) {
    const uint32_t address = shared_address(barrier);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::
                 "r"(address) : "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ void tensor_commit(uint64_t* barrier) {
  if (threadIdx.x == 0) {
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
        "shared::cluster.b64 [%0];" :: "r"(address) : "memory");
  }
}

__device__ __forceinline__ void barrier_wait(uint64_t* barrier, int phase) {
  if (threadIdx.x == 0) {
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
  __syncthreads();
}

template <int M, int N>
__device__ __forceinline__ void issue_tf32_mma(
    uint32_t tmem_base, uint64_t a_desc, uint64_t b_desc, bool accumulate) {
  if (threadIdx.x == 0) {
    const uint32_t instruction = tf32_instruction_descriptor<M, N>();
    const uint32_t scale = accumulate ? 1u : 0u;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 "
        "[%0], %1, %2, %3, {%5,%6,%7,%8}, p;\n\t"
        "}\n" ::
        "r"(tmem_base), "l"(a_desc), "l"(b_desc), "r"(instruction),
        "r"(scale), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
  }
}

__device__ __forceinline__ float tmem_load_one(uint32_t address) {
  uint32_t value;
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
      : "=r"(value) : "r"(address));
  return __uint_as_float(value);
}

// Unblocked CTA-wide POTF2 over a diagonal block of the shared matrix.
__device__ __forceinline__ void potf2_single(float* s, int begin, int size) {
  for (int column = 0; column < size; ++column) {
    const int j = begin + column;
    if (threadIdx.x == 0) {
      float diagonal;
      float inverse;
      root_pair(single_at(s, j, j), diagonal, inverse);
      single_at(s, j, j) = diagonal;
      s[kStorageFloats - 1] = inverse;
    }
    __syncthreads();
    const float inverse = s[kStorageFloats - 1];
    for (int row = column + 1 + static_cast<int>(threadIdx.x);
         row < size; row += static_cast<int>(blockDim.x)) {
      single_at(s, begin + row, j) *= inverse;
    }
    __syncthreads();

    const int trailing = size - column - 1;
    const int pairs = trailing * (trailing + 1) / 2;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < pairs; linear += static_cast<int>(blockDim.x)) {
      int local_row = 0;
      int remainder = linear;
      for (int width = 1; remainder >= width; ++width) {
        remainder -= width;
        ++local_row;
      }
      const int row = j + 1 + local_row;
      const int col = j + 1 + remainder;
      single_at(s, row, col) =
          fmaf(-single_at(s, row, j), single_at(s, col, j),
               single_at(s, row, col));
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void trsm_single(
    float* s, int row_begin, int rows, int col_begin, int cols) {
  for (int local_row = static_cast<int>(threadIdx.x);
       local_row < rows; local_row += static_cast<int>(blockDim.x)) {
    const int row = row_begin + local_row;
    for (int local_col = 0; local_col < cols; ++local_col) {
      const int col = col_begin + local_col;
      float value = single_at(s, row, col);
      for (int k = 0; k < local_col; ++k) {
        value = fmaf(-single_at(s, row, col_begin + k),
                     single_at(s, col, col_begin + k), value);
      }
      single_at(s, row, col) = value / single_at(s, col, col);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void simt_update_single(
    float* s, int target, int size, int panel, int panel_cols) {
  constexpr int kMicro = 16;
  const int tile_count = (size + kMicro - 1) / kMicro;
  const int lower_tiles = tile_count * (tile_count + 1) / 2;
  for (int tile_linear = 0; tile_linear < lower_tiles; ++tile_linear) {
    int tile_row = 0;
    int tile_col = tile_linear;
    while (tile_col > tile_row) {
      tile_col -= tile_row + 1;
      ++tile_row;
    }
    for (int element = static_cast<int>(threadIdx.x);
         element < kMicro * kMicro;
         element += static_cast<int>(blockDim.x)) {
      const int local_row = tile_row * kMicro + element / kMicro;
      const int local_col = tile_col * kMicro + element % kMicro;
      if (local_row < size && local_col < size && local_col <= local_row) {
        float value = single_at(s, target + local_row, target + local_col);
#pragma unroll 4
        for (int k = 0; k < panel_cols; ++k) {
          value = fmaf(
              -single_at(s, target + local_row, panel + k),
              single_at(s, target + local_col, panel + k), value);
        }
        single_at(s, target + local_row, target + local_col) = value;
      }
    }
  }
  __syncthreads();
}

// Trailing SYRK through one tcgen05 TF32 MMA per eight-deep k slice.
template <int M>
__device__ __forceinline__ void tc_update_single(
    float* s, int target, int panel, float* scratch,
    uint32_t* tmem_slot, uint64_t* barrier, int& phase) {
  tmem_allocate(tmem_slot, kTile);
  const uint32_t tmem_base = *tmem_slot;
  for (int k = 0; k < M; k += 8) {
    for (int linear = static_cast<int>(threadIdx.x);
         linear < M * 8; linear += static_cast<int>(blockDim.x)) {
      const int row = linear >> 3;
      const int column = linear & 7;
      const int packed = kmajor_offset(row, column, M);
      reinterpret_cast<uint32_t*>(scratch)[packed] =
          to_tf32(single_at(s, target + row, panel + k + column));
    }
    __syncthreads();
    proxy_fence();
    __syncthreads();
    const uint64_t descriptor = make_kmajor_descriptor(scratch, M);
    issue_tf32_mma<M, M>(tmem_base, descriptor, descriptor, k != 0);
    tensor_commit(barrier);
    barrier_wait(barrier, phase);
    phase ^= 1;
  }

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if constexpr (M == 128) {
    if (warp < 4) {
      const int row = warp * 32 + lane;
      for (int col = 0; col < M; ++col) {
        const uint32_t address =
            tmem_base + static_cast<uint32_t>(warp * 32) * kTmemDp +
            static_cast<uint32_t>(col);
        const float product = tmem_load_one(address);
        if (col <= row) {
          single_at(s, target + row, target + col) -= product;
        }
      }
    }
  } else {
    if (warp < 4) {
      const int row = warp * 16 + lane;
      for (int col = 0; col < M; ++col) {
        const uint32_t address =
            tmem_base + static_cast<uint32_t>(warp * 32) * kTmemDp +
            static_cast<uint32_t>(col);
        const float product = tmem_load_one(address);
        if (lane < 16 && col <= row) {
          single_at(s, target + row, target + col) -= product;
        }
      }
    }
  }
  __syncthreads();
  tmem_deallocate(tmem_base, kTile);
}

// 128x128 diagonal block: recursive 32/32 base factors with SIMT updates and
// one TF32 tensor-core update for the 64-wide trailing block.
__device__ __forceinline__ void potrf128_single(
    float* s, int begin, float* scratch, uint32_t* tmem_slot,
    uint64_t* barrier, int& phase) {
  potf2_single(s, begin, 32);
  trsm_single(s, begin + 32, 32, begin, 32);
  simt_update_single(s, begin + 32, 32, begin, 32);
  potf2_single(s, begin + 32, 32);

  trsm_single(s, begin + 64, 64, begin, 64);
  tc_update_single<64>(
      s, begin + 64, begin, scratch, tmem_slot, barrier, phase);

  potf2_single(s, begin + 64, 32);
  trsm_single(s, begin + 96, 32, begin + 64, 32);
  simt_update_single(s, begin + 96, 32, begin + 64, 32);
  potf2_single(s, begin + 96, 32);
}

__global__ __launch_bounds__(kThreads, 1)
void single_kernel(const float* __restrict__ input,
                   float* __restrict__ output) {
  extern __shared__ __align__(16) float storage[];
  float* scratch = storage + kStorageFloats;
  uint32_t* tmem_slot =
      reinterpret_cast<uint32_t*>(scratch + kTcScratchFloats);
  uint64_t* barrier =
      reinterpret_cast<uint64_t*>(scratch + kTcScratchFloats + 2);
  int phase = 0;

  const int matrix = static_cast<int>(blockIdx.x);
  const float* matrix_input = input + static_cast<int64_t>(matrix) * kN * kN;
  float* matrix_output = output + static_cast<int64_t>(matrix) * kN * kN;

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kN * kN; linear += static_cast<int>(blockDim.x)) {
    matrix_output[linear] = 0.0f;
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      single_at(storage, row, col) = matrix_input[row * kN + col];
      single_at(storage, row + kTile, col + kTile) =
          matrix_input[(row + kTile) * kN + col + kTile];
    }
    single_at(storage, row + kTile, col) =
        matrix_input[(row + kTile) * kN + col];
  }
  __syncthreads();
  barrier_init(barrier);

  potrf128_single(storage, 0, scratch, tmem_slot, barrier, phase);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      matrix_output[row * kN + col] = single_at(storage, row, col);
    }
  }
  __syncthreads();

  trsm_single(storage, kTile, kTile, 0, kTile);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    matrix_output[(row + kTile) * kN + col] =
        single_at(storage, row + kTile, col);
  }
  __syncthreads();

  tc_update_single<128>(storage, kTile, 0, scratch, tmem_slot, barrier, phase);
  potrf128_single(storage, kTile, scratch, tmem_slot, barrier, phase);

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      matrix_output[(row + kTile) * kN + col + kTile] =
          single_at(storage, row + kTile, col + kTile);
    }
  }
  __syncthreads();
  tmem_relinquish();
}

}  // namespace

void cholesky_b64n256_prepare() {
  auto status = cudaFuncSetAttribute(
      single_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  TORCH_CHECK(status == cudaSuccess,
              "dynamic shared-memory opt-in failed: ",
              cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      single_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(status == cudaSuccess,
              "shared-memory carveout failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b64n256(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (64, 256, 256)");
  auto out = at::empty_like(data);
  single_kernel<<<kBatch, kThreads, kSharedBytes>>>(
      data.data_ptr<float>(), out.data_ptr<float>());
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return out;
}
"""


@lru_cache(maxsize=1)
def _module_b64n256():
    module = _build(
        "cholesky_b64n256", _CPP_SOURCE_B64N256, _CUDA_SOURCE_B64N256,
        extra_cuda_flags=("--restrict",))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# (16, 512, 512) - b16n512 variant 2
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
# (640, 512, 512) - b640n512 variant 8
# ---------------------------------------------------------------------------

_CPP_SOURCE_B640N512 = r"""
#include <torch/extension.h>

void cholesky_b640n512_prepare();
at::Tensor cholesky_b640n512(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b640n512_prepare,
        "Configure the fused 640x512 Cholesky kernel");
  m.def("run", &cholesky_b640n512, "Batched 640x512 Cholesky");
}
"""

_CUDA_SOURCE_B640N512 = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 640;
constexpr int kN = 512;
constexpr int kTile = 64;
constexpr int kLd = kTile + 1;
constexpr int kTileCount = kN / kTile;
constexpr int kThreads = 256;
// 640 CTAs fit inside the 740-CTA resident capacity at five blocks per SM,
// which keeps the whole batch in one wave.
constexpr int kMinBlocks = 5;

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

// The two local updates below are driven by the first four warps only; the
// remaining warps of the 256-thread CTA wait at the enclosing barrier.
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

// Recursive 16 -> 32 -> 64 factor of one diagonal block in shared memory.
__device__ __forceinline__ void factor_tile(
    float* tile, float* inverse_diagonal) {
  factor32_recursive16(tile, inverse_diagonal, 0);
  __syncthreads();
  local_trsm_sub4<32, 32>(tile, inverse_diagonal, 32, 0);
  __syncthreads();
  local_update32(tile, 32, 0);
  __syncthreads();
  factor32_recursive16(tile, inverse_diagonal, 32);
}

// Copy the lower triangle of the input into the output buffer, which the
// panel loop then factors in place. One thread owns a whole float4 column
// strip, so every row is a single vector transaction except on the diagonal.
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
              column + item <= row ? input[offset + item] : 0.0f;
        }
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void load_tile(
    const float* matrix, float* tile, int row_begin, int column_begin) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += kThreads) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    tile[row * kLd + column] = load_global(
        matrix + (row_begin + row) * kN + column_begin + column);
  }
}

__device__ __forceinline__ void load_diagonal(
    const float* matrix, float* tile, int begin) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += kThreads) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    tile[row * kLd + column] =
        column <= row
            ? load_global(matrix + (begin + row) * kN + begin + column)
            : 0.0f;
  }
}

__device__ __forceinline__ void store_tile(
    const float* tile, float* matrix, int row_begin, int column_begin) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += kThreads) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    store_global(
        matrix + (row_begin + row) * kN + column_begin + column,
        tile[row * kLd + column]);
  }
}

__device__ __forceinline__ void store_diagonal(
    const float* tile, float* matrix, int begin) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += kThreads) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    if (column <= row) {
      store_global(matrix + (begin + row) * kN + begin + column,
                   tile[row * kLd + column]);
    }
  }
}

// One thread per row of the 64x64 right-hand side.
__device__ __forceinline__ void solve_panel(
    float* rhs, const float* diagonal, const float* inverse_diagonal) {
  if (static_cast<int>(threadIdx.x) < kTile) {
    const int row = static_cast<int>(threadIdx.x);
#pragma unroll 1
    for (int column = 0; column < kTile; ++column) {
      float value = rhs[row * kLd + column];
#pragma unroll 4
      for (int k = 0; k < column; ++k) {
        value = fmaf(
            -rhs[row * kLd + k], diagonal[column * kLd + k], value);
      }
      rhs[row * kLd + column] = value * inverse_diagonal[column];
    }
  }
}

// 4x4 micro-tiled rank-64 update of one trailing tile. The destination is
// preloaded into the accumulators before the 64-step product and the FMAs
// are negated, which moves the dependent global load off the epilogue.
__device__ __forceinline__ void update_global(
    const float* left, const float* right, float* matrix,
    int row_begin, int column_begin, bool diagonal) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = (warp >> 1) * 16;
  const int column_base = (warp & 1) * 32;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;
  float product[4][4];
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const int output_row = row_base + lane_row + row * 4;
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_column = column_base + lane_column + column * 8;
      product[row][column] =
          !diagonal || output_column <= output_row
              ? load_global(matrix + (row_begin + output_row) * kN +
                            column_begin + output_column)
              : 0.0f;
    }
  }
#pragma unroll 1
  for (int k = 0; k < kTile; ++k) {
    float left_values[4];
    float right_values[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      left_values[row] = left[(row_base + lane_row + row * 4) * kLd + k];
    }
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      right_values[column] =
          right[(column_base + lane_column + column * 8) * kLd + k];
    }
#pragma unroll
    for (int row = 0; row < 4; ++row) {
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        product[row][column] = fmaf(
            -left_values[row], right_values[column], product[row][column]);
      }
    }
  }
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const int output_row = row_base + lane_row + row * 4;
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_column = column_base + lane_column + column * 8;
      if (!diagonal || output_column <= output_row) {
        store_global(matrix + (row_begin + output_row) * kN +
                         column_begin + output_column,
                     product[row][column]);
      }
    }
  }
}

// One CTA owns a whole 512x512 matrix and runs the entire right-looking
// blocked factorization over eight 64-column panels: factor the diagonal
// block in shared memory, solve every trailing row tile against it, then
// apply the rank-64 update to the trailing submatrix.
__global__ __launch_bounds__(kThreads, kMinBlocks)
void fused_potrf_kernel(const float* __restrict__ input,
                        float* __restrict__ output) {
  __shared__ __align__(128) float tile_a[kTile * kLd];
  __shared__ __align__(128) float tile_b[kTile * kLd];
  __shared__ float inverse_diagonal[kTile];

  const int matrix_index = static_cast<int>(blockIdx.x);
  const int64_t matrix_offset =
      static_cast<int64_t>(matrix_index) * kN * kN;
  const float* input_matrix = input + matrix_offset;
  float* matrix = output + matrix_offset;

  copy_lower(input_matrix, matrix);

  for (int panel = 0; panel < kTileCount; ++panel) {
    const int panel_begin = panel * kTile;
    load_diagonal(matrix, tile_a, panel_begin);
    __syncthreads();

    factor_tile(tile_a, inverse_diagonal);
    __syncthreads();
    store_diagonal(tile_a, matrix, panel_begin);

    // store_tile and load_tile share one linear index mapping, so each
    // thread rereads exactly the tile_b slots it just wrote; no barrier is
    // needed between the store and the next iteration's load.
    for (int row_tile = panel + 1; row_tile < kTileCount; ++row_tile) {
      const int row_begin = row_tile * kTile;
      load_tile(matrix, tile_b, row_begin, panel_begin);
      __syncthreads();
      solve_panel(tile_b, tile_a, inverse_diagonal);
      __syncthreads();
      store_tile(tile_b, matrix, row_begin, panel_begin);
    }

    for (int row_tile = panel + 1; row_tile < kTileCount; ++row_tile) {
      const int row_begin = row_tile * kTile;
      load_tile(matrix, tile_a, row_begin, panel_begin);
      __syncthreads();

      for (int column_tile = panel + 1;
           column_tile <= row_tile; ++column_tile) {
        const int column_begin = column_tile * kTile;
        const bool is_diagonal = column_tile == row_tile;
        const float* right = tile_a;
        if (!is_diagonal) {
          load_tile(matrix, tile_b, column_begin, panel_begin);
          __syncthreads();
          right = tile_b;
        }
        update_global(
            tile_a, right, matrix, row_begin, column_begin, is_diagonal);
        __syncthreads();
      }
    }
  }
}

}  // namespace

void cholesky_b640n512_prepare() {
  const cudaError_t status = cudaFuncSetAttribute(
      fused_potrf_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  TORCH_CHECK(status == cudaSuccess,
              "shared-memory carveout failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b640n512(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (640, 512, 512)");
  auto output = at::empty_like(data);
  c10::cuda::CUDAGuard device_guard(data.device());
  fused_potrf_kernel<<<kBatch, kThreads>>>(
      data.data_ptr<float>(), output.data_ptr<float>());
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return output;
}
"""


@lru_cache(maxsize=1)
def _module_b640n512():
    module = _build(
        "cholesky_b640n512", _CPP_SOURCE_B640N512, _CUDA_SOURCE_B640N512,
        extra_cuda_flags=("-DNDEBUG", "--restrict"))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SPECIALIZATIONS = {
    (4096, 32, 32): _module_b4096n32,
    (1024, 64, 64): _module_b1024n64,
    (256, 128, 128): _module_b256n128,
    (64, 256, 256): _module_b64n256,
    (16, 512, 512): _module_b16n512,
    (640, 512, 512): _module_b640n512,
}


def custom_kernel(data: input_t) -> output_t:
    if data.is_cuda and data.dtype == torch.float32 and data.is_contiguous():
        specialization = _SPECIALIZATIONS.get(tuple(data.shape))
        if specialization is not None:
            return specialization().run(data)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
