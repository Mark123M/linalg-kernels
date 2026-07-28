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
                             TEMPORARILY DISABLED: this shape is routed to
                             torch. The extension is still built into this
                             file; only its dispatch entry is commented out.
  (16, 512, 512)   b16n512   variant 2   r16_micro4x4_raw_fused_u256
                             staged 64x64 right-looking Cholesky with fused
                             factor/solve launches and a 4x4 micro-tiled
                             FP32 update.
  (640, 512, 512)  b640n512  variant 20  staged_p128_precise_sub4_cublas_tf32_t256
                             staged 128-column panels across separate
                             factor/solve launches, with one fast-TF32
                             cublasGemmStridedBatchedEx per panel over the
                             whole trailing square.
  (60, 1024, 1024) b60n1024  variant 6   staged_precise_sub4_cublas_tf32_t256
                             the same staged schedule over eight 128-column
                             panels, sized for 1024x1024 matrices.
  (8, 2048, 2048)  b8n2048   variant 5   ll_fixed128_custom_tf32
                             left-looking fixed 128-column panels, shared
                             memory 128x128 leaf factors, register-blocked
                             custom TRSM, one fast-TF32 strided-batched GEMM
                             of the history per panel.
  (1, 8192, 8192)  b1n8192   variant 8   ll_nb512_m64_microfused_split2_tf32
                             left-looking 512-column panels with a fused
                             producer/consumer 64-wide micro block; the
                             inverse application of each 64x64 tile is split
                             across two consumer CTAs.
  (1, 16384, 16384) b1n16384 variant 0   ll_nb1024_invgemm_tf32
                             left-looking 1024-column panels, wide 128x128
                             factor/inverse, TF32 GEMM apply plus copy-back.
  (1, 32768, 32768) b1n32768 variant 14  ll_nb1024_invgemm_tf32
                             the same production schedule at n=32768.

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


def _build(name, cpp_source, cuda_source, extra_cuda_flags=(),
           extra_ldflags=()):
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
            extra_ldflags=list(extra_ldflags),
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
# (16, 512, 512) - b16n512 variant 2, fused factor/solve
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

// Fused factor + solve: one grid covers the diagonal factorization and
// every remaining row tile's triangular solve.  All blocks redundantly
// factor the diagonal so they can proceed without inter-block sync.
__global__ __launch_bounds__(128)
void factor_solve_kernel(float* __restrict__ output, int panel) {
  __shared__ __align__(128) float tile[kTile * kLd];
  __shared__ __align__(128) float rhs[kTile * kLd];
  __shared__ float inverse_diagonal[kTile];

  const int matrix_index = static_cast<int>(blockIdx.x);
  const int local_row = static_cast<int>(blockIdx.y) + panel;
  float* matrix = output + static_cast<int64_t>(matrix_index) * kN * kN;
  const int panel_begin = panel * kTile;
  const int row_begin = local_row * kTile;
  const bool is_factor = (local_row == panel);

  // Load diagonal block (all blocks).
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += 128) {
    const int row = linear / kTile;
    const int column = linear % kTile;
    tile_at(tile, row, column) =
        column <= row
            ? load_global(matrix + (panel_begin + row) * kN +
                          panel_begin + column)
            : 0.0f;
  }

  // Load the rhs panel (solve blocks only).
  if (!is_factor) {
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kTile * kTile; linear += 128) {
      const int row = linear / kTile;
      const int column = linear % kTile;
      rhs[row * kLd + column] = load_global(
          matrix + (row_begin + row) * kN + panel_begin + column);
    }
  }
  __syncthreads();

  // Factor the 64x64 diagonal (all blocks — redundant, but avoids
  // inter-block synchronisation).
  factor32_recursive16(tile, inverse_diagonal, 0);
  __syncthreads();
  local_trsm_sub4<32, 32>(tile, inverse_diagonal, 32, 0);
  __syncthreads();
  local_update32(tile, 32, 0);
  __syncthreads();
  factor32_recursive16(tile, inverse_diagonal, 32);
  __syncthreads();

  // Write factored diagonal (only the diagonal block).
  if (is_factor) {
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kTile * kTile; linear += 128) {
      const int row = linear / kTile;
      const int column = linear % kTile;
      if (column <= row) {
        store_global(matrix + (panel_begin + row) * kN +
                         panel_begin + column,
                     tile_at(tile, row, column));
      }
    }
  }

  // Triangular solve (solve blocks only).
  if (!is_factor) {
    if (static_cast<int>(threadIdx.x) < kTile) {
      const int column = static_cast<int>(threadIdx.x);
      inverse_diagonal[column] =
          __fdividef(1.0f, tile[column * kLd + column]);
    }
    __syncthreads();

    if (static_cast<int>(threadIdx.x) < kTile) {
      const int row = static_cast<int>(threadIdx.x);
#pragma unroll 1
      for (int column = 0; column < kTile; ++column) {
        float value = rhs[row * kLd + column];
#pragma unroll 4
        for (int k = 0; k < column; ++k) {
          value =
              fmaf(-rhs[row * kLd + k], tile[column * kLd + k], value);
        }
        rhs[row * kLd + column] = value * inverse_diagonal[column];
      }
    }
    __syncthreads();

    for (int linear = static_cast<int>(threadIdx.x);
         linear < kTile * kTile; linear += 128) {
      const int row = linear / kTile;
      const int column = linear % kTile;
      store_global(matrix + (row_begin + row) * kN + panel_begin + column,
                   rhs[row * kLd + column]);
    }
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
    const int remaining = kTileCount - panel;
    cudaLaunchConfig_t fuse_config{};
    fuse_config.gridDim = dim3(kBatch, remaining, 1);
    fuse_config.blockDim = dim3(128, 1, 1);
    cudaLaunchKernelEx(&fuse_config, factor_solve_kernel, output, panel);

    const int trailing = remaining - 1;
    if (trailing == 0) {
      continue;
    }

    const int tasks = trailing * (trailing + 1) / 2;
    cudaLaunchConfig_t update_config{};
    update_config.gridDim = dim3(kBatch * tasks, 1, 1);
    update_config.blockDim = dim3(kUpdateThreads, 1, 1);
    cudaLaunchKernelEx(
        &update_config, fp32_update_kernel, output, panel, tasks);
  }
}

}  // namespace

void cholesky_b16n512_prepare() {
  prefer_shared(factor_solve_kernel);
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
# (640, 512, 512) - b640n512 variant 20
# ---------------------------------------------------------------------------

_CPP_SOURCE_B640N512 = r"""
#include <torch/extension.h>

void cholesky_b640n512_prepare();
at::Tensor cholesky_b640n512(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b640n512_prepare,
        "Configure the staged 640x512 Cholesky kernels");
  m.def("run", &cholesky_b640n512, "Batched 640x512 Cholesky");
}
"""

_CUDA_SOURCE_B640N512 = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 640;
constexpr int kN = 512;
constexpr int kOuter = 128;
constexpr int kMicro = 64;
constexpr int kPanelCount = kN / kOuter;
constexpr int kMicroCount = kN / kMicro;
constexpr int kWidth = 4;
constexpr int kThreads = 256;
constexpr int kFactorBytes =
    static_cast<int>(sizeof(float)) *
    (kOuter * (kOuter + 1) + kOuter);
constexpr int kSolveBytes =
    static_cast<int>(sizeof(float)) *
    (32 * (kOuter + 1) + kMicro * (kOuter + kWidth));
static_assert(kFactorBytes == 66560);
static_assert(kSolveBytes == 50304);

__device__ __forceinline__ float load_global(const float* pointer) {
  return __ldcg(pointer);
}

__device__ __forceinline__ void store_global(float* pointer, float value) {
  __stcg(pointer, value);
}

__device__ __forceinline__ void root_pair(
    float value, float& diagonal, float& inverse) {
  diagonal = __fsqrt_rn(value);
  inverse = __fdiv_rn(1.0f, diagonal);
}

__device__ __forceinline__ float& tile_at(
    float* tile, int row, int column) {
  return tile[row * (kOuter + 1) + column];
}

// Unblocked 32x32 right-looking factor executed by warp 0.
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
        root_pair(tile_at(tile, column, column), diagonal, inverse);
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

// Shared-tile triangular solve, one four-lane subgroup per row.
template <int Rows, int Columns>
__device__ __forceinline__ void local_trsm(
    float* tile, const float* inverse_diagonal,
    int row_begin, int column_begin) {
  const int lane = static_cast<int>(threadIdx.x) & (kWidth - 1);
  const int row_index = static_cast<int>(threadIdx.x) / kWidth;
  if (row_index < Rows) {
    const int row = row_begin + row_index;
#pragma unroll 1
    for (int local_column = 0; local_column < Columns; ++local_column) {
      const int column = column_begin + local_column;
      float partial = 0.0f;
#pragma unroll 4
      for (int k = lane; k < local_column; k += kWidth) {
        partial = fmaf(
            tile_at(tile, row, column_begin + k),
            tile_at(tile, column, column_begin + k), partial);
      }
#pragma unroll
      for (int offset = kWidth / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(
            0xffffffffu, partial, offset, kWidth);
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

// Symmetric rank-K update of one diagonal block inside the shared tile.
template <int Size, int K>
__device__ __forceinline__ void local_update(
    float* tile, int target, int panel) {
  constexpr int kElements = Size * Size;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kElements; linear += static_cast<int>(blockDim.x)) {
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

// Four 32-column factors linked by subgroup solves and FP32 updates.
__device__ __forceinline__ void factor_local(
    float* tile, float* inverse_diagonal) {
  potf2_32(tile, inverse_diagonal, 0);
  local_trsm<32, 32>(tile, inverse_diagonal, 32, 0);
  local_update<32, 32>(tile, 32, 0);
  potf2_32(tile, inverse_diagonal, 32);
  local_trsm<64, 64>(tile, inverse_diagonal, 64, 0);
  local_update<64, 64>(tile, 64, 0);
  potf2_32(tile, inverse_diagonal, 64);
  local_trsm<32, 32>(tile, inverse_diagonal, 96, 64);
  local_update<32, 32>(tile, 96, 64);
  potf2_32(tile, inverse_diagonal, 96);
}

__device__ __forceinline__ void factor_global(
    float* matrix, int begin, float* work) {
  float* tile = work;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kOuter * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    tile_at(tile, row, column) =
        column <= row
            ? load_global(matrix + (begin + row) * kN + begin + column)
            : 0.0f;
  }
  __syncthreads();
  float* inverse_diagonal = tile + kOuter * (kOuter + 1);
  factor_local(tile, inverse_diagonal);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kOuter * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    if (column <= row) {
      store_global(
          matrix + (begin + row) * kN + begin + column,
          tile_at(tile, row, column));
    }
  }
}

// One solved column of a 32-wide right-hand-side block held in registers.
template <int Block, int LocalColumn, int RegisterCount>
__device__ __forceinline__ void trsm_column(
    float (&values)[RegisterCount], const float* diagonal,
    const float* panel, int row, int lane) {
  constexpr int kDiagonalLd = kOuter + 1;
  constexpr int kPanelLd = kOuter + kWidth;
  constexpr int kBlockBegin = Block * 32;
  constexpr int kColumn = kBlockBegin + LocalColumn;
  constexpr int kOwner = LocalColumn & (kWidth - 1);
  constexpr int kOwnerSlot = LocalColumn / kWidth;
  static_assert(RegisterCount == 32 / kWidth);
  float partial = 0.0f;
#pragma unroll 4
  for (int k = lane; k < kBlockBegin; k += kWidth) {
    partial = fmaf(
        panel[row * kPanelLd + k],
        diagonal[LocalColumn * kDiagonalLd + k], partial);
  }
#pragma unroll
  for (int slot = 0; slot < RegisterCount; ++slot) {
    const int local_k = lane + slot * kWidth;
    if (local_k < LocalColumn) {
      partial = fmaf(
          values[slot],
          diagonal[LocalColumn * kDiagonalLd + kBlockBegin + local_k],
          partial);
    }
  }
#pragma unroll
  for (int offset = kWidth / 2; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffffu, partial, offset, kWidth);
  }
  const float owned_rhs = values[kOwnerSlot];
  const float rhs = __shfl_sync(0xffffffffu, owned_rhs, kOwner, kWidth);
  float solved = 0.0f;
  if (lane == 0) {
    solved =
        (rhs - partial) / diagonal[LocalColumn * kDiagonalLd + kColumn];
  }
  solved = __shfl_sync(0xffffffffu, solved, 0, kWidth);
  if (lane == kOwner) {
    values[kOwnerSlot] = solved;
  }
}

// Stages 32 diagonal rows at a time so the whole solve fits 50,304 bytes.
template <int Block>
__device__ __forceinline__ void trsm_block(
    float* matrix, int panel_begin, float* diagonal, float* panel) {
  constexpr int kDiagonalLd = kOuter + 1;
  constexpr int kPanelLd = kOuter + kWidth;
  constexpr int kBlockBegin = Block * 32;
  constexpr int kRegisterCount = 32 / kWidth;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < 32 * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int local_row = linear / kOuter;
    const int column = linear % kOuter;
    const int matrix_row = kBlockBegin + local_row;
    diagonal[local_row * kDiagonalLd + column] =
        column <= matrix_row
            ? load_global(
                  matrix + (panel_begin + matrix_row) * kN +
                  panel_begin + column)
            : 0.0f;
  }
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) & (kWidth - 1);
  const int row = static_cast<int>(threadIdx.x) / kWidth;
  if (row < kMicro) {
    float values[kRegisterCount];
#pragma unroll
    for (int slot = 0; slot < kRegisterCount; ++slot) {
      values[slot] =
          panel[row * kPanelLd + kBlockBegin + lane + slot * kWidth];
    }
#define TRSM_COLUMN(COLUMN)                                       \
    trsm_column<Block, COLUMN>(values, diagonal, panel, row, lane)
    TRSM_COLUMN(0);
    TRSM_COLUMN(1);
    TRSM_COLUMN(2);
    TRSM_COLUMN(3);
    TRSM_COLUMN(4);
    TRSM_COLUMN(5);
    TRSM_COLUMN(6);
    TRSM_COLUMN(7);
    TRSM_COLUMN(8);
    TRSM_COLUMN(9);
    TRSM_COLUMN(10);
    TRSM_COLUMN(11);
    TRSM_COLUMN(12);
    TRSM_COLUMN(13);
    TRSM_COLUMN(14);
    TRSM_COLUMN(15);
    TRSM_COLUMN(16);
    TRSM_COLUMN(17);
    TRSM_COLUMN(18);
    TRSM_COLUMN(19);
    TRSM_COLUMN(20);
    TRSM_COLUMN(21);
    TRSM_COLUMN(22);
    TRSM_COLUMN(23);
    TRSM_COLUMN(24);
    TRSM_COLUMN(25);
    TRSM_COLUMN(26);
    TRSM_COLUMN(27);
    TRSM_COLUMN(28);
    TRSM_COLUMN(29);
    TRSM_COLUMN(30);
    TRSM_COLUMN(31);
#undef TRSM_COLUMN
#pragma unroll
    for (int slot = 0; slot < kRegisterCount; ++slot) {
      panel[row * kPanelLd + kBlockBegin + lane + slot * kWidth] =
          values[slot];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void trsm_global(
    float* matrix, int row_begin, int panel_begin, float* work) {
  constexpr int kDiagonalLd = kOuter + 1;
  constexpr int kPanelLd = kOuter + kWidth;
  float* diagonal = work;
  float* panel = diagonal + 32 * kDiagonalLd;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    panel[row * kPanelLd + column] = load_global(
        matrix + (row_begin + row) * kN + panel_begin + column);
  }
  trsm_block<0>(matrix, panel_begin, diagonal, panel);
  trsm_block<1>(matrix, panel_begin, diagonal, panel);
  trsm_block<2>(matrix, panel_begin, diagonal, panel);
  trsm_block<3>(matrix, panel_begin, diagonal, panel);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    store_global(
        matrix + (row_begin + row) * kN + panel_begin + column,
        panel[row * kPanelLd + column]);
  }
}

__global__ __launch_bounds__(kThreads)
void copy_kernel(
    const float* __restrict__ input, float* __restrict__ output) {
  constexpr int kCtasPerMatrix = 4;
  constexpr int kVectors = kN * kN / 4;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / kCtasPerMatrix;
  const int rank = static_cast<int>(blockIdx.x) % kCtasPerMatrix;
  const int64_t base = static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear = rank * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
       linear < kVectors;
       linear += kCtasPerMatrix * static_cast<int>(blockDim.x)) {
    const int64_t offset = base + linear * 4;
    *reinterpret_cast<float4*>(output + offset) =
        *reinterpret_cast<const float4*>(input + offset);
  }
}

// The trailing GEMM writes the full square, so restore exact upper zeros.
__global__ __launch_bounds__(kThreads)
void zero_upper_kernel(float* __restrict__ output) {
  constexpr int kCtasPerMatrix = 2;
  constexpr int kVectorsPerRow = kN / 4;
  constexpr int kVectors = kN * kVectorsPerRow;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / kCtasPerMatrix;
  const int rank = static_cast<int>(blockIdx.x) % kCtasPerMatrix;
  const int64_t base = static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear = rank * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
       linear < kVectors;
       linear += kCtasPerMatrix * static_cast<int>(blockDim.x)) {
    const int row = linear / kVectorsPerRow;
    const int column = (linear % kVectorsPerRow) * 4;
    float* destination = output + base + row * kN + column;
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

__global__ __launch_bounds__(kThreads)
void factor_kernel(float* __restrict__ output, int panel) {
  extern __shared__ __align__(128) unsigned char dynamic_bytes[];
  const int matrix_index = static_cast<int>(blockIdx.x);
  float* matrix = output + static_cast<int64_t>(matrix_index) * kN * kN;
  factor_global(
      matrix, panel * kOuter,
      reinterpret_cast<float*>(dynamic_bytes));
}

__global__ __launch_bounds__(kThreads)
void solve_kernel(
    float* __restrict__ output, int panel, int remaining) {
  extern __shared__ __align__(128) unsigned char dynamic_bytes[];
  const int matrix_index = static_cast<int>(blockIdx.x) / remaining;
  const int row_index = static_cast<int>(blockIdx.x) % remaining;
  float* matrix = output + static_cast<int64_t>(matrix_index) * kN * kN;
  trsm_global(
      matrix, (panel * 2 + 2 + row_index) * kMicro, panel * kOuter,
      reinterpret_cast<float*>(dynamic_bytes));
}

void check_cublas(cublasStatus_t status, const char* role) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      role, " failed with cuBLAS status ", static_cast<int>(status));
}

// Selects high-performance cuBLAS math for the trailing GEMMs and restores
// the caller's handle state afterwards.
class CublasFastState {
 public:
  explicit CublasFastState(cublasHandle_t handle) : handle_(handle) {
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
        cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_ALLOWED),
        "enable cuBLAS atomic algorithms");
    check_cublas(
        cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
        "select host cuBLAS scalars");
  }

  ~CublasFastState() {
    cublasSetPointerMode(handle_, pointer_mode_);
    cublasSetAtomicsMode(handle_, atomics_mode_);
    cublasSetMathMode(handle_, math_mode_);
  }

  CublasFastState(const CublasFastState&) = delete;
  CublasFastState& operator=(const CublasFastState&) = delete;

 private:
  cublasHandle_t handle_;
  cublasMath_t math_mode_{};
  cublasAtomicsMode_t atomics_mode_{};
  cublasPointerMode_t pointer_mode_{};
};

template <typename Kernel>
void configure_kernel(Kernel kernel, int dynamic_bytes) {
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(
      status == cudaSuccess,
      "shared-memory carveout failed: ", cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);
  TORCH_CHECK(
      status == cudaSuccess,
      "dynamic shared-memory opt-in failed: ",
      cudaGetErrorString(status));
}

// One fast-TF32 batched GEMM over the whole trailing square.
void launch_blas_update(cublasHandle_t handle, float* output, int panel) {
  const int begin = (panel + 1) * kOuter;
  const int remaining = kN - begin;
  const int panel_begin = panel * kOuter;
  float* panel_pointer = output + begin * kN + panel_begin;
  float* destination = output + begin * kN + begin;
  const float alpha = -1.0f;
  const float beta = 1.0f;
  constexpr long long kMatrixStride = static_cast<long long>(kN) * kN;
  check_cublas(
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          remaining, remaining, kOuter,
          &alpha,
          panel_pointer, CUDA_R_32F, kN, kMatrixStride,
          panel_pointer, CUDA_R_32F, kN, kMatrixStride,
          &beta,
          destination, CUDA_R_32F, kN, kMatrixStride,
          kBatch,
          CUBLAS_COMPUTE_32F_FAST_TF32,
          CUBLAS_GEMM_DEFAULT),
      "staged batched trailing GEMM");
}

}  // namespace

void cholesky_b640n512_prepare() {
  configure_kernel(factor_kernel, kFactorBytes);
  configure_kernel(solve_kernel, kSolveBytes);
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
  const float* input = data.data_ptr<float>();
  float* result = output.data_ptr<float>();

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasFastState fast_state(handle);

  cudaLaunchConfig_t copy_config{};
  copy_config.gridDim = dim3(kBatch * 4, 1, 1);
  copy_config.blockDim = dim3(kThreads, 1, 1);
  cudaLaunchKernelEx(&copy_config, copy_kernel, input, result);

  for (int panel = 0; panel < kPanelCount; ++panel) {
    cudaLaunchConfig_t factor_config{};
    factor_config.gridDim = dim3(kBatch, 1, 1);
    factor_config.blockDim = dim3(kThreads, 1, 1);
    factor_config.dynamicSmemBytes = kFactorBytes;
    cudaLaunchKernelEx(&factor_config, factor_kernel, result, panel);

    const int remaining = kMicroCount - panel * 2 - 2;
    if (remaining == 0) {
      continue;
    }
    cudaLaunchConfig_t solve_config{};
    solve_config.gridDim = dim3(kBatch * remaining, 1, 1);
    solve_config.blockDim = dim3(kThreads, 1, 1);
    solve_config.dynamicSmemBytes = kSolveBytes;
    cudaLaunchKernelEx(
        &solve_config, solve_kernel, result, panel, remaining);

    launch_blas_update(handle, result, panel);
  }

  cudaLaunchConfig_t zero_config{};
  zero_config.gridDim = dim3(kBatch * 2, 1, 1);
  zero_config.blockDim = dim3(kThreads, 1, 1);
  cudaLaunchKernelEx(&zero_config, zero_upper_kernel, result);

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
        extra_cuda_flags=("-DNDEBUG", "--restrict"),
        extra_ldflags=("-lcublas",))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# (60, 1024, 1024) - b60n1024 variant 6
# ---------------------------------------------------------------------------

_CPP_SOURCE_B60N1024 = r"""
#include <torch/extension.h>

void cholesky_b60n1024_prepare();
at::Tensor cholesky_b60n1024(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b60n1024_prepare,
        "Configure the staged 60x1024 Cholesky kernels");
  m.def("run", &cholesky_b60n1024, "Batched 60x1024 Cholesky");
}
"""

_CUDA_SOURCE_B60N1024 = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 60;
constexpr int kN = 1024;
constexpr int kOuter = 128;
constexpr int kMicro = 64;
constexpr int kPanelCount = kN / kOuter;
constexpr int kMicroCount = kN / kMicro;
constexpr int kWidth = 4;
constexpr int kThreads = 256;
// The tuned configuration reserves the first 8 KiB of the factor block's
// dynamic allocation and requests 76 KiB in total; both are kept exactly as
// they were measured on the B200.
constexpr int kFactorReservedBytes = 8192;
constexpr int kFactorBytes = 76 * 1024;
constexpr int kSolveBytes =
    static_cast<int>(sizeof(float)) *
    (32 * (kOuter + 1) + kMicro * (kOuter + kWidth));
static_assert(kSolveBytes == 50304);
static_assert(
    kFactorReservedBytes +
        static_cast<int>(sizeof(float)) *
            (kOuter * (kOuter + 1) + kOuter) <= kFactorBytes);

__device__ __forceinline__ float load_global(const float* pointer) {
  return __ldcg(pointer);
}

__device__ __forceinline__ void store_global(float* pointer, float value) {
  __stcg(pointer, value);
}

__device__ __forceinline__ void root_pair(
    float value, float& diagonal, float& inverse) {
  diagonal = __fsqrt_rn(value);
  inverse = __fdiv_rn(1.0f, diagonal);
}

__device__ __forceinline__ float& tile_at(
    float* tile, int row, int column) {
  return tile[row * (kOuter + 1) + column];
}

// Unblocked 32x32 right-looking factor executed by warp 0.
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
        root_pair(tile_at(tile, column, column), diagonal, inverse);
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

// Shared-tile triangular solve, one four-lane subgroup per row.
template <int Rows, int Columns>
__device__ __forceinline__ void local_trsm(
    float* tile, const float* inverse_diagonal,
    int row_begin, int column_begin) {
  const int lane = static_cast<int>(threadIdx.x) & (kWidth - 1);
  const int row_index = static_cast<int>(threadIdx.x) / kWidth;
  if (row_index < Rows) {
    const int row = row_begin + row_index;
#pragma unroll 1
    for (int local_column = 0; local_column < Columns; ++local_column) {
      const int column = column_begin + local_column;
      float partial = 0.0f;
#pragma unroll 4
      for (int k = lane; k < local_column; k += kWidth) {
        partial = fmaf(
            tile_at(tile, row, column_begin + k),
            tile_at(tile, column, column_begin + k), partial);
      }
      for (int offset = kWidth / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(
            0xffffffffu, partial, offset, kWidth);
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

// Symmetric rank-K update of one diagonal block inside the shared tile.
template <int Size, int K>
__device__ __forceinline__ void local_update(
    float* tile, int target, int panel) {
  constexpr int kElements = Size * Size;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kElements; linear += static_cast<int>(blockDim.x)) {
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

// Four 32-column factors linked by subgroup solves and FP32 updates.
__device__ __forceinline__ void factor_local(
    float* tile, float* inverse_diagonal) {
  potf2_32(tile, inverse_diagonal, 0);
  local_trsm<32, 32>(tile, inverse_diagonal, 32, 0);
  local_update<32, 32>(tile, 32, 0);
  potf2_32(tile, inverse_diagonal, 32);
  local_trsm<64, 64>(tile, inverse_diagonal, 64, 0);
  local_update<64, 64>(tile, 64, 0);
  potf2_32(tile, inverse_diagonal, 64);
  local_trsm<32, 32>(tile, inverse_diagonal, 96, 64);
  local_update<32, 32>(tile, 96, 64);
  potf2_32(tile, inverse_diagonal, 96);
}

__device__ __forceinline__ void factor_global(
    float* matrix, int begin, float* work) {
  float* tile = work;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kOuter * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    tile_at(tile, row, column) =
        column <= row
            ? load_global(matrix + (begin + row) * kN + begin + column)
            : 0.0f;
  }
  __syncthreads();
  float* inverse_diagonal = tile + kOuter * (kOuter + 1);
  factor_local(tile, inverse_diagonal);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kOuter * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    if (column <= row) {
      store_global(
          matrix + (begin + row) * kN + begin + column,
          tile_at(tile, row, column));
    }
  }
  __syncthreads();
}

// One solved column of a 32-wide right-hand-side block held in registers.
template <int Block, int LocalColumn, int RegisterCount>
__device__ __forceinline__ void trsm_column(
    float (&values)[RegisterCount], const float* diagonal,
    const float* panel, int row, int lane) {
  constexpr int kDiagonalLd = kOuter + 1;
  constexpr int kPanelLd = kOuter + kWidth;
  constexpr int kBlockBegin = Block * 32;
  constexpr int kColumn = kBlockBegin + LocalColumn;
  constexpr int kOwner = LocalColumn & (kWidth - 1);
  constexpr int kOwnerSlot = LocalColumn / kWidth;
  static_assert(RegisterCount == 32 / kWidth);
  float partial = 0.0f;
#pragma unroll 4
  for (int k = lane; k < kBlockBegin; k += kWidth) {
    partial = fmaf(
        panel[row * kPanelLd + k],
        diagonal[LocalColumn * kDiagonalLd + k], partial);
  }
#pragma unroll
  for (int slot = 0; slot < RegisterCount; ++slot) {
    const int local_k = lane + slot * kWidth;
    if (local_k < LocalColumn) {
      partial = fmaf(
          values[slot],
          diagonal[LocalColumn * kDiagonalLd + kBlockBegin + local_k],
          partial);
    }
  }
#pragma unroll
  for (int offset = kWidth / 2; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffffu, partial, offset, kWidth);
  }
  const float owned_rhs = values[kOwnerSlot];
  const float rhs = __shfl_sync(0xffffffffu, owned_rhs, kOwner, kWidth);
  float solved = 0.0f;
  if (lane == 0) {
    solved =
        (rhs - partial) / diagonal[LocalColumn * kDiagonalLd + kColumn];
  }
  solved = __shfl_sync(0xffffffffu, solved, 0, kWidth);
  if (lane == kOwner) {
    values[kOwnerSlot] = solved;
  }
}

// Stages 32 diagonal rows at a time so the whole solve fits 50,304 bytes.
template <int Block>
__device__ __forceinline__ void trsm_block(
    float* matrix, int panel_begin, float* diagonal, float* panel) {
  constexpr int kDiagonalLd = kOuter + 1;
  constexpr int kPanelLd = kOuter + kWidth;
  constexpr int kBlockBegin = Block * 32;
  constexpr int kRegisterCount = 32 / kWidth;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < 32 * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int local_row = linear / kOuter;
    const int column = linear % kOuter;
    const int matrix_row = kBlockBegin + local_row;
    diagonal[local_row * kDiagonalLd + column] =
        column <= matrix_row
            ? load_global(
                  matrix + (panel_begin + matrix_row) * kN +
                  panel_begin + column)
            : 0.0f;
  }
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) & (kWidth - 1);
  const int row = static_cast<int>(threadIdx.x) / kWidth;
  if (row < kMicro) {
    float values[kRegisterCount];
#pragma unroll
    for (int slot = 0; slot < kRegisterCount; ++slot) {
      values[slot] =
          panel[row * kPanelLd + kBlockBegin + lane + slot * kWidth];
    }
#define TRSM_COLUMN(COLUMN)                                       \
    trsm_column<Block, COLUMN>(values, diagonal, panel, row, lane)
    TRSM_COLUMN(0);
    TRSM_COLUMN(1);
    TRSM_COLUMN(2);
    TRSM_COLUMN(3);
    TRSM_COLUMN(4);
    TRSM_COLUMN(5);
    TRSM_COLUMN(6);
    TRSM_COLUMN(7);
    TRSM_COLUMN(8);
    TRSM_COLUMN(9);
    TRSM_COLUMN(10);
    TRSM_COLUMN(11);
    TRSM_COLUMN(12);
    TRSM_COLUMN(13);
    TRSM_COLUMN(14);
    TRSM_COLUMN(15);
    TRSM_COLUMN(16);
    TRSM_COLUMN(17);
    TRSM_COLUMN(18);
    TRSM_COLUMN(19);
    TRSM_COLUMN(20);
    TRSM_COLUMN(21);
    TRSM_COLUMN(22);
    TRSM_COLUMN(23);
    TRSM_COLUMN(24);
    TRSM_COLUMN(25);
    TRSM_COLUMN(26);
    TRSM_COLUMN(27);
    TRSM_COLUMN(28);
    TRSM_COLUMN(29);
    TRSM_COLUMN(30);
    TRSM_COLUMN(31);
#undef TRSM_COLUMN
#pragma unroll
    for (int slot = 0; slot < kRegisterCount; ++slot) {
      panel[row * kPanelLd + kBlockBegin + lane + slot * kWidth] =
          values[slot];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void trsm_global(
    float* matrix, int row_begin, int panel_begin, float* work) {
  constexpr int kDiagonalLd = kOuter + 1;
  constexpr int kPanelLd = kOuter + kWidth;
  float* diagonal = work;
  float* panel = diagonal + 32 * kDiagonalLd;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    panel[row * kPanelLd + column] = load_global(
        matrix + (row_begin + row) * kN + panel_begin + column);
  }
  trsm_block<0>(matrix, panel_begin, diagonal, panel);
  trsm_block<1>(matrix, panel_begin, diagonal, panel);
  trsm_block<2>(matrix, panel_begin, diagonal, panel);
  trsm_block<3>(matrix, panel_begin, diagonal, panel);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    store_global(
        matrix + (row_begin + row) * kN + panel_begin + column,
        panel[row * kPanelLd + column]);
  }
  __syncthreads();
}

__global__ __launch_bounds__(kThreads)
void copy_kernel(
    const float* __restrict__ input, float* __restrict__ output) {
  constexpr int kCtasPerMatrix = 16;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / kCtasPerMatrix;
  const int rank = static_cast<int>(blockIdx.x) % kCtasPerMatrix;
  const int64_t base = static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear = rank * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
       linear < kN * kN;
       linear += kCtasPerMatrix * static_cast<int>(blockDim.x)) {
    store_global(output + base + linear, input[base + linear]);
  }
}

// The trailing GEMM writes the full square, so restore exact upper zeros.
__global__ __launch_bounds__(kThreads)
void zero_upper_kernel(float* __restrict__ output) {
  constexpr int kCtasPerMatrix = 8;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / kCtasPerMatrix;
  const int rank = static_cast<int>(blockIdx.x) % kCtasPerMatrix;
  const int64_t base = static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear = rank * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
       linear < kN * kN;
       linear += kCtasPerMatrix * static_cast<int>(blockDim.x)) {
    const int row = linear / kN;
    const int column = linear % kN;
    if (column > row) {
      store_global(output + base + linear, 0.0f);
    }
  }
}

__global__ __launch_bounds__(kThreads)
void factor_kernel(float* __restrict__ output, int panel) {
  extern __shared__ __align__(16) unsigned char dynamic_bytes[];
  float* work =
      reinterpret_cast<float*>(dynamic_bytes + kFactorReservedBytes);
  const int matrix_index = static_cast<int>(blockIdx.x);
  float* matrix = output + static_cast<int64_t>(matrix_index) * kN * kN;
  factor_global(matrix, panel * kOuter, work);
}

__global__ __launch_bounds__(kThreads)
void solve_kernel(
    float* __restrict__ output, int panel, int remaining) {
  extern __shared__ __align__(16) unsigned char dynamic_bytes[];
  const int matrix_index = static_cast<int>(blockIdx.x) / remaining;
  const int row_index = static_cast<int>(blockIdx.x) % remaining;
  float* matrix = output + static_cast<int64_t>(matrix_index) * kN * kN;
  trsm_global(
      matrix, (panel * 2 + 2 + row_index) * kMicro, panel * kOuter,
      reinterpret_cast<float*>(dynamic_bytes));
}

void check_cublas(cublasStatus_t status, const char* role) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      role, " failed with cuBLAS status ", static_cast<int>(status));
}

template <typename Kernel>
void configure_kernel(Kernel kernel, int dynamic_bytes) {
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);
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

// One fast-TF32 batched GEMM over the whole trailing square.
void launch_blas_update(cublasHandle_t handle, float* output, int panel) {
  const int begin = (panel + 1) * kOuter;
  const int remaining = kN - begin;
  const int panel_begin = panel * kOuter;
  float* panel_pointer = output + begin * kN + panel_begin;
  float* destination = output + begin * kN + begin;
  const float alpha = -1.0f;
  const float beta = 1.0f;
  constexpr long long kMatrixStride = static_cast<long long>(kN) * kN;
  check_cublas(
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          remaining, remaining, kOuter,
          &alpha,
          panel_pointer, CUDA_R_32F, kN, kMatrixStride,
          panel_pointer, CUDA_R_32F, kN, kMatrixStride,
          &beta,
          destination, CUDA_R_32F, kN, kMatrixStride,
          kBatch,
          CUBLAS_COMPUTE_32F_FAST_TF32,
          CUBLAS_GEMM_DEFAULT),
      "staged batched trailing GEMM");
}

}  // namespace

void cholesky_b60n1024_prepare() {
  configure_kernel(factor_kernel, kFactorBytes);
  configure_kernel(solve_kernel, kSolveBytes);
}

at::Tensor cholesky_b60n1024(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (60, 1024, 1024)");
  auto output = at::empty_like(data);
  c10::cuda::CUDAGuard device_guard(data.device());
  const float* input = data.data_ptr<float>();
  float* result = output.data_ptr<float>();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  cudaLaunchConfig_t copy_config{};
  copy_config.gridDim = dim3(kBatch * 16, 1, 1);
  copy_config.blockDim = dim3(kThreads, 1, 1);
  cudaLaunchKernelEx(&copy_config, copy_kernel, input, result);

  for (int panel = 0; panel < kPanelCount; ++panel) {
    cudaLaunchConfig_t factor_config{};
    factor_config.gridDim = dim3(kBatch, 1, 1);
    factor_config.blockDim = dim3(kThreads, 1, 1);
    factor_config.dynamicSmemBytes = kFactorBytes;
    cudaLaunchKernelEx(&factor_config, factor_kernel, result, panel);

    const int remaining = kMicroCount - panel * 2 - 2;
    if (remaining == 0) {
      continue;
    }
    cudaLaunchConfig_t solve_config{};
    solve_config.gridDim = dim3(kBatch * remaining, 1, 1);
    solve_config.blockDim = dim3(kThreads, 1, 1);
    solve_config.dynamicSmemBytes = kSolveBytes;
    cudaLaunchKernelEx(
        &solve_config, solve_kernel, result, panel, remaining);

    launch_blas_update(handle, result, panel);
  }

  cudaLaunchConfig_t zero_config{};
  zero_config.gridDim = dim3(kBatch * 8, 1, 1);
  zero_config.blockDim = dim3(kThreads, 1, 1);
  cudaLaunchKernelEx(&zero_config, zero_upper_kernel, result);

  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return output;
}
"""


@lru_cache(maxsize=1)
def _module_b60n1024():
    module = _build(
        "cholesky_b60n1024", _CPP_SOURCE_B60N1024, _CUDA_SOURCE_B60N1024,
        extra_cuda_flags=("-DNDEBUG", "--restrict"),
        extra_ldflags=("-lcublas",))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# (8, 2048, 2048) - b8n2048 variant 5
# ---------------------------------------------------------------------------

_CPP_SOURCE_B8N2048 = r"""
#include <torch/extension.h>

void cholesky_b8n2048_prepare();
at::Tensor cholesky_b8n2048(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b8n2048_prepare,
        "Configure the left-looking 8x2048 Cholesky kernels");
  m.def("run", &cholesky_b8n2048, "Batched 8x2048 Cholesky");
}
"""

_CUDA_SOURCE_B8N2048 = r"""
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
constexpr int kFactorThreads = 256;
constexpr int kSolveThreads = 256;
constexpr int kFactorBytes =
    static_cast<int>(sizeof(float)) *
    (kLeaf * (kLeaf + 1) + kLeaf);
constexpr int kSolveBytes =
    static_cast<int>(sizeof(float)) *
    (32 * (kLeaf + 1) + kRowTile * (kLeaf + 4));
constexpr int64_t kMatrixStride = static_cast<int64_t>(kN) * kN;

static_assert(kFactorBytes == 66560);
static_assert(kSolveBytes == 50304);

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

template <int Begin>
void launch_factor(float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch, 1, 1);
  config.blockDim = dim3(kFactorThreads, 1, 1);
  config.dynamicSmemBytes = kFactorBytes;
  cudaLaunchKernelEx(&config, factor_kernel<Begin>, output);
}

template <int Begin, int Rows>
void launch_trsm(float* output) {
  static_assert(Rows >= 0 && Rows % kRowTile == 0);
  if constexpr (Rows > 0) {
    constexpr int row_tiles = Rows / kRowTile;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kBatch * row_tiles, 1, 1);
    config.blockDim = dim3(kSolveThreads, 1, 1);
    config.dynamicSmemBytes = kSolveBytes;
    cudaLaunchKernelEx(
        &config, solve_kernel, output, Begin,
        Begin + kLeaf, row_tiles);
  }
}

// One fast-TF32 strided-batched GEMM folds the whole factored history
// into the next 128-column panel.
void launch_gemm_update(
    cublasHandle_t handle, float* output,
    int target_row, int target_column,
    int rows, int columns, int panel_begin, int rank) {
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
          CUBLAS_COMPUTE_32F_FAST_TF32,
          CUBLAS_GEMM_DEFAULT),
      "strided batched update");
}

// Left-looking 128-column panel: apply the accumulated history, factor
// the 128x128 leaf in shared memory, then solve the column below it.
template <int Begin>
void left_panel(cublasHandle_t handle, float* output) {
  static_assert(Begin % kLeaf == 0 && Begin >= 0 && Begin < kN);
  if constexpr (Begin > 0) {
    launch_gemm_update(
        handle, output, Begin, Begin, kN - Begin, kLeaf, 0, Begin);
  }
  launch_factor<Begin>(output);
  launch_trsm<Begin, kN - Begin - kLeaf>(output);
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

void launch_all(const float* input, float* output) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle);
  launch_copy(input, output);
  left_panel<0>(handle, output);
  left_panel<128>(handle, output);
  left_panel<256>(handle, output);
  left_panel<384>(handle, output);
  left_panel<512>(handle, output);
  left_panel<640>(handle, output);
  left_panel<768>(handle, output);
  left_panel<896>(handle, output);
  left_panel<1024>(handle, output);
  left_panel<1152>(handle, output);
  left_panel<1280>(handle, output);
  left_panel<1408>(handle, output);
  left_panel<1536>(handle, output);
  left_panel<1664>(handle, output);
  left_panel<1792>(handle, output);
  left_panel<1920>(handle, output);
  launch_zero_upper(output);
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

void configure_all_factors() {
#define B8N2048_CONFIG_FACTOR(BEGIN)                         \
  configure_dynamic(factor_kernel<BEGIN>, kFactorBytes)
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

}  // namespace

void cholesky_b8n2048_prepare() {
  configure_all_factors();
  configure_dynamic(solve_kernel, kSolveBytes);
}

at::Tensor cholesky_b8n2048(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (8, 2048, 2048)");
  c10::cuda::CUDAGuard device_guard(data.device());
  auto output = at::empty_like(data);
  launch_all(data.data_ptr<float>(), output.data_ptr<float>());
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return output;
}
"""


@lru_cache(maxsize=1)
def _module_b8n2048():
    module = _build(
        "cholesky_b8n2048", _CPP_SOURCE_B8N2048, _CUDA_SOURCE_B8N2048,
        extra_cuda_flags=(
            "-DNDEBUG", "--restrict",
            "-Xptxas=--allow-expensive-optimizations=true"),
        extra_ldflags=("-lcublas",))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# (1, 8192, 8192) - b1n8192 variant 8
# ---------------------------------------------------------------------------

_CPP_SOURCE_B1N8192 = r"""
#include <torch/extension.h>

void cholesky_b1n8192_prepare();
at::Tensor cholesky_b1n8192(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b1n8192_prepare,
        "Configure the fused 8192 Cholesky kernel");
  m.def("run", &cholesky_b1n8192, "Single 8192 Cholesky");
}
"""

_CUDA_SOURCE_B1N8192 = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kN = 8192;
constexpr int kNb = 512;
constexpr int kMicro = 64;
constexpr int kTileLd = kMicro + 1;
constexpr int kPanelLd = 9;
constexpr int kThreads = 256;
constexpr int kConsumerSplit = 2;
constexpr int kFactorBytes =
    static_cast<int>(sizeof(float)) *
    (2 * kMicro * kTileLd + kMicro + kMicro * kPanelLd + 32 * 32);
static_assert(kFactorBytes == 39936);

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

__device__ __forceinline__ float& tile_at(
    float* tile, int row, int column) {
  return tile[row * kTileLd + column];
}

// Eight-column redundant-corner factorization. Four threads share
// each row solve and split its rank-8 trailing update.
// corner_sm and inverse are in shared memory to cut register pressure.
__device__ __forceinline__ void factor_wide(
    float* __restrict__ tile,
    float* __restrict__ inverse_diagonal,
    float* __restrict__ panel,
    float* __restrict__ corner_sm) {
  constexpr int kGroup = 8;
  const int thread = static_cast<int>(threadIdx.x);
  const int row_index = thread >> 2;
  const int quarter = thread & 3;
  float* inverse_sm = corner_sm + kGroup * kGroup;
#pragma unroll 1
  for (int base = 0; base < kMicro; base += kGroup) {
    if (thread < kGroup) {
#pragma unroll
      for (int i = thread; i < kGroup; ++i) {
        corner_sm[i * kGroup + thread] =
            tile_at(tile, base + i, base + thread);
      }
    }
    __syncthreads();
    if (thread == 0) {
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        const float diagonal =
            __fsqrt_rn(corner_sm[j * kGroup + j]);
        const float inv = __fdiv_rn(1.0f, diagonal);
        corner_sm[j * kGroup + j] = diagonal;
        inverse_sm[j] = inv;
#pragma unroll
        for (int i = j + 1; i < kGroup; ++i) {
          corner_sm[i * kGroup + j] *= inv;
        }
#pragma unroll
        for (int i = j + 1; i < kGroup; ++i) {
#pragma unroll
          for (int target = j + 1; target <= i; ++target) {
            corner_sm[i * kGroup + target] = fmaf(
                -corner_sm[i * kGroup + j],
                corner_sm[target * kGroup + j],
                corner_sm[i * kGroup + target]);
          }
        }
      }
    }
    __syncthreads();
    if (thread < kGroup) {
      inverse_diagonal[base + thread] = inverse_sm[thread];
#pragma unroll
      for (int i = thread; i < kGroup; ++i) {
        tile_at(tile, base + i, base + thread) =
            corner_sm[i * kGroup + thread];
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
          value = fmaf(
              -solved[i], corner_sm[j * kGroup + i], value);
        }
        solved[j] = value * inverse_sm[j];
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

// Invert the two 32-wide diagonal blocks, then combine them into the
// 64-wide triangular inverse the consumers apply.
__device__ __forceinline__ void build_inverse(
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
  if (warp < kMicro / 32) {
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
  for (int linear = thread; linear < 32 * 32;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 5;
    const int column = linear & 31;
    float partial = 0.0f;
#pragma unroll 4
    for (int k = column; k < 32; ++k) {
      partial = fmaf(
          tile[(32 + row) * kTileLd + k],
          tinv[k * kTileLd + column], partial);
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
          tinv[(32 + row) * kTileLd + 32 + k],
          mid[k * 32 + column], partial);
    }
    tinv[(32 + row) * kTileLd + column] = -partial;
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

__device__ __forceinline__ void load_x_tile(
    float* x_tile, const float* output, int begin, int tile_index) {
  const int row_begin = begin + kMicro + tile_index * kMicro;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kMicro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kMicro;
    const int column = linear & (kMicro - 1);
    x_tile[row * kTileLd + column] = load_global(
        output + matrix_index(row_begin + row, begin + column));
  }
}

// Each 64x64 application is split across two CTAs. Both halves use all
// eight warps: four 16-row warp bands by two 16-column stripes.
__device__ __forceinline__ void apply_tile_half(
    const float* __restrict__ x_tile,
    const float* __restrict__ t_tile,
    float* __restrict__ output, int begin, int tile_index,
    int half) {
  constexpr int kStripe = 16;
  constexpr int kThreadColumns = 2;
  const int row_begin = begin + kMicro + tile_index * kMicro;
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
          (warp_row * 16 + lane_row + row * 4) * kTileLd + k];
    }
#pragma unroll
    for (int column = 0; column < kThreadColumns; ++column) {
      const int t_row =
          warp_column * kStripe + lane_column + column * 8;
      right[column] = t_tile[t_row * kTileLd + k];
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

// CTA 0 factors the 64-wide micro block and publishes its inverse;
// every other CTA preloads its X tile, waits on the release flag, and
// applies the inverse to one half of one 64x64 tile.
__global__ __launch_bounds__(kThreads)
void fused_micro_kernel(
    float* __restrict__ output, int begin,
    float* __restrict__ t_inv, int* __restrict__ flags) {
  extern __shared__ __align__(16) float dynamic_floats[];
  const int tiles = (kN - begin - kMicro) / kMicro;
  int* flag = flags + begin / kMicro;
  if (blockIdx.x == 0) {
    float* tile = dynamic_floats;
    float* inverse_diagonal = tile + kMicro * kTileLd;
    float* panel = inverse_diagonal + kMicro;
    float* tinv = panel + kMicro * kPanelLd;
    float* mid = tinv + kMicro * kTileLd;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kMicro * kMicro;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear / kMicro;
      const int column = linear & (kMicro - 1);
      tile_at(tile, row, column) =
          column <= row
              ? load_global(
                    output +
                    matrix_index(begin + row, begin + column))
              : 0.0f;
    }
    __syncthreads();
    factor_wide(tile, inverse_diagonal, panel, mid);
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kMicro * kMicro;
         linear += static_cast<int>(blockDim.x)) {
      const int row = linear / kMicro;
      const int column = linear & (kMicro - 1);
      if (column <= row) {
        store_global(
            output + matrix_index(begin + row, begin + column),
            tile_at(tile, row, column));
      }
    }
    if (tiles > 0) {
      build_inverse(tile, inverse_diagonal, tinv, mid);
      for (int linear = static_cast<int>(threadIdx.x);
           linear < kMicro * kMicro;
           linear += static_cast<int>(blockDim.x)) {
        const int row = linear / kMicro;
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
    const int consumer = static_cast<int>(blockIdx.x) - 1;
    const int consumer_count = static_cast<int>(gridDim.x) - 1;
    const int part = consumer % kConsumerSplit;
    const int consumer_stride = consumer_count / kConsumerSplit;
    int tile_index = consumer / kConsumerSplit;
    load_x_tile(x_tile, output, begin, tile_index);
    if (threadIdx.x == 0) {
      while (poll_flag(flag) == 0) {
        __nanosleep(64);
      }
      acquire_fence();
    }
    __syncthreads();
    constexpr int kRowsPerConsumer = kMicro / kConsumerSplit;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kRowsPerConsumer * kMicro;
         linear += static_cast<int>(blockDim.x)) {
      const int local_row = linear / kMicro;
      const int row = part * kRowsPerConsumer + local_row;
      const int column = linear & (kMicro - 1);
      t_tile[row * kTileLd + column] =
          load_global(t_inv + row * kMicro + column);
    }
    __syncthreads();
    while (true) {
      apply_tile_half(
          x_tile, t_tile, output, begin, tile_index, part);
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

__global__ __launch_bounds__(256)
void zero_wedges_kernel(float* __restrict__ output) {
  constexpr int ctas_per_block = 8;
  constexpr int shift = 9;
  static_assert((1 << shift) == kNb);
  const int block = static_cast<int>(blockIdx.x) / ctas_per_block;
  const int rank = static_cast<int>(blockIdx.x) % ctas_per_block;
  const int base = block * kNb;
  constexpr int64_t elements = static_cast<int64_t>(kNb) * kNb;
  for (int64_t linear =
           static_cast<int64_t>(rank) * blockDim.x + threadIdx.x;
       linear < elements;
       linear +=
       static_cast<int64_t>(ctas_per_block) * blockDim.x) {
    const int row = static_cast<int>(linear >> shift);
    const int column = static_cast<int>(linear & (kNb - 1));
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

void gemm_history(
    cublasHandle_t handle, float* output, int64_t panel_begin) {
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const int columns = static_cast<int>(kN - panel_begin);
  const int history = static_cast<int>(panel_begin);
  const float* panel_rows = output + panel_begin * kN;
  float* destination = output + panel_begin * kN + panel_begin;
  check_cublas(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          kNb, columns, history,
          &alpha,
          panel_rows, CUDA_R_32F, kN,
          panel_rows, CUDA_R_32F, kN,
          &beta,
          destination, CUDA_R_32F, kN,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "panel history GEMM");
}

void gemm_inner(
    cublasHandle_t handle, float* output,
    int64_t panel_begin, int64_t micro_begin) {
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
          kMicro, columns, history,
          &alpha,
          micro_rows, CUDA_R_32F, kN,
          micro_rows, CUDA_R_32F, kN,
          &beta,
          destination, CUDA_R_32F, kN,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "micro history GEMM");
}

void launch_copy(const float* input, float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(512, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, copy_lower_kernel, input, output);
}

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
        &active, fused_micro_kernel, kThreads, kFactorBytes);
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

void launch_fused_micro(
    float* output, int begin, float* t_inv, int* flags) {
  const int tiles = (kN - begin - kMicro) / kMicro;
  const int limit = fused_micro_grid_limit();
  const int jobs = tiles * kConsumerSplit;
  int consumers = jobs < limit - 1 ? jobs : limit - 1;
  consumers -= consumers % kConsumerSplit;
  const int grid = 1 + consumers;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kFactorBytes;
  cudaLaunchKernelEx(
      &config, fused_micro_kernel, output, begin, t_inv, flags);
}

void launch_wedges(float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((kN / kNb) * 8, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, zero_wedges_kernel, output);
}

// Left-looking 512-column panels; inside each panel the trailing
// history is folded in before every fused 64-wide micro block.
void launch_staged(
    float* output, const float* input, float* t_inv, int* flags) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuard guard(handle);
  launch_copy(input, output);
  for (int64_t panel = 0; panel < kN; panel += kNb) {
    if (panel > 0) {
      gemm_history(handle, output, panel);
    }
    for (int64_t micro = panel; micro < panel + kNb;
         micro += kMicro) {
      if (micro > panel) {
        gemm_inner(handle, output, panel, micro);
      }
      launch_fused_micro(
          output, static_cast<int>(micro), t_inv, flags);
    }
  }
  launch_wedges(output);
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

}  // namespace

void cholesky_b1n8192_prepare() {
  configure_dynamic(fused_micro_kernel, kFactorBytes);
  TORCH_CHECK(
      fused_micro_grid_limit() >= 1 + kConsumerSplit,
      "fused micro kernel needs a consumer CTA");
}

at::Tensor cholesky_b1n8192(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() &&
                  data.scalar_type() == at::kFloat,
              "input must be a contiguous float32 CUDA tensor");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == 1 &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (1, 8192, 8192)");
  c10::cuda::CUDAGuard device_guard(data.device());
  auto output = at::empty_like(data);
  at::Tensor t_inv = at::empty(
      {static_cast<int64_t>(kMicro) * kMicro}, data.options());
  at::Tensor flags = at::zeros(
      {kN / kMicro}, data.options().dtype(at::kInt));
  launch_staged(
      output.data_ptr<float>(), data.data_ptr<float>(),
      t_inv.data_ptr<float>(), flags.data_ptr<int>());
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
  return output;
}
"""


@lru_cache(maxsize=1)
def _module_b1n8192():
    module = _build(
        "cholesky_b1n8192", _CPP_SOURCE_B1N8192, _CUDA_SOURCE_B1N8192,
        extra_cuda_flags=("-DNDEBUG", "--restrict"),
        extra_ldflags=("-lcublas",))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# (1, 16384, 16384) - b1n16384 variant 0
# (1, 32768, 32768) - b1n32768 variant 14
# ---------------------------------------------------------------------------

_CPP_SOURCE_B1N16384_B1N32768 = r"""
#include <torch/extension.h>

void cholesky_b1n16384_b1n32768_prepare();
at::Tensor cholesky_b1n16384_b1n32768(const at::Tensor& data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b1n16384_b1n32768_prepare,
        "Configure the large single-matrix Cholesky kernels");
  m.def("run", &cholesky_b1n16384_b1n32768,
        "Single 16384/32768 Cholesky");
}
"""

_CUDA_SOURCE_B1N16384_B1N32768 = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kMicro = 128;
constexpr int kTileLd = kMicro + 1;
constexpr int kPanelLd = 9;
constexpr int kOuter = 1024;
constexpr int kFactorBytes =
    static_cast<int>(sizeof(float)) *
    (kMicro * kTileLd + kMicro + kMicro * kPanelLd +
     kMicro * kTileLd + 64 * 64);
static_assert(kFactorBytes == 153600);

template <int N>
__device__ __forceinline__ int64_t matrix_index(
    int row, int column) {
  return static_cast<int64_t>(row) * N + column;
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

// Redundant 8x8 corner factors avoid warp-serial communication. Four
// threads share each remaining row and split its rank-8 update.
__device__ __forceinline__ void factor_wide_b1n16384_b1n32768(
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

// Construct the dense inverse of the lower 128x128 factor. Exact zeros
// in its strict upper triangle make the later dense GEMM triangular.
__device__ __forceinline__ void build_inverse_b1n16384_b1n32768(
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

template <int N>
__global__ __launch_bounds__(512)
void factor128_kernel_b1n16384_b1n32768(
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
                  output +
                  matrix_index<N>(begin + row, begin + column))
            : 0.0f;
  }
  __syncthreads();
  factor_wide_b1n16384_b1n32768(
      tile, inverse_diagonal, panel);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kMicro;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 7;
    const int column = linear & (kMicro - 1);
    if (column <= row) {
      store_global(
          output + matrix_index<N>(begin + row, begin + column),
          tile_at(tile, row, column));
    }
  }
  if (begin + kMicro < N) {
    build_inverse_b1n16384_b1n32768(
        tile, inverse_diagonal, tinv, mid);
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

template <int N>
__global__ __launch_bounds__(256)
void copy_back_kernel_b1n16384_b1n32768(
    float* __restrict__ output,
    const float* __restrict__ scratch, int begin) {
  constexpr int kQuadsPerRow = kMicro / 4;
  const int rows = N - begin - kMicro;
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
            scratch + static_cast<int64_t>(row) * kMicro + column));
    __stcg(
        reinterpret_cast<float4*>(
            output +
            matrix_index<N>(begin + kMicro + row, begin + column)),
        value);
  }
}

template <int N>
__global__ __launch_bounds__(256)
void copy_lower_kernel_b1n16384_b1n32768(
    const float* __restrict__ input, float* __restrict__ output) {
  constexpr int64_t kQuads = static_cast<int64_t>(N) * N / 4;
  constexpr int kQuadsPerRow = N / 4;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t quad = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       quad < kQuads; quad += stride) {
    const int row = static_cast<int>(quad / kQuadsPerRow);
    const int column =
        static_cast<int>(quad % kQuadsPerRow) * 4;
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

template <int N>
__global__ __launch_bounds__(256)
void zero_wedges_kernel_b1n16384_b1n32768(
    float* __restrict__ output) {
  constexpr int kCtasPerBlock = 8;
  const int block = static_cast<int>(blockIdx.x) / kCtasPerBlock;
  const int rank = static_cast<int>(blockIdx.x) % kCtasPerBlock;
  const int base = block * kOuter;
  constexpr int64_t kElements =
      static_cast<int64_t>(kOuter) * kOuter;
  for (int64_t linear =
           static_cast<int64_t>(rank) * blockDim.x + threadIdx.x;
       linear < kElements;
       linear +=
       static_cast<int64_t>(kCtasPerBlock) * blockDim.x) {
    const int row = static_cast<int>(linear >> 10);
    const int column = static_cast<int>(linear & (kOuter - 1));
    if (column > row) {
      store_global(
          output + matrix_index<N>(base + row, base + column), 0.0f);
    }
  }
}

void check_cublas_b1n16384_b1n32768(
    cublasStatus_t status, const char* role) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      role, " failed with cuBLAS status ", static_cast<int>(status));
}

class CublasStateGuardB1N16384B1N32768 {
 public:
  explicit CublasStateGuardB1N16384B1N32768(cublasHandle_t handle)
      : handle_(handle) {
    check_cublas_b1n16384_b1n32768(
        cublasGetMathMode(handle_, &math_mode_),
        "query cuBLAS math mode");
    check_cublas_b1n16384_b1n32768(
        cublasGetAtomicsMode(handle_, &atomics_mode_),
        "query cuBLAS atomics mode");
    check_cublas_b1n16384_b1n32768(
        cublasGetPointerMode(handle_, &pointer_mode_),
        "query cuBLAS pointer mode");
    check_cublas_b1n16384_b1n32768(
        cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH),
        "select cuBLAS math mode");
    check_cublas_b1n16384_b1n32768(
        cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_ALLOWED),
        "enable cuBLAS atomic algorithms");
    check_cublas_b1n16384_b1n32768(
        cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
        "select host cuBLAS scalars");
  }

  ~CublasStateGuardB1N16384B1N32768() {
    cublasSetPointerMode(handle_, pointer_mode_);
    cublasSetAtomicsMode(handle_, atomics_mode_);
    cublasSetMathMode(handle_, math_mode_);
  }

  CublasStateGuardB1N16384B1N32768(
      const CublasStateGuardB1N16384B1N32768&) = delete;
  CublasStateGuardB1N16384B1N32768& operator=(
      const CublasStateGuardB1N16384B1N32768&) = delete;

 private:
  cublasHandle_t handle_;
  cublasMath_t math_mode_{};
  cublasAtomicsMode_t atomics_mode_{};
  cublasPointerMode_t pointer_mode_{};
};

template <int N>
void gemm_history_b1n16384_b1n32768(
    cublasHandle_t handle, float* output, int64_t panel_begin) {
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const int columns = static_cast<int>(N - panel_begin);
  const int history = static_cast<int>(panel_begin);
  const float* panel_rows = output + panel_begin * N;
  float* destination = output + panel_begin * N + panel_begin;
  check_cublas_b1n16384_b1n32768(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          kOuter, columns, history,
          &alpha,
          panel_rows, CUDA_R_32F, N,
          panel_rows, CUDA_R_32F, N,
          &beta,
          destination, CUDA_R_32F, N,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "panel history GEMM");
}

template <int N>
void gemm_inner_b1n16384_b1n32768(
    cublasHandle_t handle, float* output, int64_t panel_begin,
    int64_t micro_begin) {
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const int columns = static_cast<int>(N - micro_begin);
  const int history = static_cast<int>(micro_begin - panel_begin);
  const float* micro_rows = output + micro_begin * N + panel_begin;
  float* destination = output + micro_begin * N + micro_begin;
  check_cublas_b1n16384_b1n32768(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          kMicro, columns, history,
          &alpha,
          micro_rows, CUDA_R_32F, N,
          micro_rows, CUDA_R_32F, N,
          &beta,
          destination, CUDA_R_32F, N,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "micro history GEMM");
}

template <int N>
void gemm_apply_b1n16384_b1n32768(
    cublasHandle_t handle, float* output, const float* t_inv,
    float* scratch, int64_t micro_begin) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int rows = static_cast<int>(N - micro_begin - kMicro);
  const float* x_rows =
      output + (micro_begin + kMicro) * N + micro_begin;
  check_cublas_b1n16384_b1n32768(
      cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          kMicro, rows, kMicro,
          &alpha,
          t_inv, CUDA_R_32F, kMicro,
          x_rows, CUDA_R_32F, N,
          &beta,
          scratch, CUDA_R_32F, kMicro,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
      "apply GEMM");
}

template <int N>
void launch_shape_b1n16384_b1n32768(
    const float* input, float* output, float* t_inv,
    float* scratch) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasStateGuardB1N16384B1N32768 guard(handle);

  cudaLaunchConfig_t copy_config{};
  copy_config.gridDim = dim3(2048, 1, 1);
  copy_config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &copy_config, copy_lower_kernel_b1n16384_b1n32768<N>,
      input, output);

  for (int64_t panel = 0; panel < N; panel += kOuter) {
    if (panel > 0) {
      gemm_history_b1n16384_b1n32768<N>(
          handle, output, panel);
    }
    for (int64_t micro = panel; micro < panel + kOuter;
         micro += kMicro) {
      if (micro > panel) {
        gemm_inner_b1n16384_b1n32768<N>(
            handle, output, panel, micro);
      }
      cudaLaunchConfig_t factor_config{};
      factor_config.gridDim = dim3(1, 1, 1);
      factor_config.blockDim = dim3(512, 1, 1);
      factor_config.dynamicSmemBytes = kFactorBytes;
      cudaLaunchKernelEx(
          &factor_config,
          factor128_kernel_b1n16384_b1n32768<N>,
          output, static_cast<int>(micro), t_inv);
      if (micro + kMicro < N) {
        gemm_apply_b1n16384_b1n32768<N>(
            handle, output, t_inv, scratch, micro);
        cudaLaunchConfig_t back_config{};
        back_config.gridDim = dim3(256, 1, 1);
        back_config.blockDim = dim3(256, 1, 1);
        cudaLaunchKernelEx(
            &back_config,
            copy_back_kernel_b1n16384_b1n32768<N>,
            output, scratch, static_cast<int>(micro));
      }
    }
  }

  cudaLaunchConfig_t wedge_config{};
  wedge_config.gridDim = dim3((N / kOuter) * 8, 1, 1);
  wedge_config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &wedge_config, zero_wedges_kernel_b1n16384_b1n32768<N>,
      output);
}

template <int N>
void configure_shape_b1n16384_b1n32768() {
  cudaError_t status = cudaFuncSetAttribute(
      factor128_kernel_b1n16384_b1n32768<N>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kFactorBytes);
  TORCH_CHECK(
      status == cudaSuccess,
      "dynamic shared-memory opt-in failed: ",
      cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      factor128_kernel_b1n16384_b1n32768<N>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(
      status == cudaSuccess,
      "shared-memory carveout failed: ", cudaGetErrorString(status));
}

template <int N>
at::Tensor run_shape_b1n16384_b1n32768(const at::Tensor& data) {
  auto output = at::empty_like(data);
  at::Tensor t_inv = at::empty({kMicro, kMicro}, data.options());
  at::Tensor scratch =
      at::empty({N - kMicro, kMicro}, data.options());
  launch_shape_b1n16384_b1n32768<N>(
      data.data_ptr<float>(), output.data_ptr<float>(),
      t_inv.data_ptr<float>(), scratch.data_ptr<float>());
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "Cholesky launch failed: ", cudaGetErrorString(status));
  return output;
}

}  // namespace

void cholesky_b1n16384_b1n32768_prepare() {
  configure_shape_b1n16384_b1n32768<16384>();
  configure_shape_b1n16384_b1n32768<32768>();
}

at::Tensor cholesky_b1n16384_b1n32768(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be CUDA");
  TORCH_CHECK(
      data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      data.dim() == 3 && data.size(0) == 1 &&
      data.size(1) == data.size(2) &&
      (data.size(1) == 16384 || data.size(1) == 32768),
      "native input must have shape (1, 16384, 16384) or "
      "(1, 32768, 32768)");
  c10::cuda::CUDAGuard device_guard(data.device());
  if (data.size(1) == 16384) {
    return run_shape_b1n16384_b1n32768<16384>(data);
  }
  return run_shape_b1n16384_b1n32768<32768>(data);
}
"""


@lru_cache(maxsize=1)
def _module_b1n16384_b1n32768():
    module = _build(
        "cholesky_b1n16384_b1n32768",
        _CPP_SOURCE_B1N16384_B1N32768,
        _CUDA_SOURCE_B1N16384_B1N32768,
        extra_cuda_flags=("-DNDEBUG", "--restrict"),
        extra_ldflags=("-lcublas",))
    module.prepare()
    return module


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SPECIALIZATIONS = {
    (4096, 32, 32): _module_b4096n32,
    (1024, 64, 64): _module_b1024n64,
    (256, 128, 128): _module_b256n128,
    # (64, 256, 256) temporarily routed to torch; re-enable by restoring
    # this entry (the b64n256 extension below is left intact).
    # (64, 256, 256): _module_b64n256,
    (16, 512, 512): _module_b16n512,
    (640, 512, 512): _module_b640n512,
    (60, 1024, 1024): _module_b60n1024,
    (8, 2048, 2048): _module_b8n2048,
    (1, 8192, 8192): _module_b1n8192,
    (1, 16384, 16384): _module_b1n16384_b1n32768,
    (1, 32768, 32768): _module_b1n16384_b1n32768,
}


def custom_kernel(data: input_t) -> output_t:
    if data.is_cuda and data.dtype == torch.float32 and data.is_contiguous():
        specialization = _SPECIALIZATIONS.get(tuple(data.shape))
        if specialization is not None:
            return specialization().run(data)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
