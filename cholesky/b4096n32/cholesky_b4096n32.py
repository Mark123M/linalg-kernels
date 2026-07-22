import os
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The Popcorn tuner replaces this exact line in temporary, untracked copies.
# Variant 3 won the full Popcorn B200 leaderboard sweep on 2026-07-20.
_DEFAULT_VARIANT = 32  # POPCORN_VARIANT
_VARIANT_COUNT = 34

_CPP_SOURCE = r"""
#include <torch/extension.h>

at::Tensor cholesky_4096x32(const at::Tensor& data,
                            int64_t variant);
void cholesky_4096x32_out(const at::Tensor& data,
                          at::Tensor out,
                          int64_t variant);
at::Tensor cholesky_4096x32_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &cholesky_4096x32, "Batched 32x32 Cholesky");
  m.def("run_out", &cholesky_4096x32_out, "Batched 32x32 Cholesky out");
  m.def("metadata", &cholesky_4096x32_metadata, "Kernel resource metadata");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBatch = 4096;
constexpr int kN = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kPreciseRoot = 0;
constexpr int kRefinedRoot = 1;
constexpr int kRawRoot = 2;
constexpr int kMetadataColumns = 19;

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (4096, 32, 32)");
}

void check_output(const at::Tensor& data, const at::Tensor& out) {
  TORCH_CHECK(out.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "output must be float32");
  TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(out.sizes() == data.sizes(), "output shape must match input");
  TORCH_CHECK(out.device() == data.device(), "output device must match input");
}

__device__ __forceinline__ float load_early_ptx(const float* address) {
  float value;
  asm volatile("ld.volatile.global.f32 %0, [%1];"
               : "=f"(value)
               : "l"(address)
               : "memory");
  return value;
}

template <int Warps, int RootMode, int TailPrefetch, int PtxPrefetchLead,
          int DotAccumulators, int ShuffleLookahead, int MinBlocks>
__global__ __launch_bounds__(Warps * 32, MinBlocks)
void crout_32_kernel(const float* __restrict__ input,
                     float* __restrict__ output) {
  static_assert(RootMode >= kPreciseRoot && RootMode <= kRawRoot);
  static_assert(TailPrefetch == 0 || TailPrefetch == 2 ||
                TailPrefetch == 4 || TailPrefetch == 6 ||
                TailPrefetch == 8);
  static_assert(PtxPrefetchLead >= 0 && PtxPrefetchLead <= 2);
  static_assert(PtxPrefetchLead == 0 || TailPrefetch == 0);
  static_assert(DotAccumulators == 1 || DotAccumulators == 2 ||
                DotAccumulators == 4);
  static_assert(ShuffleLookahead == 1 || ShuffleLookahead == 2);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int matrix = static_cast<int>(blockIdx.x) * Warps + warp;
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  // The final columns have the longest dot products, but the baseline SASS
  // cannot issue their input loads early because all 32 factor values are live.
  // Keep selected tail inputs in explicit scalar registers from kernel entry.
  // Scalars avoid a dynamically indexed local array if ptxas cannot scalarize
  // the fully unrolled outer loop.
  float input24 = 0.0f;
  float input25 = 0.0f;
  float input26 = 0.0f;
  float input27 = 0.0f;
  float input28 = 0.0f;
  float input29 = 0.0f;
  float input30 = 0.0f;
  float input31 = 0.0f;
  if constexpr (TailPrefetch >= 8) {
    input24 = a[24 * kN + lane];
    input25 = a[25 * kN + lane];
  }
  if constexpr (TailPrefetch >= 6) {
    input26 = a[26 * kN + lane];
    input27 = a[27 * kN + lane];
  }
  if constexpr (TailPrefetch >= 4) {
    input28 = a[28 * kN + lane];
    input29 = a[29 * kN + lane];
  }
  if constexpr (TailPrefetch >= 2) {
    input30 = a[30 * kN + lane];
    input31 = a[31 * kN + lane];
  }
  // Give the CUDA front end an explicit scalar dependency point before the
  // factorization. The generated SASS still decides final load scheduling, so
  // a prefetch variant is accepted only after profiling confirms early LDGs.
  if constexpr (TailPrefetch == 8) {
    asm volatile("" : "+f"(input24), "+f"(input25), "+f"(input26),
                  "+f"(input27), "+f"(input28), "+f"(input29),
                  "+f"(input30), "+f"(input31));
  } else if constexpr (TailPrefetch == 6) {
    asm volatile("" : "+f"(input26), "+f"(input27), "+f"(input28),
                  "+f"(input29), "+f"(input30), "+f"(input31));
  } else if constexpr (TailPrefetch == 4) {
    asm volatile("" : "+f"(input28), "+f"(input29),
                  "+f"(input30), "+f"(input31));
  } else if constexpr (TailPrefetch == 2) {
    asm volatile("" : "+f"(input30), "+f"(input31));
  }

  // Lane i owns row i of L. Reading A[j, i] instead of A[i, j] is valid for
  // the symmetric input and makes every warp input transaction contiguous.
  float row[kN];
#pragma unroll
  for (int k = 0; k < kN; ++k) {
    row[k] = 0.0f;
  }

#pragma unroll
  for (int j = 0; j < kN; ++j) {
    // The bounded prefetch variants issue real volatile PTX loads one or two
    // complete Cholesky iterations before columns 26--31 consume them. Unlike
    // the entry-prefetch experiment, these loads have short register lifetimes
    // and cannot be deleted or folded into the later ordinary input loads.
    if constexpr (PtxPrefetchLead != 0) {
      if (j == 26 - PtxPrefetchLead) {
        input26 = load_early_ptx(a + 26 * kN + lane);
      } else if (j == 27 - PtxPrefetchLead) {
        input27 = load_early_ptx(a + 27 * kN + lane);
      } else if (j == 28 - PtxPrefetchLead) {
        input28 = load_early_ptx(a + 28 * kN + lane);
      } else if (j == 29 - PtxPrefetchLead) {
        input29 = load_early_ptx(a + 29 * kN + lane);
      } else if (j == 30 - PtxPrefetchLead) {
        input30 = load_early_ptx(a + 30 * kN + lane);
      } else if (j == 31 - PtxPrefetchLead) {
        input31 = load_early_ptx(a + 31 * kN + lane);
      }
    }

    float value;
    if constexpr (PtxPrefetchLead != 0) {
      value = j == 26 ? input26
              : j == 27 ? input27
              : j == 28 ? input28
              : j == 29 ? input29
              : j == 30 ? input30
              : j == 31 ? input31
                        : a[j * kN + lane];
    } else if constexpr (TailPrefetch == 0) {
      value = a[j * kN + lane];
    } else if constexpr (TailPrefetch == 2) {
      value = j == 30 ? input30 : (j == 31 ? input31 : a[j * kN + lane]);
    } else if constexpr (TailPrefetch == 4) {
      value = j == 28 ? input28
              : j == 29 ? input29
              : j == 30 ? input30
              : j == 31 ? input31
                        : a[j * kN + lane];
    } else if constexpr (TailPrefetch == 6) {
      value = j == 26 ? input26
              : j == 27 ? input27
              : j == 28 ? input28
              : j == 29 ? input29
              : j == 30 ? input30
              : j == 31 ? input31
                        : a[j * kN + lane];
    } else {
      value = j == 24 ? input24
              : j == 25 ? input25
              : j == 26 ? input26
              : j == 27 ? input27
              : j == 28 ? input28
              : j == 29 ? input29
              : j == 30 ? input30
              : j == 31 ? input31
                        : a[j * kN + lane];
    }
    if constexpr (DotAccumulators == 1 && ShuffleLookahead == 1) {
#pragma unroll
      for (int k = 0; k < j; ++k) {
        const float pivot = __shfl_sync(kFullMask, row[k], j);
        value = fmaf(-row[k], pivot, value);
      }
    } else if constexpr (DotAccumulators == 2 && ShuffleLookahead == 1) {
      float accumulator0 = value;
      float accumulator1 = 0.0f;
#pragma unroll
      for (int k = 0; k < j; ++k) {
        const float pivot = __shfl_sync(kFullMask, row[k], j);
        if ((k & 1) == 0) {
          accumulator0 = fmaf(-row[k], pivot, accumulator0);
        } else {
          accumulator1 = fmaf(-row[k], pivot, accumulator1);
        }
      }
      value = accumulator0 + accumulator1;
    } else if constexpr (DotAccumulators == 4 && ShuffleLookahead == 1) {
      float accumulator0 = value;
      float accumulator1 = 0.0f;
      float accumulator2 = 0.0f;
      float accumulator3 = 0.0f;
#pragma unroll
      for (int k = 0; k < j; ++k) {
        const float pivot = __shfl_sync(kFullMask, row[k], j);
        if ((k & 3) == 0) {
          accumulator0 = fmaf(-row[k], pivot, accumulator0);
        } else if ((k & 3) == 1) {
          accumulator1 = fmaf(-row[k], pivot, accumulator1);
        } else if ((k & 3) == 2) {
          accumulator2 = fmaf(-row[k], pivot, accumulator2);
        } else {
          accumulator3 = fmaf(-row[k], pivot, accumulator3);
        }
      }
      value = (accumulator0 + accumulator1) +
              (accumulator2 + accumulator3);
    } else if constexpr (DotAccumulators == 1) {
      int k = 0;
#pragma unroll
      for (; k + 1 < j; k += 2) {
        const float pivot0 = __shfl_sync(kFullMask, row[k], j);
        const float pivot1 = __shfl_sync(kFullMask, row[k + 1], j);
        value = fmaf(-row[k], pivot0, value);
        value = fmaf(-row[k + 1], pivot1, value);
      }
      if (k < j) {
        const float pivot = __shfl_sync(kFullMask, row[k], j);
        value = fmaf(-row[k], pivot, value);
      }
    } else if constexpr (DotAccumulators == 2) {
      float accumulator0 = value;
      float accumulator1 = 0.0f;
      int k = 0;
#pragma unroll
      for (; k + 1 < j; k += 2) {
        const float pivot0 = __shfl_sync(kFullMask, row[k], j);
        const float pivot1 = __shfl_sync(kFullMask, row[k + 1], j);
        accumulator0 = fmaf(-row[k], pivot0, accumulator0);
        accumulator1 = fmaf(-row[k + 1], pivot1, accumulator1);
      }
      if (k < j) {
        const float pivot = __shfl_sync(kFullMask, row[k], j);
        accumulator0 = fmaf(-row[k], pivot, accumulator0);
      }
      value = accumulator0 + accumulator1;
    } else {
      float accumulator0 = value;
      float accumulator1 = 0.0f;
      float accumulator2 = 0.0f;
      float accumulator3 = 0.0f;
      int k = 0;
#pragma unroll
      for (; k + 1 < j; k += 2) {
        const float pivot0 = __shfl_sync(kFullMask, row[k], j);
        const float pivot1 = __shfl_sync(kFullMask, row[k + 1], j);
        if ((k & 3) == 0) {
          accumulator0 = fmaf(-row[k], pivot0, accumulator0);
          accumulator1 = fmaf(-row[k + 1], pivot1, accumulator1);
        } else {
          accumulator2 = fmaf(-row[k], pivot0, accumulator2);
          accumulator3 = fmaf(-row[k + 1], pivot1, accumulator3);
        }
      }
      if (k < j) {
        const float pivot = __shfl_sync(kFullMask, row[k], j);
        if ((k & 3) == 0) {
          accumulator0 = fmaf(-row[k], pivot, accumulator0);
        } else {
          accumulator2 = fmaf(-row[k], pivot, accumulator2);
        }
      }
      value = (accumulator0 + accumulator1) +
              (accumulator2 + accumulator3);
    }

    float inverse = 0.0f;
    if (lane == j) {
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
    }
    inverse = __shfl_sync(kFullMask, inverse, j);
    if (lane > j) {
      row[j] = value * inverse;
    }
  }

  // Each output row is 128-byte aligned. Eight explicit float4 writes retain
  // alignment while zeroing the unused upper triangle in the same kernel.
  float* out_row = result + lane * kN;
#pragma unroll
  for (int vector_index = 0; vector_index < 8; ++vector_index) {
    const int column = vector_index * 4;
    float4 values;
    values.x = column <= lane ? row[column] : 0.0f;
    values.y = column + 1 <= lane ? row[column + 1] : 0.0f;
    values.z = column + 2 <= lane ? row[column + 2] : 0.0f;
    values.w = column + 3 <= lane ? row[column + 3] : 0.0f;
    reinterpret_cast<float4*>(out_row)[vector_index] = values;
  }
}

// Pack independent matrices into power-of-two warp subdivisions. A width-16
// shuffle performs two unrelated matrix broadcasts in one hardware
// instruction; width 8 performs four. Each thread consequently owns two or
// four rows of its subdivision's matrix.
template <int Warps, int GroupWidth, int MinBlocks>
__global__ __launch_bounds__(Warps * 32, MinBlocks)
void subwarp_left_32_kernel(const float* __restrict__ input,
                            float* __restrict__ output) {
  static_assert(GroupWidth == 16 || GroupWidth == 8);
  constexpr int kMatricesPerWarp = 32 / GroupWidth;
  constexpr int kRowsPerLane = kN / GroupWidth;

  const int physical_lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int group = physical_lane / GroupWidth;
  const int lane = physical_lane & (GroupWidth - 1);
  const int physical_warp = static_cast<int>(blockIdx.x) * Warps + warp;
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
    float value[kRowsPerLane];
#pragma unroll
    for (int owned = 0; owned < kRowsPerLane; ++owned) {
      const int matrix_row = lane + owned * GroupWidth;
      value[owned] = a[j * kN + matrix_row];
    }

#pragma unroll
    for (int k = 0; k < j; ++k) {
      const int pivot_owner = j / GroupWidth;
      const int pivot_lane = j & (GroupWidth - 1);
      const float pivot = __shfl_sync(
          kFullMask, row[pivot_owner][k], pivot_lane, GroupWidth);
#pragma unroll
      for (int owned = 0; owned < kRowsPerLane; ++owned) {
        value[owned] = fmaf(-row[owned][k], pivot, value[owned]);
      }
    }

    const int diagonal_owner = j / GroupWidth;
    const int diagonal_lane = j & (GroupWidth - 1);
    float inverse = 0.0f;
    if (lane == diagonal_lane) {
      const float diagonal_value = value[diagonal_owner];
      inverse = rsqrtf(diagonal_value);
      row[diagonal_owner][j] = diagonal_value * inverse;
    }
    inverse = __shfl_sync(kFullMask, inverse, diagonal_lane, GroupWidth);
#pragma unroll
    for (int owned = 0; owned < kRowsPerLane; ++owned) {
      const int matrix_row = lane + owned * GroupWidth;
      if (matrix_row > j) {
        row[owned][j] = value[owned] * inverse;
      }
    }
  }

#pragma unroll
  for (int owned = 0; owned < kRowsPerLane; ++owned) {
    const int matrix_row = lane + owned * GroupWidth;
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

// Right-looking rank-1 Cholesky. state[owned][j] begins as A[row,j], is
// updated once per preceding factor column, and becomes L[row,j] when column
// j is normalized. The update order for every element remains increasing in
// k, matching the left-looking recurrence, while consecutive FFMAs target
// independent trailing-column registers.
template <int Warps, int GroupWidth, bool RootLookahead, int MinBlocks>
__global__ __launch_bounds__(Warps * 32, MinBlocks)
void right_looking_32_kernel(const float* __restrict__ input,
                             float* __restrict__ output) {
  static_assert(GroupWidth == 32 || GroupWidth == 16 || GroupWidth == 8);
  constexpr int kMatricesPerWarp = 32 / GroupWidth;
  constexpr int kRowsPerLane = kN / GroupWidth;

  const int physical_lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int group = physical_lane / GroupWidth;
  const int lane = physical_lane & (GroupWidth - 1);
  const int physical_warp = static_cast<int>(blockIdx.x) * Warps + warp;
  const int matrix = physical_warp * kMatricesPerWarp + group;
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  float state[kRowsPerLane][kN];
#pragma unroll
  for (int owned = 0; owned < kRowsPerLane; ++owned) {
    const int matrix_row = lane + owned * GroupWidth;
#pragma unroll
    for (int j = 0; j < kN; ++j) {
      // Symmetry converts the logical lower element A[matrix_row,j] into
      // coalesced row-major input A[j,matrix_row].
      state[owned][j] = a[j * kN + matrix_row];
    }
  }

  if constexpr (!RootLookahead) {
#pragma unroll
    for (int k = 0; k < kN; ++k) {
      const int diagonal_owner = k / GroupWidth;
      const int diagonal_lane = k & (GroupWidth - 1);
      float inverse = 0.0f;
      if (lane == diagonal_lane) {
        const float diagonal_value = state[diagonal_owner][k];
        inverse = rsqrtf(diagonal_value);
        state[diagonal_owner][k] = diagonal_value * inverse;
      }
      inverse = __shfl_sync(kFullMask, inverse, diagonal_lane, GroupWidth);
#pragma unroll
      for (int owned = 0; owned < kRowsPerLane; ++owned) {
        const int matrix_row = lane + owned * GroupWidth;
        if (matrix_row > k) {
          state[owned][k] *= inverse;
        }
      }

#pragma unroll
      for (int j = k + 1; j < kN; ++j) {
        const int pivot_owner = j / GroupWidth;
        const int pivot_lane = j & (GroupWidth - 1);
        const float pivot = __shfl_sync(
            kFullMask, state[pivot_owner][k], pivot_lane, GroupWidth);
#pragma unroll
        for (int owned = 0; owned < kRowsPerLane; ++owned) {
          state[owned][j] =
              fmaf(-state[owned][k], pivot, state[owned][j]);
        }
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
    inverse = __shfl_sync(kFullMask, inverse, 0, GroupWidth);
#pragma unroll
    for (int owned = 0; owned < kRowsPerLane; ++owned) {
      const int matrix_row = lane + owned * GroupWidth;
      if (matrix_row > 0) {
        state[owned][0] *= inverse;
      }
    }

#pragma unroll
    for (int k = 0; k < kN - 1; ++k) {
      const int next = k + 1;
      const int next_owner = next / GroupWidth;
      const int next_lane = next & (GroupWidth - 1);
      const float next_pivot = __shfl_sync(
          kFullMask, state[next_owner][k], next_lane, GroupWidth);
#pragma unroll
      for (int owned = 0; owned < kRowsPerLane; ++owned) {
        state[owned][next] =
            fmaf(-state[owned][k], next_pivot, state[owned][next]);
      }

      float next_inverse = 0.0f;
      float next_diagonal = 0.0f;
      if (lane == next_lane) {
        next_diagonal = state[next_owner][next];
        next_inverse = rsqrtf(next_diagonal);
      }

#pragma unroll
      for (int j = k + 2; j < kN; ++j) {
        const int pivot_owner = j / GroupWidth;
        const int pivot_lane = j & (GroupWidth - 1);
        const float pivot = __shfl_sync(
            kFullMask, state[pivot_owner][k], pivot_lane, GroupWidth);
#pragma unroll
        for (int owned = 0; owned < kRowsPerLane; ++owned) {
          state[owned][j] =
              fmaf(-state[owned][k], pivot, state[owned][j]);
        }
      }

      if (lane == next_lane) {
        state[next_owner][next] = next_diagonal * next_inverse;
      }
      next_inverse = __shfl_sync(
          kFullMask, next_inverse, next_lane, GroupWidth);
#pragma unroll
      for (int owned = 0; owned < kRowsPerLane; ++owned) {
        const int matrix_row = lane + owned * GroupWidth;
        if (matrix_row > next) {
          state[owned][next] *= next_inverse;
        }
      }
    }
  }

#pragma unroll
  for (int owned = 0; owned < kRowsPerLane; ++owned) {
    const int matrix_row = lane + owned * GroupWidth;
    float* out_row = result + matrix_row * kN;
#pragma unroll
    for (int vector_index = 0; vector_index < 8; ++vector_index) {
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

template <int Warps, int RootMode, int TailPrefetch, int PtxPrefetchLead,
          int DotAccumulators, int ShuffleLookahead, int MinBlocks>
void launch_typed(const float* input, float* output) {
  constexpr int threads = Warps * 32;
  constexpr int blocks = kBatch / Warps;
  crout_32_kernel<Warps, RootMode, TailPrefetch, PtxPrefetchLead,
                  DotAccumulators, ShuffleLookahead, MinBlocks>
      <<<blocks, threads>>>(input, output);
}

template <int Warps, int GroupWidth, int MinBlocks>
void launch_subwarp_left(const float* input, float* output) {
  constexpr int threads = Warps * 32;
  constexpr int matrices_per_warp = 32 / GroupWidth;
  constexpr int blocks = kBatch / (Warps * matrices_per_warp);
  static_assert(kBatch % (Warps * matrices_per_warp) == 0);
  subwarp_left_32_kernel<Warps, GroupWidth, MinBlocks>
      <<<blocks, threads>>>(input, output);
}

template <int Warps, int GroupWidth, bool RootLookahead, int MinBlocks>
void launch_right_looking(const float* input, float* output) {
  constexpr int threads = Warps * 32;
  constexpr int matrices_per_warp = 32 / GroupWidth;
  constexpr int blocks = kBatch / (Warps * matrices_per_warp);
  static_assert(kBatch % (Warps * matrices_per_warp) == 0);
  right_looking_32_kernel<Warps, GroupWidth, RootLookahead, MinBlocks>
      <<<blocks, threads>>>(input, output);
}

void launch_variant(const float* input,
                    float* output,
                    int variant) {
  switch (variant) {
    case  0: launch_typed< 1, kPreciseRoot, 0, 0, 1, 1, 32>(input, output); break;
    case  1: launch_typed< 1, kRefinedRoot, 0, 0, 1, 1, 32>(input, output); break;
    case  2: launch_typed< 2, kPreciseRoot, 0, 0, 1, 1, 16>(input, output); break;
    case  3: launch_typed< 2, kRefinedRoot, 0, 0, 1, 1, 16>(input, output); break;
    case  4: launch_typed< 4, kPreciseRoot, 0, 0, 1, 1,  8>(input, output); break;
    case  5: launch_typed< 4, kRefinedRoot, 0, 0, 1, 1,  8>(input, output); break;
    case  6: launch_typed< 8, kPreciseRoot, 0, 0, 1, 1,  4>(input, output); break;
    case  7: launch_typed< 8, kRefinedRoot, 0, 0, 1, 1,  4>(input, output); break;
    case  8: launch_typed<16, kPreciseRoot, 0, 0, 1, 1,  2>(input, output); break;
    case  9: launch_typed<16, kRefinedRoot, 0, 0, 1, 1,  2>(input, output); break;
    case 10: launch_typed< 2, kRefinedRoot, 0, 0, 1, 1, 14>(input, output); break;
    case 11: launch_typed< 2, kRefinedRoot, 2, 0, 1, 1, 14>(input, output); break;
    case 12: launch_typed< 2, kRefinedRoot, 4, 0, 1, 1, 14>(input, output); break;
    case 13: launch_typed< 2, kRefinedRoot, 6, 0, 1, 1, 14>(input, output); break;
    case 14: launch_typed< 2, kRefinedRoot, 8, 0, 1, 1, 14>(input, output); break;
    case 15: launch_typed< 2, kRawRoot,     0, 0, 1, 1, 14>(input, output); break;
    case 16: launch_typed< 2, kRawRoot,     2, 0, 1, 1, 14>(input, output); break;
    case 17: launch_typed< 2, kRawRoot,     4, 0, 1, 1, 14>(input, output); break;
    case 18: launch_typed< 2, kRawRoot,     6, 0, 1, 1, 14>(input, output); break;
    case 19: launch_typed< 2, kRawRoot,     8, 0, 1, 1, 14>(input, output); break;
    case 20: launch_typed< 2, kRawRoot,     0, 1, 1, 1, 14>(input, output); break;
    case 21: launch_typed< 2, kRawRoot,     0, 2, 1, 1, 14>(input, output); break;
    case 22: launch_typed< 2, kRawRoot,     0, 0, 2, 1, 14>(input, output); break;
    case 23: launch_typed< 2, kRawRoot,     0, 0, 4, 1, 14>(input, output); break;
    case 24: launch_typed< 2, kRawRoot,     0, 0, 1, 2, 14>(input, output); break;
    case 25: launch_typed< 2, kRawRoot,     0, 0, 2, 2, 14>(input, output); break;
    case 26: launch_typed< 2, kRawRoot,     0, 0, 4, 2, 14>(input, output); break;
    case 27: launch_right_looking<2, 32, false, 14>(input, output); break;
    case 28: launch_right_looking<2, 32, true,  14>(input, output); break;
    case 29: launch_subwarp_left<   2, 16,        7>(input, output); break;
    case 30: launch_right_looking<2, 16, false,  7>(input, output); break;
    case 31: launch_right_looking<2, 16, true,   7>(input, output); break;
    case 32: launch_subwarp_left<   2,  8,        4>(input, output); break;
    case 33: launch_right_looking<2,  8, false,  4>(input, output); break;
    default: TORCH_CHECK(false, "variant must be in [0, 33]");
  }
}

template <int Warps, int RootMode, int TailPrefetch, int PtxPrefetchLead,
          int DotAccumulators, int ShuffleLookahead, int MinBlocks>
void write_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  auto status = cudaFuncGetAttributes(
      &attributes,
      crout_32_kernel<Warps, RootMode, TailPrefetch, PtxPrefetchLead,
                      DotAccumulators, ShuffleLookahead, MinBlocks>);
  TORCH_CHECK(status == cudaSuccess,
              "cudaFuncGetAttributes failed: ", cudaGetErrorString(status));
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      crout_32_kernel<Warps, RootMode, TailPrefetch, PtxPrefetchLead,
                      DotAccumulators, ShuffleLookahead, MinBlocks>,
      Warps * 32,
      0);
  TORCH_CHECK(status == cudaSuccess,
              "occupancy query failed: ", cudaGetErrorString(status));

  int64_t* row = rows + static_cast<int64_t>(variant) * kMetadataColumns;
  row[0] = variant;
  row[1] = Warps;
  row[2] = RootMode == kRefinedRoot ? 1 : 0;
  row[3] = RootMode == kRawRoot ? 1 : 0;
  row[4] = Warps * 32;
  row[5] = attributes.numRegs;
  row[6] = attributes.localSizeBytes;
  row[7] = attributes.sharedSizeBytes;
  row[8] = active_blocks;
  row[9] = active_blocks * Warps;
  row[10] = TailPrefetch;
  row[11] = MinBlocks;
  row[12] = PtxPrefetchLead;
  row[13] = DotAccumulators;
  row[14] = ShuffleLookahead;
  row[15] = 0;
  row[16] = 0;
  row[17] = 32;
  row[18] = 1;
}

template <int Warps, int GroupWidth, int MinBlocks>
void write_subwarp_left_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  auto status = cudaFuncGetAttributes(
      &attributes, subwarp_left_32_kernel<Warps, GroupWidth, MinBlocks>);
  TORCH_CHECK(status == cudaSuccess,
              "cudaFuncGetAttributes failed: ", cudaGetErrorString(status));
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      subwarp_left_32_kernel<Warps, GroupWidth, MinBlocks>,
      Warps * 32,
      0);
  TORCH_CHECK(status == cudaSuccess,
              "occupancy query failed: ", cudaGetErrorString(status));

  int64_t* row = rows + static_cast<int64_t>(variant) * kMetadataColumns;
  row[0] = variant;
  row[1] = Warps;
  row[2] = 0;
  row[3] = 1;
  row[4] = Warps * 32;
  row[5] = attributes.numRegs;
  row[6] = attributes.localSizeBytes;
  row[7] = attributes.sharedSizeBytes;
  row[8] = active_blocks;
  row[9] = active_blocks * Warps;
  row[10] = 0;
  row[11] = MinBlocks;
  row[12] = 0;
  row[13] = 1;
  row[14] = 1;
  row[15] = 0;
  row[16] = 0;
  row[17] = GroupWidth;
  row[18] = 32 / GroupWidth;
}

template <int Warps, int GroupWidth, bool RootLookahead, int MinBlocks>
void write_right_looking_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  auto status = cudaFuncGetAttributes(
      &attributes,
      right_looking_32_kernel<Warps, GroupWidth, RootLookahead, MinBlocks>);
  TORCH_CHECK(status == cudaSuccess,
              "cudaFuncGetAttributes failed: ", cudaGetErrorString(status));
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      right_looking_32_kernel<Warps, GroupWidth, RootLookahead, MinBlocks>,
      Warps * 32,
      0);
  TORCH_CHECK(status == cudaSuccess,
              "occupancy query failed: ", cudaGetErrorString(status));

  int64_t* row = rows + static_cast<int64_t>(variant) * kMetadataColumns;
  row[0] = variant;
  row[1] = Warps;
  row[2] = 0;
  row[3] = 1;
  row[4] = Warps * 32;
  row[5] = attributes.numRegs;
  row[6] = attributes.localSizeBytes;
  row[7] = attributes.sharedSizeBytes;
  row[8] = active_blocks;
  row[9] = active_blocks * Warps;
  row[10] = 0;
  row[11] = MinBlocks;
  row[12] = 0;
  row[13] = 1;
  row[14] = 1;
  row[15] = 1;
  row[16] = RootLookahead ? 1 : 0;
  row[17] = GroupWidth;
  row[18] = 32 / GroupWidth;
}

}  // namespace

void cholesky_4096x32_out(const at::Tensor& data,
                          at::Tensor out,
                          int64_t variant) {
  check_input(data);
  check_output(data, out);
  TORCH_CHECK(variant >= 0 && variant < 34, "variant must be in [0, 33]");
  launch_variant(data.data_ptr<float>(), out.data_ptr<float>(),
                 static_cast<int>(variant));
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_4096x32(const at::Tensor& data,
                            int64_t variant) {
  auto out = at::empty_like(data);
  cholesky_4096x32_out(data, out, variant);
  return out;
}

at::Tensor cholesky_4096x32_metadata() {
  auto result = at::empty({34, kMetadataColumns},
                          at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  write_metadata< 1, kPreciseRoot, 0, 0, 1, 1, 32>(rows, 0);
  write_metadata< 1, kRefinedRoot, 0, 0, 1, 1, 32>(rows, 1);
  write_metadata< 2, kPreciseRoot, 0, 0, 1, 1, 16>(rows, 2);
  write_metadata< 2, kRefinedRoot, 0, 0, 1, 1, 16>(rows, 3);
  write_metadata< 4, kPreciseRoot, 0, 0, 1, 1,  8>(rows, 4);
  write_metadata< 4, kRefinedRoot, 0, 0, 1, 1,  8>(rows, 5);
  write_metadata< 8, kPreciseRoot, 0, 0, 1, 1,  4>(rows, 6);
  write_metadata< 8, kRefinedRoot, 0, 0, 1, 1,  4>(rows, 7);
  write_metadata<16, kPreciseRoot, 0, 0, 1, 1,  2>(rows, 8);
  write_metadata<16, kRefinedRoot, 0, 0, 1, 1,  2>(rows, 9);
  write_metadata< 2, kRefinedRoot, 0, 0, 1, 1, 14>(rows, 10);
  write_metadata< 2, kRefinedRoot, 2, 0, 1, 1, 14>(rows, 11);
  write_metadata< 2, kRefinedRoot, 4, 0, 1, 1, 14>(rows, 12);
  write_metadata< 2, kRefinedRoot, 6, 0, 1, 1, 14>(rows, 13);
  write_metadata< 2, kRefinedRoot, 8, 0, 1, 1, 14>(rows, 14);
  write_metadata< 2, kRawRoot,     0, 0, 1, 1, 14>(rows, 15);
  write_metadata< 2, kRawRoot,     2, 0, 1, 1, 14>(rows, 16);
  write_metadata< 2, kRawRoot,     4, 0, 1, 1, 14>(rows, 17);
  write_metadata< 2, kRawRoot,     6, 0, 1, 1, 14>(rows, 18);
  write_metadata< 2, kRawRoot,     8, 0, 1, 1, 14>(rows, 19);
  write_metadata< 2, kRawRoot,     0, 1, 1, 1, 14>(rows, 20);
  write_metadata< 2, kRawRoot,     0, 2, 1, 1, 14>(rows, 21);
  write_metadata< 2, kRawRoot,     0, 0, 2, 1, 14>(rows, 22);
  write_metadata< 2, kRawRoot,     0, 0, 4, 1, 14>(rows, 23);
  write_metadata< 2, kRawRoot,     0, 0, 1, 2, 14>(rows, 24);
  write_metadata< 2, kRawRoot,     0, 0, 2, 2, 14>(rows, 25);
  write_metadata< 2, kRawRoot,     0, 0, 4, 2, 14>(rows, 26);
  write_right_looking_metadata<2, 32, false, 14>(rows, 27);
  write_right_looking_metadata<2, 32, true,  14>(rows, 28);
  write_subwarp_left_metadata<   2, 16,        7>(rows, 29);
  write_right_looking_metadata<2, 16, false,  7>(rows, 30);
  write_right_looking_metadata<2, 16, true,   7>(rows, 31);
  write_subwarp_left_metadata<   2,  8,        4>(rows, 32);
  write_right_looking_metadata<2,  8, false,  4>(rows, 33);
  return result;
}
"""


@lru_cache(maxsize=1)
def _native_module():
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name="cholesky_b4096n32_algorithms_v7",
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
    if not 0 <= variant < _VARIANT_COUNT:
        raise ValueError(f"variant must be in [0, {_VARIANT_COUNT - 1}], got {variant}")
    module = _native_module()
    if out is None:
        return module.run(data, variant)
    module.run_out(data, out, variant)
    return out


def _variant_metadata() -> torch.Tensor:
    return _native_module().metadata()


def custom_kernel(data: input_t) -> output_t:
    if tuple(data.shape) != (4096, 32, 32):
        return torch.linalg.cholesky_ex(data, check_errors=False).L
    return _run_variant(data, _DEFAULT_VARIANT)
