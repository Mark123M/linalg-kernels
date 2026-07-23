import hashlib
import os
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The Popcorn tuner replaces this exact line in temporary, untracked copies.
_DEFAULT_VARIANT = 8  # POPCORN_VARIANT
_VARIANT_COUNT = 18
_VARIANT_IDS = tuple(range(_VARIANT_COUNT))

_VARIANT_NAMES = (
    "cta256_rec32_scalar_simt_precise",
    "cta256_rec32_scalar_simt_refined",
    "cta256_rec32_scalar_simt_raw",
    "cta256_rec32_sub8_simt_refined",
    "cta256_potf2_64_scalar_simt_refined",
    "cta256_rank1_64_scalar_simt_refined",
    "cta512_rec32_scalar_simt_refined",
    "cta256_rec32_scalar_tc_outer_refined",
    "cta256_rec32_scalar_tc_all_refined",
    "cta256_rec32_sub8_tc_all_refined",
    "cta256_rec32_scalar_tc_all_raw",
    "cta512_rec32_scalar_tc_all_refined",
    "cluster128_rec32_scalar_simt_refined",
    "cluster256_rec32_scalar_simt_refined",
    "cluster128_rec32_scalar_tc_outer_refined",
    "cluster128_rec32_scalar_tc_all_refined",
    "cluster256_rec32_scalar_tc_all_refined",
    "cluster256_rec32_scalar_tc_all_raw",
)

_METADATA_COLUMNS = (
    "variant",
    "threads",
    "registers",
    "local_bytes",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "active_ctas_per_sm",
    "active_warps_per_sm",
    "cluster_size",
    "uses_tcgen05",
    "recursive_base",
    "factor_mode",
    "trsm_mode",
    "update_mode",
    "root_mode",
    "launch_bounds",
)

_CPP_SOURCE = r"""
#include <torch/extension.h>

void cholesky_b64n256_prepare();
at::Tensor cholesky_b64n256(const at::Tensor& data, int64_t variant);
void cholesky_b64n256_out(const at::Tensor& data,
                          at::Tensor out,
                          int64_t variant);
at::Tensor cholesky_b64n256_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b64n256_prepare,
        "Configure B200 batched 256x256 Cholesky kernels");
  m.def("run", &cholesky_b64n256, "Batched 256x256 Cholesky");
  m.def("run_out", &cholesky_b64n256_out,
        "Batched 256x256 Cholesky out");
  m.def("metadata", &cholesky_b64n256_metadata,
        "Kernel resource metadata");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <cooperative_groups.h>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace cg = cooperative_groups;

namespace {

constexpr int kBatch = 64;
constexpr int kN = 256;
constexpr int kTile = 128;
constexpr int kHalf = 64;
constexpr int kLd = 129;
constexpr int kA00 = 0;
constexpr int kA10 = kTile * kLd;
constexpr int kA11 = kA10 + kTile * kTile;
constexpr int kSingleFloats = kA11 + kTile * kLd;
constexpr int kSingleBytes = kSingleFloats * static_cast<int>(sizeof(float));
constexpr int kTcScratchFloats = kHalf * kHalf;
constexpr int kTcBarrierFloats = 4;
constexpr int kSingleTcBytes =
    (kSingleFloats + kTcScratchFloats + kTcBarrierFloats) *
    static_cast<int>(sizeof(float));

constexpr int kClusterA00 = 0;
constexpr int kClusterA10 = kHalf * kLd;
constexpr int kClusterA11 = kClusterA10 + kHalf * kTile;
constexpr int kClusterBaseFloats = kClusterA11 + kHalf * kLd;
constexpr int kClusterTcTailFloats = 6 * kHalf * kHalf;
constexpr int kClusterFloats = kClusterBaseFloats;
constexpr int kClusterTcFloats =
    kClusterBaseFloats + kClusterTcTailFloats + kTcBarrierFloats;
constexpr int kClusterBytes =
    kClusterFloats * static_cast<int>(sizeof(float));
constexpr int kClusterTcBytes =
    kClusterTcFloats * static_cast<int>(sizeof(float));

constexpr int kVariantCount = 18;
constexpr int kMetadataColumns = 16;
constexpr int kPreciseRoot = 0;
constexpr int kRefinedRoot = 1;
constexpr int kRawRoot = 2;
constexpr int kRecursive32 = 0;
constexpr int kPotf264 = 1;
constexpr int kRank164 = 2;
constexpr int kScalarTrsm = 0;
constexpr int kSub8Trsm = 1;
constexpr int kTensorBlockTrsm = 2;
constexpr int kSimtUpdate = 0;
constexpr int kTcOuterUpdate = 1;
constexpr int kTcAllUpdate = 2;
constexpr uint32_t kTmemDp = 1u << 16;

template <int VariantId>
struct Variant;

#define SPEC(ID, THREADS, ROOT, BASE, FACTOR, TRSM, UPDATE, CLUSTER) \
  template <> struct Variant<ID> {                                  \
    static constexpr int threads = THREADS;                          \
    static constexpr int root = ROOT;                                \
    static constexpr int base = BASE;                                \
    static constexpr int factor = FACTOR;                            \
    static constexpr int trsm = TRSM;                                \
    static constexpr int update = UPDATE;                            \
    static constexpr bool cluster = CLUSTER;                         \
    static constexpr bool tensor = UPDATE != kSimtUpdate;            \
  }

SPEC(0, 256, kPreciseRoot, 32, kRecursive32, kScalarTrsm, kSimtUpdate, false);
SPEC(1, 256, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kSimtUpdate, false);
SPEC(2, 256, kRawRoot, 32, kRecursive32, kScalarTrsm, kSimtUpdate, false);
SPEC(3, 256, kRefinedRoot, 32, kRecursive32, kSub8Trsm, kSimtUpdate, false);
SPEC(4, 256, kRefinedRoot, 64, kPotf264, kScalarTrsm, kSimtUpdate, false);
SPEC(5, 256, kRefinedRoot, 64, kRank164, kScalarTrsm, kSimtUpdate, false);
SPEC(6, 512, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kSimtUpdate, false);
SPEC(7, 256, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kTcOuterUpdate, false);
SPEC(8, 256, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kTcAllUpdate, false);
SPEC(9, 256, kRefinedRoot, 32, kRecursive32, kSub8Trsm, kTcAllUpdate, false);
SPEC(10, 256, kRawRoot, 32, kRecursive32, kScalarTrsm, kTcAllUpdate, false);
SPEC(11, 512, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kTcAllUpdate, false);
SPEC(12, 128, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kSimtUpdate, true);
SPEC(13, 256, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kSimtUpdate, true);
SPEC(14, 128, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kTcOuterUpdate, true);
SPEC(15, 128, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kTcAllUpdate, true);
SPEC(16, 256, kRefinedRoot, 32, kRecursive32, kScalarTrsm, kTcAllUpdate, true);
SPEC(17, 256, kRawRoot, 32, kRecursive32, kScalarTrsm, kTcAllUpdate, true);

#undef SPEC

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(data.dim() == 3 && data.size(0) == kBatch &&
                  data.size(1) == kN && data.size(2) == kN,
              "native path requires shape (64, 256, 256)");
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

__device__ __forceinline__ float& single_at(float* s, int row, int col) {
  if (row < kTile) {
    return s[kA00 + row * kLd + col];
  }
  if (col < kTile) {
    return s[kA10 + (row - kTile) * kTile + col];
  }
  return s[kA11 + (row - kTile) * kLd + col - kTile];
}

__device__ __forceinline__ const float& single_at(
    const float* s, int row, int col) {
  if (row < kTile) {
    return s[kA00 + row * kLd + col];
  }
  if (col < kTile) {
    return s[kA10 + (row - kTile) * kTile + col];
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

__device__ __forceinline__ int kmajor_offset(
    int row, int column, int rows) {
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

__device__ __forceinline__ void cluster_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
}

__device__ __forceinline__ void tensor_after_sync_fence() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

template <bool TwoCta>
__device__ __forceinline__ void tmem_allocate(uint32_t* destination,
                                               int columns) {
  if (static_cast<int>(threadIdx.x) < 32) {
    const uint32_t address = shared_address(destination);
    if constexpr (TwoCta) {
      asm volatile(
          "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 "
          "[%0], %1;" :: "r"(address), "r"(columns) : "memory");
    } else {
      asm volatile(
          "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
          "[%0], %1;" :: "r"(address), "r"(columns) : "memory");
    }
  }
  __syncthreads();
}

template <bool TwoCta>
__device__ __forceinline__ void tmem_deallocate(uint32_t base, int columns) {
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < 32) {
    if constexpr (TwoCta) {
      asm volatile(
          "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::
          "r"(base), "r"(columns));
    } else {
      asm volatile(
          "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
          "r"(base), "r"(columns));
    }
  }
  __syncthreads();
}

template <bool TwoCta>
__device__ __forceinline__ void tmem_relinquish() {
  if (static_cast<int>(threadIdx.x) < 32) {
    if constexpr (TwoCta) {
      asm volatile(
          "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    } else {
      asm volatile(
          "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
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

template <bool TwoCta>
__device__ __forceinline__ void tensor_commit(
    uint64_t* barrier, bool leader = true) {
  if (threadIdx.x == 0 && leader) {
    const uint32_t address = shared_address(barrier);
    if constexpr (TwoCta) {
      const uint16_t mask = 0x3u;
      asm volatile(
          "tcgen05.commit.cta_group::2.mbarrier::arrive::one."
          "shared::cluster.multicast::cluster.b64 [%0], %1;" ::
          "r"(address), "h"(mask) : "memory");
    } else {
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
          "shared::cluster.b64 [%0];" :: "r"(address) : "memory");
    }
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

template <int M, int N, bool TwoCta>
__device__ __forceinline__ void issue_tf32_mma(
    uint32_t tmem_base, uint64_t a_desc, uint64_t b_desc,
    bool accumulate, bool leader = true) {
  if (threadIdx.x == 0 && leader) {
    const uint32_t instruction = tf32_instruction_descriptor<M, N>();
    const uint32_t scale = accumulate ? 1u : 0u;
    if constexpr (TwoCta) {
      asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::2.kind::tf32 "
          "[%0], %1, %2, %3, "
          "{%5,%6,%7,%8,%9,%10,%11,%12}, p;\n\t"
          "}\n" ::
          "r"(tmem_base), "l"(a_desc), "l"(b_desc), "r"(instruction),
          "r"(scale), "r"(0u), "r"(0u), "r"(0u), "r"(0u),
          "r"(0u), "r"(0u), "r"(0u), "r"(0u));
    } else {
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
}

__device__ __forceinline__ float tmem_load_one(uint32_t address) {
  uint32_t value;
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
      : "=r"(value) : "r"(address));
  return __uint_as_float(value);
}

template <int RootMode, int FactorMode>
__device__ __forceinline__ void potf2_single(
    float* s, int begin, int size) {
  for (int column = 0; column < size; ++column) {
    const int j = begin + column;
    if (threadIdx.x == 0) {
      float diagonal;
      float inverse;
      root_pair<RootMode>(single_at(s, j, j), diagonal, inverse);
      single_at(s, j, j) = diagonal;
      s[kSingleFloats - 1] = inverse;
    }
    __syncthreads();
    const float inverse = s[kSingleFloats - 1];
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
      int local_col = 0;
      int remainder = linear;
      if constexpr (FactorMode == kRank164) {
        local_row = static_cast<int>(
            (sqrtf(8.0f * static_cast<float>(linear) + 1.0f) - 1.0f) *
            0.5f);
        int first = local_row * (local_row + 1) / 2;
        while (first > linear) {
          --local_row;
          first = local_row * (local_row + 1) / 2;
        }
        while ((local_row + 1) * (local_row + 2) / 2 <= linear) {
          ++local_row;
        }
        local_col = linear - local_row * (local_row + 1) / 2;
      } else {
        for (int width = 1; remainder >= width; ++width) {
          remainder -= width;
          ++local_row;
        }
        local_col = remainder;
      }
      const int row = j + 1 + local_row;
      const int col = j + 1 + local_col;
      single_at(s, row, col) =
          fmaf(-single_at(s, row, j),
               single_at(s, col, j),
               single_at(s, row, col));
    }
    __syncthreads();
  }
}

template <int TrsmMode>
__device__ __forceinline__ void trsm_single(
    float* s, int row_begin, int rows, int col_begin, int cols) {
  if constexpr (TrsmMode == kScalarTrsm) {
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
  } else {
    const int lane = static_cast<int>(threadIdx.x) & 7;
    const int group = static_cast<int>(threadIdx.x) >> 3;
    const int groups = static_cast<int>(blockDim.x) >> 3;
    for (int local_row = group; local_row < rows; local_row += groups) {
      const int row = row_begin + local_row;
      for (int local_col = 0; local_col < cols; ++local_col) {
        float partial = lane == 0 ? single_at(s, row, col_begin + local_col)
                                  : 0.0f;
        for (int k = lane; k < local_col; k += 8) {
          partial = fmaf(-single_at(s, row, col_begin + k),
                         single_at(s, col_begin + local_col, col_begin + k),
                         partial);
        }
        partial += __shfl_down_sync(0xffffffffu, partial, 4, 8);
        partial += __shfl_down_sync(0xffffffffu, partial, 2, 8);
        partial += __shfl_down_sync(0xffffffffu, partial, 1, 8);
        if (lane == 0) {
          single_at(s, row, col_begin + local_col) =
              partial /
              single_at(s, col_begin + local_col, col_begin + local_col);
        }
        __syncwarp();
      }
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
      if (local_row < size && local_col < size &&
          local_col <= local_row) {
        float value = single_at(s, target + local_row, target + local_col);
#pragma unroll 4
        for (int k = 0; k < panel_cols; ++k) {
          value = fmaf(-single_at(s, target + local_row, panel + k),
                       single_at(s, target + local_col, panel + k), value);
        }
        single_at(s, target + local_row, target + local_col) = value;
      }
    }
  }
  __syncthreads();
}

template <int M>
__device__ __forceinline__ void tc_update_single(
    float* s, int target, int panel, float* scratch,
    uint32_t* tmem_slot, uint64_t* barrier, int& phase) {
  tmem_allocate<false>(tmem_slot, kTile);
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
    const uint64_t descriptor =
        make_kmajor_descriptor(scratch, M);
    issue_tf32_mma<M, M, false>(
        tmem_base, descriptor, descriptor, k != 0);
    tensor_commit<false>(barrier);
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
  tmem_deallocate<false>(tmem_base, kTile);
}

template <int RootMode, int FactorMode, int TrsmMode, int UpdateMode>
__device__ __forceinline__ void potrf128_single(
    float* s, int begin, float* scratch, uint32_t* tmem_slot,
    uint64_t* barrier, int& phase) {
  if constexpr (FactorMode == kRecursive32) {
    potf2_single<RootMode, FactorMode>(s, begin, 32);
    trsm_single<TrsmMode>(s, begin + 32, 32, begin, 32);
    simt_update_single(s, begin + 32, 32, begin, 32);
    potf2_single<RootMode, FactorMode>(s, begin + 32, 32);

    trsm_single<TrsmMode>(s, begin + 64, 64, begin, 64);
    if constexpr (UpdateMode == kTcAllUpdate) {
      tc_update_single<64>(
          s, begin + 64, begin, scratch, tmem_slot, barrier, phase);
    } else {
      simt_update_single(s, begin + 64, 64, begin, 64);
    }

    potf2_single<RootMode, FactorMode>(s, begin + 64, 32);
    trsm_single<TrsmMode>(s, begin + 96, 32, begin + 64, 32);
    simt_update_single(s, begin + 96, 32, begin + 64, 32);
    potf2_single<RootMode, FactorMode>(s, begin + 96, 32);
  } else {
    potf2_single<RootMode, FactorMode>(s, begin, 64);
    trsm_single<TrsmMode>(s, begin + 64, 64, begin, 64);
    simt_update_single(s, begin + 64, 64, begin, 64);
    potf2_single<RootMode, FactorMode>(s, begin + 64, 64);
  }
}

template <int VariantId>
__global__ __launch_bounds__(Variant<VariantId>::threads, 1)
void single_kernel(const float* __restrict__ input,
                   float* __restrict__ output) {
  using V = Variant<VariantId>;
  extern __shared__ __align__(16) float storage[];
  float* scratch = storage + kSingleFloats;
  uint32_t* tmem_slot =
      reinterpret_cast<uint32_t*>(scratch + kTcScratchFloats);
  uint64_t* barrier =
      reinterpret_cast<uint64_t*>(scratch + kTcScratchFloats + 2);
  int phase = 0;

  const int matrix = static_cast<int>(blockIdx.x);
  const float* matrix_input =
      input + static_cast<int64_t>(matrix) * kN * kN;
  float* matrix_output =
      output + static_cast<int64_t>(matrix) * kN * kN;

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kN * kN; linear += static_cast<int>(blockDim.x)) {
    matrix_output[linear] = 0.0f;
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      storage[kA00 + row * kLd + col] =
          matrix_input[row * kN + col];
      storage[kA11 + row * kLd + col] =
          matrix_input[(row + kTile) * kN + col + kTile];
    }
    storage[kA10 + linear] =
        matrix_input[(row + kTile) * kN + col];
  }
  __syncthreads();
  if constexpr (V::tensor) {
    barrier_init(barrier);
  }

  potrf128_single<V::root, V::factor, V::trsm, V::update>(
      storage, 0, scratch, tmem_slot, barrier, phase);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      matrix_output[row * kN + col] =
          storage[kA00 + row * kLd + col];
    }
  }
  __syncthreads();

  trsm_single<V::trsm>(storage, kTile, kTile, 0, kTile);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    matrix_output[(row + kTile) * kN + col] =
        storage[kA10 + linear];
  }
  __syncthreads();

  if constexpr (V::update == kSimtUpdate) {
    simt_update_single(storage, kTile, kTile, 0, kTile);
  } else {
    tc_update_single<128>(
        storage, kTile, 0, scratch, tmem_slot, barrier, phase);
  }
  potrf128_single<V::root, V::factor, V::trsm, V::update>(
      storage, kTile, scratch, tmem_slot, barrier, phase);

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kTile * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      matrix_output[(row + kTile) * kN + col + kTile] =
          storage[kA11 + row * kLd + col];
    }
  }
  __syncthreads();
  if constexpr (V::tensor) {
    tmem_relinquish<false>();
  }
}

__device__ __forceinline__ int cluster_owner(int row) {
  return (row & 127) >> 6;
}

__device__ __forceinline__ int cluster_offset(int row, int col) {
  const int local_row = row & 63;
  if (row < kTile) {
    return kClusterA00 + local_row * kLd + col;
  }
  if (col < kTile) {
    return kClusterA10 + local_row * kTile + col;
  }
  return kClusterA11 + local_row * kLd + col - kTile;
}

__device__ __forceinline__ float* cluster_pointer(
    cg::cluster_group cluster, float* local, int row, int col) {
  const int owner = cluster_owner(row);
  float* base = owner == static_cast<int>(cluster.block_rank())
                    ? local
                    : cluster.map_shared_rank(local, owner);
  return base + cluster_offset(row, col);
}

__device__ __forceinline__ float cluster_get(
    cg::cluster_group cluster, float* local, int row, int col) {
  return *cluster_pointer(cluster, local, row, col);
}

__device__ __forceinline__ void cluster_set(
    cg::cluster_group cluster, float* local, int row, int col, float value) {
  *cluster_pointer(cluster, local, row, col) = value;
}

template <int RootMode>
__device__ __forceinline__ void cluster_potf2(
    cg::cluster_group cluster, float* local, int begin, int size) {
  const int owner = cluster_owner(begin);
  if (static_cast<int>(cluster.block_rank()) == owner) {
    for (int column = 0; column < size; ++column) {
      const int j = begin + column;
      if (threadIdx.x == 0) {
        float diagonal;
        float inverse;
        root_pair<RootMode>(
            cluster_get(cluster, local, j, j), diagonal, inverse);
        cluster_set(cluster, local, j, j, diagonal);
        local[kClusterBaseFloats - 1] = inverse;
      }
      __syncthreads();
      const float inverse = local[kClusterBaseFloats - 1];
      for (int row = column + 1 + static_cast<int>(threadIdx.x);
           row < size; row += static_cast<int>(blockDim.x)) {
        const int global_row = begin + row;
        cluster_set(
            cluster, local, global_row, j,
            cluster_get(cluster, local, global_row, j) * inverse);
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
        const float value =
            fmaf(-cluster_get(cluster, local, row, j),
                 cluster_get(cluster, local, col, j),
                 cluster_get(cluster, local, row, col));
        cluster_set(cluster, local, row, col, value);
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ void cluster_trsm_rows(
    cg::cluster_group cluster, float* local,
    int row_begin, int rows, int col_begin, int cols) {
  for (int local_row = static_cast<int>(threadIdx.x);
       local_row < rows; local_row += static_cast<int>(blockDim.x)) {
    const int row = row_begin + local_row;
    if (cluster_owner(row) == static_cast<int>(cluster.block_rank())) {
      for (int local_col = 0; local_col < cols; ++local_col) {
        const int col = col_begin + local_col;
        float value = cluster_get(cluster, local, row, col);
        for (int k = 0; k < local_col; ++k) {
          value = fmaf(
              -cluster_get(cluster, local, row, col_begin + k),
              cluster_get(cluster, local, col, col_begin + k), value);
        }
        cluster_set(
            cluster, local, row, col,
            value / cluster_get(cluster, local, col, col));
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void cluster_simt_update(
    cg::cluster_group cluster, float* local,
    int target, int size, int panel, int panel_cols) {
  const int rank = static_cast<int>(cluster.block_rank());
  for (int local_row = static_cast<int>(threadIdx.x);
       local_row < size; local_row += static_cast<int>(blockDim.x)) {
    const int row = target + local_row;
    if (cluster_owner(row) == rank) {
      for (int local_col = 0; local_col <= local_row; ++local_col) {
        const int col = target + local_col;
        float value = cluster_get(cluster, local, row, col);
#pragma unroll 4
        for (int k = 0; k < panel_cols; ++k) {
          value = fmaf(
              -cluster_get(cluster, local, row, panel + k),
              cluster_get(cluster, local, col, panel + k), value);
        }
        cluster_set(cluster, local, row, col, value);
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void cluster_tc_update64(
    cg::cluster_group cluster, float* local, int target, int panel) {
  const int rank = static_cast<int>(cluster.block_rank());
  float* scratch = local + kClusterBaseFloats;
  float* a_panel = scratch;
  float* b_panel = scratch + kHalf * kTile;
  uint32_t* tmem_slot =
      reinterpret_cast<uint32_t*>(scratch + kClusterTcTailFloats);
  uint64_t* barrier =
      reinterpret_cast<uint64_t*>(scratch + kClusterTcTailFloats + 2);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kHalf * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kTile;
    const int col = linear % kTile;
    const float value =
        rank == 0 && col < kHalf
            ? cluster_get(cluster, local, target + row, panel + col)
            : 0.0f;
    reinterpret_cast<uint32_t*>(a_panel)[
        kmajor_offset(row, col, kHalf)] = to_tf32(value);
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kHalf * kTile; linear += static_cast<int>(blockDim.x)) {
    const int block = linear / (32 * kTile);
    const int remainder = linear % (32 * kTile);
    const int row = remainder / kTile;
    const int col = remainder % kTile;
    const float value =
        col < kHalf
            ? cluster_get(
                  cluster, local, target + block * 32 + row, panel + col)
            : 0.0f;
    reinterpret_cast<uint32_t*>(b_panel)[
        block * 32 * kTile + kmajor_offset(row, col, 32)] =
        to_tf32(value);
  }
  __syncthreads();
  cluster_proxy_fence();
  cluster.sync();
  tensor_after_sync_fence();
  barrier_init(barrier);
  tmem_allocate<true>(tmem_slot, kTile);
  const uint32_t tmem_base = *tmem_slot;
  for (int block = 0; block < 2; ++block) {
    for (int k = 0; k < kTile; k += 8) {
      const void* a_slice = a_panel + k * kHalf;
      const void* b_slice =
          b_panel + block * 32 * kTile + k * 32;
      issue_tf32_mma<128, 64, true>(
          tmem_base + static_cast<uint32_t>(block * 32),
          make_kmajor_descriptor(a_slice, kHalf),
          make_kmajor_descriptor(b_slice, 32),
          k != 0, rank == 0);
    }
  }
  tensor_commit<true>(barrier, rank == 0);
  barrier_wait(barrier, 0);

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (rank == 0 && warp < 2) {
    const int row = warp * 32 + lane;
    for (int block = 0; block < 2; ++block) {
      for (int local_col = 0; local_col < 32; ++local_col) {
        const int col = block * 32 + local_col;
        const uint32_t address =
            tmem_base + static_cast<uint32_t>(block * 32) +
            static_cast<uint32_t>(warp * 32) * kTmemDp +
            static_cast<uint32_t>(local_col);
        const float product = tmem_load_one(address);
        if (col <= row) {
          const float value =
              cluster_get(cluster, local, target + row, target + col) -
              product;
          cluster_set(
              cluster, local, target + row, target + col, value);
        }
      }
    }
  }
  __syncthreads();
  cluster.sync();
  tmem_deallocate<true>(tmem_base, kTile);
}

__device__ __forceinline__ void cluster_tensor_trsm128(
    cg::cluster_group cluster, float* local,
    int row_begin, int rank) {
  cluster_trsm_rows(
      cluster, local, row_begin, kHalf, 0, kHalf);
  cluster.sync();

  float* scratch = local + kClusterBaseFloats;
  float* a_panel = scratch;
  float* b_panel = scratch + kHalf * kHalf;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kHalf * kHalf; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kHalf;
    const int col = linear % kHalf;
    reinterpret_cast<uint32_t*>(a_panel)[
        kmajor_offset(row, col, kHalf)] =
        to_tf32(cluster_get(
            cluster, local, row_begin + row, col));
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kHalf * kHalf; linear += static_cast<int>(blockDim.x)) {
    const int block = linear / (32 * kHalf);
    const int remainder = linear % (32 * kHalf);
    const int row = remainder / kHalf;
    const int col = remainder % kHalf;
    reinterpret_cast<uint32_t*>(b_panel)[
        block * 32 * kHalf + kmajor_offset(row, col, 32)] =
        to_tf32(cluster_get(
            cluster, local, kHalf + block * 32 + row, col));
  }
  __syncthreads();
  cluster_proxy_fence();
  cluster.sync();
  tensor_after_sync_fence();

  uint32_t* tmem_slot =
      reinterpret_cast<uint32_t*>(scratch + kClusterTcTailFloats);
  uint64_t* barrier =
      reinterpret_cast<uint64_t*>(scratch + kClusterTcTailFloats + 2);
  barrier_init(barrier);
  tmem_allocate<true>(tmem_slot, kTile);
  const uint32_t tmem_base = *tmem_slot;
  for (int block = 0; block < 2; ++block) {
    for (int k = 0; k < kHalf; k += 8) {
      const void* a_slice = a_panel + k * kHalf;
      const void* b_slice =
          b_panel + block * 32 * kHalf + k * 32;
      issue_tf32_mma<128, 64, true>(
          tmem_base + static_cast<uint32_t>(block * 32),
          make_kmajor_descriptor(a_slice, kHalf),
          make_kmajor_descriptor(b_slice, 32),
          k != 0,
          rank == 0);
    }
  }
  tensor_commit<true>(barrier, rank == 0);
  barrier_wait(barrier, 0);

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (warp < 2) {
    const int row = warp * 32 + lane;
    for (int block = 0; block < 2; ++block) {
      for (int local_col = 0; local_col < 32; ++local_col) {
        const int col = block * 32 + local_col;
        const uint32_t address =
            tmem_base + static_cast<uint32_t>(block * 32) +
            static_cast<uint32_t>(warp * 32) * kTmemDp +
            static_cast<uint32_t>(local_col);
        const float product = tmem_load_one(address);
        const int global_row = row_begin + row;
        const int global_col = kHalf + col;
        cluster_set(
            cluster, local, global_row, global_col,
            cluster_get(cluster, local, global_row, global_col) - product);
      }
    }
  }
  __syncthreads();
  cluster.sync();
  tmem_deallocate<true>(tmem_base, kTile);
  cluster.sync();
  cluster_trsm_rows(
      cluster, local, row_begin, kHalf, kHalf, kHalf);
  cluster.sync();
}

template <int RootMode, bool TensorAll>
__device__ __forceinline__ void cluster_factor64(
    cg::cluster_group cluster, float* local, int begin) {
  cluster_potf2<RootMode>(cluster, local, begin, 32);
  cluster_trsm_rows(cluster, local, begin + 32, 32, begin, 32);
  cluster_simt_update(cluster, local, begin + 32, 32, begin, 32);
  cluster_potf2<RootMode>(cluster, local, begin + 32, 32);
}

template <int RootMode>
__device__ __forceinline__ void cluster_potrf128_simt(
    cg::cluster_group cluster, float* local, int begin) {
  cluster_factor64<RootMode, false>(cluster, local, begin);
  cluster.sync();
  cluster_trsm_rows(cluster, local, begin + 64, 64, begin, 64);
  cluster_simt_update(cluster, local, begin + 64, 64, begin, 64);
  cluster.sync();
  cluster_factor64<RootMode, false>(cluster, local, begin + 64);
  cluster.sync();
}

template <int RootMode>
__device__ __forceinline__ void cluster_potrf128_tensor(
    cg::cluster_group cluster, float* local, int begin) {
  cluster_factor64<RootMode, true>(cluster, local, begin);
  cluster.sync();
  cluster_trsm_rows(cluster, local, begin + 64, 64, begin, 64);
  cluster.sync();
  cluster_tc_update64(cluster, local, begin + 64, begin);
  cluster.sync();
  cluster_factor64<RootMode, true>(cluster, local, begin + 64);
  cluster.sync();
}

template <int Threads, int RootMode, bool TensorOuter, bool TensorAll>
__global__ __launch_bounds__(Threads, 1)
void cluster_kernel(const float* __restrict__ input,
                    float* __restrict__ output) {
  extern __shared__ __align__(16) float local[];
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int matrix = static_cast<int>(blockIdx.x) >> 1;
  const float* matrix_input =
      input + static_cast<int64_t>(matrix) * kN * kN;
  float* matrix_output =
      output + static_cast<int64_t>(matrix) * kN * kN;

  for (int half = 0; half < 2; ++half) {
    const int row_base = half * kTile + rank * kHalf;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kHalf * kN; linear += static_cast<int>(blockDim.x)) {
      const int row = row_base + linear / kN;
      const int col = linear % kN;
      matrix_output[row * kN + col] = 0.0f;
    }
  }
  for (int local_row = static_cast<int>(threadIdx.x);
       local_row < kHalf; local_row += static_cast<int>(blockDim.x)) {
    const int top_row = rank * kHalf + local_row;
    const int bottom_row = kTile + rank * kHalf + local_row;
    for (int col = 0; col < kTile; ++col) {
      if (col <= top_row) {
        local[kClusterA00 + local_row * kLd + col] =
            matrix_input[top_row * kN + col];
      }
      local[kClusterA10 + local_row * kTile + col] =
          matrix_input[bottom_row * kN + col];
      if (col <= rank * kHalf + local_row) {
        local[kClusterA11 + local_row * kLd + col] =
            matrix_input[bottom_row * kN + kTile + col];
      }
    }
  }
  cluster.sync();

  if constexpr (TensorAll) {
    cluster_potrf128_tensor<RootMode>(cluster, local, 0);
  } else {
    cluster_potrf128_simt<RootMode>(cluster, local, 0);
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kHalf * kTile; linear += static_cast<int>(blockDim.x)) {
    const int row = rank * kHalf + linear / kTile;
    const int col = linear % kTile;
    if (col <= row) {
      matrix_output[row * kN + col] =
          cluster_get(cluster, local, row, col);
    }
  }
  __syncthreads();

  if constexpr (TensorOuter) {
    cluster_tensor_trsm128(
        cluster, local, kTile + rank * kHalf, rank);
  } else {
    cluster_trsm_rows(
        cluster, local, kTile + rank * kHalf, kHalf, 0, kTile);
    cluster.sync();
  }
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kHalf * kTile; linear += static_cast<int>(blockDim.x)) {
    const int local_row = linear / kTile;
    const int col = linear % kTile;
    const int row = kTile + rank * kHalf + local_row;
    matrix_output[row * kN + col] =
        local[kClusterA10 + local_row * kTile + col];
  }
  __syncthreads();

  if constexpr (!TensorOuter) {
    cluster_simt_update(cluster, local, kTile, kTile, 0, kTile);
  } else {
    float* panel = local + kClusterA10;
    float* tail = local + kClusterBaseFloats;
    float* a_panel = tail;
    float* b_panel = tail + kHalf * kTile;
    for (int linear = static_cast<int>(threadIdx.x);
         linear < kHalf * kTile; linear += static_cast<int>(blockDim.x)) {
      const int row = linear / kTile;
      const int col = linear % kTile;
      reinterpret_cast<uint32_t*>(a_panel)[
          kmajor_offset(row, col, kHalf)] =
          to_tf32(panel[linear]);
    }
    for (int linear = static_cast<int>(threadIdx.x);
         linear < 4 * 32 * kTile;
         linear += static_cast<int>(blockDim.x)) {
      const int block = linear / (32 * kTile);
      const int remainder = linear % (32 * kTile);
      const int row = remainder / kTile;
      const int col = remainder % kTile;
      reinterpret_cast<uint32_t*>(b_panel)[
          block * 32 * kTile + kmajor_offset(row, col, 32)] =
          to_tf32(cluster_get(
              cluster, local, kTile + block * 32 + row, col));
    }
    __syncthreads();
    cluster_proxy_fence();
    cluster.sync();
    tensor_after_sync_fence();

    uint32_t* tmem_slot =
        reinterpret_cast<uint32_t*>(tail + kClusterTcTailFloats);
    uint64_t* barrier =
        reinterpret_cast<uint64_t*>(tail + kClusterTcTailFloats + 2);
    barrier_init(barrier);
    tmem_allocate<true>(tmem_slot, kTile);
    const uint32_t tmem_base = *tmem_slot;
    for (int block = 0; block < 4; ++block) {
      for (int k = 0; k < kTile; k += 8) {
        const void* a_slice = a_panel + k * kHalf;
        const void* b_slice =
            b_panel + block * 32 * kTile + k * 32;
        issue_tf32_mma<128, 64, true>(
            tmem_base + static_cast<uint32_t>(block * 32),
            make_kmajor_descriptor(a_slice, kHalf),
            make_kmajor_descriptor(b_slice, 32),
            k != 0, rank == 0);
      }
    }
    tensor_commit<true>(barrier, rank == 0);
    barrier_wait(barrier, 0);

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp < 2) {
      const int local_row = warp * 32 + lane;
      const int global_row = rank * kHalf + local_row;
      for (int block = 0; block < 4; ++block) {
        for (int local_col = 0; local_col < 32; ++local_col) {
          const int col = block * 32 + local_col;
          const uint32_t address =
              tmem_base + static_cast<uint32_t>(block * 32) +
              static_cast<uint32_t>(warp * 32) * kTmemDp +
              static_cast<uint32_t>(local_col);
          const float product = tmem_load_one(address);
          if (col <= global_row) {
            local[kClusterA11 + local_row * kLd + col] -= product;
          }
        }
      }
    }
    __syncthreads();
    tmem_deallocate<true>(tmem_base, kTile);
  }
  cluster.sync();

  if constexpr (TensorAll) {
    cluster_potrf128_tensor<RootMode>(cluster, local, kTile);
  } else {
    cluster_potrf128_simt<RootMode>(cluster, local, kTile);
  }

  for (int linear = static_cast<int>(threadIdx.x);
       linear < kHalf * kTile; linear += static_cast<int>(blockDim.x)) {
    const int local_row = linear / kTile;
    const int col = linear % kTile;
    const int relative_row = rank * kHalf + local_row;
    if (col <= relative_row) {
      const int row = kTile + relative_row;
      matrix_output[row * kN + kTile + col] =
          local[kClusterA11 + local_row * kLd + col];
    }
  }
  __syncthreads();
  if constexpr (TensorOuter) {
    tmem_relinquish<true>();
  }
}

template <typename Kernel>
void configure_kernel(Kernel kernel, int dynamic_bytes) {
  auto status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);
  TORCH_CHECK(status == cudaSuccess,
              "dynamic shared-memory opt-in failed: ",
              cudaGetErrorString(status));
  status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(status == cudaSuccess,
              "shared-memory carveout failed: ", cudaGetErrorString(status));
}

template <int Id>
void configure_one() {
  using V = Variant<Id>;
  const int bytes = V::cluster
      ? (V::tensor ? kClusterTcBytes : kClusterBytes)
      : (V::tensor ? kSingleTcBytes : kSingleBytes);
  if constexpr (V::cluster) {
    configure_kernel(
        cluster_kernel<V::threads, V::root,
                       V::update != kSimtUpdate,
                       V::update == kTcAllUpdate>,
        bytes);
  } else {
    configure_kernel(single_kernel<Id>, bytes);
  }
}

void configure_all() {
  configure_one<0>(); configure_one<1>(); configure_one<2>();
  configure_one<3>(); configure_one<4>(); configure_one<5>();
  configure_one<6>(); configure_one<7>(); configure_one<8>();
  configure_one<9>(); configure_one<10>(); configure_one<11>();
  configure_one<12>(); configure_one<13>(); configure_one<14>();
  configure_one<15>(); configure_one<16>(); configure_one<17>();
}

template <int Id>
void launch_one(const float* input, float* output) {
  using V = Variant<Id>;
  const int bytes = V::cluster
      ? (V::tensor ? kClusterTcBytes : kClusterBytes)
      : (V::tensor ? kSingleTcBytes : kSingleBytes);
  if constexpr (V::cluster) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 2;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kBatch * 2, 1, 1);
    config.blockDim = dim3(V::threads, 1, 1);
    config.dynamicSmemBytes = bytes;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cudaLaunchKernelEx(
        &config,
        cluster_kernel<V::threads, V::root,
                       V::update != kSimtUpdate,
                       V::update == kTcAllUpdate>,
        input, output);
  } else {
    single_kernel<Id><<<kBatch, V::threads, bytes>>>(input, output);
  }
}

void launch_variant(const float* input, float* output, int variant) {
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
    default:
      TORCH_CHECK(false, "native variant must be in [0, 17]");
  }
}

template <int Id>
void write_metadata(int64_t* rows) {
  using V = Variant<Id>;
  cudaFuncAttributes attributes{};
  int active = 0;
  const int bytes = V::cluster
      ? (V::tensor ? kClusterTcBytes : kClusterBytes)
      : (V::tensor ? kSingleTcBytes : kSingleBytes);
  cudaError_t status;
  if constexpr (V::cluster) {
    auto kernel = cluster_kernel<
        V::threads, V::root,
        V::update != kSimtUpdate, V::update == kTcAllUpdate>;
    status = cudaFuncGetAttributes(&attributes, kernel);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, kernel, V::threads, bytes);
    }
  } else {
    auto kernel = single_kernel<Id>;
    status = cudaFuncGetAttributes(&attributes, kernel);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, kernel, V::threads, bytes);
    }
  }
  TORCH_CHECK(status == cudaSuccess,
              "kernel resource query failed: ", cudaGetErrorString(status));

  int64_t* row = rows + static_cast<int64_t>(Id) * kMetadataColumns;
  row[0] = Id;
  row[1] = V::threads;
  row[2] = attributes.numRegs;
  row[3] = attributes.localSizeBytes;
  row[4] = attributes.sharedSizeBytes;
  row[5] = bytes;
  row[6] = active;
  row[7] = active * (V::threads / 32);
  row[8] = V::cluster ? 2 : 1;
  row[9] = V::tensor ? 1 : 0;
  row[10] = V::base;
  row[11] = V::factor;
  row[12] =
      V::cluster && V::tensor ? kTensorBlockTrsm : V::trsm;
  row[13] = V::update;
  row[14] = V::root;
  row[15] = V::threads;
}

}  // namespace

void cholesky_b64n256_prepare() {
  configure_all();
}

void cholesky_b64n256_out(const at::Tensor& data,
                          at::Tensor out,
                          int64_t variant) {
  check_input(data);
  check_output(data, out);
  TORCH_CHECK(variant >= 0 && variant < kVariantCount,
              "native variant must be in [0, 17]");
  launch_variant(
      data.data_ptr<float>(), out.data_ptr<float>(),
      static_cast<int>(variant));
  const auto status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b64n256(const at::Tensor& data, int64_t variant) {
  auto out = at::empty_like(data);
  cholesky_b64n256_out(data, out, variant);
  return out;
}

at::Tensor cholesky_b64n256_metadata() {
  auto result = at::zeros(
      {kVariantCount, kMetadataColumns},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  write_metadata<0>(rows); write_metadata<1>(rows);
  write_metadata<2>(rows); write_metadata<3>(rows);
  write_metadata<4>(rows); write_metadata<5>(rows);
  write_metadata<6>(rows); write_metadata<7>(rows);
  write_metadata<8>(rows); write_metadata<9>(rows);
  write_metadata<10>(rows); write_metadata<11>(rows);
  write_metadata<12>(rows); write_metadata<13>(rows);
  write_metadata<14>(rows); write_metadata<15>(rows);
  write_metadata<16>(rows); write_metadata<17>(rows);
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
            name=f"cholesky_b64n256_{tag}",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=None,
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "--use_fast_math",
                "--extra-device-vectorization",
                "--restrict",
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
    if tuple(data.shape) != (64, 256, 256):
        return torch.linalg.cholesky_ex(data, check_errors=False).L
    return _run_variant(data, _DEFAULT_VARIANT)
