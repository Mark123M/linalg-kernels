import hashlib
import os
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The tuner replaces this exact line in retained candidate copies.
_DEFAULT_VARIANT = 6  # POPCORN_VARIANT
_VARIANT_NAMES = (
    "staged_precise_sub4_fp32_t256",
    "staged_refined_sub8_fp32_t512",
    "staged_precise_sub4_tc_outer_t256",
    "staged_precise_sub4_tc_all_t256",
    "staged_refined_sub8_tc_all_t512",
    "staged_precise_sub4_cublas_fp32_t256",
    "staged_precise_sub4_cublas_tf32_t256",
    "cluster_dag_refined_sub8_tc_all_t512",
)
_VARIANT_COUNT = len(_VARIANT_NAMES)
_VARIANT_IDS = tuple(range(_VARIANT_COUNT))

_METADATA_COLUMNS = (
    "variant",
    "factor_threads",
    "solve_threads",
    "update_threads",
    "factor_registers",
    "solve_registers",
    "update_registers",
    "factor_shared_bytes",
    "solve_shared_bytes",
    "update_shared_bytes",
    "factor_local_bytes",
    "solve_local_bytes",
    "update_local_bytes",
    "dynamic_shared_bytes",
    "active_factor_blocks",
    "active_solve_blocks",
    "active_update_blocks",
    "scheduler",
    "root_mode",
    "solve_width",
    "update_mode",
    "tensor_inner",
    "outer_block",
    "microtile",
    "launch_count",
    "cluster_size",
    "node_count",
    "tmem_columns",
)

_CPP_SOURCE = r"""
#include <torch/extension.h>

void cholesky_b60n1024_prepare(int64_t variant);
at::Tensor cholesky_b60n1024(const at::Tensor& data, int64_t variant);
void cholesky_b60n1024_out(
    const at::Tensor& data, at::Tensor out, int64_t variant);
at::Tensor cholesky_b60n1024_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &cholesky_b60n1024_prepare,
        "Configure one B200 60x1024 Cholesky variant");
  m.def("run", &cholesky_b60n1024, "Batched 60x1024 Cholesky");
  m.def("run_out", &cholesky_b60n1024_out,
        "Batched 60x1024 Cholesky out");
  m.def("metadata", &cholesky_b60n1024_metadata,
        "B200 kernel resource metadata");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace cg = cooperative_groups;

namespace {

constexpr int kBatch = 60;
constexpr int kN = 1024;
constexpr int kMicro = 64;
constexpr int kOuter = 128;
constexpr int kMicroCount = 16;
constexpr int kPanelCount = 8;
constexpr int kVariantCount = 8;
constexpr int kMetadataColumns = 28;
constexpr int kFactorPlainBytes = 76 * 1024;
constexpr int kFactorTensorBytes = 92 * 1024;
constexpr int kFp32UpdateBytes = 68 * 1024;
constexpr int kTcUpdateBytes = 72 * 1024;
constexpr int kClusterBytes = 108 * 1024;
constexpr int kCluster = 8;
constexpr int kClusters = 16;
constexpr int kDagNodes = 372;
constexpr int kStateStride = 384;
constexpr int kControlBytes = 8192;
constexpr int kTaskSlot = 1900;
constexpr int kTmemSlot = 1901;
constexpr int kPhaseSlot = 1902;
constexpr int kBarrierByte = 7936;
constexpr int kTmemColumns = 128;
constexpr uint32_t kTmemDp = 1u << 16;

constexpr int kPreciseRoot = 0;
constexpr int kRefinedRoot = 1;
constexpr int kSub4 = 4;
constexpr int kSub8 = 8;
constexpr int kFp32Update = 0;
constexpr int kTcUpdate = 1;
constexpr int kBlasFp32Update = 2;
constexpr int kBlasTf32Update = 3;
constexpr int kStagedScheduler = 0;
constexpr int kDagScheduler = 1;

constexpr int kPotrf = 0;
constexpr int kTrsm = 1;
constexpr int kUpdate = 2;

template <int Width>
constexpr int solve_bytes() {
  return static_cast<int>(sizeof(float)) *
         (32 * (kOuter + 1) + kMicro * (kOuter + Width));
}

static_assert(solve_bytes<kSub4>() == 50304);
static_assert(solve_bytes<kSub8>() == 51328);
static_assert(
    solve_bytes<kSub8>() <= kClusterBytes - kControlBytes);

struct Node {
  int kind;
  int row;
  int column;
  int panel;
  int dep0;
  int dep1;
  int dep2;
};

template <int Id>
struct Variant;

#define SPEC(ID, THREADS, ROOT, WIDTH, UPDATE, INNER, SCHED) \
  template <> struct Variant<ID> {                          \
    static constexpr int threads = THREADS;                 \
    static constexpr int root = ROOT;                       \
    static constexpr int width = WIDTH;                     \
    static constexpr int update = UPDATE;                   \
    static constexpr bool tensor_inner = INNER;             \
    static constexpr int scheduler = SCHED;                 \
    static constexpr bool tensor =                          \
        UPDATE == kTcUpdate || INNER || SCHED == kDagScheduler; \
  }

SPEC(0, 256, kPreciseRoot, kSub4, kFp32Update, false,
     kStagedScheduler);
SPEC(1, 512, kRefinedRoot, kSub8, kFp32Update, false,
     kStagedScheduler);
SPEC(2, 256, kPreciseRoot, kSub4, kTcUpdate, false,
     kStagedScheduler);
SPEC(3, 256, kPreciseRoot, kSub4, kTcUpdate, true,
     kStagedScheduler);
SPEC(4, 512, kRefinedRoot, kSub8, kTcUpdate, true,
     kStagedScheduler);
SPEC(5, 256, kPreciseRoot, kSub4, kBlasFp32Update, false,
     kStagedScheduler);
SPEC(6, 256, kPreciseRoot, kSub4, kBlasTf32Update, false,
     kStagedScheduler);
SPEC(7, 512, kRefinedRoot, kSub8, kTcUpdate, true,
     kDagScheduler);

#undef SPEC

__device__ __forceinline__ uint32_t shared_address(
    const void* pointer) {
  return static_cast<uint32_t>(
      __cvta_generic_to_shared(const_cast<void*>(pointer)));
}

__device__ __forceinline__ float load_global(const float* pointer) {
  return __ldcg(pointer);
}

__device__ __forceinline__ void store_global(
    float* pointer, float value) {
  __stcg(pointer, value);
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
  return start | (static_cast<uint64_t>(rows) << 16) |
         (8ull << 32) | (1ull << 46);
}

__device__ __forceinline__ constexpr uint32_t tf32_descriptor() {
  return (1u << 4) | (2u << 7) | (2u << 10) |
         (8u << 17) | (4u << 24);
}

__device__ __forceinline__ void proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tensor_after_sync_fence() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_allocate(
    uint32_t* destination) {
  if (static_cast<int>(threadIdx.x) < 32) {
    const uint32_t address = shared_address(destination);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
        "[%0], %1;" :: "r"(address), "r"(kTmemColumns) : "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ void tmem_deallocate(uint32_t base) {
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < 32) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
        "r"(base), "r"(kTmemColumns));
  }
  __syncthreads();
}

__device__ __forceinline__ void tmem_relinquish() {
  if (static_cast<int>(threadIdx.x) < 32) {
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier) {
  if (threadIdx.x == 0) {
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;" ::
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

__device__ __forceinline__ void barrier_wait(
    uint64_t* barrier, int phase) {
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

__device__ __forceinline__ void issue_tf32_mma(
    uint32_t tmem_base, uint64_t a_desc, uint64_t b_desc,
    bool accumulate) {
  if (threadIdx.x == 0) {
    const uint32_t instruction = tf32_descriptor();
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

template <int RootMode>
__device__ __forceinline__ void root_pair(
    float value, float& diagonal, float& inverse) {
  if constexpr (RootMode == kPreciseRoot) {
    diagonal = __fsqrt_rn(value);
    inverse = __fdiv_rn(1.0f, diagonal);
  } else {
    inverse = rsqrtf(value);
    inverse *= fmaf(-0.5f * value, inverse * inverse, 1.5f);
    diagonal = value * inverse;
    diagonal = 0.5f * (diagonal + __fdiv_rn(value, diagonal));
    inverse = __fdiv_rn(1.0f, diagonal);
  }
}

__device__ __forceinline__ float& tile_at(
    float* tile, int row, int column) {
  return tile[row * (kOuter + 1) + column];
}

template <int RootMode>
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
        root_pair<RootMode>(
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

__device__ __forceinline__ void pack_global_slice(
    uint32_t* packed, const float* matrix,
    int row_begin, int panel_begin, int k_begin) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * 8; linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 3;
    const int column = linear & 7;
    packed[kmajor_offset(row, column, kMicro)] =
        to_tf32(load_global(
            matrix + (row_begin + row) * kN +
            panel_begin + k_begin + column));
  }
}

__device__ __forceinline__ void pack_local_slice(
    uint32_t* packed, float* tile,
    int row_begin, int column_begin) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * 8; linear += static_cast<int>(blockDim.x)) {
    const int row = linear >> 3;
    const int column = linear & 7;
    packed[kmajor_offset(row, column, kMicro)] =
        to_tf32(tile_at(
            tile, row_begin + row, column_begin + column));
  }
}

__device__ __forceinline__ void tc_update_global(
    float* matrix, int row_begin, int column_begin, int panel_begin,
    float* work, uint32_t tmem_base, uint64_t* barrier, int* phase) {
  uint32_t* a_first = reinterpret_cast<uint32_t*>(work);
  uint32_t* b_first = a_first + kMicro * kMicro;
  uint32_t* a_second = b_first + kMicro * kMicro;
  uint32_t* b_second = a_second + kMicro * kMicro;
  for (int k = 0; k < kMicro; k += 8) {
    pack_global_slice(
        a_first + k * kMicro, matrix,
        row_begin, panel_begin, k);
    pack_global_slice(
        b_first + k * kMicro, matrix,
        column_begin, panel_begin, k);
  }
  __syncthreads();
  proxy_fence();
  __syncthreads();
  for (int k = 0; k < kMicro; k += 8) {
    issue_tf32_mma(
        tmem_base,
        make_kmajor_descriptor(a_first + k * kMicro, kMicro),
        make_kmajor_descriptor(b_first + k * kMicro, kMicro),
        k != 0);
  }
  tensor_commit(barrier);
  for (int k = 0; k < kMicro; k += 8) {
    pack_global_slice(
        a_second + k * kMicro, matrix,
        row_begin, panel_begin + kMicro, k);
    pack_global_slice(
        b_second + k * kMicro, matrix,
        column_begin, panel_begin + kMicro, k);
  }
  __syncthreads();
  proxy_fence();
  barrier_wait(barrier, *phase);
  if (threadIdx.x == 0) {
    *phase ^= 1;
  }
  __syncthreads();
  for (int k = 0; k < kMicro; k += 8) {
    issue_tf32_mma(
        tmem_base,
        make_kmajor_descriptor(a_second + k * kMicro, kMicro),
        make_kmajor_descriptor(b_second + k * kMicro, kMicro),
        true);
  }
  tensor_commit(barrier);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = warp * 16 + lane;
  float prior[4] = {};
  if (warp < 4 && lane < 16) {
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      if (row_begin != column_begin || column <= row) {
        prior[column] = load_global(
            matrix + (row_begin + row) * kN +
            column_begin + column);
      }
    }
  }
  barrier_wait(barrier, *phase);
  if (threadIdx.x == 0) {
    *phase ^= 1;
  }
  __syncthreads();
  tensor_after_sync_fence();
  if (warp < 4) {
    for (int column = 0; column < kMicro; ++column) {
      const uint32_t address =
          tmem_base + static_cast<uint32_t>(warp * 32) * kTmemDp +
          static_cast<uint32_t>(column);
      const float product = tmem_load_one(address);
      if (lane < 16 &&
          (row_begin != column_begin || column <= row)) {
        float* destination =
            matrix + (row_begin + row) * kN + column_begin + column;
        const float destination_value =
            column < 4 ? prior[column] : load_global(destination);
        store_global(destination, destination_value - product);
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void tc_update_local(
    float* tile, float* work, uint32_t tmem_base,
    uint64_t* barrier, int* phase) {
  uint32_t* first = reinterpret_cast<uint32_t*>(work);
  uint32_t* second = first + 32 * kMicro;
  for (int k = 0; k < 32; k += 8) {
    pack_local_slice(
        first + k * kMicro, tile, 64, k);
  }
  __syncthreads();
  proxy_fence();
  __syncthreads();
  for (int k = 0; k < 32; k += 8) {
    const uint64_t descriptor =
        make_kmajor_descriptor(first + k * kMicro, kMicro);
    issue_tf32_mma(
        tmem_base, descriptor, descriptor, k != 0);
  }
  tensor_commit(barrier);
  for (int k = 0; k < 32; k += 8) {
    pack_local_slice(
        second + k * kMicro, tile, 64, 32 + k);
  }
  __syncthreads();
  proxy_fence();
  barrier_wait(barrier, *phase);
  if (threadIdx.x == 0) {
    *phase ^= 1;
  }
  __syncthreads();
  for (int k = 0; k < 32; k += 8) {
    const uint64_t descriptor =
        make_kmajor_descriptor(second + k * kMicro, kMicro);
    issue_tf32_mma(
        tmem_base, descriptor, descriptor, true);
  }
  tensor_commit(barrier);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = warp * 16 + lane;
  float prior[4] = {};
  if (warp < 4 && lane < 16) {
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      if (column <= row) {
        prior[column] =
            tile_at(tile, 64 + row, 64 + column);
      }
    }
  }
  barrier_wait(barrier, *phase);
  if (threadIdx.x == 0) {
    *phase ^= 1;
  }
  __syncthreads();
  tensor_after_sync_fence();
  if (warp < 4) {
    for (int column = 0; column < kMicro; ++column) {
      const uint32_t address =
          tmem_base + static_cast<uint32_t>(warp * 32) * kTmemDp +
          static_cast<uint32_t>(column);
      const float product = tmem_load_one(address);
      if (lane < 16 && column <= row) {
        const float destination_value =
            column < 4
                ? prior[column]
                : tile_at(tile, 64 + row, 64 + column);
        tile_at(tile, 64 + row, 64 + column) =
            destination_value - product;
      }
    }
  }
  __syncthreads();
}

template <int RootMode, int Width, bool TensorInner>
__device__ __forceinline__ void factor_local(
    float* tile, float* inverse_diagonal, float* tc_work,
    uint32_t tmem_base, uint64_t* barrier, int* phase) {
  potf2_32<RootMode>(tile, inverse_diagonal, 0);
  local_trsm<32, 32, Width>(
      tile, inverse_diagonal, 32, 0);
  local_update<32, 32>(tile, 32, 0);
  potf2_32<RootMode>(tile, inverse_diagonal, 32);
  local_trsm<64, 64, Width>(
      tile, inverse_diagonal, 64, 0);
  if constexpr (TensorInner) {
    tc_update_local(
        tile, tc_work, tmem_base, barrier, phase);
  } else {
    local_update<64, 64>(tile, 64, 0);
  }
  potf2_32<RootMode>(tile, inverse_diagonal, 64);
  local_trsm<32, 32, Width>(
      tile, inverse_diagonal, 96, 64);
  local_update<32, 32>(tile, 96, 64);
  potf2_32<RootMode>(tile, inverse_diagonal, 96);
}

template <int RootMode, int Width, bool TensorInner>
__device__ __forceinline__ void factor_global(
    float* matrix, int begin, float* work,
    uint32_t tmem_base, uint64_t* barrier, int* phase) {
  float* tile = work;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kOuter * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    tile_at(tile, row, column) =
        column <= row
            ? load_global(
                  matrix + (begin + row) * kN + begin + column)
            : 0.0f;
  }
  __syncthreads();
  float* inverse_diagonal = tile + kOuter * (kOuter + 1);
  float* tc_work = inverse_diagonal + kOuter;
  factor_local<RootMode, Width, TensorInner>(
      tile, inverse_diagonal, tc_work,
      tmem_base, barrier, phase);
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

template <
    int Block, int LocalColumn, int Width, int RegisterCount>
__device__ __forceinline__ void trsm_register_column(
    float (&values)[RegisterCount], const float* diagonal,
    const float* panel, int row, int lane) {
  constexpr int diagonal_ld = kOuter + 1;
  constexpr int panel_ld = kOuter + Width;
  constexpr int block_begin = Block * 32;
  constexpr int column = block_begin + LocalColumn;
  constexpr int owner = LocalColumn & (Width - 1);
  constexpr int owner_slot = LocalColumn / Width;
  static_assert(Width == kSub4 || Width == kSub8);
  static_assert(RegisterCount == 32 / Width);
  float partial = 0.0f;
#pragma unroll 4
  for (int k = lane; k < block_begin; k += Width) {
    partial = fmaf(
        panel[row * panel_ld + k],
        diagonal[LocalColumn * diagonal_ld + k], partial);
  }
#pragma unroll
  for (int slot = 0; slot < RegisterCount; ++slot) {
    const int local_k = lane + slot * Width;
    if (local_k < LocalColumn) {
      partial = fmaf(
          values[slot],
          diagonal[
              LocalColumn * diagonal_ld + block_begin + local_k],
          partial);
    }
  }
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(
        0xffffffffu, partial, offset, Width);
  }
  const float owned_rhs = values[owner_slot];
  const float rhs = __shfl_sync(
      0xffffffffu, owned_rhs, owner, Width);
  float solved = 0.0f;
  if (lane == 0) {
    solved =
        (rhs - partial) /
        diagonal[LocalColumn * diagonal_ld + column];
  }
  solved = __shfl_sync(0xffffffffu, solved, 0, Width);
  if (lane == owner) {
    values[owner_slot] = solved;
  }
}

template <int Block, int Width>
__device__ __forceinline__ void trsm_register_block(
    float* matrix, int panel_begin, float* diagonal, float* panel) {
  constexpr int diagonal_ld = kOuter + 1;
  constexpr int panel_ld = kOuter + Width;
  constexpr int block_begin = Block * 32;
  constexpr int register_count = 32 / Width;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < 32 * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int local_row = linear / kOuter;
    const int column = linear % kOuter;
    const int matrix_row = block_begin + local_row;
    diagonal[local_row * diagonal_ld + column] =
        column <= matrix_row
            ? load_global(
                  matrix + (panel_begin + matrix_row) * kN +
                  panel_begin + column)
            : 0.0f;
  }
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) & (Width - 1);
  const int row = static_cast<int>(threadIdx.x) / Width;
  if (row < kMicro) {
    float values[register_count];
#pragma unroll
    for (int slot = 0; slot < register_count; ++slot) {
      values[slot] =
          panel[
              row * panel_ld + block_begin +
              lane + slot * Width];
    }
#define TRSM_COLUMN(COLUMN)                                      \
    trsm_register_column<Block, COLUMN, Width>(                  \
        values, diagonal, panel, row, lane)
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
    for (int slot = 0; slot < register_count; ++slot) {
      panel[
          row * panel_ld + block_begin +
          lane + slot * Width] = values[slot];
    }
  }
  __syncthreads();
}

template <int Width>
__device__ __forceinline__ void trsm_global(
    float* matrix, int row_begin, int panel_begin, float* work) {
  constexpr int diagonal_ld = kOuter + 1;
  constexpr int panel_ld = kOuter + Width;
  float* diagonal = work;
  float* panel = diagonal + 32 * diagonal_ld;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    panel[row * panel_ld + column] = load_global(
        matrix + (row_begin + row) * kN + panel_begin + column);
  }
  trsm_register_block<0, Width>(
      matrix, panel_begin, diagonal, panel);
  trsm_register_block<1, Width>(
      matrix, panel_begin, diagonal, panel);
  trsm_register_block<2, Width>(
      matrix, panel_begin, diagonal, panel);
  trsm_register_block<3, Width>(
      matrix, panel_begin, diagonal, panel);
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    store_global(
        matrix + (row_begin + row) * kN + panel_begin + column,
        panel[row * panel_ld + column]);
  }
  __syncthreads();
}

__device__ __forceinline__ void fp32_update_global(
    float* matrix, int row_begin, int column_begin,
    int panel_begin, float* work) {
  constexpr int panel_ld = kOuter + 1;
  float* a_panel = work;
  float* b_panel = a_panel + kMicro * panel_ld;
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kMicro * kOuter;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kOuter;
    const int column = linear % kOuter;
    a_panel[row * panel_ld + column] = load_global(
        matrix + (row_begin + row) * kN + panel_begin + column);
    b_panel[row * panel_ld + column] = load_global(
        matrix + (column_begin + row) * kN +
        panel_begin + column);
  }
  __syncthreads();
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row_base = (warp >> 1) * 16;
  const int column_base = (warp & 1) * 32;
  const int lane_row = lane >> 3;
  const int lane_column = lane & 7;
  float value[4][4];
  bool valid[4][4];
#pragma unroll
  for (int row = 0; row < 4; ++row) {
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int output_row = row_base + lane_row + row * 4;
      const int output_column =
          column_base + lane_column + column * 8;
      valid[row][column] =
          row_begin != column_begin || output_column <= output_row;
      value[row][column] = valid[row][column]
          ? load_global(
                matrix + (row_begin + output_row) * kN +
                column_begin + output_column)
          : 0.0f;
    }
  }
#pragma unroll 1
  for (int k = 0; k < kOuter; ++k) {
    float left[4];
    float right[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      left[row] = a_panel[
          (row_base + lane_row + row * 4) * panel_ld + k];
    }
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      right[column] = b_panel[
          (column_base + lane_column + column * 8) * panel_ld + k];
    }
#pragma unroll
    for (int row = 0; row < 4; ++row) {
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        value[row][column] = fmaf(
            -left[row], right[column], value[row][column]);
      }
    }
  }
#pragma unroll
  for (int row = 0; row < 4; ++row) {
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      if (valid[row][column]) {
        const int output_row = row_base + lane_row + row * 4;
        const int output_column =
            column_base + lane_column + column * 8;
        store_global(
            matrix + (row_begin + output_row) * kN +
            column_begin + output_column,
            value[row][column]);
      }
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(256)
void copy_lower_kernel(
    const float* __restrict__ input,
    float* __restrict__ output, int copy_upper) {
  constexpr int ctas_per_matrix = 16;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / ctas_per_matrix;
  const int rank =
      static_cast<int>(blockIdx.x) % ctas_per_matrix;
  const int64_t base =
      static_cast<int64_t>(matrix_index) * kN * kN;
  for (int linear = rank * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
       linear < kN * kN;
       linear += ctas_per_matrix * static_cast<int>(blockDim.x)) {
    const int row = linear / kN;
    const int column = linear % kN;
    store_global(
        output + base + linear,
        copy_upper || column <= row ? input[base + linear] : 0.0f);
  }
}

__global__ __launch_bounds__(256)
void zero_upper_kernel(float* __restrict__ output) {
  constexpr int ctas_per_matrix = 8;
  const int matrix_index =
      static_cast<int>(blockIdx.x) / ctas_per_matrix;
  const int rank =
      static_cast<int>(blockIdx.x) % ctas_per_matrix;
  const int64_t base =
      static_cast<int64_t>(matrix_index) * kN * kN;
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

template <int RootMode, int Width, bool TensorInner, int Threads>
__global__ __launch_bounds__(Threads)
void factor_kernel(float* __restrict__ output, int panel) {
  extern __shared__ __align__(16) unsigned char dynamic_bytes[];
  float* work =
      reinterpret_cast<float*>(dynamic_bytes + kControlBytes);
  int* words = reinterpret_cast<int*>(dynamic_bytes);
  uint64_t* barrier =
      reinterpret_cast<uint64_t*>(dynamic_bytes + kBarrierByte);
  uint32_t tmem_base = 0;
  if constexpr (TensorInner) {
    barrier_init(barrier);
    if (threadIdx.x == 0) {
      words[kPhaseSlot] = 0;
    }
    __syncthreads();
    tmem_allocate(reinterpret_cast<uint32_t*>(words + kTmemSlot));
    tmem_base = static_cast<uint32_t>(words[kTmemSlot]);
    tensor_after_sync_fence();
  }
  const int matrix_index = static_cast<int>(blockIdx.x);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  factor_global<RootMode, Width, TensorInner>(
      matrix, panel * kOuter, work, tmem_base,
      barrier, words + kPhaseSlot);
  if constexpr (TensorInner) {
    tmem_deallocate(tmem_base);
    tmem_relinquish();
  }
}

template <int Width, int Threads>
__global__ __launch_bounds__(Threads)
void solve_kernel(
    float* __restrict__ output, int panel, int remaining) {
  extern __shared__ __align__(16) unsigned char dynamic_bytes[];
  const int matrix_index =
      static_cast<int>(blockIdx.x) / remaining;
  const int row_index =
      static_cast<int>(blockIdx.x) % remaining;
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  trsm_global<Width>(
      matrix,
      (panel * 2 + 2 + row_index) * kMicro,
      panel * kOuter,
      reinterpret_cast<float*>(dynamic_bytes));
}

__device__ __forceinline__ void decode_update_task(
    int task, int panel, int& row, int& column) {
  int cursor = task;
  const int first = panel * 2 + 2;
  for (int c = first; c < kMicroCount; ++c) {
    const int count = kMicroCount - c;
    if (cursor < count) {
      column = c;
      row = c + cursor;
      return;
    }
    cursor -= count;
  }
  row = -1;
  column = -1;
}

__global__ __launch_bounds__(256)
void fp32_update_kernel(
    float* __restrict__ output, int panel, int tasks) {
  extern __shared__ __align__(16) unsigned char dynamic_bytes[];
  const int matrix_index =
      static_cast<int>(blockIdx.x) / tasks;
  const int task = static_cast<int>(blockIdx.x) % tasks;
  int row;
  int column;
  decode_update_task(task, panel, row, column);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  fp32_update_global(
      matrix, row * kMicro, column * kMicro, panel * kOuter,
      reinterpret_cast<float*>(dynamic_bytes));
}

template <int Threads>
__global__ __launch_bounds__(Threads)
void tc_update_kernel(
    float* __restrict__ output, int panel, int tasks) {
  extern __shared__ __align__(16) unsigned char dynamic_bytes[];
  int* words = reinterpret_cast<int*>(dynamic_bytes);
  uint64_t* barrier =
      reinterpret_cast<uint64_t*>(dynamic_bytes + kBarrierByte);
  barrier_init(barrier);
  if (threadIdx.x == 0) {
    words[kPhaseSlot] = 0;
  }
  __syncthreads();
  tmem_allocate(reinterpret_cast<uint32_t*>(words + kTmemSlot));
  const uint32_t tmem_base =
      static_cast<uint32_t>(words[kTmemSlot]);
  tensor_after_sync_fence();
  const int matrix_index =
      static_cast<int>(blockIdx.x) / tasks;
  const int task = static_cast<int>(blockIdx.x) % tasks;
  int row;
  int column;
  decode_update_task(task, panel, row, column);
  float* matrix =
      output + static_cast<int64_t>(matrix_index) * kN * kN;
  tc_update_global(
      matrix, row * kMicro, column * kMicro, panel * kOuter,
      reinterpret_cast<float*>(dynamic_bytes + kControlBytes),
      tmem_base, barrier, words + kPhaseSlot);
  tmem_deallocate(tmem_base);
  tmem_relinquish();
}

__device__ __forceinline__ int graph_panel_base(int panel) {
  int result = 0;
  for (int p = 0; p < panel; ++p) {
    const int remaining = 14 - 2 * p;
    result += 1 + remaining + remaining * (remaining + 1) / 2;
  }
  return result;
}

__device__ __forceinline__ int factor_id(int panel) {
  return graph_panel_base(panel);
}

__device__ __forceinline__ int trsm_id(int row, int panel) {
  const int start = 2 * panel;
  return graph_panel_base(panel) + 1 + row - start - 2;
}

__device__ __forceinline__ int update_id(
    int row, int column, int panel) {
  const int start = 2 * panel;
  int result = graph_panel_base(panel) + 1 + 14 - start;
  for (int c = start + 2; c < column; ++c) {
    result += kMicroCount - c;
  }
  return result + row - column;
}

__device__ __forceinline__ Node graph_node(int id) {
  for (int panel = 0; panel < kPanelCount; ++panel) {
    const int start = 2 * panel;
    const int base = graph_panel_base(panel);
    const int remaining = 14 - start;
    if (id == base) {
      if (panel == 0) {
        return Node{kPotrf, start, start, panel, -1, -1, -1};
      }
      return Node{
          kPotrf, start, start, panel,
          update_id(start, start, panel - 1),
          update_id(start + 1, start, panel - 1),
          update_id(start + 1, start + 1, panel - 1)};
    }
    const int solve_end = base + 1 + remaining;
    if (id < solve_end) {
      const int row = start + 2 + id - (base + 1);
      return Node{
          kTrsm, row, start, panel,
          factor_id(panel),
          panel == 0 ? -1 : update_id(row, start, panel - 1),
          panel == 0 ? -1 : update_id(row, start + 1, panel - 1)};
    }
    const int update_end =
        solve_end + remaining * (remaining + 1) / 2;
    if (id < update_end) {
      int cursor = solve_end;
      for (int column = start + 2;
           column < kMicroCount; ++column) {
        const int count = kMicroCount - column;
        if (id < cursor + count) {
          const int row = column + id - cursor;
          return Node{
              kUpdate, row, column, panel,
              trsm_id(row, panel),
              trsm_id(column, panel),
              panel == 0
                  ? -1
                  : update_id(row, column, panel - 1)};
        }
        cursor += count;
      }
    }
  }
  return Node{-1, 0, 0, 0, -1, -1, -1};
}

__device__ __forceinline__ bool dependency_done(
    int* states, int dependency) {
  if (dependency < 0) {
    return true;
  }
  int value;
  asm volatile(
      "ld.acquire.cluster.b32 %0, [%1];"
      : "=r"(value) : "l"(states + dependency) : "memory");
  return value == 2;
}

__device__ __forceinline__ int cluster_atomic_load(int* pointer) {
  int value;
  asm volatile(
      "ld.acquire.cluster.b32 %0, [%1];"
      : "=r"(value) : "l"(pointer) : "memory");
  return value;
}

__device__ __forceinline__ bool cluster_atomic_claim(int* pointer) {
  int previous;
  asm volatile(
      "atom.cas.acquire.cluster.b32 %0, [%1], %2, %3;"
      : "=r"(previous)
      : "l"(pointer), "r"(0), "r"(1)
      : "memory");
  return previous == 0;
}

__device__ __forceinline__ void cluster_atomic_complete(
    int* state_pointer, int* count_pointer) {
  asm volatile(
      "st.release.cluster.b32 [%0], %1;"
      :: "l"(state_pointer), "r"(2)
      : "memory");
  int previous;
  asm volatile(
      "atom.add.release.cluster.s32 %0, [%1], %2;"
      : "=r"(previous)
      : "l"(count_pointer), "r"(1)
      : "memory");
}

__device__ __forceinline__ void execute_dag_node(
    const Node& node, float* matrix, float* work,
    uint32_t tmem_base, uint64_t* barrier, int* phase) {
  if (node.kind == kPotrf) {
    factor_global<kRefinedRoot, kSub8, true>(
        matrix, node.row * kMicro, work,
        tmem_base, barrier, phase);
  } else if (node.kind == kTrsm) {
    trsm_global<kSub8>(
        matrix, node.row * kMicro, node.column * kMicro, work);
  } else {
    tc_update_global(
        matrix, node.row * kMicro, node.column * kMicro,
        node.panel * kOuter, work,
        tmem_base, barrier, phase);
  }
}

__device__ __forceinline__ void run_dag(
    cg::cluster_group cluster, float* local_base, int* states,
    float* matrix, float* work, uint32_t tmem_base,
    uint64_t* barrier, int* phase) {
  int* local_words = reinterpret_cast<int*>(local_base);
  if (cluster.block_rank() == 0) {
    for (int node = static_cast<int>(threadIdx.x);
         node < kDagNodes; node += static_cast<int>(blockDim.x)) {
      states[node] = 0;
    }
    if (threadIdx.x == 0) {
      states[kDagNodes] = 0;
    }
  }
  cluster.sync();
  int cursor = static_cast<int>(cluster.block_rank());
  while (true) {
    if (threadIdx.x == 0) {
      local_words[kTaskSlot] =
          cluster_atomic_load(states + kDagNodes) < kDagNodes
              ? -1 : -2;
    }
    __syncthreads();
    if (local_words[kTaskSlot] == -2) {
      break;
    }
    if (threadIdx.x == 0) {
      int claimed = -1;
      for (int attempt = 0; attempt < kDagNodes; ++attempt) {
        const int node_id = (cursor + attempt) % kDagNodes;
        const Node node = graph_node(node_id);
        if (dependency_done(states, node.dep0) &&
            dependency_done(states, node.dep1) &&
            dependency_done(states, node.dep2) &&
            cluster_atomic_claim(states + node_id)) {
          claimed = node_id;
          cursor = (node_id + kCluster) % kDagNodes;
          break;
        }
      }
      local_words[kTaskSlot] = claimed;
    }
    __syncthreads();
    const int claimed = local_words[kTaskSlot];
    if (claimed >= 0) {
      const Node node = graph_node(claimed);
      __threadfence();
      execute_dag_node(
          node, matrix, work, tmem_base, barrier, phase);
      __syncthreads();
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) {
        cluster_atomic_complete(
            states + claimed, states + kDagNodes);
      }
    } else if (threadIdx.x == 0) {
      __nanosleep(64);
    }
    __syncthreads();
  }
  cluster.sync();
}

__global__ __launch_bounds__(512)
void cluster_dag_kernel(
    const float* __restrict__ input,
    float* __restrict__ output) {
  extern __shared__ __align__(16) unsigned char dynamic_bytes[];
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_index =
      static_cast<int>(blockIdx.x) / kCluster;
  float* local_base = reinterpret_cast<float*>(dynamic_bytes);
  float* work =
      reinterpret_cast<float*>(dynamic_bytes + kControlBytes);
  int* words = reinterpret_cast<int*>(dynamic_bytes);
  int* root_words = reinterpret_cast<int*>(
      cluster.map_shared_rank(local_base, 0));
  uint64_t* barrier =
      reinterpret_cast<uint64_t*>(dynamic_bytes + kBarrierByte);
  barrier_init(barrier);
  if (threadIdx.x == 0) {
    words[kPhaseSlot] = 0;
  }
  __syncthreads();
  tmem_allocate(reinterpret_cast<uint32_t*>(words + kTmemSlot));
  const uint32_t tmem_base =
      static_cast<uint32_t>(words[kTmemSlot]);
  tensor_after_sync_fence();

  for (int local_matrix = 0; local_matrix < 4; ++local_matrix) {
    const int matrix_index =
        cluster_index + local_matrix * kClusters;
    if (matrix_index >= kBatch) {
      continue;
    }
    const int64_t base =
        static_cast<int64_t>(matrix_index) * kN * kN;
    for (int linear = rank * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
         linear < kN * kN;
         linear += kCluster * static_cast<int>(blockDim.x)) {
      const int row = linear / kN;
      const int column = linear % kN;
      store_global(
          output + base + linear,
          column <= row ? input[base + linear] : 0.0f);
    }
    cluster.sync();
    run_dag(
        cluster, local_base,
        root_words + local_matrix * kStateStride,
        output + base, work, tmem_base,
        barrier, words + kPhaseSlot);
  }
  tmem_deallocate(tmem_base);
  tmem_relinquish();
}

void check_input(const at::Tensor& data) {
  TORCH_CHECK(data.is_cuda(), "input must be CUDA");
  TORCH_CHECK(
      data.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(data.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      data.dim() == 3 && data.size(0) == kBatch &&
      data.size(1) == kN && data.size(2) == kN,
      "native input must have shape (60, 1024, 1024)");
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
  if constexpr (V::scheduler == kDagScheduler) {
    configure_dynamic(cluster_dag_kernel, kClusterBytes);
    checked_attributes(cluster_dag_kernel);
  } else {
    auto factor =
        factor_kernel<V::root, V::width, V::tensor_inner, V::threads>;
    auto solve = solve_kernel<V::width, V::threads>;
    configure_dynamic(
        factor,
        V::tensor_inner ? kFactorTensorBytes : kFactorPlainBytes);
    configure_dynamic(solve, solve_bytes<V::width>());
    checked_attributes(factor);
    checked_attributes(solve);
    if constexpr (V::update == kFp32Update) {
      configure_dynamic(fp32_update_kernel, kFp32UpdateBytes);
      checked_attributes(fp32_update_kernel);
    } else if constexpr (V::update == kTcUpdate) {
      auto update = tc_update_kernel<V::threads>;
      configure_dynamic(update, kTcUpdateBytes);
      checked_attributes(update);
    }
  }
}

void check_cublas(cublasStatus_t status, const char* role) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      role, " failed with cuBLAS status ", static_cast<int>(status));
}

void launch_copy(
    const float* input, float* output, bool copy_upper) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch * 16, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(
      &config, copy_lower_kernel, input, output,
      copy_upper ? 1 : 0);
}

void launch_zero_upper(float* output) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch * 8, 1, 1);
  config.blockDim = dim3(256, 1, 1);
  cudaLaunchKernelEx(&config, zero_upper_kernel, output);
}

void launch_blas_update(
    float* output, int panel, bool fast_tf32) {
  const int begin = (panel + 1) * kOuter;
  const int remaining = kN - begin;
  const int panel_begin = panel * kOuter;
  float* panel_pointer =
      output + begin * kN + panel_begin;
  float* destination = output + begin * kN + begin;
  const float alpha = -1.0f;
  const float beta = 1.0f;
  const long long stride = static_cast<long long>(kN) * kN;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          remaining, remaining, kOuter,
          &alpha,
          panel_pointer, CUDA_R_32F, kN, stride,
          panel_pointer, CUDA_R_32F, kN, stride,
          &beta,
          destination, CUDA_R_32F, kN, stride,
          kBatch,
          fast_tf32
              ? CUBLAS_COMPUTE_32F_FAST_TF32
              : CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT),
      "batched trailing GEMM");
}

template <int Id>
void launch_staged(float* output, const float* input) {
  using V = Variant<Id>;
  constexpr bool use_blas =
      V::update == kBlasFp32Update ||
      V::update == kBlasTf32Update;
  launch_copy(input, output, use_blas);
  for (int panel = 0; panel < kPanelCount; ++panel) {
    cudaLaunchConfig_t factor_config{};
    factor_config.gridDim = dim3(kBatch, 1, 1);
    factor_config.blockDim = dim3(V::threads, 1, 1);
    factor_config.dynamicSmemBytes =
        V::tensor_inner ? kFactorTensorBytes : kFactorPlainBytes;
    cudaLaunchKernelEx(
        &factor_config,
        factor_kernel<
            V::root, V::width, V::tensor_inner, V::threads>,
        output, panel);
    const int remaining = kMicroCount - panel * 2 - 2;
    if (remaining == 0) {
      continue;
    }
    cudaLaunchConfig_t solve_config{};
    solve_config.gridDim = dim3(kBatch * remaining, 1, 1);
    solve_config.blockDim = dim3(V::threads, 1, 1);
    solve_config.dynamicSmemBytes = solve_bytes<V::width>();
    cudaLaunchKernelEx(
        &solve_config, solve_kernel<V::width, V::threads>,
        output, panel, remaining);
    if constexpr (use_blas) {
      launch_blas_update(
          output, panel, V::update == kBlasTf32Update);
    } else {
      const int tasks = remaining * (remaining + 1) / 2;
      cudaLaunchConfig_t update_config{};
      update_config.gridDim = dim3(kBatch * tasks, 1, 1);
      if constexpr (V::update == kFp32Update) {
        update_config.blockDim = dim3(256, 1, 1);
        update_config.dynamicSmemBytes = kFp32UpdateBytes;
        cudaLaunchKernelEx(
            &update_config, fp32_update_kernel,
            output, panel, tasks);
      } else {
        update_config.blockDim = dim3(V::threads, 1, 1);
        update_config.dynamicSmemBytes = kTcUpdateBytes;
        cudaLaunchKernelEx(
            &update_config, tc_update_kernel<V::threads>,
            output, panel, tasks);
      }
    }
  }
  if constexpr (use_blas) {
    launch_zero_upper(output);
  }
}

void launch_cluster(float* output, const float* input) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = kCluster;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kClusters * kCluster, 1, 1);
  config.blockDim = dim3(512, 1, 1);
  config.dynamicSmemBytes = kClusterBytes;
  config.attrs = &attribute;
  config.numAttrs = 1;
  cudaLaunchKernelEx(
      &config, cluster_dag_kernel, input, output);
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
    default:
      TORCH_CHECK(false, "native variant must be in [0, 7]");
  }
}

void launch_variant(
    const float* input, float* output, int variant) {
  switch (variant) {
    case 0: launch_staged<0>(output, input); break;
    case 1: launch_staged<1>(output, input); break;
    case 2: launch_staged<2>(output, input); break;
    case 3: launch_staged<3>(output, input); break;
    case 4: launch_staged<4>(output, input); break;
    case 5: launch_staged<5>(output, input); break;
    case 6: launch_staged<6>(output, input); break;
    case 7: launch_cluster(output, input); break;
    default:
      TORCH_CHECK(false, "native variant must be in [0, 7]");
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
  cudaFuncAttributes factor{};
  cudaFuncAttributes solve{};
  cudaFuncAttributes update{};
  int factor_active = 0;
  int solve_active = 0;
  int update_active = 0;
  if constexpr (V::scheduler == kDagScheduler) {
    configure_dynamic(cluster_dag_kernel, kClusterBytes);
    factor = checked_attributes(cluster_dag_kernel);
    factor_active = active_blocks(
        cluster_dag_kernel, V::threads, kClusterBytes);
  } else {
    auto factor_kernel_pointer =
        factor_kernel<V::root, V::width, V::tensor_inner, V::threads>;
    auto solve_kernel_pointer =
        solve_kernel<V::width, V::threads>;
    constexpr int factor_bytes =
        V::tensor_inner ? kFactorTensorBytes : kFactorPlainBytes;
    configure_dynamic(factor_kernel_pointer, factor_bytes);
    configure_dynamic(
        solve_kernel_pointer, solve_bytes<V::width>());
    factor = checked_attributes(factor_kernel_pointer);
    solve = checked_attributes(solve_kernel_pointer);
    factor_active =
        active_blocks(
            factor_kernel_pointer, V::threads, factor_bytes);
    solve_active =
        active_blocks(
            solve_kernel_pointer, V::threads,
            solve_bytes<V::width>());
    if constexpr (V::update == kFp32Update) {
      configure_dynamic(fp32_update_kernel, kFp32UpdateBytes);
      update = checked_attributes(fp32_update_kernel);
      update_active = active_blocks(
          fp32_update_kernel, 256, kFp32UpdateBytes);
    } else if constexpr (V::update == kTcUpdate) {
      auto update_kernel_pointer =
          tc_update_kernel<V::threads>;
      configure_dynamic(update_kernel_pointer, kTcUpdateBytes);
      update = checked_attributes(update_kernel_pointer);
      update_active =
          active_blocks(
              update_kernel_pointer, V::threads, kTcUpdateBytes);
    }
  }
  int64_t* row =
      rows + static_cast<int64_t>(Id) * kMetadataColumns;
  row[0] = Id;
  row[1] = V::threads;
  row[2] = V::threads;
  row[3] =
      V::update == kFp32Update ? 256 :
      (V::update == kTcUpdate ? V::threads : 0);
  row[4] = factor.numRegs;
  row[5] = solve.numRegs;
  row[6] = update.numRegs;
  row[7] = factor.sharedSizeBytes;
  row[8] = solve.sharedSizeBytes;
  row[9] = update.sharedSizeBytes;
  row[10] = factor.localSizeBytes;
  row[11] = solve.localSizeBytes;
  row[12] = update.localSizeBytes;
  row[13] =
      V::scheduler == kDagScheduler
          ? kClusterBytes : solve_bytes<V::width>();
  row[14] = factor_active;
  row[15] = solve_active;
  row[16] = update_active;
  row[17] = V::scheduler;
  row[18] = V::root;
  row[19] = V::width;
  row[20] = V::update;
  row[21] = V::tensor_inner ? 1 : 0;
  row[22] = kOuter;
  row[23] = kMicro;
  row[24] =
      V::scheduler == kDagScheduler ? 1 :
      (V::update == kBlasFp32Update ||
       V::update == kBlasTf32Update ? 24 : 23);
  row[25] = V::scheduler == kDagScheduler ? kCluster : 0;
  row[26] = V::scheduler == kDagScheduler ? kDagNodes + 1 : 0;
  row[27] = V::tensor ? kTmemColumns : 0;
}

}  // namespace

void cholesky_b60n1024_prepare(int64_t variant) {
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 7]");
  configure_variant(static_cast<int>(variant));
}

void cholesky_b60n1024_out(
    const at::Tensor& data, at::Tensor output, int64_t variant) {
  check_input(data);
  check_output(data, output);
  TORCH_CHECK(
      variant >= 0 && variant < kVariantCount,
      "native variant must be in [0, 7]");
  c10::cuda::CUDAGuard device_guard(data.device());
  launch_variant(
      data.data_ptr<float>(), output.data_ptr<float>(),
      static_cast<int>(variant));
  const cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "Cholesky launch failed: ", cudaGetErrorString(status));
}

at::Tensor cholesky_b60n1024(
    const at::Tensor& data, int64_t variant) {
  auto output = at::empty_like(data);
  cholesky_b60n1024_out(data, output, variant);
  return output;
}

at::Tensor cholesky_b60n1024_metadata() {
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
            name=f"cholesky_b60n1024_b200_{tag}",
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
        and tuple(data.shape) == (60, 1024, 1024)
    ):
        return _run_variant(data, _DEFAULT_VARIANT)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
