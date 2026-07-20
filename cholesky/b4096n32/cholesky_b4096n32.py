import os
from functools import lru_cache

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


# The Popcorn tuner replaces this exact line in temporary, untracked copies.
# Variant 3 won the full Popcorn B200 leaderboard sweep on 2026-07-20.
_DEFAULT_VARIANT = 3  # POPCORN_VARIANT
_VARIANT_COUNT = 10

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

template <int Warps, bool FastRoot>
__global__ __launch_bounds__(Warps * 32, 32 / Warps)
void crout_32_kernel(const float* __restrict__ input,
                     float* __restrict__ output) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int matrix = static_cast<int>(blockIdx.x) * Warps + warp;
  const float* a = input + static_cast<int64_t>(matrix) * kN * kN;
  float* result = output + static_cast<int64_t>(matrix) * kN * kN;

  // Lane i owns row i of L. Reading A[j, i] instead of A[i, j] is valid for
  // the symmetric input and makes every warp input transaction contiguous.
  float row[kN];
#pragma unroll
  for (int k = 0; k < kN; ++k) {
    row[k] = 0.0f;
  }

#pragma unroll
  for (int j = 0; j < kN; ++j) {
    float value = a[j * kN + lane];
#pragma unroll
    for (int k = 0; k < j; ++k) {
      const float pivot_row = __shfl_sync(kFullMask, row[k], j);
      value = fmaf(-row[k], pivot_row, value);
    }

    float inverse = 0.0f;
    if (lane == j) {
      float diagonal;
      if constexpr (FastRoot) {
        inverse = rsqrtf(value);
        const float square = inverse * inverse;
        inverse *= fmaf(-0.5f * value, square, 1.5f);
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

template <int Warps, bool FastRoot>
void launch_typed(const float* input, float* output) {
  constexpr int threads = Warps * 32;
  constexpr int blocks = kBatch / Warps;
  crout_32_kernel<Warps, FastRoot><<<blocks, threads>>>(input, output);
}

void launch_variant(const float* input,
                    float* output,
                    int variant) {
  switch (variant) {
    case 0: launch_typed<1, false>(input, output); break;
    case 1: launch_typed<1, true >(input, output); break;
    case 2: launch_typed<2, false>(input, output); break;
    case 3: launch_typed<2, true >(input, output); break;
    case 4: launch_typed<4, false>(input, output); break;
    case 5: launch_typed<4, true >(input, output); break;
    case 6: launch_typed<8, false>(input, output); break;
    case 7: launch_typed<8, true >(input, output); break;
    case 8: launch_typed<16, false>(input, output); break;
    case 9: launch_typed<16, true >(input, output); break;
    default: TORCH_CHECK(false, "variant must be in [0, 9]");
  }
}

template <int Warps, bool FastRoot>
void write_metadata(int64_t* rows, int variant) {
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  auto status = cudaFuncGetAttributes(
      &attributes, crout_32_kernel<Warps, FastRoot>);
  TORCH_CHECK(status == cudaSuccess,
              "cudaFuncGetAttributes failed: ", cudaGetErrorString(status));
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, crout_32_kernel<Warps, FastRoot>, Warps * 32, 0);
  TORCH_CHECK(status == cudaSuccess,
              "occupancy query failed: ", cudaGetErrorString(status));

  int64_t* row = rows + static_cast<int64_t>(variant) * 9;
  row[0] = variant;
  row[1] = Warps;
  row[2] = FastRoot ? 1 : 0;
  row[3] = Warps * 32;
  row[4] = attributes.numRegs;
  row[5] = attributes.localSizeBytes;
  row[6] = attributes.sharedSizeBytes;
  row[7] = active_blocks;
  row[8] = active_blocks * Warps;
}

}  // namespace

void cholesky_4096x32_out(const at::Tensor& data,
                          at::Tensor out,
                          int64_t variant) {
  check_input(data);
  check_output(data, out);
  TORCH_CHECK(variant >= 0 && variant < 10, "variant must be in [0, 9]");
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
  auto result = at::empty({10, 9},
                          at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* rows = result.data_ptr<int64_t>();
  write_metadata<1, false>(rows, 0);
  write_metadata<1, true >(rows, 1);
  write_metadata<2, false>(rows, 2);
  write_metadata<2, true >(rows, 3);
  write_metadata<4, false>(rows, 4);
  write_metadata<4, true >(rows, 5);
  write_metadata<8, false>(rows, 6);
  write_metadata<8, true >(rows, 7);
  write_metadata<16, false>(rows, 8);
  write_metadata<16, true >(rows, 9);
  return result;
}
"""


@lru_cache(maxsize=1)
def _native_module():
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_inline(
            name="cholesky_b4096n32_crout_v3",
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
