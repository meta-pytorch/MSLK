/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/device_memory.h>
#include <mslk/utils/utils.h>
#include <mslk/utils/tuning_cache.cuh>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include "mx6mx6bf16/mx6mx6bf16_manifest.cuh"
#endif

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

namespace {

// Tile selector tuned on B200 via autotune sweep + 10-rep production bench.
Kernel_mx6mx6bf16 get_kernel_via_heuristics(int M, int N, int K) {
  if (M <= 16 && (N == 9216 || N == 10240)) {
    return mx6mx6bf16_256_128_4_1_1;
  }
  if (M <= 192) {
    return mx6mx6bf16_256_128_2_1_1;
  }
  if (M <= 256) {
    if (N >= 10240) {
      return mx6mx6bf16_256_256_2_1_1;
    }
    // Tested but dropped
    //   if ((N == 7168 || N == 8192) && K >= 12288) {
    //     return mx6mx6bf16_256_128_2_2_1;
    //   }
    return mx6mx6bf16_256_128_2_1_1;
  }
  if (M <= 512) {
    if (N >= 5120 && K >= 4096) {
      return mx6mx6bf16_256_256_2_1_1;
    }
    return mx6mx6bf16_256_128_2_1_1;
  }
  return mx6mx6bf16_256_256_2_1_1;
}

const std::unordered_map<std::string, Kernel_mx6mx6bf16>&
get_mx6mx6bf16_kernels() {
  static const std::unordered_map<std::string, Kernel_mx6mx6bf16> kernels = {
      {"mx6mx6bf16_128_128_1_1_1", mx6mx6bf16_128_128_1_1_1},
      {"mx6mx6bf16_128_128_2_2_1", mx6mx6bf16_128_128_2_2_1},
      {"mx6mx6bf16_128_128_4_1_1", mx6mx6bf16_128_128_4_1_1},
      {"mx6mx6bf16_256_128_2_1_1", mx6mx6bf16_256_128_2_1_1},
      {"mx6mx6bf16_256_128_2_2_1", mx6mx6bf16_256_128_2_2_1},
      {"mx6mx6bf16_256_128_4_1_1", mx6mx6bf16_256_128_4_1_1},
      {"mx6mx6bf16_256_256_2_1_1", mx6mx6bf16_256_256_2_1_1},
      {"mx6mx6bf16_256_256_2_4_1", mx6mx6bf16_256_256_2_4_1},
  };
  return kernels;
}

Kernel_mx6mx6bf16 get_kernel_via_tuning(
    int M,
    int N,
    int K,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output) {
  static TuningCache cache("mx6mx6bf16");

  M = nextPowerOf2OrRoundUp(M, 1024, 1024);
  N = nextPowerOf2OrRoundUp(N, 1024, 1024);
  K = nextPowerOf2OrRoundUp(K, 1024, 1024);

  const std::string shape_key =
      std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);

  const auto& kernels = get_mx6mx6bf16_kernels();

  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key, kernels, XQ, WQ, x_scale, w_scale, output);
  return kernel;
}

} // namespace

at::Tensor mx6mx6bf16(
    at::Tensor XQ, // MX FP6 (e2m3)
    at::Tensor WQ, // MX FP6 (e2m3)
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output) {
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(x_scale.is_cuda() && x_scale.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());

  const auto M = XQ.size(0);
  const auto N = WQ.size(0);
  // XQ is packed 6-bit: K elements occupy K*6/8 = K*3/4 bytes, so K = bytes * 4
  // / 3.
  const auto K = XQ.size(1) * 4 / 3;
  TORCH_CHECK(
      K % 32 == 0,
      "K must be a multiple of block size 32 for MX block-scaled GEMM");

  if (M == 0 || N == 0 || K == 0) {
    return at::zeros({M, N}, XQ.options().dtype(at::kBFloat16));
  }

  at::Tensor out = output.has_value()
      ? output.value()
      : at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  auto kernel = [&]() {
    if (std::getenv("MSLK_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(M, N, K, XQ, WQ, x_scale, w_scale, out);
    }
    return get_kernel_via_heuristics(M, N, K);
  }();
  return kernel(XQ, WQ, x_scale, w_scale, out);
}

#else

at::Tensor mx6mx6bf16(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output) {
  throw std::runtime_error("CUDA version is older than 12.8");
}

#endif

} // namespace mslk::gemm
