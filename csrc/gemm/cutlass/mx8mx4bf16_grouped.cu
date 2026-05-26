/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <mslk/utils/utils.h>
#include <mslk/utils/tuning_cache.cuh>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include "mx8mx4bf16_grouped/mx8mx4bf16_grouped_manifest.cuh"
#endif

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

namespace {

Kernel_mx8mx4bf16_grouped
get_mx8mx4_grouped_kernel_via_heuristics(int M, int N, int K) {
  if (M <= 64) {
    if (N <= 1024) {
      return mx8mx4bf16_grouped_128_64_256_1_1_1;
    } else if (N >= 4096 && K <= 4096) {
      return mx8mx4bf16_grouped_256_64_128_2_1_1;
    } else {
      return mx8mx4bf16_grouped_256_64_256_2_1_1;
    }
  } else if (M <= 128) {
    if (N >= 4096 && K <= 4096) {
      return mx8mx4bf16_grouped_256_128_128_2_1_1;
    } else {
      return mx8mx4bf16_grouped_256_128_256_2_1_1;
    }
  } else {
    if (N >= 4096 && K <= 4096) {
      return mx8mx4bf16_grouped_256_256_128_2_1_1;
    } else {
      return mx8mx4bf16_grouped_256_256_256_2_1_1;
    }
  }
}

Kernel_mx8mx4bf16_grouped get_mx8mx4_grouped_kernel_via_tuning(
    int M,
    int N,
    int K,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets) {
  static TuningCache cache("mx8mx4bf16_grouped");

  M = nextPowerOf2OrRoundUp(M, 1024, 1024);
  N = nextPowerOf2OrRoundUp(N, 1024, 1024);
  K = nextPowerOf2OrRoundUp(K, 1024, 1024);

  const std::string shape_key =
      std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);

  const auto& kernels = get_mx8mx4bf16_grouped_kernels();
  return cache.findBestKernelMaybeAutotune(
      shape_key, kernels, XQ, WQ, x_scale, w_scale, output, G, offsets);
}

} // namespace

at::Tensor mx8mx4bf16_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    std::optional<at::Tensor> output_maybe) {
  TORCH_CHECK(
      XQ.is_cuda() && XQ.is_contiguous(), "XQ must be contiguous CUDA.");
  TORCH_CHECK(WQ.is_cuda(), "WQ must be CUDA.");
  TORCH_CHECK(WQ.transpose(-2, -1).is_contiguous(), "WQ must be column major.");
  TORCH_CHECK(x_scale.is_cuda() && x_scale.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());
  TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA.");
  TORCH_CHECK(offsets.dtype() == at::kInt, "offsets must be int32.");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D.");

  TORCH_CHECK(XQ.dtype() == at::kFloat8_e4m3fn, "XQ must be MXFP8 E4M3.");
  TORCH_CHECK(WQ.dtype() == at::kFloat4_e2m1fn_x2, "WQ must be MXFP4 E2M1.");

  TORCH_CHECK(
      XQ.dim() == 2 && WQ.dim() == 3,
      "mx8mx4bf16_grouped_mm currently supports only 2D-3D inputs.");

  const int64_t G = offsets.size(0);
  TORCH_CHECK(
      G > 0 && G <= 1024, "mx8mx4bf16_grouped_mm supports 1 to 1024 groups.");
  const int64_t total_M = XQ.size(0);
  const int64_t N = WQ.size(-1);
  const int64_t K_storage = WQ.size(-2);
  const int64_t K = K_storage * 2;

  TORCH_CHECK(WQ.size(0) == G, "WQ group dimension must match offsets.");
  TORCH_CHECK(
      XQ.size(1) == K,
      "XQ shape must be [total_M, K] and WQ shape must be [G, K/2, N].");
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32.");

  at::Tensor output = output_maybe.has_value()
      ? output_maybe.value()
      : at::empty({total_M, N}, XQ.options().dtype(at::kBFloat16));
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) == total_M && output.size(1) == N,
      "output must have shape [total_M, N].");

  if (output.numel() == 0) {
    return output;
  }

  at::Tensor WQ_contig = WQ.transpose(-2, -1);
  const int64_t heuristic_M = total_M / G;
  auto kernel = [&]() {
    if (std::getenv("MSLK_AUTOTUNE_ENABLE")) {
      return get_mx8mx4_grouped_kernel_via_tuning(
          heuristic_M,
          N,
          K,
          XQ,
          WQ_contig,
          x_scale,
          w_scale,
          output,
          G,
          offsets);
    }
    return get_mx8mx4_grouped_kernel_via_heuristics(heuristic_M, N, K);
  }();

  return kernel(XQ, WQ_contig, x_scale, w_scale, output, G, offsets);
}

#else

at::Tensor mx8mx4bf16_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    std::optional<at::Tensor> output) {
  throw std::runtime_error("CUDA version is older than 12.8");
}

#endif

} // namespace mslk::gemm
