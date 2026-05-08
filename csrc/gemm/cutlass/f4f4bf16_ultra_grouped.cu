/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
#include "f4f4bf16_ultra_grouped/f4f4bf16_ultra_grouped_manifest.cuh"
#endif

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)

Kernel_f4f4bf16_ultra_grouped
get_ultra_kernel_via_heuristics(int M, int N, int K) {
  if (M <= 128) {
    return f4f4bf16_ultra_grouped_256_128_768_2_1_1;
  }
  return f4f4bf16_ultra_grouped_256_256_768_2_1_1;
}

at::Tensor f4f4bf16_ultra_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    std::optional<at::Tensor> output_maybe) {
  TORCH_CHECK(offsets.dtype() == at::kInt, "offsets must be int32.");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D tensor.");
  TORCH_CHECK(XQ.is_contiguous(), "XQ must be row major.");
  TORCH_CHECK(WQ.transpose(-2, -1).is_contiguous(), "WQ must be column major.");
  TORCH_CHECK(XQ.dtype() == at::kFloat4_e2m1fn_x2, "XQ must be FP4.");
  TORCH_CHECK(WQ.dtype() == at::kFloat4_e2m1fn_x2, "WQ must be FP4.");
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat8_e4m3fn, "x_scale must be FP8 e4m3.");
  TORCH_CHECK(
      w_scale.dtype() == at::kFloat8_e4m3fn, "w_scale must be FP8 e4m3.");
  TORCH_CHECK(
      x_global_scale.dtype() == at::kFloat, "x_global_scale must be float32.");
  TORCH_CHECK(
      w_global_scale.dtype() == at::kFloat, "w_global_scale must be float32.");
  TORCH_CHECK(
      XQ.dim() == 2 && WQ.dim() == 3,
      "Only 2D-3D grouped GEMM (MoE forward) is supported.");

  int64_t G = offsets.size(0);
  int64_t M = XQ.size(0);
  int64_t N = WQ.size(-1);
  int64_t K = WQ.size(-2);

  TORCH_CHECK(
      XQ.size(-1) == K && WQ.size(0) == G,
      "XQ shape must be (total_M, K) and WQ shape must be (G, K, N).");
  TORCH_CHECK(
      x_global_scale.size(0) == M,
      "x_global_scale must have total_M elements.");
  TORCH_CHECK(
      w_global_scale.size(0) == G, "w_global_scale must have G elements.");

  at::Tensor out = output_maybe.has_value()
      ? output_maybe.value()
      : at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  if (out.numel() == 0) {
    return out;
  }

  // Normalize per-group M for heuristics.
  int M_per_group = M / G;

  auto kernel = get_ultra_kernel_via_heuristics(M_per_group, N, K * 2);

  return kernel(
      XQ,
      WQ.transpose(-2, -1), // Column-major to row-major for CUTLASS.
      x_scale,
      w_scale,
      offsets,
      x_global_scale,
      w_global_scale,
      out);
}

#else

at::Tensor f4f4bf16_ultra_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    std::optional<at::Tensor> output) {
  throw std::runtime_error(
      "f4f4bf16_ultra_grouped_mm requires CUDA 13.0+ (SM103/B300)");
}

#endif

} // namespace mslk::gemm
