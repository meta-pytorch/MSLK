/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

// SM100 (B200) NVFP4 variants. Use TileShape K = 256.
at::Tensor f4f4bf16_ultra_grouped_256_128_256_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output);

at::Tensor f4f4bf16_ultra_grouped_256_256_256_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output);

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
// SM103 (B300) NVFP4 ultra variants. Use TileShape K = 768.
at::Tensor f4f4bf16_ultra_grouped_128_256_768_1_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output);

at::Tensor f4f4bf16_ultra_grouped_256_256_768_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output);

at::Tensor f4f4bf16_ultra_grouped_256_128_768_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output);
#endif // CUDA >= 13.0

using Kernel_f4f4bf16_ultra_grouped = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor);

const std::unordered_map<std::string, Kernel_f4f4bf16_ultra_grouped>&
get_f4f4bf16_ultra_grouped_kernels() {
  static const std::unordered_map<std::string, Kernel_f4f4bf16_ultra_grouped>
      kernels = {
          {"f4f4bf16_ultra_grouped_256_128_256_2_1_1",
           f4f4bf16_ultra_grouped_256_128_256_2_1_1},
          {"f4f4bf16_ultra_grouped_256_256_256_2_1_1",
           f4f4bf16_ultra_grouped_256_256_256_2_1_1},
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
          {"f4f4bf16_ultra_grouped_128_256_768_1_1_1",
           f4f4bf16_ultra_grouped_128_256_768_1_1_1},
          {"f4f4bf16_ultra_grouped_256_256_768_2_1_1",
           f4f4bf16_ultra_grouped_256_256_768_2_1_1},
          {"f4f4bf16_ultra_grouped_256_128_768_2_1_1",
           f4f4bf16_ultra_grouped_256_128_768_2_1_1},
#endif
      };
  return kernels;
}

#endif // CUDA >= 12.8
} // namespace mslk::gemm
