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

// TB_M=128, TBS_M=1 (1-SM schedule)
at::Tensor mx8mx8bf16_grouped_128_64_256_1_1_1_ba(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets);

at::Tensor mx8mx8bf16_grouped_128_128_256_1_1_1_ba(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets);

at::Tensor mx8mx8bf16_grouped_128_256_256_1_1_1_ba(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets);

at::Tensor mx8mx8bf16_grouped_128_256_256_1_1_1_ab(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets);

// TB_M=256, TBS_M=2 (2-SM cooperative schedule)
at::Tensor mx8mx8bf16_grouped_256_64_256_2_1_1_ba(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets);

at::Tensor mx8mx8bf16_grouped_256_128_256_2_1_1_ba(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets);

at::Tensor mx8mx8bf16_grouped_256_256_256_2_1_1_ab(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets);

at::Tensor mx8mx8bf16_grouped_256_256_256_2_1_1_ba(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets);

using Kernel_mx8mx8bf16_grouped = at::Tensor (*)(
    at::Tensor, // XQ
    at::Tensor, // WQ
    at::Tensor, // x_scale
    at::Tensor, // w_scale
    at::Tensor, // output
    int64_t, // G
    at::Tensor); // offsets

const std::unordered_map<std::string, Kernel_mx8mx8bf16_grouped>&
get_mx8mx8bf16_grouped_kernels() {
  static const std::unordered_map<std::string, Kernel_mx8mx8bf16_grouped>
      kernels = {
          // TB_M=128 (1-SM)
          {"mx8mx8bf16_grouped_128_64_256_1_1_1_ba",
           mx8mx8bf16_grouped_128_64_256_1_1_1_ba},
          {"mx8mx8bf16_grouped_128_128_256_1_1_1_ba",
           mx8mx8bf16_grouped_128_128_256_1_1_1_ba},
          {"mx8mx8bf16_grouped_128_256_256_1_1_1_ba",
           mx8mx8bf16_grouped_128_256_256_1_1_1_ba},
          {"mx8mx8bf16_grouped_128_256_256_1_1_1_ab",
           mx8mx8bf16_grouped_128_256_256_1_1_1_ab},
          // TB_M=256 (2-SM cooperative)
          {"mx8mx8bf16_grouped_256_64_256_2_1_1_ba",
           mx8mx8bf16_grouped_256_64_256_2_1_1_ba},
          {"mx8mx8bf16_grouped_256_128_256_2_1_1_ba",
           mx8mx8bf16_grouped_256_128_256_2_1_1_ba},
          {"mx8mx8bf16_grouped_256_256_256_2_1_1_ab",
           mx8mx8bf16_grouped_256_256_256_2_1_1_ab},
          {"mx8mx8bf16_grouped_256_256_256_2_1_1_ba",
           mx8mx8bf16_grouped_256_256_256_2_1_1_ba},
      };
  return kernels;
}

#endif
} // namespace mslk::gemm
