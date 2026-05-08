/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f4f4bf16_ultra_grouped_common.cuh"

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)

at::Tensor f4f4bf16_ultra_grouped_128_256_768_1_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    at::Tensor x_global_scale,
    at::Tensor w_global_scale,
    at::Tensor output) {
  return f4f4bf16_ultra_grouped_impl<128, 256, 768, 1, 1, 1>(
      XQ,
      WQ,
      x_scale,
      w_scale,
      offsets,
      x_global_scale,
      w_global_scale,
      output);
}

#endif

} // namespace mslk::gemm
