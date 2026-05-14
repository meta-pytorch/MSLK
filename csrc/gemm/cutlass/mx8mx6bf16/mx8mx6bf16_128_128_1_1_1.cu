/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "mx8mx6bf16_common.cuh"

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor mx8mx6bf16_128_128_1_1_1(
    at::Tensor XQ, // MX FP8
    at::Tensor WQ, // MX FP6
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output) {
  return _mx8mx6bf16<128, 128, 1, 1, 1>(XQ, WQ, x_scale, w_scale, output);
}

#endif

} // namespace mslk::gemm
