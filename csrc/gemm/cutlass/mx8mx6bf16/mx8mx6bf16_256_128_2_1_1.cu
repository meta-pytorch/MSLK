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

at::Tensor mx8mx6bf16_256_128_2_1_1(
    at::Tensor XQ, // MX FP8
    at::Tensor WQ, // MX FP6
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t block_size) {
  if (block_size == 16) {
    return _mx8mx6bf16<MXFP8_16, MXFP6_16, 256, 128, 2, 1, 1>(
        XQ, WQ, x_scale, w_scale, output);
  }
  return _mx8mx6bf16<MXFP8, MXFP6, 256, 128, 2, 1, 1>(
      XQ, WQ, x_scale, w_scale, output);
}

#endif

} // namespace mslk::gemm
