/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f4f4bf16_common.cuh"

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)

at::Tensor f4f4bf16_ultra_256_192_768_2_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size) {
  if (global_scale) {
    return _f4f4bf16<NVFP4, 256, 192, 2, 2, 1, /*UseUltra=*/true>(
        XQ, WQ, x_scale, w_scale, output, global_scale);
  } else if (mxfp4_block_size == 16) {
    return _f4f4bf16<MXFP4_16, 256, 192, 2, 2, 1, /*UseUltra=*/true>(
        XQ, WQ, x_scale, w_scale, output, global_scale);
  } else {
    return _f4f4bf16<MXFP4, 256, 192, 2, 2, 1, /*UseUltra=*/true>(
        XQ, WQ, x_scale, w_scale, output, global_scale);
  }
}

#endif

} // namespace mslk::gemm
