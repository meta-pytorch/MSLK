/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f4f4bf16_common.cuh"

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor f4f4bf16_256_256_4_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale) {
  if (global_scale) {
    return _f4f4bf16<NVFP4, 256, 256, 4, 1, 1>(
        XQ, WQ, x_scale, w_scale, output, global_scale);
  } else {
    return _f4f4bf16<MXFP4, 256, 256, 4, 1, 1>(
        XQ, WQ, x_scale, w_scale, output, global_scale);
  }
}

#endif

} // namespace mslk::gemm
