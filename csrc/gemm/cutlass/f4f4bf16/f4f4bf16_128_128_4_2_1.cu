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

at::Tensor f4f4bf16_128_128_4_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale) {
  if (global_scale) {
    return _f4f4bf16<NVFP4, 128, 128, 4, 2, 1>(
        XQ, WQ, x_scale, w_scale, output, global_scale);
  } else {
    return _f4f4bf16<MXFP4, 128, 128, 4, 2, 1>(
        XQ, WQ, x_scale, w_scale, output, global_scale);
  }
}

#endif

} // namespace mslk::gemm
