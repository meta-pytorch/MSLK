/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "f4f4bf16_grouped_common.cuh"

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor f4f4bf16_grouped_128_64_256_1_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> offsets,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding,
    int64_t mxfp4_block_size) {
  if (global_scale) {
    return f4f4bf16_grouped_impl<NVFP4, 128, 64, 256, 1, 1, 1>(
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        offsets,
        M_sizes,
        global_scale,
        starting_row_after_padding);
  } else if (mxfp4_block_size == 16) {
    // NOTE: deliberately 128,128,256,1,1,1 and not this file's own tile shape.
    // The other MXFP4_16 tile instantiations fault with an illegal memory
    // access (see mslk/test/gemm/test_mxfp4_16_grouped_mm.py, which was written
    // to reproduce it), so every MXFP4_16 workload is routed to the one shape
    // that is known good. This does mean per-name tuning does not select a
    // tile for MXFP4_16 yet.
    return f4f4bf16_grouped_impl<MXFP4_16, 128, 128, 256, 1, 1, 1>(
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        offsets,
        M_sizes,
        global_scale,
        starting_row_after_padding);
  } else {
    return f4f4bf16_grouped_impl<MXFP4, 128, 64, 256, 1, 1, 1>(
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        offsets,
        M_sizes,
        global_scale,
        starting_row_after_padding);
  }
}

#endif

} // namespace mslk::gemm
