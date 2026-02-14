/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/core/Tensor.h>

#pragma once

namespace mslk::gemm {

#ifdef USE_ROCM

// PyTorch FP8 grouped GEMM API is only available on AMD
at::Tensor f8f8bf16_rowwise_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> offsets,
    at::Tensor& output);

#else

// Torch compliant MXFP8 grouped GEMM only on CUDA for now.
at::Tensor mx8mx8bf16_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    std::optional<at::Tensor> output = std::nullopt);

// Torch compliant FP4 grouped GEMM.
at::Tensor f4f4bf16_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    std::optional<at::Tensor> output = std::nullopt,
    std::optional<at::Tensor> global_scale = std::nullopt);

// FP4 GEMM supporting NVFP4, MXFP4 (1x32 block), and MXFP4_16 (1x16 block)
//
// Format selection:
// - NVFP4: Provide global_scale (mxfp4_block_size is ignored)
// - MXFP4 (1x32): No global_scale, mxfp4_block_size = 32 (default)
// - MXFP4_16 (1x16): No global_scale, mxfp4_block_size = 16
//
// MXFP4_16 is a hybrid format that uses MXFP4's E8M0 scale factors with
// NVFP4's 16-element block size, enabling finer-grained quantization.
at::Tensor f4f4bf16(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output = std::nullopt,
    std::optional<at::Tensor> global_scale = std::nullopt,
    int64_t mxfp4_block_size = 32);

#endif

} // namespace mslk::gemm
