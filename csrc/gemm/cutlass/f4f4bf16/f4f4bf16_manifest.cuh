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

at::Tensor f4f4bf16_128_128_1_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_128_128_1_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_128_128_1_4_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_128_128_2_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_128_128_4_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_128_128_4_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_128_192_2_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_128_192_2_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_128_192_4_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_128_256_2_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_128_2_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_128_2_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_128_2_4_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_128_4_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_192_2_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_192_2_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_192_2_4_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_192_4_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_256_2_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_256_2_2_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_256_2_4_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_256_256_4_1_1(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
// SM103 (B300) ultra FP4 variants. TileShape K = 768, explicit ultra schedules
// (NVFP4 / MXFP4_16 -> Vs16, MXFP4 -> Vs32). Same 7-arg signature as SM100.
// Only the 2SM / 256-M tiles are declared. The 1SM / 128-M family was pruned:
// it almost never wins autotune, so dropping all 24 costs at most ~1% (measured
// worst case <1% on a single shape) and 0% on the default heuristic path.

at::Tensor f4f4bf16_ultra_256_128_768_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_128_768_2_2_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_128_768_4_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_256_768_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_256_768_2_2_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_128_768_2_4_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_256_768_2_4_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_192_768_2_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_192_768_2_2_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_192_768_2_4_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_192_768_4_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_128_768_4_2_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_192_768_4_2_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_256_768_4_1_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);

at::Tensor f4f4bf16_ultra_256_256_768_4_2_1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size);
#endif // CUDA >= 13.0

// Kernel function pointer type
// mxfp4_block_size: 32 for standard MXFP4 (1x32 block), 16 for MXFP4_16 (1x16
// block)
using Kernel_f4f4bf16 = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    int64_t);

#endif
} // namespace mslk::gemm
