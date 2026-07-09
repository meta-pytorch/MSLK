/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <vector>

#pragma once

namespace mslk::gemm {

at::Tensor f8f8bf16_blockwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m = 128,
    int64_t block_n = 128,
    int64_t block_k = 128);

at::Tensor f8f8bf16_rowwise_preshuffle(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);

at::Tensor f8f8f16_rowwise_preshuffle(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);

std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale);

at::Tensor f8f8bf16_rowwise_grouped_cat(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale);

at::Tensor f8f8bf16_rowwise_grouped_stacked(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes);

at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor zero_start_index_M,
    bool zeroing_output_tensor = true);

at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt);

std::vector<at::Tensor> bf16bf16bf16_grouped(
    at::TensorList X,
    at::TensorList W);

at::Tensor bf16bf16bf16_grouped_cat(at::TensorList X, at::TensorList W);

at::Tensor bf16bf16bf16_grouped_dynamic(
    at::Tensor X,
    at::Tensor W,
    at::Tensor zero_start_index_M);

at::Tensor bf16bf16bf16_grouped_stacked(
    at::Tensor X,
    at::Tensor W,
    at::Tensor M_sizes,
    std::optional<at::Tensor> out = std::nullopt,
    std::optional<int64_t> num_sms = std::nullopt);

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);

void f8f8bf16_rowwise_out(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);

at::Tensor f8f8f16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);

at::Tensor f8f8bf16_groupwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);

at::Tensor f8f8bf16_groupwise_grouped(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes);

at::Tensor
i8i8bf16(at::Tensor XQ, at::Tensor WQ, double scale, int64_t split_k);

at::Tensor i8i8bf16_dynamic(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor scale,
    int64_t split_k = 1);

at::Tensor f4f4bf16_grouped_stacked(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> global_scale = std::nullopt,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt,
    bool use_mx = true);

at::Tensor bf16x9_gemm(
    at::Tensor A,
    at::Tensor B,
    std::optional<at::Tensor> output = std::nullopt);

at::Tensor bf16i4bf16_shuffled(
    at::Tensor X,
    at::Tensor W,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group);

at::Tensor f8i4bf16_shuffled_grouped(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group,
    at::Tensor M_sizes);

at::Tensor bf16i4bf16_shuffled_grouped(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group,
    at::Tensor M_sizes);

at::Tensor bf16i4bf16_shuffled_batched(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale,
    at::Tensor w_zp);

at::Tensor bf16i4bf16_rowwise_batched(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale,
    at::Tensor w_zp);

at::Tensor bf16i4bf16_rowwise(
    at::Tensor X,
    at::Tensor W,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group);

at::Tensor f8i4bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp);

at::Tensor f8i4bf16_shuffled(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group);

std::tuple<at::Tensor, at::Tensor> preshuffle_i4(
    at::Tensor WQ,
    at::Tensor w_scale);

// Mixed MX8 x MX4 GEMM using mxf8f6f4 block-scaled tensor core instruction
// ElementA = mx_float8_t<float_e4m3_t>, ElementB = mx_float4_t<float_e2m1_t>
at::Tensor mx8mx4bf16(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output = std::nullopt);

// Mixed MX8 x MX6 GEMM using mxf8f6f4 block-scaled tensor core instruction
// ElementA = mx_float8_t<float_e4m3_t>, ElementB = mx_float6_t<float_e2m3_t>
//
// Tensor shape / format contract:
//   XQ:       [M, K]     uint8 — MX8 (E4M3) bytes, 1 per element
//   WQ:       [N, K*6/8] uint8 — MX6 (E2M3) BIT-PACKED, 4 unpacked 6-bit
//             values per 3 packed bytes (LSB-first). See pack_fp6_e2m3()
//             in mslk/mslk/quantize/mx_mixed_dtype_utils.py.
//   x_scale:  E8M0 BS32 scales for XQ, _to_blocked swizzled, dtype uint8.
//   w_scale:  E8M0 BS32 scales for WQ (per *unpacked* element count K),
//             _to_blocked swizzled, dtype uint8.
//   output:   [M, N] bfloat16 (optional; allocated if None).
//
// Kernel TORCH_CHECKs WQ.size(1)*8 == K*6.
at::Tensor mx8mx6bf16(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output = std::nullopt);

// Symmetric MX6 x MX6 GEMM using mxf8f6f4 block-scaled tensor core instruction
// ElementA = ElementB = mx_float6_t<float_e2m3_t>
at::Tensor mx6mx6bf16(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output = std::nullopt,
    // Deep-K split-K control: 0 = heuristic (auto), >0 = force that many
    // splits.
    int64_t splits = 0);

} // namespace mslk::gemm
