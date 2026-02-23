/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <vector>

namespace mslk::quantize {

#ifdef USE_ROCM
void flush_icache_ck();
#endif

at::Tensor silu_mul_quantize_i8(at::Tensor X1, at::Tensor X2, double scale);

at::Tensor per_tensor_quantize_i8(at::Tensor X, double scale);

std::tuple<at::Tensor, at::Tensor> per_tensor_dynamic_quantize_i8(at::Tensor X);

std::vector<at::Tensor> quantize_fp8_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub, // scale upperbound
    const bool stochastic_rounding); // whether apply stochastic rounding

std::vector<at::Tensor> quantize_fp8_per_row(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub, // scale upperbound
    std::optional<c10::ScalarType> output_dtype, // output dtype
    bool stochastic_rounding); // whether apply stochastic rounding

std::vector<at::Tensor> quantize_fp8_per_col(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub); // scale upperbound

at::Tensor quantize_fp8_per_tensor_fixed_scale(
    at::Tensor input,
    at::Tensor scale,
    std::optional<at::Tensor> bs,
    bool stochatic_rounding);

at::Tensor get_fp8_per_tensor_scale(
    at::Tensor input,
    std::optional<at::Tensor> bs,
    std::optional<at::Tensor> scale_ub); // scale upperbound

std::vector<at::Tensor> fake_quantize_nvfp4_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> static_scales,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub); // scale upperbound

} // namespace mslk::quantize
