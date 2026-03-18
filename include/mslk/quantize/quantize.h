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

at::Tensor per_tensor_quantize_i8(at::Tensor X, double scale);

std::tuple<at::Tensor, at::Tensor> per_tensor_dynamic_quantize_i8(at::Tensor X);

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
