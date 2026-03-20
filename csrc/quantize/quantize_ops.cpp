/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mslk/quantize/quantize.h> // @manual
#include <mslk/utils/torch/op_registration.h> // @manual
#include <torch/library.h>

namespace mslk::quantize {

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
#define torch_fp8_e4m3 at::kFloat8_e4m3fnuz
#else
#define torch_fp8_e4m3 at::kFloat8_e4m3fn
#endif

TORCH_LIBRARY_FRAGMENT(mslk, m) {
  m.set_python_module("mslk.quantize");

  m.def(
      "get_fp8_per_tensor_scale(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor");

  m.def(
      "quantize_fp8_per_tensor_fixed_scale(Tensor input, Tensor scale, Tensor? bs=None, bool stochatic_rounding=False) -> Tensor");
  m.def("per_tensor_quantize_i8(Tensor X, float scale) -> Tensor");
  m.def("per_tensor_dynamic_quantize_i8(Tensor X) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  DISPATCH_TO_CUDA("per_tensor_quantize_i8", per_tensor_quantize_i8);
  DISPATCH_TO_CUDA(
      "per_tensor_dynamic_quantize_i8", per_tensor_dynamic_quantize_i8);
  DISPATCH_TO_CUDA("get_fp8_per_tensor_scale", get_fp8_per_tensor_scale);
  DISPATCH_TO_CUDA(
      "quantize_fp8_per_tensor_fixed_scale",
      quantize_fp8_per_tensor_fixed_scale);
}

} // namespace mslk::quantize
