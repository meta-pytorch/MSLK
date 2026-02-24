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
      "quantize_fp8_per_tensor(Tensor input, Tensor? bs=None, Tensor? scale_ub=None, bool stochastic_rounding=False) -> Tensor[]");
  m.def(
      "quantize_fp8_per_row(Tensor input, Tensor? bs=None, Tensor? scale_ub=None, ScalarType? output_dtype=None, bool stochastic_rounding = False) -> Tensor[] ");

  m.def(
      "get_fp8_per_tensor_scale(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor");

  m.def(
      "quantize_fp8_per_tensor_fixed_scale(Tensor input, Tensor scale, Tensor? bs=None, bool stochatic_rounding=False) -> Tensor");
  m.def("per_tensor_quantize_i8(Tensor X, float scale) -> Tensor");
  m.def("per_tensor_dynamic_quantize_i8(Tensor X) -> (Tensor, Tensor)");
#ifndef USE_ROCM
  m.def(
      "fake_quantize_nvfp4_per_tensor(Tensor input, Tensor? static_scales=None, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor[]");
#else
  m.def("flush_icache_hip() -> ()");
#endif
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  DISPATCH_TO_CUDA("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  DISPATCH_TO_CUDA("quantize_fp8_per_row", quantize_fp8_per_row);
  DISPATCH_TO_CUDA("per_tensor_quantize_i8", per_tensor_quantize_i8);
  DISPATCH_TO_CUDA(
      "per_tensor_dynamic_quantize_i8", per_tensor_dynamic_quantize_i8);
  DISPATCH_TO_CUDA("get_fp8_per_tensor_scale", get_fp8_per_tensor_scale);
  DISPATCH_TO_CUDA(
      "quantize_fp8_per_tensor_fixed_scale",
      quantize_fp8_per_tensor_fixed_scale);

#ifndef USE_ROCM
  DISPATCH_TO_CUDA(
      "fake_quantize_nvfp4_per_tensor", fake_quantize_nvfp4_per_tensor);
#else
  DISPATCH_TO_CUDA("flush_icache_hip", flush_icache_ck);
#endif
}

std::vector<at::Tensor> quantize_fp8_per_tensor_meta(
    at::Tensor input,
    std::optional<at::Tensor> /* bs */,
    std::optional<at::Tensor> /*scale_ub*/,
    const bool /*stochastic_rounding*/) {
  int dims = input.dim();
  TORCH_CHECK(dims == 2 || dims == 3, "The dim of input should be 2 or 3");
  at::Tensor Y = at::empty_like(input, input.options().dtype(torch_fp8_e4m3));
  at::Tensor scale;
  if (dims <= 2) {
    scale = at::empty_symint({}, input.options().dtype(at::kFloat));
  } else {
    const at::SymInt B = input.sym_size(0);
    scale = at::empty_symint({B}, input.options().dtype(at::kFloat));
  }
  return {Y, scale};
}

std::vector<at::Tensor> quantize_fp8_per_row_meta(
    at::Tensor input,
    std::optional<at::Tensor> /* bs */,
    std::optional<at::Tensor> /* scale_ub */,
    std::optional<c10::ScalarType> /* output_dtype */,
    bool /* stochastic_rounding */) {
  int dims = input.dim();
  TORCH_CHECK(dims == 2 || dims == 3, "The dim of input should be 2 or 3");
  at::Tensor Y = at::empty_like(input, input.options().dtype(torch_fp8_e4m3));
  at::Tensor scale;
  if (dims == 2) {
    const at::SymInt M = input.sym_size(0);
    scale = at::empty_symint({M}, input.options().dtype(at::kFloat));
  } else {
    const at::SymInt B = input.sym_size(0);
    const at::SymInt M = input.sym_size(1);
    scale = at::empty_symint({B, M}, input.options().dtype(at::kFloat));
  }
  return {Y, scale};
}

TORCH_LIBRARY_IMPL(mslk, Meta, m) {
  DISPATCH_TO_META("quantize_fp8_per_tensor", quantize_fp8_per_tensor_meta);
  DISPATCH_TO_META("quantize_fp8_per_row", quantize_fp8_per_row_meta);
}

} // namespace mslk::quantize
