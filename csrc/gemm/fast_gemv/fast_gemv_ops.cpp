/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <mslk/utils/torch/op_registration.h> // @manual
#include <torch/library.h>

namespace mslk {

at::Tensor bf16_fast_gemv(at::Tensor X, at::Tensor W);
at::Tensor
bf16fp8bf16_fast_gemv(at::Tensor X, at::Tensor W, at::Tensor w_scale);
at::Tensor fp8fp8bf16_fast_gemv(
    at::Tensor X,
    at::Tensor W,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool is_batched = false);

at::Tensor bf16_fast_gemv_meta(at::Tensor X, at::Tensor W) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kHalf));
  return Y;
}

at::Tensor bf16fp8bf16_fast_gemv_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor /* w_scale */) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor fp8fp8bf16_fast_gemv_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    bool is_batched) {
  at::Tensor Y;
  if (is_batched) {
    const at::SymInt B = X.sym_size(0);
    const at::SymInt M = X.sym_size(1);
    const at::SymInt N = W.sym_size(1);
    Y = at::empty_symint({B, M, N}, X.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt M = X.sym_size(0);
    const at::SymInt N = W.sym_size(0);
    auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  }
  return Y;
}

#ifndef USE_ROCM
TORCH_LIBRARY_FRAGMENT(mslk, m) {
  m.def("bf16_fast_gemv(Tensor X, Tensor W) -> Tensor");
  m.def("bf16fp8bf16_fast_gemv(Tensor X, Tensor W, Tensor w_scale) -> Tensor");
  m.def(
      "fp8fp8bf16_fast_gemv(Tensor X, Tensor W, Tensor x_scale, Tensor w_scale, bool is_batched=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(mslk, CPU, m) {
  m.impl("bf16_fast_gemv", bf16_fast_gemv);
  m.impl("bf16fp8bf16_fast_gemv", bf16fp8bf16_fast_gemv);
  m.impl("fp8fp8bf16_fast_gemv", fp8fp8bf16_fast_gemv);
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  m.impl("bf16_fast_gemv", bf16_fast_gemv);
  m.impl("bf16fp8bf16_fast_gemv", bf16fp8bf16_fast_gemv);
  m.impl("fp8fp8bf16_fast_gemv", fp8fp8bf16_fast_gemv);
}

TORCH_LIBRARY_IMPL(mslk, Meta, m) {
  m.impl("bf16_fast_gemv", bf16_fast_gemv_meta);
  m.impl("bf16fp8bf16_fast_gemv", bf16fp8bf16_fast_gemv_meta);
  m.impl("fp8fp8bf16_fast_gemv", fp8fp8bf16_fast_gemv_meta);
}
#endif

} // namespace mslk
