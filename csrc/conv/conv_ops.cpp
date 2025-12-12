/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mslk/conv/conv.h> // @manual
#include <mslk/utils/torch/op_registration.h> // @manual
#include <torch/library.h>

namespace mslk::conv {

TORCH_LIBRARY_FRAGMENT(mslk, m) {
#ifndef USE_ROCM
  m.def(
      "f8f8bf16_conv(Tensor activation, Tensor filter, Tensor scale, int[] padding, int[] stride, int[] dilation) -> Tensor");
#endif
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
#ifndef USE_ROCM
  m.impl("f8f8bf16_conv", f8f8bf16_conv);
#endif
}

at::Tensor f8f8bf16_conv_meta(
    at::Tensor activation,
    at::Tensor filter,
    at::Tensor /* scale */,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation) {
  TORCH_CHECK(activation.dim() == 5, "Activation must be 5D tensor");
  TORCH_CHECK(filter.dim() == 5, "Filter must be 5D tensor");

  // Check if input is channels first or channels last.
  bool channels_last = activation.sym_strides()[4] == 1;

  at::SymInt n, d, h, w, k, t, r, s;

  if (channels_last) {
    n = activation.sym_size(0);
    d = activation.sym_size(1);
    h = activation.sym_size(2);
    w = activation.sym_size(3);

    k = filter.sym_size(0);
    t = filter.sym_size(1);
    r = filter.sym_size(2);
    s = filter.sym_size(3);
  } else {
    n = activation.sym_size(0);
    d = activation.sym_size(2);
    h = activation.sym_size(3);
    w = activation.sym_size(4);

    k = filter.sym_size(0);
    t = filter.sym_size(2);
    r = filter.sym_size(3);
    s = filter.sym_size(4);
  }

  TORCH_CHECK(
      padding.size() == 3,
      "Padding must have 3 elements corresponding to [d, h, w]");
  TORCH_CHECK(
      stride.size() == 3,
      "Stride must have 3 elements corresponding to [d, h, w]");
  TORCH_CHECK(
      dilation.size() == 3,
      "Dilation must have 3 elements corresponding to [d, h, w]");

  int64_t pad_d = padding[0];
  int64_t pad_h = padding[1];
  int64_t pad_w = padding[2];
  int64_t stride_d = stride[0];
  int64_t stride_h = stride[1];
  int64_t stride_w = stride[2];
  int64_t dilation_d = dilation[0];
  int64_t dilation_h = dilation[1];
  int64_t dilation_w = dilation[2];

  at::SymInt z = 1 + (d + 2 * pad_d - ((t - 1) * dilation_d + 1)) / stride_d;
  at::SymInt p = 1 + (h + 2 * pad_h - ((r - 1) * dilation_h + 1)) / stride_h;
  at::SymInt q = 1 + (w + 2 * pad_w - ((s - 1) * dilation_w + 1)) / stride_w;

  auto Y = at::empty_symint(
      {n, z, p, q, k}, activation.options().dtype(at::kBFloat16));
  // Permute to channels first shape if inputs are channels first.
  if (!channels_last) {
    Y = Y.permute({0, 4, 1, 2, 3});
  }
  return Y;
}

TORCH_LIBRARY_IMPL(mslk, Meta, m) {
#ifndef USE_ROCM
  m.impl("f8f8bf16_conv", f8f8bf16_conv_meta);
#endif
}

} // namespace mslk::conv
