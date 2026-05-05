/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mslk/conv/conv.h> // @manual
#include <torch/library.h>

namespace mslk::conv {

TORCH_LIBRARY_FRAGMENT(mslk, m) {
#ifndef USE_ROCM
  m.def(
      "f8f8bf16_conv(Tensor activation, Tensor filter, Tensor scale, int[] padding, int[] stride, int[] dilation) -> Tensor");
#endif
}

#if !defined(USE_MTIA)
TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
#ifndef USE_ROCM
  m.impl("f8f8bf16_conv", f8f8bf16_conv);
#endif
}
#endif // !defined(USE_MTIA)

} // namespace mslk::conv
