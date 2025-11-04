/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mslk/coalesce/coalesce.h> // @manual
#include <mslk/utils/torch/op_registration.h> // @manual
#include <torch/library.h>

namespace mslk::coalesce {

TORCH_LIBRARY_FRAGMENT(mslk, m) {
  m.def(
      "coalesce_batches(Tensor(a!)[] input_tensors, Tensor(a!)[] output_tensors, Tensor old_bids, Tensor new_bids) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mslk, CPU, m) {
  DISPATCH_TO_CPU("coalesce_batches", coalesce_batches_cpu);
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  DISPATCH_TO_CUDA("coalesce_batches", coalesce_batches_gpu);
}

} // namespace mslk::coalesce
