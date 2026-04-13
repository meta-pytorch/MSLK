/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <mslk/moe/moe.h> // @manual
#include <torch/library.h>
#include <cstdint>
#include <optional>

namespace mslk::moe {

TORCH_LIBRARY_FRAGMENT(mslk, m) {
  m.def(
      "index_shuffling(Tensor routing_scores,             "
      "                int? expert_index_start=None,      "
      "                int? expert_index_end=None,        "
      "                Tensor? valid_token_count=None,    "
      "                int top_k=1) ->                    "
      "(Tensor, Tensor, Tensor)");
#ifndef USE_ROCM
  m.def(
      "scatter_add_along_first_dim(Tensor Dst, Tensor Src, Tensor Index) -> ()");
#endif
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  m.impl("index_shuffling", index_shuffling_torch);
#ifndef USE_ROCM
  m.impl("scatter_add_along_first_dim", scatter_add_along_first_dim);
#endif
}

} // namespace mslk::moe
