/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mslk/moe/moe.h> // @manual
#include <torch/csrc/stable/library.h>
#include <cstdint>
#include <optional>

namespace mslk::moe {

STABLE_TORCH_LIBRARY_FRAGMENT(mslk, m) {
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

STABLE_TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  m.impl("index_shuffling", TORCH_BOX(&index_shuffling_torch));
#ifndef USE_ROCM
  m.impl("scatter_add_along_first_dim", TORCH_BOX(&scatter_add_along_first_dim));
#endif
}

} // namespace mslk::moe
