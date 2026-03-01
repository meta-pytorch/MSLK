/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace mslk::moe {

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch(
    const at::Tensor& routing_scores,
    const std::optional<int64_t>& expert_index_start,
    const std::optional<int64_t>& expert_index_end,
    const std::optional<at::Tensor>& valid_token_count,
    const int64_t top_k);

void scatter_add_along_first_dim(
    at::Tensor dst,
    at::Tensor src,
    at::Tensor index);

} // namespace mslk::moe
