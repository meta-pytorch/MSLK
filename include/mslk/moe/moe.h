/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/csrc/stable/tensor.h>

namespace mslk::moe {

using torch::stable::Tensor;

std::tuple<Tensor, Tensor, Tensor> index_shuffling_torch(
    const Tensor& routing_scores,
    const std::optional<int64_t>& expert_index_start,
    const std::optional<int64_t>& expert_index_end,
    const std::optional<Tensor>& valid_token_count,
    const int64_t top_k);

void scatter_add_along_first_dim(
    Tensor dst,
    Tensor src,
    Tensor index);

} // namespace mslk::moe
