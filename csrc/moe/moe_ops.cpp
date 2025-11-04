/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <mslk/moe/moe.h> // @manual
#include <mslk/utils/torch/op_registration.h> // @manual
#include <torch/library.h>
#include <cstdint>
#include <optional>

namespace mslk::moe {

TORCH_LIBRARY_FRAGMENT(mslk, m) {
  m.set_python_module("mslk.moe");
  m.def(
      "index_shuffling(Tensor routing_scores,             "
      "                int? expert_index_start=None,      "
      "                int? expert_index_end=None,        "
      "                Tensor? valid_token_count=None,    "
      "                int top_k=1) ->                    "
      "(Tensor, Tensor, Tensor)");
#ifndef USE_ROCM
  m.def("gather_along_first_dim(Tensor Data, Tensor Index) -> Tensor");
  m.def(
      "scatter_add_along_first_dim(Tensor Dst, Tensor Src, Tensor Index) -> ()");
#endif
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  DISPATCH_TO_CUDA("index_shuffling", index_shuffling_torch);
#ifndef USE_ROCM
  DISPATCH_TO_CUDA("gather_along_first_dim", gather_along_first_dim);
  DISPATCH_TO_CUDA("scatter_add_along_first_dim", scatter_add_along_first_dim);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch_meta(
    const at::Tensor& routing_scores,
    const std::optional<int64_t>& expert_index_start,
    const std::optional<int64_t>& expert_index_end,
    const std::optional<at::Tensor>& valid_token_count,
    const int64_t top_k = 1) {
  auto T = routing_scores.sym_size(0);
  auto E = routing_scores.sym_size(1);
  at::Tensor token_counts_per_expert =
      at::empty_symint({E + 2}, routing_scores.options().dtype(at::kInt));
  at::Tensor expert_indices =
      at::empty_symint({T * top_k}, routing_scores.options().dtype(at::kInt));
  at::Tensor token_indices =
      at::empty_symint({T * top_k}, routing_scores.options().dtype(at::kInt));
  return {token_counts_per_expert, expert_indices, token_indices};
}

at::Tensor gather_along_first_dim_meta(at::Tensor data, at::Tensor index) {
  int K = data.size(1);
  int N = index.size(0);
  at::Tensor output = at::empty({N, K}, data.options());
  return output;
}

void scatter_add_along_first_dim_meta(
    at::Tensor /*dst*/,
    at::Tensor /*src*/,
    at::Tensor /*index*/) {
  return;
}

TORCH_LIBRARY_IMPL(mslk, Meta, m) {
  DISPATCH_TO_META("index_shuffling", index_shuffling_torch_meta);
#ifndef USE_ROCM
  DISPATCH_TO_META("gather_along_first_dim", gather_along_first_dim_meta);
  DISPATCH_TO_META(
      "scatter_add_along_first_dim", scatter_add_along_first_dim_meta);
#endif
}

} // namespace mslk::moe
