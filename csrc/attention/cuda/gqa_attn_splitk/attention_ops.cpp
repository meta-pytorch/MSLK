/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mslk/attention/gqa_attn_splitk.h> // @manual
#include <mslk/utils/torch/op_registration.h> // @manual
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(mslk, m) {
  m.def(
      "gqa_attn_splitk("
      "    Tensor XQ, "
      "    Tensor cache_K, "
      "    Tensor cache_V, "
      "    Tensor seq_positions, "
      "    float qk_scale, "
      "    int num_split_ks, "
      "    int kv_cache_quant_num_groups=1, "
      "    bool use_tensor_cores=True,"
      "    int cache_logical_dtype_int=0"
      ") -> (Tensor, Tensor, Tensor)");
  m.def(
      "mqa_attn("
      "    Tensor XQ, "
      "    Tensor cache_K, "
      "    Tensor cache_V, "
      "    Tensor seq_positions, "
      "    float qk_scale, "
      "    int? num_groups=1, "
      "    int cache_logical_dtype_int=0, "
      "    Tensor? qparam_k=None, "
      "    Tensor? qparam_v=None"
      ") -> Tensor");
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  DISPATCH_TO_CUDA("gqa_attn_splitk", mslk::attention::gqa_attn_splitk);
  DISPATCH_TO_CUDA("mqa_attn", mslk::attention::mqa_attn);
}
