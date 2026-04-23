/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mslk/gemm/gemm.h> // @manual
#include <mslk/gemm/gemm_torch.h> // @manual
#include <torch/library.h>

namespace mslk::gemm {

TORCH_LIBRARY_FRAGMENT(mslk, m) {
  m.def("bf16bf16bf16_grouped(Tensor[] X, Tensor[] W) -> Tensor[]");
  m.def("bf16bf16bf16_grouped_cat(Tensor[] X, Tensor[] W) -> Tensor");
  m.def(
      "bf16bf16bf16_grouped_dynamic(Tensor X, Tensor W, Tensor zero_start_index_M) -> Tensor");
  m.def(
      "bf16bf16bf16_grouped_stacked(Tensor X, Tensor W, Tensor M_sizes, Tensor? out=None, int? num_sms=None) -> Tensor");
  m.def(
      "f8f8bf16_blockwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, int block_m=128, int block_n=128, int block_k=128) -> Tensor");
  m.def(
      "f8f8bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_out(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor(a!) output, Tensor? bias=None, bool use_fast_accum=True) -> ()");
  m.def(
      "f8f8bf16_rowwise_batched(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True, Tensor(a!)? output=None) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_grouped(Tensor[] XQ, Tensor[] WQ, Tensor[] x_scale, Tensor[] w_scale) -> Tensor[]");
  m.def(
      "f8f8bf16_rowwise_grouped_cat(Tensor[] XQ, Tensor[] WQ, Tensor[] x_scale, Tensor[] w_scale) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_grouped_stacked(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_grouped_dynamic(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor zero_start_index_M, bool zeroing_output_tensor=True) -> Tensor");
  m.def(
      "f8f8bf16_tensorwise(Tensor XQ, Tensor WQ, float scale, bool use_fast_accum=True) -> Tensor");

#ifdef USE_ROCM
  m.def(
      "f8f8f16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8f16_rowwise_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  // Generic PyTorch grouped GEMM API is only available on AMD for now.
  m.def(
      "f8f8bf16_rowwise_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? offsets, Tensor(a!) output) -> Tensor");
#else
  m.def("i8i8bf16(Tensor XQ, Tensor WQ, float scale, int split_k=1) -> Tensor");
  m.def(
      "f4f4bf16(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? output=None, Tensor? global_scale=None, int mxfp4_block_size=32) -> Tensor");
  m.def(
      "mx8mx4bf16(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? output=None) -> Tensor");
  m.def(
      "f4f4bf16_grouped_stacked(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes, Tensor? global_scale=None, Tensor? starting_row_after_padding=None, bool use_mx=True) -> Tensor");
  m.def(
      "mx8mx8bf16_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor offsets, Tensor(a!)? output=None, int? actual_num_tokens=None) -> Tensor");
  m.def(
      "f4f4bf16_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor offsets, Tensor(a!)? output=None, Tensor(a!)? global_scale=None) -> Tensor");
  m.def(
      "f8f8bf16(Tensor XQ, Tensor WQ, Tensor scale, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8bf16_groupwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale) -> Tensor");
  m.def(
      "f8f8bf16_groupwise_grouped(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes) -> Tensor");
  m.def(
      "f8f8bf16_cublas(Tensor A, Tensor B, Tensor? Ainvs=None, Tensor? Binvs=None, bool use_fast_accum=True, Tensor(a!)? output=None) -> Tensor");
  m.def("bf16x9_gemm(Tensor A, Tensor B, Tensor(a!)? output=None) -> Tensor");
  m.def(
      "f8i4bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "f8i4bf16_shuffled(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor w_scale_group) -> Tensor");
  m.def(
      "bf16i4bf16_shuffled(Tensor X, Tensor W, Tensor w_scale_group, Tensor w_zero_group) -> Tensor");
  m.def(
      "f8i4bf16_shuffled_grouped(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor w_scale_group, Tensor M_sizes) -> Tensor");
  m.def(
      "bf16i4bf16_shuffled_grouped(Tensor X, Tensor WQ, Tensor w_scale_group, Tensor w_zero_group, Tensor M_sizes) -> Tensor");
  m.def(
      "bf16i4bf16_rowwise(Tensor X, Tensor W, Tensor w_scale_group, Tensor w_zero_group) -> Tensor");
  m.def(
      "bf16i4bf16_shuffled_batched(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "bf16i4bf16_rowwise_batched(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "i8i8bf16_dynamic(Tensor XQ, Tensor WQ, Tensor scale, int split_k=1) -> Tensor");
  m.def("preshuffle_i4(Tensor WQ, Tensor w_scale) -> (Tensor, Tensor)");
#endif
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
  m.impl("f8f8bf16_rowwise_out", f8f8bf16_rowwise_out);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched);
  m.impl("f8f8bf16_rowwise_grouped", f8f8bf16_rowwise_grouped);
  m.impl("f8f8bf16_rowwise_grouped_cat", f8f8bf16_rowwise_grouped_cat);
  m.impl("f8f8bf16_rowwise_grouped_stacked", f8f8bf16_rowwise_grouped_stacked);
  m.impl("f8f8bf16_rowwise_grouped_dynamic", f8f8bf16_rowwise_grouped_dynamic);
  m.impl("bf16bf16bf16_grouped", bf16bf16bf16_grouped);
  m.impl("bf16bf16bf16_grouped_cat", bf16bf16bf16_grouped_cat);
  m.impl("bf16bf16bf16_grouped_dynamic", bf16bf16bf16_grouped_dynamic);
  m.impl("bf16bf16bf16_grouped_stacked", bf16bf16bf16_grouped_stacked);

#ifdef USE_ROCM
  m.impl("f8f8f16_rowwise", f8f8f16_rowwise);
  m.impl("f8f8bf16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
  m.impl("f8f8f16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
  m.impl("f8f8bf16_rowwise_grouped_mm", f8f8bf16_rowwise_grouped_mm);
#else
  m.impl("f8f8bf16_groupwise", f8f8bf16_groupwise);
  m.impl("f8f8bf16_groupwise_grouped", f8f8bf16_groupwise_grouped);
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f4f4bf16", f4f4bf16);
  m.impl("mx8mx4bf16", mx8mx4bf16);
  m.impl("f4f4bf16_grouped_stacked", f4f4bf16_grouped_stacked);
  m.impl("mx8mx8bf16_grouped_mm", mx8mx8bf16_grouped_mm);
  m.impl("f4f4bf16_grouped_mm", f4f4bf16_grouped_mm);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
  m.impl("bf16x9_gemm", bf16x9_gemm);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);
  m.impl("f8i4bf16_shuffled", f8i4bf16_shuffled);
  m.impl("bf16i4bf16_shuffled", bf16i4bf16_shuffled);
  m.impl("f8i4bf16_shuffled_grouped", f8i4bf16_shuffled_grouped);
  m.impl("bf16i4bf16_shuffled_grouped", bf16i4bf16_shuffled_grouped);
  m.impl("bf16i4bf16_shuffled_batched", bf16i4bf16_shuffled_batched);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise);
  m.impl("i8i8bf16_dynamic", i8i8bf16_dynamic);
  m.impl("preshuffle_i4", preshuffle_i4);
#endif
}

// Unfortunately there's broken code in production sometimes calling these ops
// on CPU for silly reasons. To prevent breaking the models, we need to keep the
// ops registered on CPU.
TORCH_LIBRARY_IMPL(mslk, CPU, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
  m.impl("f8f8bf16_rowwise_out", f8f8bf16_rowwise_out);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched);
  m.impl("f8f8bf16_rowwise_grouped", f8f8bf16_rowwise_grouped);
  m.impl("f8f8bf16_rowwise_grouped_cat", f8f8bf16_rowwise_grouped_cat);
  m.impl("f8f8bf16_rowwise_grouped_stacked", f8f8bf16_rowwise_grouped_stacked);
  m.impl("f8f8bf16_rowwise_grouped_dynamic", f8f8bf16_rowwise_grouped_dynamic);
  m.impl("bf16bf16bf16_grouped", bf16bf16bf16_grouped);
  m.impl("bf16bf16bf16_grouped_cat", bf16bf16bf16_grouped_cat);
  m.impl("bf16bf16bf16_grouped_dynamic", bf16bf16bf16_grouped_dynamic);
  m.impl("bf16bf16bf16_grouped_stacked", bf16bf16bf16_grouped_stacked);

#ifdef USE_ROCM
  m.impl("f8f8f16_rowwise", f8f8f16_rowwise);
  m.impl("f8f8bf16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
  m.impl("f8f8f16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
  m.impl("f8f8bf16_rowwise_grouped_mm", f8f8bf16_rowwise_grouped_mm);
#else
  m.impl("f8f8bf16_groupwise", f8f8bf16_groupwise);
  m.impl("f8f8bf16_groupwise_grouped", f8f8bf16_groupwise_grouped);
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f4f4bf16", f4f4bf16);
  m.impl("mx8mx4bf16", mx8mx4bf16);
  m.impl("f4f4bf16_grouped_stacked", f4f4bf16_grouped_stacked);
  m.impl("mx8mx8bf16_grouped_mm", mx8mx8bf16_grouped_mm);
  m.impl("f4f4bf16_grouped_mm", f4f4bf16_grouped_mm);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
  m.impl("bf16x9_gemm", bf16x9_gemm);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);
  m.impl("f8i4bf16_shuffled", f8i4bf16_shuffled);
  m.impl("bf16i4bf16_shuffled", bf16i4bf16_shuffled);
  m.impl("f8i4bf16_shuffled_grouped", f8i4bf16_shuffled_grouped);
  m.impl("bf16i4bf16_shuffled_grouped", bf16i4bf16_shuffled_grouped);
  m.impl("bf16i4bf16_shuffled_batched", bf16i4bf16_shuffled_batched);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise);
  m.impl("i8i8bf16_dynamic", i8i8bf16_dynamic);
  m.impl("preshuffle_i4", preshuffle_i4);
#endif
}

} // namespace mslk::gemm
