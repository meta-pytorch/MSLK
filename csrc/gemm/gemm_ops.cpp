/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mslk/gemm/gemm.h> // @manual
#include <mslk/gemm/gemm_torch.h> // @manual
#include <mslk/utils/torch/op_registration.h> // @manual
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
      "f4f4bf16(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? output=None, Tensor? global_scale=None) -> Tensor");
  m.def(
      "f4f4bf16_grouped_stacked(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes, Tensor? global_scale=None, Tensor? starting_row_after_padding=None, bool use_mx=True) -> Tensor");
  m.def(
      "mx8mx8bf16_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor offsets, Tensor(a!)? output=None) -> Tensor");
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

at::Tensor i8i8bf16_meta(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    double scale,
    int64_t split_k) {
  const at::SymInt M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(0);
  auto Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f4f4bf16_meta(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    std::optional<at::Tensor> /* output = std::nullopt */,
    std::optional<at::Tensor> /* global_scale = std::nullopt */) {
  const at::SymInt M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(0);
  auto Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_rowwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    std::optional<at::Tensor> /* bias = std::nullopt */,
    bool /* use_fast_accum = true */) {
  int64_t x_dims = XQ.dim();
  int64_t w_dims = WQ.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of XQ must be 2 or 3, and dim of WQ must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = XQ.sym_size(0);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = XQ.sym_size(0);
    const at::SymInt M = XQ.sym_size(1);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
  }
  return Y;
}

at::Tensor f8f8f16_rowwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    std::optional<at::Tensor> /* bias = std::nullopt */,
    bool /* use_fast_accum = true */) {
  const at::SymInt M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(0);
  auto Y = at::empty_symint({M, N}, XQ.options().dtype(at::kHalf));
  return Y;
}

void f8f8bf16_rowwise_out_meta(
    at::Tensor /* XQ */,
    at::Tensor /* WQ */, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    at::Tensor /* output */,
    std::optional<at::Tensor> /* bias = std::nullopt */,
    bool /* use_fast_accum = true */) {
  return;
}

at::Tensor f8f8bf16_rowwise_batched_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    std::optional<at::Tensor> /* bias = std::nullopt */,
    bool /* use_fast_accum = true */,
    std::optional<at::Tensor> /* output = std::nullopt */) {
  const at::SymInt B = XQ.sym_size(0);
  const at::SymInt M = XQ.sym_size(1);
  const at::SymInt N = WQ.sym_size(1);
  auto Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_blockwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    int64_t /* block_m = 128*/,
    int64_t /* block_n = 128*/,
    int64_t /* block_k = 128*/) {
  int64_t x_dims = XQ.dim();
  int64_t w_dims = WQ.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of XQ must be 2 or 3, and dim of WQ must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = XQ.sym_size(0);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = XQ.sym_size(0);
    const at::SymInt M = XQ.sym_size(1);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
  }
  return Y;
}

at::Tensor f8f8bf16_cublas_meta(
    at::Tensor X,
    at::Tensor W,
    std::optional<at::Tensor> /* x_scale = std::nullopt */,
    std::optional<at::Tensor> /* w_scale = std::nullopt */,
    bool /* use_fast_accum = true */,
    std::optional<at::Tensor> /* output = std::nullopt */) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor bf16x9_gemm_meta(
    at::Tensor A,
    at::Tensor B,
    std::optional<at::Tensor> /* output = std::nullopt */) {
  const at::SymInt M = A.sym_size(0);
  const at::SymInt N = B.sym_size(0);
  auto Y = at::empty_symint({M, N}, A.options().dtype(at::kFloat));
  return Y;
}

at::Tensor f8f8bf16_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor scale,
    bool use_fast_accum = true) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_tensorwise_meta(
    at::Tensor XQ,
    at::Tensor WQ,
    double /* scale */,
    bool /* use_fast_accum = true */) {
  int64_t x_dims = XQ.dim();
  int64_t w_dims = WQ.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of XQ must be 2 or 3, and dim of WQ must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = XQ.sym_size(0);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = XQ.sym_size(0);
    const at::SymInt M = XQ.sym_size(1);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
  }
  return Y;
}

at::Tensor f8i4bf16_rowwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // INT4
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    at::Tensor /* w_zp */) {
  int64_t x_dims = XQ.dim();
  int64_t w_dims = WQ.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of X must be 2 or 3, and dim of W must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = XQ.sym_size(0);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = XQ.sym_size(0);
    const at::SymInt M = XQ.sym_size(1);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
  }
  return Y;
}

at::Tensor f8i4bf16_shuffled_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // INT4
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    at::Tensor /* w_scale_group */) {
  const at::SymInt M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(0);
  auto Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor bf16i4bf16_rowwise_meta(
    at::Tensor X, // BF16
    at::Tensor W, // INT4
    at::Tensor /*  w_scale_group */,
    at::Tensor /* w_zero_group */
) {
  int64_t x_dims = X.dim();
  int64_t w_dims = W.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of XQ must be 2 or 3, and dim of WQ must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = X.sym_size(0);
    const at::SymInt N = W.sym_size(0);
    Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = X.sym_size(0);
    const at::SymInt M = X.sym_size(1);
    const at::SymInt N = W.sym_size(0);
    Y = at::empty_symint({B, M, N}, X.options().dtype(at::kBFloat16));
  }
  return Y;
}

at::Tensor bf16i4bf16_shuffled_batched_meta(
    at::Tensor X, // BF16
    at::Tensor W, // INT4
    at::Tensor /* w_scale_group */,
    at::Tensor /* w_zero_group */
) {
  const at::SymInt B = X.sym_size(0);
  const at::SymInt M = X.sym_size(1);
  const at::SymInt N = W.sym_size(1);
  auto Y = at::empty_symint({B, M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor bf16i4bf16_rowwise_batched_meta(
    at::Tensor X, // BF16
    at::Tensor W, // INT4
    at::Tensor /* w_scale_group */,
    at::Tensor /* w_zero_group */
) {
  const at::SymInt B = X.sym_size(0);
  const at::SymInt M = X.sym_size(1);
  const at::SymInt N = W.sym_size(1);
  auto Y = at::empty_symint({B, M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

std::vector<at::Tensor> bf16bf16bf16_grouped_meta(
    at::TensorList X,
    at::TensorList W) {
  std::vector<at::Tensor> Y;
  for (int i = 0; i < X.size(); i++) {
    const at::SymInt M = X[i].sym_size(0);
    const at::SymInt N = W[i].sym_size(0);
    Y.push_back(at::empty_symint({M, N}, X[i].options().dtype(at::kBFloat16)));
  }
  return Y;
}

at::Tensor bf16bf16bf16_grouped_dynamic_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor /* zero_start_index_M */) {
  const at::SymInt G = X.sym_size(0);
  const at::SymInt M = X.sym_size(1);
  const at::SymInt N = W.sym_size(1);
  at::Tensor Y =
      at::empty_symint({G, M, N}, X[0].options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor bf16bf16bf16_grouped_stacked_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor /* M_sizes */,
    std::optional<at::Tensor> out,
    std::optional<int64_t> /* num_sms */) {
  const at::SymInt total_M = X.sym_size(0);
  const at::SymInt N = W.sym_size(1);

  if (out.has_value()) {
    return out.value();
  } else {
    at::Tensor output =
        at::empty_symint({total_M, N}, X.options().dtype(at::kBFloat16));
    return output;
  }
}

std::tuple<at::Tensor, at::Tensor> preshuffle_i4_meta(
    at::Tensor WQ,
    at::Tensor w_scale) {
  auto WS = at::empty_like(w_scale);
  if (w_scale.dtype() != at::kBFloat16) {
    WS = at::empty({w_scale.size(0), 8, w_scale.size(1)}, w_scale.options());
  }
  return {at::empty_like(WQ), WS};
}

at::Tensor f8f8bf16_rowwise_grouped_stacked_meta(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    at::Tensor /* M_sizes */) {
  const at::SymInt total_M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(1);
  at::Tensor Y =
      at::empty_symint({total_M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

TORCH_LIBRARY_IMPL(mslk, Meta, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise_meta);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise_meta);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise_meta);
  m.impl("f8f8bf16_rowwise_out", f8f8bf16_rowwise_out_meta);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched_meta);
  m.impl(
      "f8f8bf16_rowwise_grouped_stacked",
      f8f8bf16_rowwise_grouped_stacked_meta);
  m.impl("bf16bf16bf16_grouped", bf16bf16bf16_grouped_meta);
  m.impl("bf16bf16bf16_grouped_dynamic", bf16bf16bf16_grouped_dynamic_meta);
  m.impl("bf16bf16bf16_grouped_stacked", bf16bf16bf16_grouped_stacked_meta);

#ifdef USE_ROCM
  m.impl("f8f8f16_rowwise", f8f8f16_rowwise_meta);
  m.impl("f8f8bf16_rowwise_preshuffle", f8f8bf16_rowwise_meta);
  m.impl("f8f8f16_rowwise_preshuffle", f8f8f16_rowwise_meta);
#else
  m.impl("i8i8bf16", i8i8bf16_meta);
  m.impl("f4f4bf16", f4f4bf16_meta);
  m.impl("f8f8bf16", f8f8bf16_meta);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas_meta);
  m.impl("bf16x9_gemm", bf16x9_gemm_meta);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched_meta);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise_meta);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise_meta);
  m.impl("bf16i4bf16_shuffled_batched", bf16i4bf16_shuffled_batched_meta);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched_meta);
  m.impl("f8i4bf16_shuffled", f8i4bf16_shuffled_meta);
  m.impl("preshuffle_i4", preshuffle_i4_meta);
#endif
}

} // namespace mslk::gemm
