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
  // BF16 grouped GEMM grad / wgrad: shared schema. CUDA uses the CUTLASS
  // implementations; ROCm uses the Triton implementations registered by
  // mslk.gemm.triton.grouped_gemm via torch.library.impl at Python import time.
  m.def(
      "bf16bf16bf16_grouped_grad(Tensor X, Tensor W, Tensor M_sizes, Tensor? out=None, int? num_sms=None) -> Tensor");
  m.def(
      "bf16bf16bf16_grouped_wgrad(Tensor X, Tensor W, Tensor M_sizes, Tensor(a!)? output=None, bool output_accum=False, int? num_sms=None) -> Tensor");
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
  // MXFP8 x MXFP4 GEMM: shared schema; CUDA uses the CUTLASS implementation,
  // ROCm uses the Triton implementation registered by mx8mx4_gemm.py.
  m.def(
      "mx8mx4bf16(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? output=None) -> Tensor");
  // Grouped variant: CUDA uses CUTLASS, ROCm uses the Triton implementation
  // registered by mx8mx4_gemm.py via torch.library.impl.
  m.def(
      "mx8mx4bf16_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor offsets, Tensor(a!)? output=None) -> Tensor");
  // MXFP8 x MXFP8 grouped GEMM: shared schema; CUDA uses the CUTLASS
  // implementation, ROCm uses the Triton implementation registered by
  // mx8mx8_gemm.py via torch.library.impl.
  m.def(
      "mx8mx8bf16_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor offsets, Tensor(a!)? output=None, int? actual_num_tokens=None) -> Tensor");
  // FP8 groupwise GEMM: shared schema; CUDA uses CUTLASS, ROCm uses the
  // Triton implementation registered by fp8_groupwise_gemm.py.
  m.def(
      "f8f8bf16_groupwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale) -> Tensor");
  // FP8 groupwise grouped GEMM: shared schema; CUDA uses CUTLASS, ROCm uses the
  // FlyDSL implementation registered by
  // mslk/gemm/flydsl/fp8_groupwise_grouped_gemm.py.
  m.def(
      "f8f8bf16_groupwise_grouped(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes) -> Tensor");
#ifdef USE_ROCM
  // Sibling of f8f8bf16_groupwise_grouped taking weights already swizzled into
  // the MFMA B layout; schema only on ROCm, implemented by the same FlyDSL
  // module via torch.library.impl at Python import time.
  m.def(
      "f8f8bf16_groupwise_grouped_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes) -> Tensor");
  // Preshuffle siblings of the rowwise grouped ops: each takes the schema of
  // the op it is named after, with weights already swizzled into the MFMA B
  // layout. ROCm only, and implemented by
  // mslk/gemm/flydsl/fp8_rowwise_grouped_gemm.py via torch.library.impl at
  // Python import time.
  m.def(
      "f8f8bf16_rowwise_grouped_stacked_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_grouped_dynamic_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor zero_start_index_M, bool zeroing_output_tensor=True) -> Tensor");
  m.def(
      "f8f8f16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8f16_rowwise_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  // Generic PyTorch grouped GEMM API is only available on AMD for now.
  m.def(
      "f8f8bf16_rowwise_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? offsets, Tensor(a!) output) -> Tensor");
  // Sibling of f8f8bf16_rowwise_grouped_mm taking weights already swizzled into
  // the MFMA B layout. It serves the 2D-3D and 3D-3D operand ranks, the ones
  // that leave each group a whole [N, K]; grouping along N or K cuts across the
  // axes the swizzle interleaves, so those ranks raise.
  m.def(
      "f8f8bf16_rowwise_grouped_mm_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? offsets, Tensor(a!) output) -> Tensor");
  // BF16xINT4 rowwise GEMMs: schema only on ROCm; implementations are
  // registered by mslk.gemm.triton.int4_gemm via torch.library.impl at
  // Python import time.
  m.def(
      "bf16i4bf16_rowwise(Tensor X, Tensor W, Tensor w_scale_group, Tensor w_zero_group) -> Tensor");
  m.def(
      "bf16i4bf16_rowwise_batched(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  // INT8 GEMM via Triton — static and dynamic scale variants.
  m.def("i8i8bf16(Tensor XQ, Tensor WQ, float scale, int split_k=1) -> Tensor");
  m.def(
      "i8i8bf16_dynamic(Tensor XQ, Tensor WQ, Tensor scale, int split_k=1) -> Tensor");
#else
  m.def("i8i8bf16(Tensor XQ, Tensor WQ, float scale, int split_k=1) -> Tensor");
  m.def(
      "f4f4bf16(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? output=None, Tensor? global_scale=None, int mxfp4_block_size=32) -> Tensor");
  m.def(
      "mx8mx6bf16(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? output=None) -> Tensor");
  m.def(
      "mx6mx6bf16(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? output=None, int splits=0) -> Tensor");
  m.def(
      "f4f4bf16_grouped_stacked(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes, Tensor? global_scale=None, Tensor? starting_row_after_padding=None, bool use_mx=True) -> Tensor");
  m.def(
      "f4f4bf16_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor offsets, Tensor(a!)? output=None, Tensor(a!)? global_scale=None) -> Tensor");
  m.def(
      "f4f4bf16_ultra_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor offsets, Tensor x_global_scale, Tensor w_global_scale, Tensor(a!)? output=None) -> Tensor");
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

#if !defined(USE_MTIA)
TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
  m.impl("f8f8bf16_rowwise_out", f8f8bf16_rowwise_out);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched);
  m.impl("f8f8bf16_rowwise_grouped", f8f8bf16_rowwise_grouped);
  m.impl("f8f8bf16_rowwise_grouped_cat", f8f8bf16_rowwise_grouped_cat);
  m.impl("bf16bf16bf16_grouped", bf16bf16bf16_grouped);
  m.impl("bf16bf16bf16_grouped_cat", bf16bf16bf16_grouped_cat);
  m.impl("bf16bf16bf16_grouped_dynamic", bf16bf16bf16_grouped_dynamic);
  m.impl("bf16bf16bf16_grouped_stacked", bf16bf16bf16_grouped_stacked);

#ifdef USE_ROCM
  m.impl("f8f8f16_rowwise", f8f8f16_rowwise);
  m.impl("f8f8bf16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
  m.impl("f8f8f16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
  // Deliberately unregistered: a Python torch.library.impl owns the CUDA key
  // for each of these on ROCm, and a C++ registration here would leave that
  // binding overriding this one rather than being the only implementation.
  // The module that binds each has to be imported before the op is called;
  // mslk/gemm/__init__.py does that.
  //   f8f8bf16_rowwise_grouped_stacked / _dynamic / _mm and their preshuffle
  //     siblings -> mslk/gemm/flydsl/fp8_rowwise_grouped_gemm.py
  //   f8f8bf16_groupwise_grouped and its preshuffle sibling
  //     -> mslk/gemm/flydsl/fp8_groupwise_grouped_gemm.py
  //   f8f8bf16_groupwise -> mslk/gemm/triton/fp8_groupwise_gemm.py
  //   i8i8bf16 / i8i8bf16_dynamic -> mslk/gemm/triton/int8_gemm.py
  //   bf16bf16bf16_grouped_grad / _wgrad -> mslk/gemm/triton/grouped_gemm.py
#else
  // The rowwise grouped ops share a schema with ROCm, where FlyDSL implements
  // them from Python; these registrations serve CUTLASS on CUDA only.
  m.impl("f8f8bf16_rowwise_grouped_stacked", f8f8bf16_rowwise_grouped_stacked);
  m.impl("f8f8bf16_rowwise_grouped_dynamic", f8f8bf16_rowwise_grouped_dynamic);
  m.impl("f8f8bf16_groupwise", f8f8bf16_groupwise);
  m.impl("f8f8bf16_groupwise_grouped", f8f8bf16_groupwise_grouped);
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f4f4bf16", f4f4bf16);
  m.impl("mx8mx4bf16", mx8mx4bf16);
  m.impl("mx8mx4bf16_grouped_mm", mx8mx4bf16_grouped_mm);
  m.impl("mx8mx6bf16", mx8mx6bf16);
  m.impl("mx6mx6bf16", mx6mx6bf16);
  m.impl("f4f4bf16_grouped_stacked", f4f4bf16_grouped_stacked);
  m.impl("mx8mx8bf16_grouped_mm", mx8mx8bf16_grouped_mm);
  m.impl("f4f4bf16_grouped_mm", f4f4bf16_grouped_mm);
  m.impl("f4f4bf16_ultra_grouped_mm", f4f4bf16_ultra_grouped_mm);
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
#endif // !defined(USE_MTIA)

#if !defined(USE_MTIA)
// Unfortunately there's broken code in production sometimes calling these ops
// on CPU for silly reasons. To prevent breaking the models, we need to keep the
// ops registered on CPU.
TORCH_LIBRARY_IMPL(mslk, CPU, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
  m.impl("f8f8bf16_rowwise_out", f8f8bf16_rowwise_out);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched);
  m.impl("f8f8bf16_rowwise_grouped", f8f8bf16_rowwise_grouped);
  m.impl("f8f8bf16_rowwise_grouped_cat", f8f8bf16_rowwise_grouped_cat);
  m.impl("bf16bf16bf16_grouped", bf16bf16bf16_grouped);
  m.impl("bf16bf16bf16_grouped_cat", bf16bf16bf16_grouped_cat);
  m.impl("bf16bf16bf16_grouped_dynamic", bf16bf16bf16_grouped_dynamic);
  m.impl("bf16bf16bf16_grouped_stacked", bf16bf16bf16_grouped_stacked);

#ifdef USE_ROCM
  m.impl("f8f8f16_rowwise", f8f8f16_rowwise);
  m.impl("f8f8bf16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
  m.impl("f8f8f16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
  // The ops left out here are the ones a Python torch.library.impl owns on
  // ROCm, listed against the CUDA registration above. They are bound on the
  // CUDA key alone, so calling one on a CPU tensor raises rather than falling
  // back, which is what the note at the top of this block is about.
#else
  // Shared with ROCm, where FlyDSL implements them from Python.
  m.impl("f8f8bf16_rowwise_grouped_stacked", f8f8bf16_rowwise_grouped_stacked);
  m.impl("f8f8bf16_rowwise_grouped_dynamic", f8f8bf16_rowwise_grouped_dynamic);
  m.impl("f8f8bf16_groupwise", f8f8bf16_groupwise);
  m.impl("f8f8bf16_groupwise_grouped", f8f8bf16_groupwise_grouped);
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f4f4bf16", f4f4bf16);
  m.impl("mx8mx4bf16", mx8mx4bf16);
  m.impl("mx8mx4bf16_grouped_mm", mx8mx4bf16_grouped_mm);
  m.impl("mx8mx6bf16", mx8mx6bf16);
  m.impl("mx6mx6bf16", mx6mx6bf16);
  m.impl("f4f4bf16_grouped_stacked", f4f4bf16_grouped_stacked);
  m.impl("mx8mx8bf16_grouped_mm", mx8mx8bf16_grouped_mm);
  m.impl("f4f4bf16_grouped_mm", f4f4bf16_grouped_mm);
  m.impl("f4f4bf16_ultra_grouped_mm", f4f4bf16_ultra_grouped_mm);
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
#endif // !defined(USE_MTIA)

} // namespace mslk::gemm
