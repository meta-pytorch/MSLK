/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <torch/types.h>

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension.
// For PyMODINIT_FUNC to work, we need to include Python.h
// https://github.com/pytorch/vision/blob/main/torchvision/csrc/vision.cpp#L17
// Fixes error LNK2001: unresolved external symbol PyInit__C
#if defined(_WIN32)
#include <Python.h>
PyMODINIT_FUNC PyInit__C(void) {
  // No need to do anything.
  return NULL;
}
#endif // defined(_WIN32)

TORCH_LIBRARY_FRAGMENT(xformers, m) {
#if defined(USE_ROCM)
  // Schemas for ops whose implementations live in hip_fmha/ are registered
  // there, alongside their TORCH_LIBRARY_IMPL, so that they are absent from
  // builds where hip_fmha is not compiled (e.g. MSLK_BUILD_HIP_FMHA=0).
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_decoder_ck(Tensor query, "
      "Tensor key, Tensor value, Tensor? seq_positions, float scale) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_decoder_splitk_ck(Tensor query, Tensor key, "
      " Tensor value, Tensor? seq_positions, float scale, int split_k) -> Tensor"));
#endif
}
