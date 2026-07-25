/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CUTLASS split-K for mx6mx6bf16 deep-K low-occupancy shapes. Thin
// instantiation of the shared `_mx6mx6bf16` pipeline (mx6mx6bf16_common.cuh)
// with USE_SPLITK = true (CUTLASS Stream-K tile scheduler); the GEMM pipeline
// is identical to the dense path. Kept in its own translation unit (like the
// per-tile mx6mx6bf16_*.cu files) so the Stream-K kernel instantiation compiles
// in parallel. Internal helper — the `mx6mx6bf16` op routes here for the deep-K
// win region (or when an explicit `splits` is requested). CUDA-graph safety of
// the split-K barrier reset is handled in `_mx6mx6bf16` (see comment there).

#include "mx6mx6bf16_common.cuh"

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor mx6mx6bf16_splitk(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t splits) {
  const int K = XQ.size(1) * 4 / 3;
  TORCH_CHECK(splits >= 1, "splits must be >= 1");
  TORCH_CHECK(K % (32 * splits) == 0, "K must be divisible by 32*splits");
  return _mx6mx6bf16<256, 128, 2, 1, 1, /*USE_SPLITK=*/true>(
      XQ, WQ, x_scale, w_scale, output, splits);
}

#endif

} // namespace mslk::gemm
