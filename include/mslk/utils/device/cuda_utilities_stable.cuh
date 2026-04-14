/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Stable ABI version of cuda_utilities.cuh.
// Only includes functions needed by stable ABI kernels, without ATen/c10 deps.

#include <torch/headeronly/util/Exception.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace mslk::utils::device {

template <typename func_t>
inline void set_gpu_max_dynamic_shared_memory(
    func_t kernel,
    const int32_t smem_bytes,
    const int32_t device) {
  // Check
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // "Compute capability 7.x devices allow a single thread block to
  // address the full capacity of shared memory: 96 KB on Volta,
  // 64 KB on Turing. Kernels relying on shared memory allocations
  // over 48 KB per block are architecture-specific, as such they
  // must use dynamic shared memory (rather than statically sized
  // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

  STD_TORCH_CHECK(smem_bytes > 0);

  int max_smem_bytes = 0;
  STD_TORCH_CHECK(
      cudaDeviceGetAttribute(
          &max_smem_bytes,
#ifndef __HIP_PLATFORM_AMD__
          cudaDevAttrMaxSharedMemoryPerBlockOptin,
#else
          hipDeviceAttributeMaxSharedMemoryPerBlock,
#endif
          device) == cudaSuccess,
      "cudaDeviceGetAttribute failed");

  STD_TORCH_CHECK(
      smem_bytes <= max_smem_bytes,
      "Attempted to allocate ",
      smem_bytes / 1024,
      " KB of shared memory but only ",
      max_smem_bytes / 1024,
      " KB is available");

  STD_TORCH_CHECK(
      cudaFuncSetAttribute(
          reinterpret_cast<void*>(kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          // V100: 64 KB; A100: 96 KB; H100: 144 KB
          smem_bytes) == cudaSuccess,
      "cudaFuncSetAttribute failed");
}

} // namespace mslk::utils::device
