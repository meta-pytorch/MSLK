/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

////////////////////////////////////////////////////////////////////////////////
// Stable ABI Kernel Launch Utilities
//
// Minimal kernel launch helpers that only depend on the CUDA runtime and
// PyTorch stable ABI headers.  For the full-featured KernelLauncher (with
// execution timing, tensor checking, etc.) see kernel_launcher.cuh.
////////////////////////////////////////////////////////////////////////////////

#include <torch/headeronly/util/Exception.h>

#include <mslk/utils/device/cuda_utilities_stable.cuh>

#include <cuda_runtime.h>

namespace mslk::utils {

/// Launch a cooperative kernel with error checking and optional dynamic
/// shared memory opt-in.
template <typename KernelFunc, typename... Args>
inline void launch_cooperative_kernel(
    KernelFunc kernel,
    dim3 grid,
    dim3 block,
    size_t smem_size,
    cudaStream_t stream,
    int device,
    Args&&... args) {
  if (smem_size >= 48 * 1024) {
    mslk::utils::device::set_gpu_max_dynamic_shared_memory(
        kernel, smem_size, device);
  }

  void* kernel_args[] = {(void*)&args...};
  cudaError_t err = cudaLaunchCooperativeKernel(
      (void*)kernel, grid, block, kernel_args, smem_size, stream);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "cudaLaunchCooperativeKernel failed: ",
      cudaGetErrorString(err));
}

} // namespace mslk::utils
