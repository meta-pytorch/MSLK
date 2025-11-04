/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <unordered_map>

namespace mslk::utils::device {

// Based on the empirical study, max grid size that is 64x larger than the
// number of SMs gives good performance across the board
constexpr int32_t MAX_THREAD_BLOCKS_FACTOR = 64;

template <class...>
constexpr bool dependent_false_v = false;

inline auto get_max_thread_blocks(const c10::cuda::CUDAStream& stream) {
  const auto device = stream.device_index();
  return MAX_THREAD_BLOCKS_FACTOR *
      at::cuda::getDeviceProperties(device)->multiProcessorCount;
}

inline auto get_compute_versions() {
  static const auto versions = [] {
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);

    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);

    return std::make_tuple(runtime_version, driver_version);
  }();

  return versions;
}

inline auto get_device_for_stream(const cudaStream_t& stream) {
  // Keep as thread local to avoid race conditions
  static thread_local std::unordered_map<cudaStream_t, int> table;

  if (const auto search = table.find(stream); search != table.end()) {
    return search->second;

  } else {
    int device = 0;

    // CUDA 12.8+ introduced cudaStreamGetDevice() to straightforwardly fetch
    // the device from a given stream, but since the runtime drivers may not be
    // at the latest, it will not support the API.  As such, we fetch the device
    // ID can be fetched by context capture instead.

    // Save the current device
    int current_device;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    // Force stream association by capturing dummy work
    cudaStreamCaptureStatus status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));

    // Save the device associated with the stream, and revert back to the
    // current device
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaSetDevice(current_device));

    table.insert({stream, device});
    return device;
  }
}

template <typename S>
inline c10::cuda::CUDAStream to_cuda_stream(
    S&& stream,
    const int device_index = -1) {
  if constexpr (std::is_same_v<std::decay_t<S>, c10::cuda::CUDAStream>) {
    // Already a CUDAStream, return as is
    return std::forward<S>(stream);

  } else if constexpr (std::is_same_v<std::decay_t<S>, cudaStream_t>) {
    // A raw cudaStream_t, figure out the associated device_index and pack into
    // CUDAStream
    const auto idx =
        (device_index < 0) ? get_device_for_stream(stream) : device_index;
    return c10::cuda::getStreamFromExternal(stream, idx);

  } else {
    static_assert(
        dependent_false_v<S>,
        "Unsupported stream type. Expected cudaStream_t or c10::cuda::CUDAStream.");
  }
}

template <typename func_t>
inline void set_gpu_max_dynamic_shared_memory(
    func_t kernel,
    const int32_t smem_bytes,
    const int32_t device = at::cuda::current_device()) {
  // Check
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // "Compute capability 7.x devices allow a single thread block to
  // address the full capacity of shared memory: 96 KB on Volta,
  // 64 KB on Turing. Kernels relying on shared memory allocations
  // over 48 KB per block are architecture-specific, as such they
  // must use dynamic shared memory (rather than statically sized
  // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

  TORCH_CHECK(smem_bytes > 0);

  int max_smem_bytes = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_smem_bytes,
#ifndef __HIP_PLATFORM_AMD__
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
#else
      hipDeviceAttributeMaxSharedMemoryPerBlock,
#endif
      device));

  TORCH_CHECK(
      smem_bytes <= max_smem_bytes,
      "Attempted to allocate ",
      smem_bytes / 1024,
      " KB of shared memory but only ",
      max_smem_bytes / 1024,
      " KB is available");

  C10_CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<void*>(kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      // V100: 64 KB; A100: 96 KB; H100: 144 KB
      smem_bytes));
}

} // namespace mslk::utils::device
