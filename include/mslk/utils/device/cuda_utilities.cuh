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

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace mslk::utils::device {

constexpr int32_t MAX_THREAD_BLOCKS_FACTOR = 64;
constexpr int64_t kMaxGridDimX = std::numeric_limits<int32_t>::max();
constexpr uint64_t kMaxThreadsPerLaunch = std::numeric_limits<uint32_t>::max();

// Never skips all optional capping. ROCm still validates the fixed grid plane
// against its legacy per-launch limit. On every platform, callers that choose
// Never must use check_launch_thread_product before launch to validate the
// complete grid.
enum class BlockCapPolicy { Always, OverflowOnly, Never };

template <class...>
constexpr bool dependent_false_v = false;

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

template <typename Integer1, typename Integer2>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  static_assert(std::is_integral_v<Integer1>);
  static_assert(std::is_integral_v<Integer2>);

  if constexpr (std::is_signed_v<Integer1>) {
    TORCH_CHECK(
        num_items >= 0,
        "When calculating block counts, the number of items must be nonnegative!");
  }
  TORCH_CHECK(
      threads_per_block > 0,
      "When calculating block counts, the number of threads must be positive!");
  TORCH_CHECK(threads_per_block <= 1024, "Number of threads must be <=1024!");

  const auto items = static_cast<uint64_t>(num_items);
  const auto threads = static_cast<uint64_t>(threads_per_block);
  const auto blocks = items / threads + (items % threads != 0);
  return static_cast<uint32_t>(
      std::min<uint64_t>(blocks, static_cast<uint64_t>(kMaxGridDimX)));
}

template <typename StreamType>
inline uint32_t cap_grid_dim_x_with_yz_blocks(
    int64_t blocks_x_uncapped,
    int64_t threads_per_block,
    int64_t yz_blocks,
    const StreamType& raw_stream,
    BlockCapPolicy policy = BlockCapPolicy::OverflowOnly) {
  TORCH_CHECK(threads_per_block > 0, "threads_per_block must be positive");
  TORCH_CHECK(yz_blocks > 0, "grid.y * grid.z must be positive");

  // Direct block-count callers preserve the legacy behavior of launching one
  // block for non-positive counts. Workload-derived callers reject empty work
  // in cap_grid_dim_x_from_workload before reaching this conversion.
  const auto to_grid_dim = [](int64_t blocks) {
    return static_cast<uint32_t>(
        std::clamp<int64_t>(blocks, int64_t{1}, kMaxGridDimX));
  };

  const auto stream = to_cuda_stream(raw_stream);
  const auto perf_block_cap = [&stream]() {
    return static_cast<int64_t>(
        MAX_THREAD_BLOCKS_FACTOR *
        at::cuda::getDeviceProperties(stream.device_index())
            ->multiProcessorCount);
  };

#ifdef __HIP_PLATFORM_AMD__
  const auto max_total_blocks =
      kMaxThreadsPerLaunch / static_cast<uint64_t>(threads_per_block);
  TORCH_CHECK(
      static_cast<uint64_t>(yz_blocks) <= max_total_blocks,
      "The fixed grid.y * grid.z plane cannot fit within the ROCm per-launch "
      "thread limit");
  const auto overflow_block_cap =
      static_cast<int64_t>(max_total_blocks / static_cast<uint64_t>(yz_blocks));
#endif

  if (policy == BlockCapPolicy::Never) {
    return to_grid_dim(blocks_x_uncapped);
  }

  if (policy == BlockCapPolicy::Always) {
#ifdef __HIP_PLATFORM_AMD__
    return to_grid_dim(
        std::min(
            blocks_x_uncapped, std::min(perf_block_cap(), overflow_block_cap)));
#else
    return to_grid_dim(std::min(blocks_x_uncapped, perf_block_cap()));
#endif
  }

#ifdef __HIP_PLATFORM_AMD__
  if (blocks_x_uncapped > overflow_block_cap) {
    // Preserve the legacy get_max_thread_blocks behavior: once a ROCm launch
    // would overflow, apply both its safety limit and the occupancy cap.
    return to_grid_dim(
        std::min(
            blocks_x_uncapped, std::min(perf_block_cap(), overflow_block_cap)));
  }
#endif
  return to_grid_dim(blocks_x_uncapped);
}

template <typename StreamType>
inline uint32_t cap_grid_dim_x(
    int64_t blocks_uncapped,
    int64_t threads_per_block,
    const StreamType& stream,
    BlockCapPolicy policy = BlockCapPolicy::OverflowOnly) {
  return cap_grid_dim_x_with_yz_blocks(
      blocks_uncapped, threads_per_block, 1, stream, policy);
}

template <typename Integer1, typename Integer2, typename StreamType>
inline uint32_t cap_grid_dim_x_from_workload(
    Integer1 num_items,
    Integer2 threads_per_block,
    const StreamType& stream,
    BlockCapPolicy policy = BlockCapPolicy::OverflowOnly) {
  static_assert(std::is_integral_v<Integer1>);
  static_assert(std::is_integral_v<Integer2>);
  TORCH_CHECK(
      num_items > 0,
      "When calculating a capped grid, the number of items must be positive!");
  return cap_grid_dim_x(
      cuda_calc_xblock_count(num_items, threads_per_block),
      static_cast<int64_t>(threads_per_block),
      stream,
      policy);
}

inline uint64_t check_launch_thread_product(
    const dim3& grid,
    const dim3& block,
    const std::string_view context = {}) {
  const auto check_dimensions = [&](const bool condition,
                                    const auto&... message) {
    TORCH_CHECK(
        condition,
        context,
        " [grid dim ",
        grid.x,
        " x ",
        grid.y,
        " x ",
        grid.z,
        "] [block dim ",
        block.x,
        " x ",
        block.y,
        " x ",
        block.z,
        "]: ",
        message...);
  };
  const std::array<uint32_t, 6> dimensions = {
      grid.x, grid.y, grid.z, block.x, block.y, block.z};
  uint64_t total_threads = 1;
  for (const auto dimension : dimensions) {
    check_dimensions(dimension > 0, "Launch dimensions must be positive");
    check_dimensions(
        total_threads <= std::numeric_limits<uint64_t>::max() / dimension,
        "Total thread product overflows uint64_t");
    total_threads *= dimension;
  }

#if defined(__HIP_PLATFORM_AMD__) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
  check_dimensions(
      total_threads <= kMaxThreadsPerLaunch,
      "Total number of threads ",
      total_threads,
      " must not exceed the legacy per-launch limit (",
      kMaxThreadsPerLaunch,
      ").");
#endif
  return total_threads;
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
