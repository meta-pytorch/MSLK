/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/util/Exception.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

namespace mslk::utils::device {

constexpr uint64_t kMaxThreadsPerLaunch = std::numeric_limits<uint32_t>::max();

template <typename Target>
inline Target checked_nonnegative_integral_cast(
    int64_t value,
    const std::string_view context,
    const std::string_view dimension) {
  static_assert(std::is_integral_v<Target>);
  TORCH_CHECK(
      value >= 0 &&
          static_cast<uint64_t>(value) <=
              static_cast<uint64_t>(std::numeric_limits<Target>::max()),
      context,
      " ",
      dimension,
      " must fit the target integer type; got ",
      value);
  return static_cast<Target>(value);
}

template <typename Grid, typename Block>
inline uint64_t check_launch_thread_product(
    const Grid& grid,
    const Block& block,
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

} // namespace mslk::utils::device
