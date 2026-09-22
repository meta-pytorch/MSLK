/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

#include <mslk/utils/device/cuda_utilities.cuh>

namespace mslk::utils::device {

static_assert(kMaxGridDimX == 2147483647);
static_assert(kMaxThreadsPerLaunch == 4294967295ULL);

TEST(CudaUtilitiesTest, CalculatesXBlockCountWithoutOverflow) {
  EXPECT_EQ(cuda_calc_xblock_count(1025, 1024), uint32_t{2});
  EXPECT_EQ(cuda_calc_xblock_count(0, 256), uint32_t{0});
  EXPECT_EQ(
      cuda_calc_xblock_count(std::numeric_limits<uint64_t>::max(), 1),
      static_cast<uint32_t>(kMaxGridDimX));
  EXPECT_THROW(cuda_calc_xblock_count(1, 0), c10::Error);
  EXPECT_THROW(cuda_calc_xblock_count(-1, 256), c10::Error);
}

TEST(CudaUtilitiesTest, FloorsAndClampsBlockCounts) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  EXPECT_EQ(cap_grid_dim_x(0, 256, stream, BlockCapPolicy::Never), uint32_t{1});
  EXPECT_EQ(
      cap_grid_dim_x(-1, 256, stream, BlockCapPolicy::Never), uint32_t{1});
  EXPECT_EQ(
      cap_grid_dim_x(int64_t{1} << 31, 256, stream, BlockCapPolicy::Never),
      static_cast<uint32_t>(kMaxGridDimX));
}

TEST(CudaUtilitiesTest, AllowsMaximumRocmLaunchThreadCount) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t threads_per_block = 256;
  constexpr int64_t last_valid_blocks =
      (kMaxThreadsPerLaunch - 1) / threads_per_block;
  EXPECT_EQ(
      cap_grid_dim_x(
          last_valid_blocks,
          threads_per_block,
          stream,
          BlockCapPolicy::OverflowOnly),
      last_valid_blocks);

  constexpr int64_t first_invalid_blocks = last_valid_blocks + 1;
  static_assert(
      first_invalid_blocks * threads_per_block > kMaxThreadsPerLaunch);
#ifdef __HIP_PLATFORM_AMD__
  const auto capped = cap_grid_dim_x(
      first_invalid_blocks,
      threads_per_block,
      stream,
      BlockCapPolicy::OverflowOnly);
  EXPECT_LT(capped, first_invalid_blocks);
  EXPECT_LT(
      static_cast<uint64_t>(capped) * threads_per_block, kMaxThreadsPerLaunch);
#else
  EXPECT_EQ(
      cap_grid_dim_x(
          first_invalid_blocks,
          threads_per_block,
          stream,
          BlockCapPolicy::OverflowOnly),
      first_invalid_blocks);
#endif
}

TEST(CudaUtilitiesTest, RejectsImpossibleFixedGridPlane) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t threads_per_block = 256;
  constexpr int64_t first_invalid_yz_blocks =
      (kMaxThreadsPerLaunch - 1) / threads_per_block + 1;
#ifdef __HIP_PLATFORM_AMD__
  EXPECT_THROW(
      cap_grid_dim_x_with_yz_blocks(
          1,
          threads_per_block,
          first_invalid_yz_blocks,
          stream,
          BlockCapPolicy::OverflowOnly),
      c10::Error);
  EXPECT_THROW(
      cap_grid_dim_x_with_yz_blocks(
          1,
          threads_per_block,
          first_invalid_yz_blocks,
          stream,
          BlockCapPolicy::Never),
      c10::Error);
#else
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          1,
          threads_per_block,
          first_invalid_yz_blocks,
          stream,
          BlockCapPolicy::OverflowOnly),
      uint32_t{1});
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          1,
          threads_per_block,
          first_invalid_yz_blocks,
          stream,
          BlockCapPolicy::Never),
      uint32_t{1});
#endif
}

TEST(CudaUtilitiesTest, HandlesMultidimensionalMultiplicity) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t blocks = 65535;
  constexpr int64_t grid_y = 257;
  constexpr int64_t threads_per_block = 256;
#ifdef __HIP_PLATFORM_AMD__
  const auto capped = cap_grid_dim_x_with_yz_blocks(
      blocks, threads_per_block, grid_y, stream, BlockCapPolicy::OverflowOnly);
  EXPECT_LT(capped, blocks);
  EXPECT_LT(
      static_cast<uint64_t>(capped) * grid_y * threads_per_block,
      kMaxThreadsPerLaunch);
#else
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          blocks,
          threads_per_block,
          grid_y,
          stream,
          BlockCapPolicy::OverflowOnly),
      blocks);
#endif
}

TEST(CudaUtilitiesTest, AlwaysCapsForBothStreamRepresentations) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const cudaStream_t raw_stream = stream.stream();
  constexpr int64_t blocks = 1000000;

  const auto from_cuda_stream =
      cap_grid_dim_x(blocks, 256, stream, BlockCapPolicy::Always);
  const auto from_raw_stream =
      cap_grid_dim_x(blocks, 256, raw_stream, BlockCapPolicy::Always);
  EXPECT_EQ(from_raw_stream, from_cuda_stream);
  EXPECT_LT(from_cuda_stream, blocks);

  const auto converted = to_cuda_stream(raw_stream);
  EXPECT_EQ(converted.stream(), raw_stream);
  EXPECT_EQ(converted.device_index(), stream.device_index());
}

TEST(CudaUtilitiesTest, AlwaysHonorsMultidimensionalLaunchLimit) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t blocks = 65535;
  constexpr int64_t grid_y = 1024;
  constexpr int64_t threads_per_block = 256;
  const auto capped = cap_grid_dim_x_with_yz_blocks(
      blocks, threads_per_block, grid_y, stream, BlockCapPolicy::Always);
#ifdef __HIP_PLATFORM_AMD__
  EXPECT_LE(
      static_cast<uint64_t>(capped) * grid_y * threads_per_block,
      kMaxThreadsPerLaunch);
#else
  EXPECT_LT(capped, blocks);
#endif
}

TEST(CudaUtilitiesTest, CapsFromWorkload) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  EXPECT_THROW(
      cap_grid_dim_x_from_workload(0, 256, stream, BlockCapPolicy::Never),
      c10::Error);
  EXPECT_EQ(
      cap_grid_dim_x_from_workload(1025, 256, stream, BlockCapPolicy::Never),
      uint32_t{5});
}

TEST(CudaUtilitiesTest, ChecksLaunchThreadProductBoundary) {
  constexpr uint32_t last_valid = std::numeric_limits<uint32_t>::max();
  EXPECT_EQ(
      check_launch_thread_product(dim3(last_valid), dim3(1), "boundary"),
      kMaxThreadsPerLaunch);

#ifdef __HIP_PLATFORM_AMD__
  try {
    check_launch_thread_product(
        dim3(std::numeric_limits<uint32_t>::max()), dim3(2), "boundary");
    FAIL() << "Expected the ROCm launch boundary check to fail";
  } catch (const c10::Error& error) {
    EXPECT_NE(
        std::string(error.what())
            .find("must not exceed the legacy per-launch limit"),
        std::string::npos);
  }
#else
  EXPECT_EQ(
      check_launch_thread_product(
          dim3(std::numeric_limits<uint32_t>::max()), dim3(2), "boundary"),
      2 * kMaxThreadsPerLaunch);
#endif
}

TEST(CudaUtilitiesTest, RejectsZeroLaunchDimensions) {
  EXPECT_THROW(
      check_launch_thread_product(dim3(0, 1, 1), dim3(1), "zero grid"),
      c10::Error);
  EXPECT_THROW(
      check_launch_thread_product(dim3(1), dim3(1, 0, 1), "zero block"),
      c10::Error);
}

} // namespace mslk::utils::device
