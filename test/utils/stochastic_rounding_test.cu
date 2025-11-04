/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <mslk/utils/stochastic_rounding.cuh>

namespace mslk::utils {

////////////////////////////////////////////////////////////////////////////////
// MSLK Stochastic Rounding Kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void convert_float_to_half_mslk_rand(
    at::Half* dst,
    const float* src,
    int size,
    at::PhiloxCudaState philox_args) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    auto random_bits = StochasticRoundingRNGState(philox_args, idx).rand4();
    dst[idx] = stochastic_rounding_scalar(src[idx], random_bits.x);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Rounding Up Kernel
////////////////////////////////////////////////////////////////////////////////

template <int rounding_choice>
__global__ void
convert_float_to_half_deterministic(at::Half* dst, const float* src, int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if constexpr (rounding_choice > 0) {
      dst[idx] = __float2half_ru(src[idx]);
    } else if constexpr (rounding_choice < 0) {
      dst[idx] = __float2half_rd(src[idx]);
    } else {
      dst[idx] = __float2half_rz(src[idx]);
    }
  }
}

half float2half_ru(float x) {
#ifdef USE_ROCM
  auto f16 =
      at::full({1}, x, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
  auto f32 =
      at::full({1}, x, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  convert_float_to_half_deterministic<1>
      <<<1, 32>>>(f16.data_ptr<at::Half>(), f32.data_ptr<float>(), 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return f16.to(at::kCPU).data_ptr<at::Half>()[0];

#else
  return __float2half_ru(x);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Benchmarking
////////////////////////////////////////////////////////////////////////////////

inline at::PhiloxCudaState philox_rng(long seed) {
  at::manual_seed(seed);
  const auto gen = at::cuda::detail::getDefaultCUDAGenerator();
  return at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(4);
}

inline bool half_equal(const half& a, const half& b) {
  // https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/float16.h
  return reinterpret_cast<__half_raw*>(const_cast<half*>(&a))->x ==
      reinterpret_cast<__half_raw*>(const_cast<half*>(&b))->x;
}

void test_stochastic_rounding(float test_value, int num_samples = 10000000) {
  // Expected FP16 values and their FP32 representation
  const half h_floor = __float2half_rz(test_value);
  const half h_ceil = float2half_ru(test_value);
  const float f_floor = __half2float(h_floor);
  const float f_ceil = __half2float(h_ceil);

  // Expected probability of rounding upwards
  const float expected_probability =
      (test_value - f_floor) / (f_ceil - f_floor);

  printf(
      "\n"
      "Testing FP32 value  : %.11f\n"
      "FP16 floor          : %.11f (0x%04x)\n"
      "FP16 ceil           : %.11f (0x%04x)\n",
      test_value,
      __half2float(h_floor),
      *reinterpret_cast<const uint16_t*>(&h_floor),
      __half2float(h_ceil),
      *reinterpret_cast<const uint16_t*>(&h_ceil));

  constexpr int block_size = 128;
  const int num_blocks = (num_samples + block_size - 1) / block_size;

  // Set up buffers with the test value
  auto f32 = at::full(
      {num_samples},
      test_value,
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  auto f16 = at::full(
      {num_samples},
      test_value,
      at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
  const auto rng_input = philox_rng(1234567890L);

  // Convert FP32 to FP16 using stochastic rounding
  convert_float_to_half_mslk_rand<<<num_blocks, block_size>>>(
      f16.data_ptr<at::Half>(), f32.data_ptr<float>(), num_samples, rng_input);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Sync back to host for access
  f16 = f16.to(at::kCPU);

  // Compare values and count number of round - ups int round_up_count = 0;
  int round_up_count = 0;
  for (auto i = 0; i < f16.numel(); i++) {
    const auto x = f16.data_ptr<at::Half>()[i];
    if (half_equal(x, h_ceil)) {
      round_up_count++;
    }
  }

  // Calculate actual probability of rounding up and difference from expected
  const float actual_probability =
      static_cast<float>(round_up_count) / num_samples;
  const float difference = std::abs(actual_probability - expected_probability);

  printf(
      "Results:\n"
      "Number of samples    : %d\n"
      "Round-up Count       : %d\n"
      "Expected probability : %.11f\n"
      "Actual probability   : %.11f\n"
      "Difference           : %.11f\n",
      num_samples,
      round_up_count,
      expected_probability,
      actual_probability,
      difference);

  EXPECT_TRUE(difference < 1e-4f)
      << "Expected difference in probability of rounding up with stochastic rounding should less than 1e-4f ";
}

TEST(StochasticRoundingTest, stochastic_rounding) {
  test_stochastic_rounding(1.1f);
  test_stochastic_rounding(2.7f);
}

} // namespace mslk::utils
