/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/hip/HIPStream.h>
#include <torch/library.h>
#include <torch/types.h>
#include <ATen/cuda/PhiloxUtils.cuh>

#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>

#include "ck_tiled_rand_uniform_kernel.h"

#ifdef HIPIFY_V2
#define getCurrentHIPStream getCurrentCUDAStream
#endif

namespace {

/**
 * Generate a [B, num_heads, M, N] uint8 tensor of CK philox random bytes (the
 * same mask CK's fused forward uses). Dimensions are passed explicitly so no
 * full-size scratch tensor is materialized, and ``device_index`` selects the
 * device generator/stream so backward regenerates the identical mask on any
 * device. The launch is asynchronous: the mask is produced on the device stream
 * and consumed by the caller on the same stream (no host sync).
 */
at::Tensor rand_uniform_int(
    double /*dropout_prob*/,
    int64_t B,
    int64_t num_heads,
    int64_t M,
    int64_t N,
    int64_t device_index) {
  const auto dev = static_cast<at::DeviceIndex>(device_index);
  hipStream_t stream = c10::cuda::getCurrentHIPStream(dev).stream();

  at::CUDAGeneratorImpl* gen =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(
          c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator(dev));

  at::PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs =
        gen->philox_cuda_state((B + 3) * (num_heads + 1) * (M + 1) * (N + 1));
  }

  const auto seeds = at::cuda::philox::unpack(rng_engine_inputs);
  int64_t philox_seed = std::get<0>(seeds);
  int64_t philox_offset = std::get<1>(seeds);

  at::Tensor randvals = at::empty(
      {B, num_heads, M, N},
      at::TensorOptions().dtype(at::ScalarType::Byte).device(at::kCUDA, dev));

  {
    // only work for batched mode
    using FmhaRandUniformKernel_ = FmhaRandUniformKernel<uint8_t, false>;

    const auto kargs = FmhaRandUniformKernel_::MakeKargs(
        randvals.data_ptr(),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(num_heads),
        static_cast<int>(B),
        static_cast<int>(randvals.stride(2)),
        static_cast<int>(randvals.stride(3)),
        static_cast<int>(randvals.stride(1)),
        static_cast<int>(randvals.stride(0)),
        {philox_seed, philox_offset});

    dim3 kGridSize = FmhaRandUniformKernel_::GridSize(
        static_cast<int>(B),
        static_cast<int>(num_heads),
        static_cast<int>(M),
        static_cast<int>(N));
    const dim3 kBlockSize = FmhaRandUniformKernel_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaRandUniformKernel_::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockPerCu>(
            FmhaRandUniformKernel_{}, kGridSize, kBlockSize, 0, kargs));
  }

  return randvals;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_ck_rand_uniform(float p, int b, int num_heads, int m, int n, int device_index) -> Tensor"));
}

// No Tensor argument to carry a dispatch key (dims + device_index are scalars),
// so register backend-agnostically; the kernel targets device_index itself.
TORCH_LIBRARY_IMPL(xformers, CompositeExplicitAutograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_ck_rand_uniform"),
      TORCH_FN(rand_uniform_int));
}
