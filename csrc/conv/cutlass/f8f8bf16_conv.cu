/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/conv/collective/collective_builder.hpp>
#include <cutlass/conv/convnd_problem_shape.hpp>
#include <cutlass/conv/convolution.h>
#include <cutlass/conv/device/conv_universal_adapter.hpp>
#include <cutlass/conv/dispatch_policy.hpp>
#include <cutlass/conv/kernel/conv_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
// clang-format on

#include "f8f8bf16_conv/f8f8bf16_conv_manifest.cuh"

namespace mslk::conv {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

struct ProblemSize {
  std::vector<int64_t> activation_shape; // [N, D, H, W, C]
  std::vector<int64_t> filter_shape; // [K, T, R, S, C]
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  bool operator==(const ProblemSize& ps) const {
    return activation_shape == ps.activation_shape &&
        filter_shape == ps.filter_shape;
  }
  void print() const {
    // clang-format off
    std::cout << "actv: " // [N, D, H, W, C]
              << activation_shape[0] << ","
              << activation_shape[1] << ","
              << activation_shape[2] << ","
              << activation_shape[3] << ","
              << activation_shape[4] << ","
              << "filter: " // [K, T, R, S, C]
              << filter_shape[0] << ","
              << filter_shape[1] << ","
              << filter_shape[2] << ","
              << filter_shape[3] << ","
              << filter_shape[4] << ","
              << std::endl;
    // clang-format on
  }
};

inline void hash_combine(std::size_t& seed, std::size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hash function for ProblemSize for use in unordered_map
struct ProblemSizeHash {
  std::size_t operator()(const ProblemSize& ps) const {
    std::size_t seed = 0;
    auto vec_hash = [](const std::vector<int64_t>& v) {
      std::size_t h = 0;
      for (auto x : v)
        hash_combine(h, std::hash<int64_t>{}(x));
      return h;
    };
    hash_combine(seed, vec_hash(ps.activation_shape));
    hash_combine(seed, vec_hash(ps.filter_shape));
    // hash_combine(seed, vec_hash(ps.padding));
    // hash_combine(seed, vec_hash(ps.stride));
    // hash_combine(seed, vec_hash(ps.dilation));
    return seed;
  }
};

// clang-format off
// Tuned on GB200
std::unordered_map<ProblemSize, Kernel_f8f8bf16_conv, ProblemSizeHash> kernel_map = {
{{{1,1,192,128,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x1x1},
{{{1,1,192,128,160}, {320,1,1,1,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x128x128_1x1x1},
{{{1,1,384,256,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{1,1,96,64,320}, {640,1,1,1,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x128x128_1x1x1},
{{{1,3,194,130,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{1,3,194,130,160}, {320,3,3,3,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x128x128_4x1x1},
{{{1,3,194,130,320}, {320,3,3,3,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x128x128_4x1x1},
{{{1,3,194,130,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
{{{1,3,386,258,160}, {160,3,3,3,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
{{{1,3,386,258,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{1,3,386,258,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
{{{1,3,48,32,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x128x128_2x1x1},
{{{1,3,50,34,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x1x1},
{{{1,3,50,34,48}, {1024,3,3,3,48}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x1024x128_4x4x1},
{{{1,3,50,34,640}, {640,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x128x128_4x1x1},
{{{1,3,50,34,640}, {96,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x128x128_2x1x1},
{{{1,3,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
{{{1,3,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
{{{1,3,98,66,320}, {640,3,3,3,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{1,3,98,66,640}, {640,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{1,4,192,128,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x1x1},
{{{1,4,384,256,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{1,4,96,64,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{1,4,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{1,6,194,130,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
{{{1,6,194,130,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
{{{1,6,386,258,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{1,6,386,258,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,1,192,128,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x1x1},
{{{2,1,192,128,160}, {320,1,1,1,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{2,1,384,256,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,1,96,64,320}, {640,1,1,1,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,3,194,130,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
{{{2,3,194,130,160}, {320,3,3,3,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
{{{2,3,194,130,320}, {320,3,3,3,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x128x128_4x1x1},
{{{2,3,194,130,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
{{{2,3,386,258,160}, {160,3,3,3,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,3,386,258,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,3,386,258,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,3,48,32,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
{{{2,3,50,34,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x2x1},
{{{2,3,50,34,48}, {1024,3,3,3,48}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x1024x128_4x4x1},
{{{2,3,50,34,640}, {640,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x1x1},
{{{2,3,50,34,640}, {96,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x128x128_4x1x1},
{{{2,3,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,3,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,3,98,66,320}, {640,3,3,3,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{2,3,98,66,640}, {640,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
{{{2,4,192,128,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{2,4,384,256,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,4,96,64,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
{{{2,4,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,6,194,130,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,6,194,130,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{2,6,386,258,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{2,6,386,258,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,1,192,128,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{4,1,192,128,160}, {320,1,1,1,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{4,1,384,256,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{4,1,96,64,320}, {640,1,1,1,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{4,3,194,130,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,3,194,130,160}, {320,3,3,3,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,3,194,130,320}, {320,3,3,3,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,3,194,130,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
{{{4,3,386,258,160}, {160,3,3,3,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{4,3,386,258,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,3,386,258,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,3,48,32,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{4,3,50,34,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
{{{4,3,50,34,48}, {1024,3,3,3,48}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x512x128_4x2x1},
{{{4,3,50,34,640}, {640,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{4,3,50,34,640}, {96,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x128x128_2x1x1},
{{{4,3,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,3,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
{{{4,3,98,66,320}, {640,3,3,3,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,3,98,66,640}, {640,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,4,192,128,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{4,4,384,256,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,4,96,64,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{4,4,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,6,194,130,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
{{{4,6,194,130,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,6,386,258,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{4,6,386,258,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,1,192,128,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,1,192,128,160}, {320,1,1,1,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,1,384,256,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,1,96,64,320}, {640,1,1,1,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,3,194,130,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,3,194,130,160}, {320,3,3,3,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,3,194,130,320}, {320,3,3,3,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,3,194,130,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,3,386,258,160}, {160,3,3,3,160}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,3,386,258,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,3,386,258,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,3,48,32,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,3,50,34,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,3,50,34,48}, {1024,3,3,3,48}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x512x128_4x2x1},
{{{8,3,50,34,640}, {640,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,3,50,34,640}, {96,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x128x128_1x1x1},
{{{8,3,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,3,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,3,98,66,320}, {640,3,3,3,320}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,3,98,66,640}, {640,3,3,3,640}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,4,192,128,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,4,384,256,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,4,96,64,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,4,98,66,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
{{{8,6,194,130,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,6,194,130,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,6,386,258,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
{{{8,6,386,258,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
};
// clang-format on

Kernel_f8f8bf16_conv get_kernel_via_heuristic(
    std::vector<int64_t> activation_shape,
    std::vector<int64_t> filter_shape,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation) {
  ProblemSize ps = {activation_shape, filter_shape, padding, stride, dilation};

  // Try exact match first
  auto it = kernel_map.find(ps);
  if (it != kernel_map.end()) {
    return it->second;
  }

  // If no exact match, look for configs with same spatial dims and filter
  // but use the one with the largest batch that is <= current batch
  int64_t current_batch = ps.activation_shape[0];
  Kernel_f8f8bf16_conv best_kernel = nullptr;
  int64_t best_batch = 0;

  for (const auto& [candidate_ps, kernel] : kernel_map) {
    // Check if spatial dimensions and filter match
    if (candidate_ps.activation_shape[1] == ps.activation_shape[1] &&
        candidate_ps.activation_shape[2] == ps.activation_shape[2] &&
        candidate_ps.activation_shape[3] == ps.activation_shape[3] &&
        candidate_ps.filter_shape == ps.filter_shape) {
      int64_t candidate_batch = candidate_ps.activation_shape[0];

      // Use config with largest batch that is <= current batch
      if (candidate_batch <= current_batch && candidate_batch > best_batch) {
        best_batch = candidate_batch;
        best_kernel = kernel;
      }
    }
  }

  if (best_kernel != nullptr) {
    return best_kernel;
  }

  // Fallback kernel if nothing else matched.
  return f8f8bf16_conv_256x256x128_2x1x1;
}

at::Tensor f8f8bf16_conv(
    at::Tensor activation, // FP8 - 5D
    at::Tensor filter, // FP8 - 5D
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation) { // [dilation_d, dilation_h, dilation_w]
  // Process input shapes, this function supports both NDHWC and NCDHW layouts
  // but for channels first, we need to make sure underlying memory is channels
  // last.
  TORCH_CHECK(activation.dim() == 5, "Activation must be 5D tensor.");
  TORCH_CHECK(filter.dim() == 5, "Filter must be 5D tensor.");
  // Check the layouts of inputs and extract shapes.
  bool channels_last_act = activation.strides()[4] == 1;
  bool channels_last_filt = filter.strides()[4] == 1;
  int64_t n, d, h, w, c;
  int64_t k, t, r, s, fc;
  auto act_sizes = activation.sizes();
  auto filt_sizes = filter.sizes();
  if (channels_last_act) {
    n = act_sizes[0];
    d = act_sizes[1];
    h = act_sizes[2];
    w = act_sizes[3];
    c = act_sizes[4];
  } else {
    // When input is NCDHW, we expect the underlying memory to be channels last.
    // Make sure thats true.
    TORCH_CHECK(
        activation.is_contiguous(c10::MemoryFormat::ChannelsLast3d),
        "NCDHW inputs must have channels last underlying memory.");
    n = act_sizes[0];
    c = act_sizes[1];
    d = act_sizes[2];
    h = act_sizes[3];
    w = act_sizes[4];
  }
  if (channels_last_filt) {
    k = filt_sizes[0];
    t = filt_sizes[1];
    r = filt_sizes[2];
    s = filt_sizes[3];
    fc = filt_sizes[4];
  } else {
    // When filter is KCTRS, we expect the underlying memory to be channels
    // last. Make sure thats true.
    TORCH_CHECK(
        filter.is_contiguous(c10::MemoryFormat::ChannelsLast3d),
        "KCTRS filters must have channels last underlying memory.");
    k = filt_sizes[0];
    fc = filt_sizes[1];
    t = filt_sizes[2];
    r = filt_sizes[3];
    s = filt_sizes[4];
  }

  TORCH_CHECK(c == fc, "Activation and filter channels must match");

  // Select kernel to run via heuristics or tuning.
  // Lookup is based on channels last layout regardless of true shape.
  auto kernel = [&]() {
    return get_kernel_via_heuristic(
        {n, d, h, w, c}, {k, t, r, s, fc}, padding, stride, dilation);
  }();

  return kernel(activation, filter, scale, padding, stride, dilation);
}

#else

at::Tensor f8f8bf16_conv(
    at::Tensor activation,
    at::Tensor filter,
    at::Tensor scale,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation) {
  throw std::runtime_error(
      "SM100 (Blackwell) architecture not supported. Requires CUTLASS 3.x with SM100 support.");
}

#endif

} // namespace mslk::conv
