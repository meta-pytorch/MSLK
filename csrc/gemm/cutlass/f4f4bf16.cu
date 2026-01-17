/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/device_memory.h>
#include <mslk/utils/utils.h>
#include <mslk/utils/tuning_cache.cuh>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include "f4f4bf16/f4f4bf16_manifest.cuh"
#endif

namespace mslk::gemm {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

namespace {

Kernel_f4f4bf16 get_kernel_via_heuristics(int M, int N, int K) {
  if (M <= 1) {
    if (N <= 1024) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else {
      return f4f4bf16_256_128_4_1_1;
    }
  } else if (M <= 64) {
    if (N <= 1024) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 15360) {
      return f4f4bf16_256_128_4_1_1;
    } else {
      if (K <= 8192) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    }
  } else if (M <= 128) {
    if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 15360) {
      return f4f4bf16_256_128_4_1_1;
    } else {
      if (K <= 11264) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    }
  } else if (M <= 256) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 5120) {
        return f4f4bf16_128_128_1_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 3072) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 3072) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    }
  } else if (M <= 512) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else if (N <= 3072) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else if (N <= 4096) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else if (N <= 6144) {
      if (K <= 3072) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_4_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 9216) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_4_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 6144) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_4_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 5120) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_2_4_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else {
      if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    }
  } else if (M <= 1024) {
    if (N <= 1024) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else if (N <= 2048) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else if (N <= 6144) {
      return f4f4bf16_256_128_2_2_1;
    } else if (N <= 7168) {
      if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 9216) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 10240) {
      if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 5120) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_128_2_4_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 3072) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 13312) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 14336) {
      if (K <= 3072) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 3072) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else {
      if (K <= 3072) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    }
  } else if (M <= 2048) {
    if (N <= 1024) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 2048) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_192_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 5120) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_192_2_4_1;
      }
    } else if (N <= 6144) {
      if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_192_4_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 9216) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 10240) {
      if (K <= 3072) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 11264) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 12288) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_192_4_1_1;
      }
    } else if (N <= 13312) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 14336) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else {
      return f4f4bf16_256_256_2_1_1;
    }
  } else if (M <= 4096) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 6144) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 6144) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_192_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 3072) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 3072) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_192_2_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_192_2_2_1;
      }
    } else if (N <= 7168) {
      if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 12288) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else {
      return f4f4bf16_256_256_2_1_1;
    }
  } else if (M <= 8192) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 4096) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 5120) {
      if (K <= 3072) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 6144) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 7168) {
      if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_128_2_4_1;
      }
    } else if (N <= 9216) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 10240) {
      if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_2_4_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 12288) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 13312) {
      if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else {
      if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    }
  } else {
    if (N <= 1024) {
      if (K <= 3072) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_192_2_4_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 2048) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 3072) {
      if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 4096) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 5120) {
      if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_128_2_4_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_192_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 12288) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 13312) {
      if (K <= 10240) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 14336) {
      return f4f4bf16_256_256_2_1_1;
    } else if (N <= 15360) {
      if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_128_2_4_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else {
      if (K <= 9216) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_192_2_4_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    }
  }
}

const std::unordered_map<std::string, Kernel_f4f4bf16>& get_f4f4bf16_kernels() {
  static const std::unordered_map<std::string, Kernel_f4f4bf16> kernels = {
      {"f4f4bf16_128_128_1_1_1", f4f4bf16_128_128_1_1_1},
      {"f4f4bf16_128_128_2_2_1", f4f4bf16_128_128_2_2_1},
      {"f4f4bf16_128_128_4_1_1", f4f4bf16_128_128_4_1_1},
      {"f4f4bf16_128_256_2_1_1", f4f4bf16_128_256_2_1_1},
      {"f4f4bf16_256_128_2_2_1", f4f4bf16_256_128_2_2_1},
      {"f4f4bf16_256_128_2_4_1", f4f4bf16_256_128_2_4_1},
      {"f4f4bf16_256_128_4_1_1", f4f4bf16_256_128_4_1_1},
      {"f4f4bf16_256_192_2_2_1", f4f4bf16_256_192_2_2_1},
      {"f4f4bf16_256_192_2_4_1", f4f4bf16_256_192_2_4_1},
      {"f4f4bf16_256_192_4_1_1", f4f4bf16_256_192_4_1_1},
      {"f4f4bf16_256_256_2_1_1", f4f4bf16_256_256_2_1_1},
      {"f4f4bf16_256_256_2_2_1", f4f4bf16_256_256_2_2_1},
      {"f4f4bf16_256_256_4_1_1", f4f4bf16_256_256_4_1_1},
  };
  return kernels;
}

Kernel_f4f4bf16 get_kernel_via_tuning(
    int M,
    int N,
    int K,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale = std::nullopt) {
  static TuningCache cache("f4f4bf16");

  M = nextPowerOf2OrRoundUp(M, 1024, 1024);
  N = nextPowerOf2OrRoundUp(N, 1024, 1024);
  K = nextPowerOf2OrRoundUp(K, 1024, 1024);

  const std::string shape_key =
      std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);

  const auto& kernels = get_f4f4bf16_kernels();

  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key, kernels, XQ, WQ, x_scale, w_scale, output, global_scale);
  return kernel;
}

} // namespace

at::Tensor f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output,
    std::optional<at::Tensor> global_scale) {
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(x_scale.is_cuda() && x_scale.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());

  const auto M = XQ.size(0);
  const auto N = WQ.size(0);
  const auto K = XQ.size(1) * 2; // Since K is packed
  constexpr auto BLOCK_SIZE = 16;
  TORCH_CHECK(
      N % BLOCK_SIZE == 0 && K % BLOCK_SIZE == 0,
      "Weight dimensions N and K must be multiples of block size 16");

  if (M == 0 || N == 0 || K == 0) {
    // Use zeros instead of empty for special case where K=0.
    return at::zeros({M, N}, XQ.options().dtype(at::kBFloat16));
  }

  at::Tensor out = output.has_value()
      ? output.value()
      : at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  auto kernel = [&]() {
    if (std::getenv("MSLK_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(
          M, N, K, XQ, WQ, x_scale, w_scale, out, global_scale);
    }
    return get_kernel_via_heuristics(M, N, K);
  }();
  return kernel(XQ, WQ, x_scale, w_scale, out, global_scale);
}

#else

at::Tensor f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output,
    std::optional<at::Tensor> global_scale) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}

#endif

} // namespace mslk::gemm
