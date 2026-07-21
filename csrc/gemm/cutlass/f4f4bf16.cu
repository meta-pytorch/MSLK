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

// Returns the compute capability of the given device as major*10 + minor
// (e.g. 100 for B200 / sm_100, 103 for B300 / sm_103). Keyed on XQ's device so
// the dispatch arch always matches the device the kernel launches on, rather
// than a process-wide current_device() that can diverge from it. (Mixed-arch
// hosts are not a tested/supported config.) getDeviceProperties caches
// internally, so the per-call cost is negligible.
int get_device_sm_version(c10::DeviceIndex device_index) {
  const auto* props = at::cuda::getDeviceProperties(device_index);
  return props->major * 10 + props->minor;
}

// Escape hatch to force the SM100 path on B300, for rollback and for
// A/B measurement of the ultra speedup on the same hardware.
bool ultra_path_disabled() {
  static const bool disabled =
      std::getenv("MSLK_F4F4BF16_DISABLE_ULTRA") != nullptr;
  return disabled;
}

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

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
// SM103 (B300) FORMAT-AWARE tile selectors. Each tree is generated by
// make_heuristic (--threshold 0.01) from the *per-format* GB300 full autotune
// sweep over the unified 52-tile pool (Sm100 K=256 + ultra K=768): a K=256
// Sm100 tile is chosen where the wide K=768 ultra tile is inefficient, ultra
// where it wins. Dispatched by format so each format picks its own optimum.
static Kernel_f4f4bf16
get_ultra_kernel_via_heuristics_nvfp4(int M, int N, int K) {
  if (M <= 1) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    }
  } else if (M <= 64) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    }
  } else if (M <= 128) {
    if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    }
  } else if (M <= 256) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 10240) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 6144) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else {
      return f4f4bf16_256_256_4_1_1;
    }
  } else if (M <= 512) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_128_128_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 10240) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 8192) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 8192) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 2048) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 2048) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 4096) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    }
  } else if (M <= 1024) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 6144) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 5120) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 5120) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 13312) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      }
    } else {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    }
  } else if (M <= 2048) {
    if (N <= 1024) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 4096) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_4_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    }
  } else if (M <= 4096) {
    if (N <= 1024) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    }
  } else if (M <= 8192) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    }
  } else {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      }
    }
  }
}

static Kernel_f4f4bf16
get_ultra_kernel_via_heuristics_mxfp4_16(int M, int N, int K) {
  if (M <= 1) {
    if (N <= 5120) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    }
  } else if (M <= 64) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    }
  } else if (M <= 128) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    }
  } else if (M <= 256) {
    if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 10240) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 10240) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_128_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 6144) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_2_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else {
      if (K <= 12288) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    }
  } else if (M <= 512) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 3072) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 8192) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 7168) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 8192) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 2048) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 2048) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 2048) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_128_2_2_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    }
  } else if (M <= 1024) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 7168) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 5120) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 5120) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      }
    } else {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    }
  } else if (M <= 2048) {
    if (N <= 1024) {
      if (K <= 8192) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 2048) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    }
  } else if (M <= 4096) {
    if (N <= 1024) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    }
  } else if (M <= 8192) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    }
  } else {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    }
  }
}

static Kernel_f4f4bf16
get_ultra_kernel_via_heuristics_mxfp4(int M, int N, int K) {
  if (M <= 1) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_128_128_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    }
  } else if (M <= 64) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_128_128_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 3072) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 3072) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 3072) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else {
      if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    }
  } else if (M <= 128) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_128_128_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_2_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 3072) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    } else {
      if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_256_128_4_1_1;
      }
    }
  } else if (M <= 256) {
    if (N <= 1024) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_128_128_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_128_128_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_4_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_256_128_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    }
  } else if (M <= 512) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_128_128_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_4_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      }
    } else if (N <= 3072) {
      if (K <= 2048) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_128_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 4096) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 4096) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_4_2_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_128_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      }
    } else if (N <= 15360) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      }
    } else {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    }
  } else if (M <= 1024) {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_128_128_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      } else if (K <= 4096) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_128_128_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 6144) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 2048) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 2048) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_128_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_128_768_2_4_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_4_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 8192) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_256_2_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    }
  } else if (M <= 2048) {
    if (N <= 1024) {
      if (K <= 5120) {
        return f4f4bf16_128_128_4_1_1;
      } else {
        return f4f4bf16_ultra_256_128_768_2_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_192_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 4096) {
      if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_256_256_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_4_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 2048) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_256_256_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    }
  } else if (M <= 4096) {
    if (N <= 1024) {
      if (K <= 7168) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_256_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_256_768_4_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    }
  } else if (M <= 8192) {
    if (N <= 1024) {
      if (K <= 2048) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_256_768_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 7168) {
      if (K <= 1024) {
        return f4f4bf16_256_192_4_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    }
  } else {
    if (N <= 1024) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_2_4_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 2048) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 3072) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      }
    } else if (N <= 4096) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 2048) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 5120) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 6144) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 8192) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 9216) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 10240) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 5120) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      }
    } else if (N <= 11264) {
      if (K <= 1024) {
        return f4f4bf16_256_192_2_2_1;
      } else if (K <= 6144) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 13312) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 14336) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 15360) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else {
        return f4f4bf16_256_256_4_1_1;
      }
    } else if (N <= 12288) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 14336) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 13312) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 12288) {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      } else if (K <= 13312) {
        return f4f4bf16_256_256_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 14336) {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 8192) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 9216) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else if (N <= 15360) {
      if (K <= 1024) {
        return f4f4bf16_256_256_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 7168) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 10240) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    } else {
      if (K <= 1024) {
        return f4f4bf16_256_256_4_1_1;
      } else if (K <= 3072) {
        return f4f4bf16_ultra_256_192_768_2_1_1;
      } else if (K <= 4096) {
        return f4f4bf16_ultra_256_192_768_4_1_1;
      } else if (K <= 11264) {
        return f4f4bf16_ultra_256_192_768_2_2_1;
      } else {
        return f4f4bf16_ultra_256_192_768_4_2_1;
      }
    }
  }
}

// Format: NVFP4 (global_scale set) / MXFP4_16 (block==16) / MXFP4 (block==32).
Kernel_f4f4bf16 get_ultra_kernel_via_heuristics(
    int M,
    int N,
    int K,
    const std::optional<at::Tensor>& global_scale,
    int64_t mxfp4_block_size) {
  if (global_scale.has_value()) {
    return get_ultra_kernel_via_heuristics_nvfp4(M, N, K);
  }
  if (mxfp4_block_size == 16) {
    return get_ultra_kernel_via_heuristics_mxfp4_16(M, N, K);
  }
  return get_ultra_kernel_via_heuristics_mxfp4(M, N, K);
}

const std::unordered_map<std::string, Kernel_f4f4bf16>&
get_f4f4bf16_ultra_kernels() {
  static const std::unordered_map<std::string, Kernel_f4f4bf16> kernels = {
      {"f4f4bf16_ultra_128_128_768_1_1_1", f4f4bf16_ultra_128_128_768_1_1_1},
      {"f4f4bf16_ultra_128_256_768_1_1_1", f4f4bf16_ultra_128_256_768_1_1_1},
      {"f4f4bf16_ultra_256_128_768_2_1_1", f4f4bf16_ultra_256_128_768_2_1_1},
      {"f4f4bf16_ultra_256_128_768_2_2_1", f4f4bf16_ultra_256_128_768_2_2_1},
      {"f4f4bf16_ultra_256_128_768_4_1_1", f4f4bf16_ultra_256_128_768_4_1_1},
      {"f4f4bf16_ultra_256_256_768_2_1_1", f4f4bf16_ultra_256_256_768_2_1_1},
      {"f4f4bf16_ultra_256_256_768_2_2_1", f4f4bf16_ultra_256_256_768_2_2_1},
      {"f4f4bf16_ultra_256_128_768_2_4_1", f4f4bf16_ultra_256_128_768_2_4_1},
      {"f4f4bf16_ultra_256_256_768_2_4_1", f4f4bf16_ultra_256_256_768_2_4_1},
      {"f4f4bf16_ultra_256_192_768_2_1_1", f4f4bf16_ultra_256_192_768_2_1_1},
      {"f4f4bf16_ultra_256_192_768_2_2_1", f4f4bf16_ultra_256_192_768_2_2_1},
      {"f4f4bf16_ultra_256_192_768_2_4_1", f4f4bf16_ultra_256_192_768_2_4_1},
      {"f4f4bf16_ultra_256_192_768_4_1_1", f4f4bf16_ultra_256_192_768_4_1_1},
      {"f4f4bf16_ultra_128_128_768_1_2_1", f4f4bf16_ultra_128_128_768_1_2_1},
      {"f4f4bf16_ultra_128_128_768_1_4_1", f4f4bf16_ultra_128_128_768_1_4_1},
      {"f4f4bf16_ultra_128_128_768_2_1_1", f4f4bf16_ultra_128_128_768_2_1_1},
      {"f4f4bf16_ultra_128_128_768_2_2_1", f4f4bf16_ultra_128_128_768_2_2_1},
      {"f4f4bf16_ultra_128_128_768_2_4_1", f4f4bf16_ultra_128_128_768_2_4_1},
      {"f4f4bf16_ultra_128_128_768_4_1_1", f4f4bf16_ultra_128_128_768_4_1_1},
      {"f4f4bf16_ultra_128_128_768_4_2_1", f4f4bf16_ultra_128_128_768_4_2_1},
      {"f4f4bf16_ultra_128_192_768_1_1_1", f4f4bf16_ultra_128_192_768_1_1_1},
      {"f4f4bf16_ultra_128_192_768_1_2_1", f4f4bf16_ultra_128_192_768_1_2_1},
      {"f4f4bf16_ultra_128_192_768_1_4_1", f4f4bf16_ultra_128_192_768_1_4_1},
      {"f4f4bf16_ultra_128_192_768_2_1_1", f4f4bf16_ultra_128_192_768_2_1_1},
      {"f4f4bf16_ultra_128_192_768_2_2_1", f4f4bf16_ultra_128_192_768_2_2_1},
      {"f4f4bf16_ultra_128_192_768_2_4_1", f4f4bf16_ultra_128_192_768_2_4_1},
      {"f4f4bf16_ultra_128_192_768_4_1_1", f4f4bf16_ultra_128_192_768_4_1_1},
      {"f4f4bf16_ultra_128_192_768_4_2_1", f4f4bf16_ultra_128_192_768_4_2_1},
      {"f4f4bf16_ultra_128_256_768_1_2_1", f4f4bf16_ultra_128_256_768_1_2_1},
      {"f4f4bf16_ultra_128_256_768_1_4_1", f4f4bf16_ultra_128_256_768_1_4_1},
      {"f4f4bf16_ultra_128_256_768_2_1_1", f4f4bf16_ultra_128_256_768_2_1_1},
      {"f4f4bf16_ultra_128_256_768_2_2_1", f4f4bf16_ultra_128_256_768_2_2_1},
      {"f4f4bf16_ultra_128_256_768_2_4_1", f4f4bf16_ultra_128_256_768_2_4_1},
      {"f4f4bf16_ultra_128_256_768_4_1_1", f4f4bf16_ultra_128_256_768_4_1_1},
      {"f4f4bf16_ultra_128_256_768_4_2_1", f4f4bf16_ultra_128_256_768_4_2_1},
      {"f4f4bf16_ultra_256_128_768_4_2_1", f4f4bf16_ultra_256_128_768_4_2_1},
      {"f4f4bf16_ultra_256_192_768_4_2_1", f4f4bf16_ultra_256_192_768_4_2_1},
      {"f4f4bf16_ultra_256_256_768_4_1_1", f4f4bf16_ultra_256_256_768_4_1_1},
      {"f4f4bf16_ultra_256_256_768_4_2_1", f4f4bf16_ultra_256_256_768_4_2_1},
  };
  return kernels;
}

// Combined SM103 candidate pool: the Sm100 tiles (K=256) plus the ultra tiles
// (K=768). On B300 the tuner selects per shape from both arch paths, so it can
// fall back to a K=256 Sm100 tile where the wide K=768 ultra tile is
// inefficient (ultra is locked to K=768 by the CUTLASS builder and cannot use a
// smaller K).
const std::unordered_map<std::string, Kernel_f4f4bf16>&
get_f4f4bf16_sm103_kernels() {
  static const std::unordered_map<std::string, Kernel_f4f4bf16> kernels = []() {
    std::unordered_map<std::string, Kernel_f4f4bf16> m = get_f4f4bf16_kernels();
    for (const auto& kv : get_f4f4bf16_ultra_kernels()) {
      m.insert(kv);
    }
    return m;
  }();
  return kernels;
}
#endif // CUDA >= 13.0

Kernel_f4f4bf16 get_kernel_via_tuning(
    int M,
    int N,
    int K,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size) {
  M = nextPowerOf2OrRoundUp(M, 1024, 1024);
  N = nextPowerOf2OrRoundUp(N, 1024, 1024);
  K = nextPowerOf2OrRoundUp(K, 1024, 1024);

  const std::string shape_key =
      std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
  // On B300 tune over the combined SM103 pool. Each FP4 format gets its own
  // cache file (format in the filename, keys stay bare M_N_K) so per-format
  // winners never collide at the same shape and the offline make_heuristic
  // parser sees plain integer M_N_K keys; the format-specific name also keeps
  // ultra and SM100 kernel names from colliding in one file.
  if (get_device_sm_version(XQ.device().index()) >= 103 &&
      !ultra_path_disabled()) {
    TuningCache& ultra_cache = [&]() -> TuningCache& {
      if (global_scale.has_value()) {
        static TuningCache c("f4f4bf16_ultra_nvfp4");
        return c;
      }
      if (mxfp4_block_size == 16) {
        static TuningCache c("f4f4bf16_ultra_mxfp4_16");
        return c;
      }
      static TuningCache c("f4f4bf16_ultra_mxfp4");
      return c;
    }();
    return ultra_cache.findBestKernelMaybeAutotune(
        shape_key,
        get_f4f4bf16_sm103_kernels(),
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        global_scale,
        mxfp4_block_size);
  }
#endif

  // Off-B300: the same per-format split over the SM100 pool.
  TuningCache& cache = [&]() -> TuningCache& {
    if (global_scale.has_value()) {
      static TuningCache c("f4f4bf16_nvfp4");
      return c;
    }
    if (mxfp4_block_size == 16) {
      static TuningCache c("f4f4bf16_mxfp4_16");
      return c;
    }
    static TuningCache c("f4f4bf16_mxfp4");
    return c;
  }();
  return cache.findBestKernelMaybeAutotune(
      shape_key,
      get_f4f4bf16_kernels(),
      XQ,
      WQ,
      x_scale,
      w_scale,
      output,
      global_scale,
      mxfp4_block_size);
}

} // namespace

at::Tensor f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size) {
  TORCH_CHECK(
      mxfp4_block_size == 32 || mxfp4_block_size == 16,
      "mxfp4_block_size must be 32 or 16, got ",
      mxfp4_block_size);
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
          M,
          N,
          K,
          XQ,
          WQ,
          x_scale,
          w_scale,
          out,
          global_scale,
          mxfp4_block_size);
    }
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
    // Transparent auto-upgrade: on B300 (sm_103) route to the ultra tree unless
    // explicitly disabled (MSLK_F4F4BF16_DISABLE_ULTRA).
    if (get_device_sm_version(XQ.device().index()) >= 103 &&
        !ultra_path_disabled()) {
      return get_ultra_kernel_via_heuristics(
          M, N, K, global_scale, mxfp4_block_size);
    }
#endif
    return get_kernel_via_heuristics(M, N, K);
  }();
  return kernel(XQ, WQ, x_scale, w_scale, out, global_scale, mxfp4_block_size);
}

#else

at::Tensor f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output,
    std::optional<at::Tensor> global_scale,
    int64_t mxfp4_block_size) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}

#endif

} // namespace mslk::gemm
