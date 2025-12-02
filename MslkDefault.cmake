# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

glob_files_nohip(mslk_cpp_source_files_cpu
  csrc/attention/cuda/cutlass_blackwell_fmha/blackwell_*.cpp
  csrc/attention/cuda/gqa_attn_splitk/*.cpp
  csrc/coalesce/*.cpp
  csrc/comm/*.cpp
  csrc/kv_cache/*.cpp
  csrc/moe/*.cpp
  csrc/quantize/*.cpp)

glob_files_nohip(mslk_cpp_source_files_gpu
  csrc/attention/cuda/cutlass_blackwell_fmha/blackwell_*.cu
  csrc/attention/cuda/gqa_attn_splitk/*.cu
  csrc/coalesce/*.cu
  csrc/comm/*.cu
  csrc/kv_cache/*.cu
  csrc/moe/*.cu
  csrc/quantize/*.cu)

# Include FB-internal sources into the build
if(BUILD_FB_CODE
  AND (MSLK_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0))
  BLOCK_PRINT("[FBPKG] INCLUDING FB-internal sources ...")

  glob_files_nohip(fb_only_sources_cpu
      fb/csrc/*/*.cpp)

  glob_files_nohip(fb_only_sources_gpu
      fb/csrc/*/*.cu)

  if(MSLK_FBPKG_BUILD)
    BLOCK_PRINT("[FBPKG] MSLK_FBPKG_BUILD is set.")

  else()
    BLOCK_PRINT(
      "[FBPKG] MSLK_FBPKG_BUILD is NOT set,"
      "certain FB-internal sources will be excluded from the build.")

    # NOTE: Some FB-internal code explicitly require an FB-internal
    # environment to build, such as code that depends on NCCLX
    list(FILTER fb_only_sources_cpu
      EXCLUDE REGEX "fb/csrc/tensor_parallel(/.*)*\\.cpp$")

    list(FILTER fb_only_sources_gpu
      EXCLUDE REGEX "fb/csrc/tensor_parallel(/.*)*\\.cu$")
  endif()

  list(APPEND mslk_cpp_source_files_cpu ${fb_only_sources_cpu})
  list(APPEND mslk_cpp_source_files_gpu ${fb_only_sources_gpu})

else()
  BLOCK_PRINT("[FBPKG] Will NOT be including FB-internal sources ...")

endif()

# CUDA-specific sources
file(GLOB_RECURSE mslk_cpp_source_files_cuda
  csrc/gemm/cutlass/*.cu
  csrc/gemm/cutlass/**/*.cu
  csrc/quantize/cutlass/*.cu
  csrc/quantize/cutlass/**/*.cu)

# HIP-specific sources
file(GLOB_RECURSE mslk_cpp_source_files_hip
  csrc/gemm/ck/*.hip
  csrc/gemm/ck/**/*.hip
  csrc/quantize/ck/*.hip
  csrc/quantize/ck/**/*.hip)

################################################################################
# Build Shared Library
################################################################################

gpu_cpp_library(
  PREFIX
    mslk
  TYPE
    SHARED
  INCLUDE_DIRS
    ${mslk_include_directories}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/cuda/cutlass_blackwell_fmha
  CPU_SRCS
    ${mslk_cpp_source_files_cpu}
  GPU_SRCS
    ${mslk_cpp_source_files_gpu}
  CUDA_SPECIFIC_SRCS
    ${mslk_cpp_source_files_cuda}
  HIP_SPECIFIC_SRCS
    ${mslk_cpp_source_files_hip}
)

################################################################################
# Install Shared Library
################################################################################

add_to_package(
  DESTINATION mslk
  TARGETS mslk)
