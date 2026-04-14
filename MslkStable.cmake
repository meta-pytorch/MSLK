# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# MSLK Stable ABI Target
#
# This file builds mslk_stable.so — a shared library compiled with the PyTorch
# stable ABI (TORCH_TARGET_VERSION).  Kernels are migrated here one-by-one from
# MslkDefault.cmake; the TORCH_TARGET_VERSION flag enforces at compile time that
# only stable ABI headers/APIs are used.
################################################################################

################################################################################
# Target Sources
################################################################################

# MOE ops (first migrated group)
glob_files_nohip(mslk_stable_cpp_source_files_cpu
  csrc/moe/*.cpp)

glob_files_nohip(mslk_stable_cpp_source_files_gpu
  csrc/moe/*.cu)

# CUDA-specific sources (none yet for stable target)
set(mslk_stable_cpp_source_files_cuda)

################################################################################
# Build Shared Library
################################################################################

list(APPEND _all_stable_sources
  ${mslk_stable_cpp_source_files_cpu}
  ${mslk_stable_cpp_source_files_gpu}
  ${mslk_stable_cpp_source_files_cuda})

if(_all_stable_sources)
  gpu_cpp_library(
    PREFIX
      mslk_stable
    TYPE
      SHARED
    INCLUDE_DIRS
      ${mslk_include_directories}
    CPU_SRCS
      ${mslk_stable_cpp_source_files_cpu}
    GPU_SRCS
      ${mslk_stable_cpp_source_files_gpu}
    CUDA_SPECIFIC_SRCS
      ${mslk_stable_cpp_source_files_cuda}
    CC_FLAGS
      -DUSE_CUDA
      -DTORCH_TARGET_VERSION=0x020b000000000000
    NVCC_FLAGS
      -DUSE_CUDA
      -DTORCH_TARGET_VERSION=0x020b000000000000
  )

  ############################################################################
  # Install Shared Library
  ############################################################################

  add_to_package(
    DESTINATION mslk
    TARGETS mslk_stable)
endif()
