# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake Prelude
################################################################################

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(CMAKEMODULES ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(MSLK ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRDPARTY ${MSLK}/external)

include(${CMAKEMODULES}/Utilities.cmake)

set(CMAKE_VERBOSE_MAKEFILE ON)

################################################################################
# Set Build Target
################################################################################

set(BUILD_TARGET_DEFAULT  "default")
set(BUILD_TARGET_VALUES   "${BUILD_TARGET_DEFAULT}")

if(NOT DEFINED MSLK_BUILD_TARGET)
  set(MSLK_BUILD_TARGET "${BUILD_TARGET_DEFAULT}")
elseif(NOT MSLK_BUILD_TARGET IN_LIST BUILD_TARGET_VALUES)
  message(FATAL_ERROR
    "Invalid MSLK_BUILD_TARGET value: ${MSLK_BUILD_TARGET}.
    Allowed values: ${BUILD_TARGET_VALUES}")
endif()

################################################################################
# Set Build Variant
################################################################################

set(BUILD_VARIANT_CPU     "cpu")
set(BUILD_VARIANT_CUDA    "cuda")
set(BUILD_VARIANT_ROCM    "rocm")
set(BUILD_VARIANT_VALUES
  "${BUILD_VARIANT_CPU};${BUILD_VARIANT_CUDA};${BUILD_VARIANT_ROCM}")

if (DEFINED MSLK_BUILD_VARIANT)
  # If MSLK_BUILD_VARIANT is set, validate it
  if(NOT MSLK_BUILD_VARIANT IN_LIST BUILD_VARIANT_VALUES)
    message(FATAL_ERROR
      "Invalid MSLK_BUILD_VARIANT value: ${MSLK_BUILD_VARIANT}.
      Allowed values: ${BUILD_VARIANT_VALUES}")
  endif()

elseif(((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) AND
  (NOT EXISTS "/bin/nvcc"))
  message(
    "AMD GPU has been detected; will default to ROCm build"
  )
  set(MSLK_BUILD_VARIANT "${BUILD_VARIANT_ROCM}")

else()
  set(MSLK_BUILD_VARIANT "${BUILD_VARIANT_CUDA}")

endif()

################################################################################
# MSLK Build Kickstart
################################################################################

# MSLK C++ Setup - must be set AFTER MSLK_BUILD_VARIANT declaration but
# BEFORE project declaration
include(${CMAKEMODULES}/CxxCompilerSetup.cmake)

if(SKBUILD)
  BLOCK_PRINT("The project is built using scikit-build")
endif()

BLOCK_PRINT(
  "Build Settings"
  ""
  "MSLK_BUILD_TARGET    : ${MSLK_BUILD_TARGET}"
  "MSLK_BUILD_VARIANT   : ${MSLK_BUILD_VARIANT}"
  ""
  "NVCC_VERBOSE           : ${NVCC_VERBOSE}"
  "CUDNN_INCLUDE_DIR      : ${CUDNN_INCLUDE_DIR}"
  "CUDNN_LIBRARY          : ${CUDNN_LIBRARY}"
  "NVML_LIB_PATH          : ${NVML_LIB_PATH}"
  "TORCH_CUDA_ARCH_LIST   : ${TORCH_CUDA_ARCH_LIST}"
  ""
  "HIP_ROOT_DIR           : ${HIP_ROOT_DIR}"
  "HIPCC_VERBOSE          : ${HIPCC_VERBOSE}"
  "AMDGPU_TARGETS         : ${AMDGPU_TARGETS}"
  "PYTORCH_ROCM_ARCH      : ${PYTORCH_ROCM_ARCH}")

set(project_languages CXX)
if(MSLK_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  list(APPEND project_languages CUDA)
endif()

# Declare CMake project
project(
  mslk
  VERSION 0.1.0
  LANGUAGES ${project_languages})

# AVX Flags Setup - must be set AFTER project declaration
include(${CMAKEMODULES}/FindAVX.cmake)

# PyTorch Dependencies Setup
include(${CMAKEMODULES}/PyTorchSetup.cmake)

# CUDA Setup
include(${CMAKEMODULES}/CudaSetup.cmake)

# ROCm and HIPify Setup
include(${CMAKEMODULES}/RocmSetup.cmake)

# Load gpu_cpp_library()
include(${CMAKEMODULES}/GpuCppLibrary.cmake)


################################################################################
# Source Includes
################################################################################

set(mslk_include_directories
  # MSLK
  ${MSLK}/include
  # PyTorch
  ${TORCH_INCLUDE_DIRS}
  # Third-party
  ${THIRDPARTY}/cutlass/include
  ${THIRDPARTY}/cutlass/tools/util/include
  ${THIRDPARTY}/composable_kernel/include
  ${THIRDPARTY}/composable_kernel/library/include
  ${NCCL_INCLUDE_DIRS})


################################################################################
# HIP Code Generation
################################################################################

if(MSLK_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
  set(include_dirs_for_hipification
    # All directories need to be included for headers to be properly HIPified
    ${MSLK}/include
    ${MSLK}/csrc)

  # HIPify all .CU and .CUH sources under the current directory (`/mslk`)
  #
  # Note that .H sources are not automatically HIPified, so if they reference
  # CUDA-specific code, e.g. `#include <c10/cuda/CUDAStream.h>`, they will need
  # to be updated with `#ifdef USE_ROCM` guards.
  hipify(
    CUDA_SOURCE_DIR
      ${PROJECT_SOURCE_DIR}
    HEADER_INCLUDE_DIR
      ${include_dirs_for_hipification})

  BLOCK_PRINT(
    "HIPify Sources"
    " "
    "CUDA_SOURCE_DIR:"
    "${PROJECT_SOURCE_DIR}"
    " "
    "HEADER_INCLUDE_DIR:"
    "${include_dirs_for_hipification}"
  )
endif()


################################################################################
# Build Targets
################################################################################

if(MSLK_BUILD_VARIANT STREQUAL BUILD_VARIANT_CPU)
  message(FATAL_ERROR
    "Currently unsupported (target, variant) combination:
    (${MSLK_BUILD_TARGET}, ${MSLK_BUILD_VARIANT})")
endif()

if(MSLK_BUILD_TARGET STREQUAL BUILD_TARGET_DEFAULT)
  # Build MSLK
  include(MslkDefault.cmake)
endif()
