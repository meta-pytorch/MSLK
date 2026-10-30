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
  csrc/conv/*.cpp
  csrc/gemm/*.cpp
  csrc/moe/*.cpp
  csrc/attention/ck/fmha/*.cpp)

glob_files_nohip(mslk_cpp_source_files_gpu
  csrc/moe/*.cu)

# Include FB-internal sources into the build
if(BUILD_FB_CODE
  AND (MSLK_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0))
  BLOCK_PRINT("[FBPKG] INCLUDING FB-internal sources ...")

  glob_files_nohip(fb_only_sources_cpu
      fb/csrc/*/*.cpp)

  glob_files_nohip(fb_only_sources_gpu
      fb/csrc/*/*.cu)

  # natten has its own build infrastructure and is not part of the conda build.
  list(FILTER fb_only_sources_cpu EXCLUDE REGEX "fb/csrc/natten/")
  list(FILTER fb_only_sources_gpu EXCLUDE REGEX "fb/csrc/natten/")

  if(MSLK_FBPKG_BUILD)
    BLOCK_PRINT("[FBPKG] MSLK_FBPKG_BUILD is set.")

  else()
    BLOCK_PRINT(
      "[FBPKG] MSLK_FBPKG_BUILD is NOT set,"
      "certain FB-internal sources will be excluded from the build.")

    # To allow a directory, add it to the regex group below, e.g.:
    #   set(FB_ONLY_ALLOW_REGEX "fb/csrc/(my_kernel_1|my_kernel_2)")
    set(FB_ONLY_ALLOW_REGEX "fb/csrc/(internal)")

    list(FILTER fb_only_sources_cpu
      INCLUDE REGEX "${FB_ONLY_ALLOW_REGEX}")

    list(FILTER fb_only_sources_gpu
      INCLUDE REGEX "${FB_ONLY_ALLOW_REGEX}")
  endif()

  list(APPEND mslk_cpp_source_files_cpu ${fb_only_sources_cpu})
  list(APPEND mslk_cpp_source_files_gpu ${fb_only_sources_gpu})

else()
  BLOCK_PRINT("[FBPKG] Will NOT be including FB-internal sources ...")

endif()

# CUDA-specific sources
file(GLOB_RECURSE mslk_cpp_source_files_cuda
  csrc/attention/cuda/cutlass_blackwell_fmha/blackwell_*.cu
  csrc/conv/cutlass/*.cu
  csrc/conv/cutlass/**/*.cu
  csrc/gemm/cublas/*.cu
  csrc/gemm/cublas/**/*.cu
  csrc/gemm/cutlass/*.cu
  csrc/gemm/cutlass/**/*.cu)

# HIP-specific sources
file(GLOB_RECURSE mslk_cpp_source_files_hip
  csrc/gemm/ck/*.hip
  csrc/gemm/ck/**/*.hip)

# HIP FMHA sources - built separately, for a restricted set of architectures,
# to reduce build time and binary size
file(GLOB_RECURSE mslk_hip_fmha_sources
  csrc/attention/ck/fmha/hip_fmha/*.cpp)

################################################################################
# Build FMHA Static Library (CK FMHA architectures only, ROCm 7.x+)
################################################################################

# CK FMHA is built only for these architectures. It is built for the requested
# architectures (PYTORCH_ROCM_ARCH) that appear here; the others get no FMHA
# device code. If none of the requested architectures appear here, the library
# is skipped rather than built for an architecture that was not requested.
set(mslk_hip_fmha_supported_archs gfx942)

set(mslk_hip_fmha_archs)
set(mslk_hip_fmha_excluded_archs)
if(MSLK_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
  # PYTORCH_ROCM_ARCH may be separated by semicolons, spaces or commas
  string(REGEX REPLACE "[ ,]+" ";" _requested_archs "${PYTORCH_ROCM_ARCH}")
  foreach(_arch IN LISTS _requested_archs)
    # Match on the processor name, ignoring target features such as :xnack-
    string(REGEX REPLACE ":.*" "" _arch_name "${_arch}")
    if(_arch_name IN_LIST mslk_hip_fmha_supported_archs)
      list(APPEND mslk_hip_fmha_archs ${_arch})
    else()
      list(APPEND mslk_hip_fmha_excluded_archs ${_arch})
    endif()
  endforeach()
endif()

set(mslk_hip_fmha_built FALSE)
if(NOT DEFINED ENV{MSLK_BUILD_HIP_FMHA} OR NOT "$ENV{MSLK_BUILD_HIP_FMHA}" STREQUAL "0")
  if(MSLK_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM AND NOT mslk_hip_fmha_archs)
    message("Skipping HIP FMHA build: none of the requested archs "
      "(${PYTORCH_ROCM_ARCH}) is supported by CK FMHA "
      "(${mslk_hip_fmha_supported_archs})")
  elseif(MSLK_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM
      AND mslk_hip_fmha_sources
      AND HIP_VERSION VERSION_GREATER_EQUAL "7.0")
    # Replace the global arch list with the CK FMHA subset of it
    set(HIP_HCC_FLAGS_FMHA ${HIP_HCC_FLAGS})
    list(FILTER HIP_HCC_FLAGS_FMHA EXCLUDE REGEX "--offload-arch=")
    foreach(_arch IN LISTS mslk_hip_fmha_archs)
      list(APPEND HIP_HCC_FLAGS_FMHA --offload-arch=${_arch})
    endforeach()
    list(APPEND HIP_HCC_FLAGS_FMHA
      -fPIC
      -DFMHA_LIMIT_MAX_HEADDIM_TO_256=1)

    message("Building FMHA for arch: ${mslk_hip_fmha_archs} (HIP_VERSION=${HIP_VERSION})")
    if(mslk_hip_fmha_excluded_archs)
      message("Not building FMHA for arch: ${mslk_hip_fmha_excluded_archs} "
        "(not supported by CK FMHA)")
    endif()

    # Exclude any pre-existing _hip.cpp files the glob may have picked up
    # from a prior hipify run, to avoid duplicate compilation targets.
    list(FILTER mslk_hip_fmha_sources EXCLUDE REGEX ".*_hip\\.cpp$")

    # Fetch HIPified versions of the sources (e.g. files that include
    # ATen/cuda/*.cuh headers need the _hip.cpp counterparts generated by
    # the hipify step; instances/ files are already pure HIP and are
    # returned unchanged by get_hipified_list).
    get_hipified_list("${mslk_hip_fmha_sources}" mslk_hip_fmha_sources)

    # Mark sources for HIPCC compilation
    set_source_files_properties(${mslk_hip_fmha_sources} PROPERTIES
      HIP_SOURCE_PROPERTY_FORMAT 1)

    hip_include_directories("${mslk_include_directories}")

    hip_add_library(mslk_hip_fmha STATIC
      ${mslk_hip_fmha_sources}
      ${MSLK_HIP_HCC_LIBRARIES}
      HIPCC_OPTIONS ${HIP_HCC_FLAGS_FMHA})

    target_include_directories(mslk_hip_fmha PUBLIC
      ${MSLK_HIP_INCLUDE}
      ${ROCRAND_INCLUDE}
      ${ROCM_SMI_INCLUDE}
      ${mslk_include_directories})

    target_include_directories(mslk_hip_fmha PRIVATE
      ${TORCH_INCLUDE_DIRS}
      ${NCCL_INCLUDE_DIRS})

    set_target_properties(mslk_hip_fmha PROPERTIES
      POSITION_INDEPENDENT_CODE ON)

    set(mslk_hip_fmha_built TRUE)
  endif()
else()
  message("Skipping HIP FMHA build (MSLK_BUILD_HIP_FMHA=0)")
endif()

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

# Link FMHA static library with --whole-archive to preserve TORCH_LIBRARY_IMPL
# static initializers that would otherwise be dropped by the linker.
if(mslk_hip_fmha_built)
  target_link_libraries(mslk PRIVATE
    -Wl,--whole-archive
    mslk_hip_fmha
    -Wl,--no-whole-archive)
  target_compile_definitions(mslk PRIVATE MSLK_HIP_FMHA_ENABLED)
endif()

################################################################################
# Install Shared Library and Python Files
################################################################################

add_to_package(
  DESTINATION mslk
  TARGETS mslk)

install(
  DIRECTORY bench
  DESTINATION mslk)

if(BUILD_FB_CODE)
  install(
    DIRECTORY fb/mslk/attention/flash_attn
    DESTINATION mslk/fb/mslk/attention)
  install(
    DIRECTORY fb/external/quack
    DESTINATION mslk/fb/external)
endif()
