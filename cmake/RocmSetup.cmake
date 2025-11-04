# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# ROCm and HIPify Setup
################################################################################

if(MSLK_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
  # Load CMake modules
  list(APPEND CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/cmake"
    "${THIRDPARTY}/hipify_torch/cmake")
  include(Hip)
  include(Hipify)

  # Configure compiler for HIP
  list(APPEND HIP_HCC_FLAGS
    " \"-Wno-#pragma-messages\" "
    " \"-Wno-#warnings\" "
    -fclang-abi-compat=17
    -Wno-cuda-compat
    -Wno-deprecated-declarations
    -Wno-format
    -Wno-ignored-attributes
    -Wno-unused-result
    # Below flags are required for strong CK performance
    # on certain kernel instances
    # Reduce register spillage on certain kernels
    -mllvm -amdgpu-coerce-illegal-types=1
    -mllvm -enable-post-misched=0
    -mllvm -greedy-reverse-local-assignment=1
    -fhip-new-launch-api)

  BLOCK_PRINT(
    "HIP found: ${HIP_FOUND}"
    "HIPCC compiler flags:"
    ""
    "${HIP_HCC_FLAGS}"
  )
endif()
