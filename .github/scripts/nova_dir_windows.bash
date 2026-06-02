#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Windows builds are python-only — no C++/CUDA compilation is performed.
# The native kernels are not needed because the CUDA-compiled .so libraries
# are Linux-only; Windows users get the Python layer (triton kernels, python
# utilities, etc.) while the heavy lifting happens on Linux GPU servers.
################################################################################
export MSLK_PYTHON_ONLY=1
export BUILD_FROM_NOVA=1

## Set MSLK_REPO path for other scripts to use.
## Windows Nova runners do not use Docker, so the checkout path differs from
## the Linux /__w/... convention.
export MSLK_REPO="${GITHUB_WORKSPACE}/${REPOSITORY}"

## Overwrite existing ENV VAR in Nova
if [[ "$CONDA_ENV" != "" ]]; then export CONDA_RUN="conda run --no-capture-output -p ${CONDA_ENV}" && echo "$CONDA_RUN"; fi

if [[ "$CU_VERSION" == "cu"* ]]; then
    echo "[NOVA] Windows python-only build for CUDA variant: ${CU_VERSION}"
fi
