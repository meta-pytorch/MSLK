#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Pre-script for building MSLK python-only wheels on Windows.
#
# Unlike the Linux pre-script, this does not invoke the full C++/CUDA build
# pipeline.  It only needs setuptools to package the Python source tree.
################################################################################

echo "[NOVA] Current working directory: $(pwd)"

# Unlike Linux, the Windows workflow does not source the env-script before
# running the pre-script, so we set all required variables here directly.
export MSLK_PYTHON_ONLY=1
export BUILD_FROM_NOVA=0

# Persist env vars for subsequent workflow steps (build, post-script) so that
# setup.py exits early in the build step and the post-script knows the variant.
if [[ -n "${GITHUB_ENV}" ]]; then
    echo "MSLK_PYTHON_ONLY=1" >> "${GITHUB_ENV}"
    echo "BUILD_FROM_NOVA=1" >> "${GITHUB_ENV}"
fi

# Force the wheel platform tag to win_amd64 so it is only installable on
# Windows and cannot accidentally become a fallback for Linux users whose
# Python/CUDA/Torch version doesn't match any native wheel.
export MSLK_PYTHON_ONLY_PLAT="win_amd64"

# Build MSLK nightly by default
if [[ -z "${CHANNEL}" ]]; then
    export CHANNEL="nightly"
fi

echo "[NOVA] Installing build dependencies ..."
$CONDA_RUN pip install build wheel setuptools 'setuptools_git_versioning>=3.0.0' numpy

echo "[NOVA] Building MSLK python-only wheel ..."
echo "[NOVA]   MSLK_PYTHON_ONLY=${MSLK_PYTHON_ONLY}"
echo "[NOVA]   CHANNEL=${CHANNEL}"
echo "[NOVA]   CU_VERSION=${CU_VERSION}"
$CONDA_RUN python -m build --wheel --no-isolation

echo "[NOVA] Enumerating the built wheels ..."
ls -lth dist/*.whl || exit 1

echo "[NOVA] Enumerating the wheel SHAs ..."
sha256sum dist/*.whl 2>/dev/null || certutil -hashfile dist/*.whl SHA256 2>/dev/null || true

echo "[NOVA] Validating the built wheel ..."
$CONDA_RUN pip install dist/*.whl
$CONDA_RUN python -c "import mslk; print(f'MSLK {mslk.__version__} variant={mslk.__variant__}')"

echo "[NOVA] MSLK python-only build completed"
