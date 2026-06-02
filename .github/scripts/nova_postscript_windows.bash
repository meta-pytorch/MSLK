#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Post-script for validating MSLK python-only wheels on Windows.
#
# Installs the built wheel and verifies that the package can be imported.
# No GPU tests are run since this is a python-only build.
################################################################################

echo "[NOVA] Current working directory: $(pwd)"
cd "${MSLK_REPO}" || echo "[NOVA] Failed to cd to ${MSLK_REPO}"

export BUILD_FROM_NOVA=1

echo "[NOVA] Installing the built wheel ..."
$CONDA_RUN pip install dist/*.whl

echo "[NOVA] Verifying import ..."
$CONDA_RUN python -c "
import mslk
print(f'MSLK version:  {mslk.__version__}')
print(f'MSLK variant:  {mslk.__variant__}')
print(f'MSLK target:   {mslk.__target__}')
assert mslk.__variant__ == 'python_only', f'Expected python_only variant, got {mslk.__variant__}'
print('Import verification passed')
"

echo "[NOVA] MSLK python-only post-build validation completed"
