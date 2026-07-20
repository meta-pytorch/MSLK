#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# FlyDSL Setup Functions
################################################################################

install_flydsl_pip () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install FlyDSL (PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # FlyDSL is a ROCm-only backend; skip it on other build variants.
  if [ "$BUILD_VARIANT" != "rocm" ]; then
    echo "[BUILD] Skipping FlyDSL install for BUILD_VARIANT '${BUILD_VARIANT}' (rocm only)"
    return 0
  fi

  # Keep this pin in sync with the `flydsl` extra in setup.py.
  local flydsl_version="0.2.4"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[BUILD] Installing flydsl==${flydsl_version} from PIP ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} python -m pip install "flydsl==${flydsl_version}") || return 1

  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" flydsl) || return 1

  echo "[CHECK] Printing out the flydsl version ..."
  # shellcheck disable=SC2086,SC2155
  installed_flydsl_version=$(conda run ${env_prefix} python -c "import flydsl; print(flydsl.__version__)")
  echo "################################################################################"
  echo "[CHECK] The installed VERSION of flydsl is: ${installed_flydsl_version}"
  echo "################################################################################"
  echo ""
}
