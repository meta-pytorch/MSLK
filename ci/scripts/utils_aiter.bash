#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# aiter Setup Functions
################################################################################

# aiter publishes a wheel per commit of its main branch here, outside PyPI. The
# "noprebuild" builds MSLK pins carry no compiled kernels: aiter compiles them
# on first use, so they install on any CPython 3 and ROCm 7.x.
AITER_WHEELS_URL="https://rocm.frameworks-nightlies.amd.com/whl-staging/gfx942-gfx950/"

install_aiter_pip () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install aiter (PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # aiter is an optional ROCm-only dependency; skip it on other build variants.
  if [ "$BUILD_VARIANT" != "rocm" ]; then
    echo "[INSTALL] Skipping aiter install for BUILD_VARIANT '${BUILD_VARIANT}' (rocm only)"
    return 0
  fi

  # Single source of truth for the aiter pin, shared with setup.py and the
  # runtime.
  # shellcheck disable=SC2155
  local aiter_version=$(tr -d '[:space:]' < "$( dirname -- "$BASH_SOURCE"; )/../../mslk/aiter/aiter_version.txt")

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # The pin must stay exact: these builds are versioned 0.1.1.dev1+g<sha>, which
  # sorts below older aiter builds in the same folder, so a looser requirement
  # resolves to the wrong build. aiter also pins FlyDSL exactly; if that differs
  # from ci/flydsl_version.txt, pip replaces the FlyDSL installed earlier, and
  # the FlyDSL version test fails.
  echo "[INSTALL] Installing amd-aiter==${aiter_version} from ${AITER_WHEELS_URL} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} python -m pip install "amd-aiter==${aiter_version}" --find-links "${AITER_WHEELS_URL}") || return 1

  # The first import compiles aiter's core module with hipcc (once per
  # installation), so this also checks that the environment can do that.
  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" aiter) || return 1

  echo "[CHECK] Printing out the aiter and flydsl versions ..."
  # shellcheck disable=SC2086,SC2155
  installed_versions=$(conda run ${env_prefix} python -c "from importlib.metadata import version; print(version('amd-aiter'), version('flydsl'))")
  echo "################################################################################"
  echo "[CHECK] The installed VERSIONS of aiter and flydsl are: ${installed_versions}"
  echo "################################################################################"
  echo ""
}
