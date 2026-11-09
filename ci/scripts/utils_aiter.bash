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

  # aiter is optional, and comes from a server outside PyPI, so failing to
  # install or import it does not fail the job: it continues without aiter, and
  # the tests of MSLK's aiter paths skip. The warning shows on the run summary.
  __continue_without_aiter () {
    echo "::warning::$1; continuing without aiter, so tests of MSLK's aiter paths will skip."
    # Leave no half-installed aiter behind; a no-op when none was installed.
    # shellcheck disable=SC2086
    conda run ${env_prefix} python -m pip uninstall -y amd-aiter
  }

  # The pin must stay exact: these builds are versioned 0.1.1.dev1+g<sha>, which
  # sorts below older aiter builds in the same folder, so a looser requirement
  # resolves to the wrong build. (== is exact for a version with a local label.)
  echo "[INSTALL] Installing amd-aiter==${aiter_version} from ${AITER_WHEELS_URL} ..."
  # shellcheck disable=SC2086
  if ! (exec_with_retries 3 conda run --no-capture-output ${env_prefix} python -m pip install "amd-aiter==${aiter_version}" --find-links "${AITER_WHEELS_URL}"); then
    __continue_without_aiter "Could not install aiter ${aiter_version} from ${AITER_WHEELS_URL}"
    return
  fi

  # aiter pins FlyDSL exactly. If that is not ci/flydsl_version.txt, pip has
  # just replaced the FlyDSL installed earlier. That is an error in this
  # repository, not in the server, so fail now rather than in the tests.
  # shellcheck disable=SC2155
  local flydsl_version=$(tr -d '[:space:]' < "$( dirname -- "$BASH_SOURCE"; )/../flydsl_version.txt")
  # shellcheck disable=SC2086,SC2155
  local installed_flydsl=$(conda run ${env_prefix} python -c "from importlib.metadata import version; print(version('flydsl'))")
  if [ "${installed_flydsl}" != "${flydsl_version}" ]; then
    echo "[CHECK] aiter ${aiter_version} requires a different FlyDSL than ci/flydsl_version.txt (${flydsl_version}); pip installed flydsl ${installed_flydsl}."
    echo "[CHECK] Set ci/flydsl_version.txt to the FlyDSL version this aiter build requires."
    return 1
  fi

  # The first import compiles aiter's core module with hipcc (once per
  # installation), so this also checks that the environment can do that.
  # shellcheck disable=SC2086
  if ! (test_python_import_package "${env_name}" aiter); then
    __continue_without_aiter "aiter ${aiter_version} was installed but cannot be imported"
    return
  fi

  echo "[CHECK] Printing out the aiter and flydsl versions ..."
  # shellcheck disable=SC2086,SC2155
  installed_versions=$(conda run ${env_prefix} python -c "from importlib.metadata import version; print(version('amd-aiter'), version('flydsl'))")
  echo "################################################################################"
  echo "[CHECK] The installed VERSIONS of aiter and flydsl are: ${installed_versions}"
  echo "################################################################################"
  echo ""
}
