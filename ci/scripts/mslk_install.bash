#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pip.bash"

################################################################################
# MSLK Install Functions
################################################################################

__install_print_dependencies_info () {
  # shellcheck disable=SC2086,SC2155
  local installed_pytorch_version=$(conda run ${env_prefix} python -c "import torch; print(torch.__version__)")
  # shellcheck disable=SC2086,SC2155
  local installed_cuda_version=$(conda run ${env_prefix} python -c "import torch; print(torch.version.cuda)")

  echo ""
  echo "################################################################################"
  echo "[CHECK] !!!!    INFO    !!!!"
  echo "[CHECK] The installed version of PyTorch is: ${installed_pytorch_version}"
  echo "[CHECK] CUDA version reported by PyTorch is: ${installed_cuda_version}"
  echo "[CHECK]"
  echo "[CHECK] NOTE: If the PyTorch package channel is different from the MSLK"
  echo "[CHECK]       package channel; the package may be broken at runtime!!!"
  echo "################################################################################"
  echo ""
}

__install_fetch_version_and_variant_info () {
  echo "[INSTALL] Checking imports and symbols ..."
  (test_python_import_package "${env_name}" mslk) || return 1
  (test_python_import_symbol "${env_name}" mslk __version__) || return 1
  (test_python_import_symbol "${env_name}" mslk __variant__) || return 1

  echo "[CHECK] Printing out the MSLK version ..."
  # shellcheck disable=SC2086,SC2155
  installed_mslk_target=$(conda run ${env_prefix} python -c "import mslk; print(mslk.__target__)")
  # shellcheck disable=SC2086,SC2155
  installed_mslk_variant=$(conda run ${env_prefix} python -c "import mslk; print(mslk.__variant__)")
  # shellcheck disable=SC2086,SC2155
  installed_mslk_version=$(conda run ${env_prefix} python -c "import mslk; print(mslk.__version__)")

  echo ""
  echo "################################################################################"
  echo "[CHECK] The installed MSLK TARGET is: ${installed_mslk_target}"
  echo "[CHECK] The installed MSLK VARIANT is: ${installed_mslk_variant}"
  echo "[CHECK] The installed MSLK VERSION is: ${installed_mslk_version}"
  echo "################################################################################"
  echo ""
}

__install_check_subpackages () {
  # shellcheck disable=SC2086,SC2155
  local mslk_packages=$(conda run ${env_prefix} python -c "import mslk; print(dir(mslk))")

  echo "################################################################################"
  echo "[CHECK] MSLK Packages"
  echo "[CHECK] mslk: ${mslk_packages}"
  echo "################################################################################"
  echo ""

  echo "[INSTALL] Check for installation of Python sources ..."
  local subpackages=(
    mslk.attention
    mslk.coalesce
    mslk.comm
    mslk.gemm
    mslk.kv_cache
    mslk.moe
    mslk.quantize
    mslk.testing
    mslk.utils
  )

  if [ "$installed_mslk_target" == "default" ]; then
    subpackages+=()
  fi

  for package in "${subpackages[@]}"; do
    (test_python_import_package "${env_name}" "${package}") || return 1
  done
}

__install_check_operator_registrations () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  local test_operators=()
  local base_import="mslk"
  echo "[INSTALL] Check operator registrations ..."

  if [ "$installed_mslk_target" == "default" ]; then
    test_operators+=(
      torch.ops.mslk.coalesce_batches
      torch.ops.mslk.rope_qkv_decoding
    )
  fi

  for operator in "${test_operators[@]}"; do
    # shellcheck disable=SC2086
    if conda run ${env_prefix} python -c "import torch; import ${base_import}; print($operator)"; then
      echo "[CHECK] MSLK operator appears to be correctly registered: $operator"
    else
      echo "################################################################################"
      echo "[CHECK] MSLK operator hasn't been registered on torch.ops.load():"
      echo "[CHECK]"
      echo "[CHECK] $operator"
      echo "[CHECK]"
      echo "[CHECK] Please check that all operators defined with m.def() have an appropriate"
      echo "[CHECK] m.impl() defined, AND that the definition sources are included in the "
      echo "[CHECK] CMake build configuration!"
      echo "################################################################################"
      echo ""
      return 1
    fi
  done
}

__mslk_post_install_checks () {
  local env_name="$1"
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Move to another directory, to avoid Python package import confusion, since
  # there exists a mslk/ subdirectory in the MSLK repo
  mkdir -p _tmp_dir_mslk || return 1
  pushd _tmp_dir_mslk || return 1

  # Check that the MSLK package is installed correctly
  local installed_dir=$(conda run --no-capture-output ${env_prefix} python -c 'import mslk; import os; print(os.path.dirname(mslk.__file__))')
  print_exec ls -la "${installed_dir}"

  # Print PyTorch and CUDA versions for sanity check
  __install_print_dependencies_info         || return 1

  # Fetch the version and variant info from the package
  __install_fetch_version_and_variant_info  || return 1

  # Check MSLK subpackages are installed correctly
  __install_check_subpackages               || return 1

  # Check operator registrations are working
  __install_check_operator_registrations    || return 1

  popd || return 1
}

uninstall_mslk_wheel () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env     # Uninstall the MSLK wheel (if installed)"
    return 1
  else
    echo "################################################################################"
    echo "# Uninstall MSLK Wheel (If Installed)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # shellcheck disable=SC2155,SC2086
  local packages=$(conda run ${env_prefix} python -m pip list --format=freeze | grep "mslk-" | cut -d"=" -f1)

  if [ -n "$packages" ]; then
    echo "[UNINSTALL] Uninstalling the following packages: $packages"
    # shellcheck disable=SC2086
    print_exec conda run ${env_prefix} python -m pip uninstall -y $packages || return 1
  else
    echo "[UNINSTALL] No matching packages found."
  fi
}

install_mslk_wheel () {
  local env_name="$1"
  local wheel_path="$2"
  if [ "$wheel_path" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME WHEEL_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env mslk.whl     # Install the package (wheel)"
    return 1
  else
    echo "################################################################################"
    echo "# Install MSLK from Wheel"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Printing out MSLK wheel SHA: ${wheel_path}"
  print_exec sha1sum "${wheel_path}"
  print_exec sha256sum "${wheel_path}"
  print_exec md5sum "${wheel_path}"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing MSLK wheel: ${wheel_path} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} python -m pip install "${wheel_path}") || return 1

  __mslk_post_install_checks "${env_name}" || return 1

  echo "[INSTALL] MSLK installation through wheel completed ..."
}

install_mslk_pip () {
  local env_name="$1"
  local mslk_channel_version="$2"
  local mslk_variant_type_version="$3"
  if [ "$mslk_variant_type_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME MSLK_CHANNEL[/VERSION] MSLK_VARIANT_TYPE[/VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 0.8.0 cpu                  # Install the CPU variant, specific version from release channel"
    echo "    ${FUNCNAME[0]} build_env release cuda/12.6.3        # Install the CUDA 12.3 variant, latest version from release channel"
    echo "    ${FUNCNAME[0]} build_env test/0.8.0 cuda/12.6.3     # Install the CUDA 12.3 variant, specific version from test channel"
    echo "    ${FUNCNAME[0]} build_env nightly rocm/6.2           # Install the ROCM 6.2 variant, latest version from nightly channel"
    return 1
  else
    echo "################################################################################"
    echo "# Install MSLK Package from PIP"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # Install the package from PyTorch PIP (not PyPI)
  # The package's canonical name is 'mslk-gpu' (hyphen, not underscore)
  install_from_pytorch_pip "${env_name}" mslk "${mslk_channel_version}" "${mslk_variant_type_version}" || return 1

  # Run post-installation checks
  __mslk_post_install_checks "${env_name}" || return 1

  echo "[INSTALL] Successfully installed MSLK through PyTorch PIP"
}
