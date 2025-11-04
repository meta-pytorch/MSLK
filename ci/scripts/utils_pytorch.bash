#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_conda.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pip.bash"

################################################################################
# PyTorch Setup Functions
################################################################################

__verify_pytorch_gpu_integration () {
  echo "[CHECK] Verifying PyTorch device properties ..."
  # shellcheck disable=SC2086,SC2155
  local torch_cuda_available=$(conda run ${env_prefix} python -c "import torch; print(torch.cuda.is_available())")
  # shellcheck disable=SC2086,SC2155
  local torch_version_cuda=$(conda run ${env_prefix} python -c "import torch; print(torch.version.cuda)")
  # shellcheck disable=SC2086,SC2155
  local torch_version_hip=$(conda run ${env_prefix} python -c "import torch; print(torch.version.hip)")
  # shellcheck disable=SC2086,SC2155
  local torch_device_compatibility=$(conda run ${env_prefix} python -c "import torch; print(torch.cuda.get_device_capability())")
  # shellcheck disable=SC2086,SC2155
  local torch_device_name=$(conda run ${env_prefix} python -c "import torch; print(torch.cuda.get_device_name(torch.cuda.current_device()))")
  # shellcheck disable=SC2086,SC2155
  local torch_device_count=$(conda run ${env_prefix} python -c "import torch; print(torch.cuda.device_count())")
  # shellcheck disable=SC2086,SC2155
  local torch_device_properties=$(conda run ${env_prefix} python -c "import torch; print(torch.cuda.get_device_properties(0))")

  echo ""
  echo "################################################################################"
  echo "[CHECK] torch.cuda.is_available(): ${torch_cuda_available}"
  echo "[CHECK] torch.cuda.get_device_capability(): ${torch_device_compatibility}"
  echo "[CHECK] torch.cuda.get_device_name(): ${torch_device_name}"
  echo "[CHECK] torch.version.cuda: ${torch_version_cuda}"
  echo "[CHECK] torch.version.hip: ${torch_version_hip}"
  echo "[CHECK] torch.cuda.device_count(): ${torch_device_count}"
  echo "[CHECK] torch.cuda.get_device_properties(0): ${torch_device_properties}"
  echo "################################################################################"
  echo ""
}

install_pytorch_pip () {
  local env_name="$1"
  local pytorch_channel_version="$2"
  local pytorch_variant_type_version="$3"
  if [ "$pytorch_variant_type_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTORCH_CHANNEL[/VERSION] PYTORCH_VARIANT_TYPE[/VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env test/2.1.0 cpu     # Install the CPU variant for a specific version"
    echo "    ${FUNCNAME[0]} build_env release cpu        # Install the CPU variant, latest release version"
    echo "    ${FUNCNAME[0]} build_env test cuda/12.4.0   # Install the CUDA 12.4 variant, latest test version"
    echo "    ${FUNCNAME[0]} build_env nightly rocm/6.2   # Install the ROCM 6.2 variant, latest nightly version"
    return 1
  else
    echo "################################################################################"
    echo "# Install PyTorch (PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # shellcheck disable=SC2155
  env_prefix=$(env_name_or_prefix "${env_name}")

  # Install the main dependencies
  #
  # NOTE: Since CUDA 12.6, the conda channels information after CUDA installation
  # gets messed up, and adding the --override-channels flag is needed to get
  # packages to be installed correctly.  See:
  #   https://github.com/conda/conda/issues/14063#issuecomment-2244508044
  #
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge --override-channels -y \
    numpy) || return 1

  # Install the torch package from PyTorch PIP (not PyPI)
  install_from_pytorch_pip "${env_name}" torch "${pytorch_channel_version}" "${pytorch_variant_type_version}" || return 1

  # Check that PyTorch is importable
  (test_python_import_package "${env_name}" torch.distributed) || return 1

  # Print out the actual installed PyTorch version
  # shellcheck disable=SC2086,SC2155
  local installed_pytorch_version=$(conda run ${env_prefix} python -c "import torch; print(torch.__version__)")
  echo "[CHECK] NOTE: The installed version is: ${installed_pytorch_version}"

  # Run check for GPU visibility from PyTorch
  __verify_pytorch_gpu_integration

  echo "[CHECK] NOTE: Checking _GLIBCXX_USE_CXX11_ABI ..."
  # shellcheck disable=SC2086,SC2155
  conda run ${env_prefix} python -c 'import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI); print(torch.compiled_with_cxx11_abi())'

  if [[ "$pytorch_variant_type_version" == cuda* ]]; then
    # Ensure that the PyTorch-CUDA headers are properly installed
    (test_filepath "${env_name}" cuda_cmake_macros.h) || return 1
  fi

  echo "[INSTALL] Successfully installed PyTorch through PyTorch PIP"
}


################################################################################
# PyTorch Diagnose Functions
################################################################################

collect_pytorch_env_info () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env         # Collect PyTorch environment information from Conda environment build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Collect PyTorch Environment Information (for Reporting Issues)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # This is the script required for collecting info and reporting to https://github.com/pytorch/pytorch/issues/new
  echo "[INFO] Downloading the PyTorch environment info collection script ..."
  print_exec wget -q "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py"

  echo "[INFO] Collecting PyTorch environment info (will be needed for reporting issues to PyTorch) ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} python collect_env.py) || return 1
}
