#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
. "$( dirname -- "$BASH_SOURCE"; )/utils_build.bash"

################################################################################
# MSLK Build Auxiliary Functions
################################################################################

prepare_mslk_build () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Prepare MSLK Build"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  if [[ "${GITHUB_WORKSPACE}" ]]; then
    # https://github.com/actions/checkout/issues/841
    git config --global --add safe.directory "${GITHUB_WORKSPACE}"
  fi

  # Run git submodule checkout only if building using OSS tooling
  if [[ -z "${BUILD_FB_CODE}" || "${BUILD_FB_CODE}" == "0" ]]; then
    echo "[BUILD] Running git submodules update ..."
    (exec_with_retries 3 git submodule sync) || return 1
    (exec_with_retries 3 git submodule update --init --recursive) || return 1
    echo "[BUILD] Successfully ran git submodules update"
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[BUILD] Installing other build dependencies ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} python -m pip install -r requirements.txt) || return 1

  (install_triton_pip "${env_name}") || return 1

  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" numpy) || return 1
  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" skbuild) || return 1

  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)

  echo "[BUILD] Finding library paths ..."
  # shellcheck disable=SC2061,SC2035
  (find "${conda_prefix}" -name *omp.so* 2> /dev/null || true)
  # shellcheck disable=SC2061,SC2035
  (find "${conda_prefix}" -name *tbb.so* 2> /dev/null || true)
  # shellcheck disable=SC2061,SC2035
  (find "${conda_prefix}" -name *bolt.so* 2> /dev/null || true)
}

__configure_mslk_build_clang () {
  echo "[BUILD] Clang is available; configuring for Clang-based build ..."
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
  # shellcheck disable=SC2206
  build_args+=(
    --cxxprefix=${conda_prefix}
  )
}

__configure_mslk_build_nvcc () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[BUILD] Looking up CUDA version ..."
  # shellcheck disable=SC2155,SC2086
  local cxx_path=$(conda run ${env_prefix} which c++)
  # shellcheck disable=SC2155,SC2086
  local cuda_version=$(conda run ${env_prefix} nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
  # shellcheck disable=SC2206
  local cuda_version_arr=(${cuda_version//./ })

  # Only NVCC 12+ supports C++20
  if [[ ${cuda_version_arr[0]} -lt 12 ]]; then
    local cppstd_ver=17
  else
    local cppstd_ver=20
  fi
  echo "[BUILD] Setting C++ version to ${cppstd_ver} ..."

  # Certain warnings are suppressed to avoid overly verbose output coming from
  # building CUTLASS
  local nvcc_prepend_flags=(
    "-std=c++${cppstd_ver}"
    -Xcompiler "-std=c++${cppstd_ver}"
    -ccbin "${cxx_path}"
    -allow-unsupported-compiler
    # warn: variable "nUpdates" was declared but never referenced
    -diag-suppress 177
    # warn: argument is incompatible with corresponding format string conversion
    -diag-suppress 181
    # warn: the implicit by-copy capture of "this" is deprecated
    -diag-suppress 2908
    # warn: __device__ annotation is ignored on a function that is explicitly defaulted on its first declaration
    -diag-suppress 20012
  )

  # NOTE: This check covers both Nova and non-Nova builds, as we set
  # BUILD_CUDA_VERSION to be CU_VERSION in the Nova build case
  if ! [[ "$BUILD_CUDA_VERSION" =~ ^12.6.*$ ]] && [[ "$BUILD_CUDA_VERSION" != "cu126" ]]; then
    # NOTE: This flag is only supported in NVCC 12.8+
    nvcc_prepend_flags+=(
      -diag-suppress 20280
      -diag-suppress 20281
    )
  fi

  if print_exec "conda run ${env_prefix} c++ --version | grep -i clang"; then
    echo "[BUILD] Host compiler is clang; setting stdlib to libstdc++..."
    # NOTE: The `-stdlib=libstdc++` flag doesn't exist for GCC
    nvcc_prepend_flags+=(
      -Xcompiler "-stdlib=libstdc++"
    )
  fi

  # Explicitly set whatever $CONDA_PREFIX/bin/c++ points to as the the host
  # compiler, but set GNU libstdc++ (as opposed to Clang libc++) as the standard
  # library
  #
  # NOTE: There appears to be no ROCm equivalent for NVCC_PREPEND_FLAGS:
  #   https://github.com/ROCm/HIP/issues/931
  #
  # shellcheck disable=SC2086,SC2145,SC2068
  print_exec conda env config vars set ${env_prefix} NVCC_PREPEND_FLAGS=\"${nvcc_prepend_flags[*]}\"
  # shellcheck disable=SC2086
  print_exec conda run ${env_prefix} printenv NVCC_PREPEND_FLAGS

  echo "[BUILD] Setting CUDA build args ..."
  # shellcheck disable=SC2206
  build_args+=(
    # Override CMake configuration
    -DCMAKE_CXX_STANDARD="${cppstd_ver}"
  )
}

__configure_mslk_cuda_home () {
  # NOTE: This only matches for non-Nova builds, as CUDA versions in Nova builds
  # are formatted as `cu12x“
  if  [[ "$BUILD_CUDA_VERSION" =~ ^11.*$ ]] ||
      [[ "$BUILD_CUDA_VERSION" =~ ^12.1.*$ ]] ||
      [[ "$BUILD_CUDA_VERSION" =~ ^12.4.*$ ]]; then
    echo "[BUILD] No need to set CUDA_TOOLKIT_ROOT_DIR and CUDAToolkit_ROOT on older CUDA installations ..."

  else
    echo "[BUILD] Setting CUDA_TOOLKIT_ROOT_DIR and CUDAToolkit_ROOT ..."
    # shellcheck disable=SC2155,SC2086
    local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
    local new_cuda_home="${conda_prefix}/targets/${MACHINE_NAME_LC}-linux"

    # shellcheck disable=SC2206
    build_args+=(
      # NOTE: The legacy find_package(CUDA) uses CUDA_TOOLKIT_ROOT_DIR
      # while the newer and recomended find_package(CUDAToolkit)
      # uses CUDAToolkit_ROOT.
      #
      # https://github.com/conda-forge/cuda-feedstock/issues/59
      # https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA.cmake#L40
      -DCUDA_TOOLKIT_ROOT_DIR="${new_cuda_home}"
      -DCUDAToolkit_ROOT="${new_cuda_home}"
    )
  fi
}

__configure_mslk_build_cpu () {
  # Update the package name and build args depending on if CUDA is specified
  echo "[BUILD] Setting CPU-only build args ..."
  build_args+=(
    --build-variant=cpu
  )
}

__configure_mslk_build_docs () {
  # Update the package name and build args depending on if CUDA is specified
  echo "[BUILD] Setting CPU-only (docs) build args ..."
  build_args+=(
    --build-variant=docs
  )
}

__configure_mslk_build_rocm () {
  local mslk_variant_targets="$1"

  # By default, we build for a limited number of target architectures to save on
  # build time.  This list needs to be updated if the CI ROCm machines have
  # different hardware.
  #
  # Target architecture mapping and ROCm compatibility table can be found at:
  #   https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
  #   https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html

  if [ "$mslk_variant_targets" != "" ]; then
    # If targets are manually supplied, override
    echo "[BUILD] Using the user-supplied ROCm targets ..."
    local arch_list="${mslk_variant_targets}"

  elif [ -n "${BUILD_FROM_NOVA+x}" ]; then
    # If BUILD_FROM_NOVA is set (regardless of 0 or 1 - some steps in Nova have
    # the value set to 0), we are building in Nova.  Nova machines take much
    # longer time to build MSLK for ROCm, so we have to limit to just the
    # latest model.
    echo "[BUILD] Building in Nova environment, ignoring the provided PYTORCH_ROCM_ARCH list and limiting ROCm targets ..."
    local arch_list="gfx942"

  else
    # If BUILD_FROM_NOVA is unset, then we are building from a compute host with
    # sufficient resources, so we can build for more AMD Instinct architectures.
    local arch_list="gfx908,gfx90a,gfx942"
  fi

  echo "[BUILD] Setting the following ROCm targets: ${arch_list}"
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} PYTORCH_ROCM_ARCH="${arch_list}"

  echo "[BUILD] Setting HIPCC verbose mode ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} HIPCC_VERBOSE=1

  # For more info on rocmcc flags:
  #   https://rocm.docs.amd.com/en/docs-6.1.1/reference/rocmcc.html
  echo "[BUILD] Setting ROCm build args ..."
  build_args+=(
    --build-variant=rocm
    # HIP_ROOT_DIR now required for HIP to be correctly detected by CMake
    -DHIP_ROOT_DIR=/opt/rocm
    # ROCm CMake complains about missing AMDGPU_TARGETS, so we explicitly set this
    -DAMDGPU_TARGETS="${arch_list}"
  )
}

__configure_mslk_build_cuda () {
  local mslk_variant_targets="$1"

  # Check nvcc is visible
  (test_binpath "${env_name}" nvcc) || return 1

  # Check that cuDNN environment variables are available
  # (test_env_var "${env_name}" CUDNN_INCLUDE_DIR) || return 1
  # (test_env_var "${env_name}" CUDNN_LIBRARY) || return 1
  (test_env_var "${env_name}" NVML_LIB_PATH) || return 1

  if [ "$mslk_variant_targets" != "" ]; then
    echo "[BUILD] Using the user-supplied CUDA targets ..."
    local arch_list="${mslk_variant_targets}"

  elif [ "$TORCH_CUDA_ARCH_LIST" != "" ]; then
    # NOTE: This is generally set by Nova CI; see:
    #   .github/scripts/nova_dir.bash
    echo "[BUILD] Using the environment-supplied TORCH_CUDA_ARCH_LIST as the CUDA targets ..."
    local arch_list="${TORCH_CUDA_ARCH_LIST}"

  else
    # Build only against the CUDA architectures that the latest PyTorch
    # supports, i.e.:
    #   7.0 (V100), 8.0 (A100), 9.0,9.0a (H100), 10.0,10.0a,12.0,12.0a (B100)
    cuda_version_nvcc=$(conda run -n "${env_name}" nvcc --version)
    echo "[BUILD] Using the default architectures for CUDA $cuda_version_nvcc ..."

    # NOTE: CUTLASS kernels for Hopper require sm_90a to be enabled
    # See:
    #
    #   https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    #   https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list
    #   https://github.com/NVIDIA/nvbench/discussions/129
    #   https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt#L187
    #   https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp#L224

    # NOTE: It turns out that the order of the arch_list matters, and that
    # appending 7.0/7.5 to the back of the list mysteriously results in
    # undefined symbol errors on .SO loads
    if [[ $mslk_build_target == "hstu" ]]; then
      if  [[ $cuda_version_nvcc == *"V13"* ]] ||
          [[ $cuda_version_nvcc == *"V12"* ]]; then
        # NOTE: Compiling 9.0a code will fail if sm_80 output is also is also
        # enabled, bc the code relies on the following function that is not
        # supported in sm_80:
        #   float4 atomicAdd(float4* address, float4 val);
        local arch_list="9.0a;10.0a;12.0a"
      else
        # NOTE: HSTU requires sm_75 or higher
        local arch_list="7.5;8.0"
      fi

    elif  [[ $cuda_version_nvcc == *"V13.0"* ]] ||
          [[ $cuda_version_nvcc == *"V12.9"* ]] ||
          [[ $cuda_version_nvcc == *"V12.8"* ]]; then
      local arch_list="7.5;8.0;9.0a;10.0a;12.0a"

    elif  [[ $cuda_version_nvcc == *"V12.6"* ]] ||
          [[ $cuda_version_nvcc == *"V12.4"* ]] ||
          [[ $cuda_version_nvcc == *"V12.1"* ]]; then
      local arch_list="7.5;8.0;9.0a"

    else
      local arch_list="7.5;8.0;9.0"
      echo "[BUILD] Unknown NVCC version $cuda_version_nvcc - setting TORCH_CUDA_ARCH_LIST to: ${arch_list}"
    fi
  fi

  echo "[BUILD] Setting the following CUDA targets: ${arch_list}"
  # Unset the environment-supplied TORCH_CUDA_ARCH_LIST because it will take
  # precedence over cmake -DTORCH_CUDA_ARCH_LIST
  unset TORCH_CUDA_ARCH_LIST

  echo "[BUILD] Looking up NVML filepath ..."
  # shellcheck disable=SC2155,SC2086
  local nvml_lib_path=$(conda run --no-capture-output ${env_prefix} printenv NVML_LIB_PATH)

  echo "[BUILD] Looking up NCCL filepath ..."
  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
  # shellcheck disable=SC2155,SC2086
  local nccl_lib_path=$(conda run ${env_prefix} find ${conda_prefix} -name "libnccl.so*")

  echo "[BUILD] Setting NVCC verbose mode ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} NVCC_VERBOSE=1

  echo "[BUILD] Setting CUDA build args ..."
  build_args+=(
    --build-variant=cuda
    --nvml_lib_path="${nvml_lib_path}"
    --nccl_lib_path="${nccl_lib_path}"
    # Pass to PyTorch CMake
    -DTORCH_CUDA_ARCH_LIST="'${arch_list}'"
  )

  # Explicitly set CUDA_HOME (for CUDA 12.6+)
  __configure_mslk_cuda_home

  # Set NVCC flags
  __configure_mslk_build_nvcc
}

# shellcheck disable=SC2120
__configure_mslk_build () {
  echo "################################################################################"
  echo "# Configure MSLK Build"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Set build args list to be populated, and set verbosity
  build_args=(
    --verbose
  )

  # Set the build target
  echo "[BUILD] Setting the build target: ${mslk_build_target} ..."
  build_args+=(
    --build-target="${mslk_build_target}"
  )

  # Append build args based on the build variant
  if [ "$mslk_build_variant" == "cpu" ]; then
    echo "[BUILD] Configuring build as CPU variant ..."
    __configure_mslk_build_cpu || return 1

  elif [ "$mslk_build_variant" == "docs" ]; then
    echo "[BUILD] Configuring build as CPU (docs) variant ..."
    __configure_mslk_build_docs || return 1

  elif [ "$mslk_build_variant" == "rocm" ]; then
    echo "[BUILD] Configuring build as ROCm variant ..."
    __configure_mslk_build_rocm "${mslk_variant_targets}" || return 1

  else
    echo "[BUILD] Configuring build as CUDA variant (this is the default behavior) ..."
    __configure_mslk_build_cuda "${mslk_variant_targets}"  || return 1
  fi

  # shellcheck disable=SC2086
  print_exec conda run ${env_prefix} c++ --version

  # Set other compiler flags as needed
  if print_exec "conda run ${env_prefix} c++ --version | grep -i clang"; then
    __configure_mslk_build_clang || return 1
  fi

  # Set debugging options
  if [ "$mslk_release_channel" != "release" ] || [ "$BUILD_DEBUG" == "1" ]; then
    echo "[BUILD] Enabling debug features in the build ..."
    build_args+=(
      --debug=1
    )
  fi

  # Set FB-only options
  if [ "$BUILD_FB_CODE" == "1" ]; then
    echo "[BUILD] Enabling build of FB-only code ..."
    build_args+=(
      --build_fb_code
    )
  fi

  # shellcheck disable=SC2145
  echo "[BUILD] MSLK build arguments have been set:  ${build_args[@]}"
}

__export_target_variant_info () {
  local mslk_build_target_variant="$1"

  # Extract the package channel and version from the tuple-string
  if  [ "$mslk_build_target_variant" == "docs" ] ||
      [ "$mslk_build_target_variant" == "cpu" ] ||
      [ "$mslk_build_target_variant" == "cuda" ] ||
      [ "$mslk_build_target_variant" == "rocm" ]; then
    export mslk_build_target="default"
    export mslk_build_variant="${mslk_build_target_variant}"

  else
    # shellcheck disable=SC2207
    local mslk_build_target_variant_arr=($(echo "${mslk_build_target_variant}" | tr '/' '\n'))

    # Assume that if the input is a single string, and is not one of the
    # designated build variants, then the it is referring to the build target
    if [ ${#mslk_build_target_variant_arr[@]} -lt 2 ]; then
      export mslk_build_target="${mslk_build_target_variant_arr[0]}"
      export mslk_build_variant="cuda"
    else
      export mslk_build_target="${mslk_build_target_variant_arr[0]}"
      export mslk_build_variant="${mslk_build_target_variant_arr[1]}"
    fi
  fi

  echo "[BUILD] BUILD_TARGET_VARIANT: ${mslk_build_target_variant}"
  echo "[BUILD] Extracted build target: ${mslk_build_target}"
  echo "[BUILD] Extracted build variant: ${mslk_build_variant}"
}

__build_mslk_set_python_tag () {
  # shellcheck disable=SC2207,SC2086
  local python_version=($(conda run --no-capture-output ${env_prefix} python --version))

  # shellcheck disable=SC2206
  local python_version_arr=(${python_version[1]//./ })

  # Set the python tag (e.g. Python 3.13 --> py313)
  export python_tag="py${python_version_arr[0]}${python_version_arr[1]}"
  echo "[BUILD] Extracted and set Python tag: ${python_tag}"
}

__build_mslk_set_python_plat_name () {
  if [[ $KERN_NAME == 'Darwin' ]]; then
    # This follows PyTorch package naming conventions
    # See https://pypi.org/project/torch/#files
    if [[ $MACHINE_NAME == 'arm64' ]]; then
      export python_plat_name="macosx_11_0_${MACHINE_NAME}"
    else
      export python_plat_name="macosx_10_9_${MACHINE_NAME}"
    fi

  elif [[ $KERN_NAME == 'Linux' ]]; then
    # NOTE: manylinux2014 is the minimum platform tag specified, bc
    # manylinux1 does not support aarch64; see https://github.com/pypa/manylinux
    #
    # As of 2024-12, upstream torch has switched to manylinux_2_28:
    #   https://dev-discuss.pytorch.org/t/pytorch-linux-wheels-switching-to-new-wheel-build-platform-manylinux-2-28-on-november-12-2024/2581
    #   https://github.com/pytorch/pytorch/pull/143423
    export python_plat_name="manylinux_2_28_${MACHINE_NAME}"

  else
    echo "[BUILD] Unsupported OS platform: ${KERN_NAME}"
    return 1
  fi

  echo "[BUILD] Extracted and set Python platform name: ${python_plat_name}"
}

__build_mslk_set_run_multicore () {
  if [[ -n "${BUILD_PARALLELISM:-}"  ]]; then
    # Set manual override if provided.  This is useful for preventing
    # overlapping compilation error messages when debugging.
    export run_multicore="-j ${BUILD_PARALLELISM}"

  else
    # shellcheck disable=SC2155
    local core=$(lscpu | grep "Core(s)" | awk '{print $NF}') && echo "core = ${core}" || echo "core not found"
    # shellcheck disable=SC2155
    local sockets=$(lscpu | grep "Socket(s)" | awk '{print $NF}') && echo "sockets = ${sockets}" || echo "sockets not found"
    local re='^[0-9]+$'

    export run_multicore=""
    if [[ $core =~ $re && $sockets =~ $re ]]; then
      local n_core=$((core * sockets))
      export run_multicore="-j ${n_core}"
    fi
  fi

  echo "[BUILD] Set multicore run option for setup.py: ${run_multicore}"
}

__build_mslk_common_pre_steps () {
  # Private function that uses variables instantiated by its caller

  # Check C/C++ compilers are visible (the build scripts look specifically for `gcc`)
  (test_binpath "${env_name}" cc)   || return 1
  (test_binpath "${env_name}" gcc)  || return 1
  (test_binpath "${env_name}" c++)  || return 1
  (test_binpath "${env_name}" g++)  || return 1

  # Set the default the MSLK build variant to be "default"
  # shellcheck disable=SC2076
  if [[ ! " default " =~ " $mslk_build_target " ]]; then
    echo "################################################################################"
    echo "[BUILD] Unknown MSLK build TARGET: ${mslk_build_target}"
    echo "[BUILD] Exiting ..."
    echo "################################################################################"
    return 1
  fi

  # Set the default the MSLK build variant to be CUDA
  # shellcheck disable=SC2076
  if [[ ! " docs cpu cuda rocm " =~ " $mslk_build_variant " ]]; then
    echo "################################################################################"
    echo "[BUILD] Unknown MSLK build VARIANT: ${mslk_build_variant}"
    echo "[BUILD] Exiting ..."
    echo "################################################################################"
    return 1
  fi

  # Extract and set the Python tag
  __build_mslk_set_python_tag

  # Extract and set the platform name
  __build_mslk_set_python_plat_name

  # Set multicore run option for setup.py if the number of cores on the machine
  # permit for this
  __build_mslk_set_run_multicore

  # Check LD_LIBRARY_PATH for numpy
  echo "[CHECK] LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"

  echo "[BUILD] Running pre-build cleanups ..."
  print_exec rm -rf dist

  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} python setup.py clean

  # echo "[BUILD] Printing git status ..."
  # print_exec git status
  # print_exec git diff
}

__verify_library_symbols () {
  __test_one_symbol () {
    local symbol="$1"
    if [ "$symbol" == "" ]; then
      echo "Usage: ${FUNCNAME[0]} SYMBOL"
      echo "Example(s):"
      echo "    ${FUNCNAME[0]} mslk::coalesce::coalesce_batches_cpu"
      return 1
    fi

    # shellcheck disable=SC2035,SC2061,SC2062,SC2155,SC2178
    local mslk_so_files=$(find . -name *.so | grep .*cmake-build/.*)
    readarray -t mslk_so_files <<<"$mslk_so_files"

    # Iterate through the built .SO files to check for the symbol's existence
    for library in "${mslk_so_files[@]}"; do
      if test_library_symbol "${library}" "${symbol}"; then
        return 0
      fi
    done

    return 1
  }

  # Prepare a sample set of symbols whose existence in the built library should be checked
  # This is by no means an exhaustive set, and should be updated accordingly
  if [ "${mslk_build_target}" == "default" ]; then
    local lib_symbols_to_check=(
      mslk::coalesce::coalesce_batches_cpu
      mslk::coalesce::coalesce_batches_gpu
      mslk::kv_cache::rope_qkv_decoding
      mslk::moe::index_shuffling_torch
      mslk::quantize::quantize_fp8_per_row
    )

    if  [ "${mslk_build_variant}" == "cuda" ]; then
      lib_symbols_to_check+=(
       mslk::moe::gather_along_first_dim
       mslk::comm::nccl_init
      )
    fi
  fi

  echo "[CHECK] Verifying sample subset of symbols in the built libraries ..."
  for symbol in "${lib_symbols_to_check[@]}"; do
    (__test_one_symbol "${symbol}") || return 1
  done
}

__run_mslk_postbuild_checks () {
  # Find the .SO file
  # shellcheck disable=SC2035,SC2061,SC2062,SC2155,SC2178
  local mslk_so_files=$(find . -name *.so | grep .*cmake-build/.*)
  readarray -t mslk_so_files <<<"$mslk_so_files"
  if [ "${#mslk_so_files[@]}" -le 0 ]; then
    echo "[CHECK] .SO library is missing from the build path!"
    return 1
  fi

  print_library_infos       || return 1
  __verify_library_symbols  || return 1
}

__run_audit_wheel () {
  mslk_wheel="$1"
  if [ "$mslk_wheel" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} MSLK_WHEEL_PATH"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} dist/mslk_nightly_cpu-2024.12.20-cp39-cp39-manylinux_2_28_x86_64.whl"
    return 1
  fi

  echo "################################################################################"
  echo "[BUILD] Wheel Audit: ${mslk_wheel}"
  echo ""

  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} auditwheel show "${mslk_wheel}"
  echo ""
  echo "################################################################################"
}

################################################################################
# MSLK Build Functions
################################################################################

build_mslk_package () {
  env_name="$1"
  mslk_release_channel="$2"
  mslk_build_target_variant="$3"
  mslk_variant_targets="$4"
  if [ "$mslk_build_target_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME RELEASE_CHANNEL TARGET/VARIANT [VARIANT_TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env release docs                             # Default build target, CPU-only (docs) build variant"
    echo "    ${FUNCNAME[0]} build_env nightly default/cuda                     # Default build target, CUDA build variant, default variant target(s)"
    echo "    ${FUNCNAME[0]} build_env test cuda '7.0;8.0'                      # Default build target, CUDA build variant, custom variant target(s)"
    echo "    ${FUNCNAME[0]} build_env test rocm                                # Default build target, ROCm build variant, default variant target(s)"
    echo "    ${FUNCNAME[0]} build_env test default/rocm 'gfx906;gfx908;gfx90a' # Default build target, ROCm build variant, custom variant target(s)"
    return 1
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Extract the build target and variant from the tuple-string, and export
  # variables to environment
  __export_target_variant_info "${mslk_build_target_variant}"

  # Set up and configure the build
  __build_mslk_common_pre_steps || return 1
  __configure_mslk_build        || return 1

  echo "################################################################################"
  echo "# Build MSLK Package (Wheel)"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # Set packaging options
  build_args+=(
    --package_channel="${mslk_release_channel}"
    --python-tag="${python_tag}"
    --plat-name="${python_plat_name}"
  )

  # Prepend build options correctly for `python -m build`
  # https://build.pypa.io/en/stable/index.html
  # https://gregoryszorc.com/blog/2023/10/30/my-user-experience-porting-off-setup.py/
  for i in "${!build_args[@]}"; do
    build_args[i]="--config-setting=--build-option=${build_args[i]}"
  done

  # Build the wheel.  Invoke using `python -m build`
  #   https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html
  echo "[BUILD] Building MSLK wheel (TARGET=${mslk_build_target}, VARIANT=${mslk_build_variant}) ..."
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} \
    python -m build --wheel --no-isolation \
      "${build_args[@]}" || return 1

  # Run checks on the built libraries
  __run_mslk_postbuild_checks || return 1

  for wheelfile in dist/*.whl; do
    __run_audit_wheel "${wheelfile}"
  done

  echo "[BUILD] Enumerating the built wheels ..."
  print_exec ls -lth dist/*.whl || return 1

  echo "[BUILD] Enumerating the wheel SHAs ..."
  print_exec sha1sum dist/*.whl || return 1
  print_exec sha256sum dist/*.whl || return 1
  print_exec md5sum dist/*.whl || return 1

  echo "[BUILD] MSLK build + package completed"
}

build_mslk_install () {
  env_name="$1"
  mslk_build_target_variant="$2"
  mslk_variant_targets="$3"
  if [ "$mslk_build_target_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME TARGET/VARIANT [TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env release cpu                              # Default build target, CPU-only build variant"
    echo "    ${FUNCNAME[0]} build_env release docs                             # Default build target, CPU-only (docs) build variant"
    echo "    ${FUNCNAME[0]} build_env nightly default/cuda                     # Default build target, CUDA build variant, default variant target(s)"
    echo "    ${FUNCNAME[0]} build_env test cuda '7.0;8.0'                      # Default build target, CUDA build variant, custom variant target(s)"
    echo "    ${FUNCNAME[0]} build_env test rocm                                # Default build target, ROCm build variant, default variant target(s)"
    echo "    ${FUNCNAME[0]} build_env test default/rocm 'gfx906;gfx908;gfx90a' # Default build target, ROCm build variant, custom variant target(s)"
    return 1
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Extract the build target and variant from the tuple-string, and export
  # variables to environment
  __export_target_variant_info "${mslk_build_target_variant}"

  # Set up and configure the build
  __build_mslk_common_pre_steps || return 1
  __configure_mslk_build        || return 1

  echo "################################################################################"
  echo "# Build + Install MSLK Package"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # Parallelism may need to be limited to prevent the build from being
  # canceled for going over ulimits
  echo "[BUILD] Building + installing MSLK wheel (TARGET=${mslk_build_target}, VARIANT=${mslk_build_variant}) ..."
  # shellcheck disable=SC2086,SC2145
  print_exec conda run --no-capture-output ${env_prefix} \
    sh -c \"python setup.py "${run_multicore}" install "${build_args[@]}"\" || return 1

  # Run checks on the built libraries
  __run_mslk_postbuild_checks || return 1

  echo "[INSTALL] Checking imports ..."
  # Exit this directory to prevent import clashing, since there is an
  # mslk/ subdirectory present
  cd - || return 1
  (test_python_import_package "${env_name}" mslk) || return 1
  (test_python_import_symbol "${env_name}" mslk __version__) || return 1
  cd - || return 1

  echo "[BUILD] MSLK build + install completed"
}
