#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

fetch_external_repos () {
  local mslk_root="$1"
  if [ "$mslk_root" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} MSLK_ROOT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} ~/MSLK        # Fetch all external repos, assuming the MSLK root directory is ~/MSLK"
    return 1
  else
    echo "################################################################################"
    echo "# Fetch MSLK Dependencies (External Repos)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  mkdir -p "${mslk_root}/external"          || return 1
  print_exec pushd "${mslk_root}/external"  || return 1

  declare -A repos=(
    ["composable_kernel"]="https://github.com/ROCm/composable_kernel.git@7fe50dc3da2069d6645d9deb8c017a876472a977"
    ["cutlass"]="https://github.com/jwfromm/cutlass.git@571edeb2d0ac872a8392fc49285b156b07884b4e"
    ["hipify_torch"]="https://github.com/ROCmSoftwarePlatform/hipify_torch.git@63b6a7b541fa7f08f8475ca7d74054db36ff2691"
  )

  echo "[MSLK FETCH REPO] Fetching repositories and checking out specified hashes..."

  for repo_name in "${!repos[@]}"; do
    local repo_url_and_default_sha="${repos[$repo_name]}"
    local repo_url="${repo_url_and_default_sha%@*}"
    local repo_default_sha="${repo_url_and_default_sha#*@}"

    if [ -d "$repo_name" ]; then
      echo "[MSLK FETCH REPO] Directory '$repo_name' already exists. Skipping clone, but will try to checkout hash."
    else
      echo "[MSLK FETCH REPO] Cloning into '$repo_name'..."
      git clone "$repo_url" "$repo_name"
    fi

    if [ -f "${repo_name}.submodule.txt" ]; then
      # shellcheck disable=SC2155
      local repo_sha=$(awk 'NR==1 {print $3}' "${repo_name}.submodule.txt")
      echo "[MSLK FETCH REPO] [$repo_name] Submodule file exists; will use the SHA found in the file: $repo_sha"
    else
      local repo_sha="$repo_default_sha"
      echo "[MSLK FETCH REPO] [$repo_name] Submodule file does NOT exist; will use the provided default SHA: $repo_sha"
    fi

    echo "[MSLK FETCH REPO] [$repo_name] Checking out: $repo_sha"
    pushd "$repo_name" || return 1
    git fetch origin
    git checkout -q "$repo_sha"

    actual_hash=$(git rev-parse HEAD)
    if [[ "$actual_hash" == "$repo_sha" ]]; then
      echo "[MSLK FETCH REPO] [$repo_name] Successfully checked out: $repo_sha"
    else
      echo "[MSLK FETCH REPO] [$repo_name] ERROR: Checkout failed. Expected $repo_sha, got $actual_hash"
      return 1
    fi

    echo ""
    popd || return 1
  done

  print_exec popd  || return 1

  echo ""
  echo "[MSLK FETCH REPO] Successfully fetched all repositories"
}
