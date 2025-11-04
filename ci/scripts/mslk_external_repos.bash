#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

fetch_external_repos () {
  echo "################################################################################"
  echo "# Fetch MSLK Dependencies (External Repos)"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  declare -A repos=(
    ["composable_kernel"]="https://github.com/ROCm/composable_kernel.git@7fe50dc3da2069d6645d9deb8c017a876472a977"
    ["cutlass"]="https://github.com/jwfromm/cutlass@98125ce499b0fdf7ffbe0e3052f5b8709f4840f8"
    ["hipify_torch"]="https://github.com/ROCmSoftwarePlatform/hipify_torch.git@63b6a7b541fa7f08f8475ca7d74054db36ff2691"
  )

  echo "[FETCH REPO] Fetching repositories and checking out specified hashes..."

  for repo_name in "${!repos[@]}"; do
    local repo_url_and_sha="${repos[$repo_name]}"
    local repo_url="${repo_url_and_sha%@*}"
    local repo_sha="${repo_url_and_sha#*@}"

    if [ -d "$repo_name" ]; then
      echo "[FETCH REPO] Directory '$repo_name' already exists. Skipping clone, but will try to checkout hash."
    else
      echo "[FETCH REPO] Cloning into '$repo_name'..."
      git clone "$repo_url" "$repo_name"
    fi

    echo "[FETCH REPO] [$repo_name] Checking out: $repo_sha"
    pushd "$repo_name" || return 1
    git fetch origin
    git checkout -q "$repo_sha"

    actual_hash=$(git rev-parse HEAD)
    if [[ "$actual_hash" == "$repo_sha" ]]; then
      echo "[FETCH REPO] [$repo_name] Successfully checked out: $repo_sha"
    else
      echo "[FETCH REPO] [$repo_name] ERROR: Checkout failed. Expected $repo_sha, got $actual_hash"
      return 1
    fi

    echo ""
    popd || return 1
  done

  echo ""
  echo "[FETCH REPO] Successfully fetched all repositories"
}
