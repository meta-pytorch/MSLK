# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from functools import wraps
from typing import Any, Callable

import torch

running_on_rocm: bool = (
    torch.cuda.is_available()
    and torch.cuda.device_count() > 0
    and torch.version.hip is not None
)


def skipIfRocm(
    reason: str = "Test does not work on ROCm",
) -> Any:
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def decorator(fn: Callable) -> Any:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if running_on_rocm:
                raise unittest.SkipTest(reason)
            else:
                fn(*args, **kwargs)

        return wrapper

    return decorator


def skipIfNotRocm(
    reason: str = "Test only works on ROCm",
) -> Any:
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def decorator(fn: Callable) -> Any:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if running_on_rocm:
                fn(*args, **kwargs)
            else:
                raise unittest.SkipTest(reason)

        return wrapper

    return decorator


def skipIfRocmLessThan(min_version: int) -> Any:
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def decorator(testfn: Callable) -> Any:
        @wraps(testfn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ROCM_VERSION_FILEPATH = "/opt/rocm/.info/version"
            if running_on_rocm:
                # Fail if ROCm version file is missing.
                if not os.path.isfile(ROCM_VERSION_FILEPATH):
                    raise AssertionError(
                        f"ROCm version file {ROCM_VERSION_FILEPATH} is missing!"
                    )

                # Parse the version number from the file.
                with open(ROCM_VERSION_FILEPATH, "r") as file:
                    version = file.read().strip()
                version = version.replace("-", "").split(".")
                version = (
                    int(version[0]) * 10000 + int(version[1]) * 100 + int(version[2])
                )

                # Fail if ROCm version is less than the minimum version.
                if version < min_version:
                    raise unittest.SkipTest(
                        f"Skip the test since the ROCm version is less than {min_version}"
                    )
                else:
                    testfn(*args, **kwargs)

            else:
                testfn(*args, **kwargs)

        return wrapper

    return decorator
