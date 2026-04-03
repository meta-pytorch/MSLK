#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

import torch


def load_library_buck(buck_target: str) -> None:
    import mslk  # noqa: F401

    if getattr(mslk, "_python_only", False):  # pyre-ignore [16]
        return

    # pyre-ignore [16]
    open_source: bool = getattr(mslk, "open_source", False)

    try:
        torch.ops.load_library(buck_target)
    except OSError as e:
        if open_source:
            pass
        else:
            logging.error(
                f"Failed to load buck target {buck_target}, ops will not be available via torch.ops! Error: {e}"
            )
