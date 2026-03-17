# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import logging
import os
from pathlib import Path


logger = logging.getLogger("mslk_fmha")

UNAVAILABLE_FEATURES_MSG = "  Memory-efficient attention won't be available."


class MslkWasNotBuiltException(Exception):
    def __str__(self) -> str:
        return (
            "Need to compile C++ extensions to use all fmha features.\n"
            "    Please install mslk properly.\n" + UNAVAILABLE_FEATURES_MSG
        )


class MslkInvalidLibException(Exception):
    def __str__(self) -> str:
        return (
            "fmha can't load C++/CUDA extensions. "
            "fmha was built for a different version of PyTorch or Python."
            "\n  Please reinstall mslk " + UNAVAILABLE_FEATURES_MSG
        )


def _register_extensions():
    import importlib

    import torch

    # Only AMD builds ship a binary extension that needs explicit loading
    if not (torch.version.hip and not hasattr(torch.version, "git_version")):
        return

    lib_dir = str(Path(__file__).parent.parent.parent.parent)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )
    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_C_hip")
    if ext_specs is None:
        raise MslkWasNotBuiltException()
    try:
        torch.ops.load_library(ext_specs.origin)
    except OSError as exc:
        raise MslkInvalidLibException() from exc


if os.environ.get("MSLK_PYTHON_ONLY", "0") != "1":
    try:
        _register_extensions()
    except (MslkInvalidLibException, MslkWasNotBuiltException) as e:
        logger.warning(f"WARNING[MSLK]: {e}", exc_info=e)
