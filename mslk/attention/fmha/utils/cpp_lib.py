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
    import os

    import torch

    # load the custom_op_library from the mslk directory
    # and register the custom ops
    lib_dir = str(Path(__file__).parent.parent.parent.parent)
    if os.name == "nt":
        # Register the main torchvision library location on the default DLL path
        import ctypes
        import sys

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p

        if sys.version_info >= (3, 8):
            os.add_dll_directory(lib_dir)
        elif with_load_library_flags:
            res = kernel32.AddDllDirectory(lib_dir)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error adding "{lib_dir}" to the DLL directories.'
                raise err

        kernel32.SetErrorMode(prev_error_mode)

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    if torch.version.hip and not hasattr(torch.version, "git_version"):
        ext_specs = extfinder.find_spec("_C_hip")
    else:
        ext_specs = extfinder.find_spec("_C")
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
