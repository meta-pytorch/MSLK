# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Vendored FlyDSL flash-attention kernel authoring modules.

These modules are copied verbatim (imports rewritten to be package-relative)
from the FlyDSL ``kernels/attention`` and ``kernels/common`` trees, which are
not shipped in the ``flydsl`` PyPI wheel. They build and launch the kernels at
runtime against the installed ``flydsl`` compiler/runtime, so the pinned
``flydsl`` version (see ``setup.py`` extras and ``ci/scripts/utils_flydsl.bash``)
must match the FlyDSL release these were vendored from. The files keep their
original Apache-2.0 headers; do not edit them here — re-vendor from FlyDSL and
re-apply the relative-import rewrite instead.

Public entry point is ``mslk.attention.flydsl.flydsl_flash_attn_func``; import
these submodules only through that wrapper (guarded by FlyDSL availability).
"""
