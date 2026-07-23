# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from mslk.utils.flydsl import require_flydsl

__all__ = ["flydsl_flash_attn_func"]


def flydsl_flash_attn_func(*args: Any, **kwargs: Any) -> Any:
    """Run the bundled FlyDSL flash-attention forward kernel."""
    require_flydsl()
    from .flash_attn_interface import flydsl_flash_attn_func as _impl

    return _impl(*args, **kwargs)
