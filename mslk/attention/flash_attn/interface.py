# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def not_implemented(*args, **kwargs):
    raise NotImplementedError("This function is not implemented.")

try:
    from mslk.fb.mslk.attention.flash_attn import flash_attn_func as _flash_attn_func
    from mslk.fb.mslk.attention.flash_attn.interface import (
        flash_attn_combine,
        mslk_flash_attn_bwd as _flash_attn_bwd,
        mslk_flash_attn_fwd as _flash_attn_fwd,
        mslk_flash_attn_decode as _flash_attn_decode,
    )
except ImportError:
    from flash_attn.cute.interface import (
        flash_attn_func as _flash_attn_func,
        flash_attn_combine,
        _flash_attn_bwd,
        _flash_attn_fwd,
    )
    _flash_attn_decode = not_implemented

flash_attn_func = _flash_attn_func
