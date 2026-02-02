# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    from mslk.fb.mslk.attention.flash_attn import flash_attn_func as _flash_attn_func
except ImportError:
    from flash_attn.cute import flash_attn_func as _flash_attn_func

flash_attn_func = _flash_attn_func
