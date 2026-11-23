# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""MSLK SDPA ops matching aten's flash and efficient attention contracts.

Usage::

    from mslk.attention.sdpa import enable_aten_override

    # Opt-in: route torch.nn.functional.scaled_dot_product_attention
    # through MSLK's fmha kernels on ROCm.
    enable_aten_override()

    # Or call the ops directly:
    from mslk.attention.sdpa import sdpa_flash_fwd, sdpa_efficient_fwd
"""

from mslk.attention.sdpa._ops import (
    disable_aten_override,
    enable_aten_override,
    sdpa_efficient_bwd,
    sdpa_efficient_fwd,
    sdpa_flash_bwd,
    sdpa_flash_fwd,
)

__all__ = [
    "sdpa_flash_fwd",
    "sdpa_flash_bwd",
    "sdpa_efficient_fwd",
    "sdpa_efficient_bwd",
    "enable_aten_override",
    "disable_aten_override",
]
