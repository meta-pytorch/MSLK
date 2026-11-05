# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Which backend serves ``mslk::f4f4bf16`` (MXFP4 x MXFP4 -> BF16) on ROCm.

- ``mslk``: MSLK's Triton kernel (``mslk/gemm/triton/f4f4bf16.py``), the
  default and the fallback.
- ``aiter``: aiter's ``gemm_afp4wfp4`` (``mslk/gemm/aiter/f4f4bf16.py``), for
  the shapes ``tables/f4f4bf16.csv`` assigns to it. That table is written by
  ``bench/aiter/tune_f4f4bf16.py``.

aiter only computes standard MXFP4 (no global scale, 32-element scale blocks);
other calls go to the Triton path, which rejects them on ROCm.
"""

from typing import Optional, Sequence, Tuple

import torch
from mslk.aiter.routing import MSLK, OpRouting

OP: str = "f4f4bf16"
AITER: str = "aiter"


# Backends are imported on first use: importing mslk.gemm must not pull in
# Triton on CPU-only nodes, nor aiter anywhere.
def _mslk(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    mxfp4_block_size: int = 32,
) -> torch.Tensor:
    from mslk.gemm.triton import f4f4bf16 as triton_f4f4bf16

    return triton_f4f4bf16.f4f4bf16(
        XQ,
        WQ,
        x_scale,
        w_scale,
        output=output,
        global_scale=global_scale,
        mxfp4_block_size=mxfp4_block_size,
    )


def _aiter(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    mxfp4_block_size: int = 32,
) -> torch.Tensor:
    from mslk.gemm.aiter import f4f4bf16 as aiter_f4f4bf16

    return aiter_f4f4bf16.f4f4bf16(XQ, WQ, x_scale, w_scale, output)


def _aiter_accepts(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    mxfp4_block_size: int = 32,
) -> bool:
    return global_scale is None and mxfp4_block_size == 32


def _key(
    XQ: torch.Tensor, WQ: torch.Tensor, *args: object, **kwargs: object
) -> Tuple[Sequence[object], int]:
    # XQ is [M, K/2] and WQ [N, K/2]: two FP4 values per byte.
    return (WQ.shape[0], XQ.shape[1] * 2), XQ.shape[0]


#: ``mslk::f4f4bf16`` on ROCm: call it with the op's arguments.
ROUTING: OpRouting = OpRouting(
    OP,
    "mslk.gemm.dispatch",
    "tables/f4f4bf16.csv",
    key_columns=("N", "K"),
    bucket_column="M_max",
    backends={MSLK: _mslk, AITER: _aiter},
    key=_key,
    accepts={AITER: _aiter_accepts},
)
BACKENDS: Tuple[str, ...] = tuple(ROUTING.backends)
