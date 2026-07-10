# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL preshuffle GEMM for FP8 rowwise scaling (gfx950).

Public API:
    flydsl_preshuffle(src)  -- shuffle weights for the FlyDSL layout
    flydsl_preshuffle_gemm(XQ, WQ, x_scale, w_scale, ...)  -- run the GEMM
"""

from typing import Optional

import torch
from torch import Tensor


def flydsl_preshuffle(src: Tensor) -> Tensor:
    """Shuffle FP8 weight tensor into FlyDSL preshuffle layout.

    This is NOT interchangeable with ``ck_preshuffle`` -- the two produce
    different memory layouts.

    Args:
        src: FP8 weight tensor of shape (N, K).

    Returns:
        Shuffled tensor with same shape and dtype.
    """
    assert src.dim() == 2, f"Expected 2D weight tensor, got {src.dim()}D"
    x_type = src.dtype
    N, K = src.shape
    BN = 16
    BK = 32
    K_pack = 16 // src.element_size()
    assert N % BN == 0, f"N ({N}) must be divisible by {BN}"
    assert K % BK == 0, f"K ({K}) must be divisible by {BK}"
    x_ = src.view(N // BN, BN, K // BK, BK // K_pack, K_pack)
    x_ = x_.permute(0, 2, 3, 1, 4).contiguous()
    x_ = x_.view(N, K).view(x_type)
    return x_


_compile_fn = None  # type: ignore[assignment]
_import_done: bool = False


def _get_compile_fn():  # type: ignore[return]
    """Lazy-import the FlyDSL preshuffle kernel compiler."""
    global _compile_fn, _import_done
    if _import_done:
        return _compile_fn
    _import_done = True
    from mslk.utils.flydsl import is_flydsl_available

    if not is_flydsl_available():
        return None
    try:
        from mslk.gemm.flydsl._kernels.preshuffle_gemm import (
            compile_preshuffle_gemm_a8,
        )

        _compile_fn = compile_preshuffle_gemm_a8
    except Exception:
        pass
    return _compile_fn


def flydsl_preshuffle_gemm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Optional[Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    tile_k: Optional[int] = None,
    lds_stage: int = 2,
    use_cshuffle_epilog: int = 0,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run FlyDSL preshuffle FP8 rowwise GEMM.

    Args:
        XQ: FP8 activation tensor (M, K).
        WQ: FP8 weight tensor (N, K), pre-shuffled via ``flydsl_preshuffle``.
        x_scale: Per-token activation scale (M, 1) or (M,), float32.
        w_scale: Per-channel weight scale (N, 1) or (N,), float32.
        out: Optional pre-allocated output tensor (M, N).
        tile_m/tile_n/tile_k: Tile dimensions. If None, auto-selected.
        dtype: Output dtype (bfloat16 or float16).

    Returns:
        Output tensor (M, N) in ``dtype``.
    """
    from mslk.utils.flydsl import require_flydsl

    require_flydsl()

    compile_fn = _get_compile_fn()
    if compile_fn is None:
        raise RuntimeError("FlyDSL preshuffle kernel compiler not available")

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0]

    if out is None:
        out = torch.empty(m, n, dtype=dtype, device=XQ.device)

    if tile_m is None or tile_n is None or tile_k is None:
        from mslk.gemm.flydsl._configs import select_default_config

        cfg = select_default_config(m, n, k)
        tile_m = tile_m or cfg.tile_m
        tile_n = tile_n or cfg.tile_n
        tile_k = tile_k or cfg.tile_k
        lds_stage = cfg.lds_stage
        use_cshuffle_epilog = cfg.use_cshuffle_epilog
        use_async_copy = cfg.use_async_copy
        waves_per_eu = cfg.waves_per_eu
        xcd_swizzle = cfg.xcd_swizzle

    if n % tile_n != 0:
        raise RuntimeError(f"N ({n}) not divisible by tile_n ({tile_n})")
    if k % tile_k != 0:
        raise RuntimeError(f"K ({k}) not divisible by tile_k ({tile_k})")

    if "float8" in str(XQ.dtype):
        in_dtype = "fp8"
    elif XQ.dtype == torch.int8:
        in_dtype = "int8"
    else:
        raise ValueError(f"Unsupported input dtype {XQ.dtype}")

    out_dtype = "bf16" if out.dtype == torch.bfloat16 else "fp16"
    wpe = None if waves_per_eu <= 0 else waves_per_eu

    exe = compile_fn(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        lds_stage=lds_stage,
        use_cshuffle_epilog=bool(use_cshuffle_epilog),
        use_async_copy=bool(use_async_copy),
        waves_per_eu=wpe,
        xcd_swizzle=int(xcd_swizzle),
    )

    from mslk.gemm.flydsl._kernels.tensor_shim import _run_compiled, ptr_arg

    import flydsl.expr as fx  # pyre-ignore[21]

    def _as_i8(t: Tensor) -> Tensor:
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    out_contig = out.contiguous()
    _dummy_bias = torch.empty(0, dtype=out.dtype, device=out.device)
    _run_compiled(
        exe,
        ptr_arg(out_contig.view(-1)),
        ptr_arg(_as_i8(XQ.contiguous()).view(-1)),
        ptr_arg(_as_i8(WQ.contiguous()).view(-1)),
        ptr_arg(x_scale.contiguous().view(-1)),
        ptr_arg(w_scale.contiguous().view(-1)),
        ptr_arg(_dummy_bias),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    if out_contig is not out:
        out.copy_(out_contig)

    return out
