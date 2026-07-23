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
    flydsl_preshuffle_batched_gemm(XQ, WQ, x_scale, w_scale, ...)  -- batched GEMM
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
        from mslk.gemm.flydsl._kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

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

    import flydsl.expr as fx  # pyre-ignore[21]
    from mslk.gemm.flydsl._kernels.tensor_shim import _run_compiled, ptr_arg

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


_batched_graph_cache: dict = {}


def _batched_dispatch_loop(
    exe, out, XQ_i8, WQ_i8, xs, ws, dummy_bias_ptr, B, M, N, stream, ptr_arg,
    _run_compiled,
):
    """Raw per-batch dispatch loop (used for graph capture and non-graphed path)."""
    for b in range(B):
        _run_compiled(
            exe,
            ptr_arg(out[b].view(-1)),
            ptr_arg(XQ_i8[b].view(-1)),
            ptr_arg(WQ_i8[b].view(-1)),
            ptr_arg(xs[b].view(-1)),
            ptr_arg(ws[b].view(-1)),
            dummy_bias_ptr, M, N, stream,
        )


def flydsl_preshuffle_batched_gemm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Optional[Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    tile_k: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    use_hip_graph: bool = True,
) -> Tensor:
    """Batched FP8 preshuffle GEMM with HIP graph acceleration.

    On the first call for a given (B, M, N, K, dtype) shape, captures all B
    kernel launches into a HIP graph using static input/output buffers.
    Subsequent calls copy data into the static buffers and replay the graph
    with a single CPU-side call, eliminating per-batch Python dispatch overhead.

    Args:
        XQ: FP8 activation tensor (B, M, K).
        WQ: FP8 weight tensor (B, N, K), pre-shuffled via ``flydsl_preshuffle``.
        x_scale: Per-token activation scale (B, M) or (B, M, 1), float32.
        w_scale: Per-channel weight scale (B, N) or (B, 1, N), float32.
        out: Optional pre-allocated output tensor (B, M, N).
        dtype: Output dtype (bfloat16 or float16).
        use_hip_graph: If True, capture and replay via HIP graph.

    Returns:
        Output tensor (B, M, N) in ``dtype``.
    """
    from mslk.utils.flydsl import require_flydsl

    require_flydsl()

    assert XQ.dim() == 3, f"Expected 3D XQ, got {XQ.dim()}D"
    assert WQ.dim() == 3, f"Expected 3D WQ, got {WQ.dim()}D"
    B, M, K = XQ.shape
    N = WQ.shape[1]

    if out is None:
        out = torch.empty(B, M, N, dtype=dtype, device=XQ.device)

    compile_fn = _get_compile_fn()
    if compile_fn is None:
        raise RuntimeError("FlyDSL preshuffle kernel compiler not available")

    if tile_m is None or tile_n is None or tile_k is None:
        from mslk.gemm.flydsl._configs import select_default_config

        cfg = select_default_config(M, N, K)
        tile_m = tile_m or cfg.tile_m
        tile_n = tile_n or cfg.tile_n
        tile_k = tile_k or cfg.tile_k

    if "float8" in str(XQ.dtype):
        in_dtype = "fp8"
    elif XQ.dtype == torch.int8:
        in_dtype = "int8"
    else:
        raise ValueError(f"Unsupported input dtype {XQ.dtype}")

    out_dtype = "bf16" if dtype == torch.bfloat16 else "fp16"

    exe = compile_fn(
        N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype=in_dtype, out_dtype=out_dtype,
    )

    import flydsl.expr as fx  # pyre-ignore[21]
    from mslk.gemm.flydsl._kernels.tensor_shim import _run_compiled, ptr_arg

    def _as_i8(t: Tensor) -> Tensor:
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    XQ_i8 = _as_i8(XQ.contiguous())
    WQ_i8 = _as_i8(WQ.contiguous())
    xs_contig = x_scale.contiguous()
    ws_contig = w_scale.contiguous()

    cache_key = (B, M, N, K, str(XQ.dtype), str(dtype), tile_m, tile_n, tile_k)

    if use_hip_graph and cache_key in _batched_graph_cache:
        entry = _batched_graph_cache[cache_key]
        entry["s_xq"].copy_(XQ_i8)
        entry["s_wq"].copy_(WQ_i8)
        entry["s_xs"].copy_(xs_contig)
        entry["s_ws"].copy_(ws_contig)
        entry["graph"].replay()
        out.copy_(entry["s_out"])
        return out

    dummy_bias = torch.empty(0, dtype=dtype, device=XQ.device)
    dummy_bias_ptr = ptr_arg(dummy_bias)
    stream = fx.Stream(torch.cuda.current_stream())

    if not use_hip_graph:
        _batched_dispatch_loop(
            exe, out, XQ_i8, WQ_i8, xs_contig, ws_contig,
            dummy_bias_ptr, B, M, N, stream, ptr_arg, _run_compiled,
        )
        return out

    # First call: allocate static buffers, warmup, capture graph
    s_xq = XQ_i8.clone()
    s_wq = WQ_i8.clone()
    s_xs = xs_contig.clone()
    s_ws = ws_contig.clone()
    s_out = torch.empty_like(out)

    # Warmup (required before graph capture)
    _batched_dispatch_loop(
        exe, s_out, s_xq, s_wq, s_xs, s_ws,
        dummy_bias_ptr, B, M, N, stream, ptr_arg, _run_compiled,
    )
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _batched_dispatch_loop(
            exe, s_out, s_xq, s_wq, s_xs, s_ws,
            dummy_bias_ptr, B, M, N, stream, ptr_arg, _run_compiled,
        )

    _batched_graph_cache[cache_key] = {
        "graph": g,
        "s_xq": s_xq, "s_wq": s_wq, "s_xs": s_xs, "s_ws": s_ws,
        "s_out": s_out,
    }

    # First call also needs correct output
    s_xq.copy_(XQ_i8)
    s_wq.copy_(WQ_i8)
    s_xs.copy_(xs_contig)
    s_ws.copy_(ws_contig)
    g.replay()
    out.copy_(s_out)
    return out


# ---------------------------------------------------------------------------
# Register FlyDSL as the ROCm implementation of mslk::f8f8bf16_rowwise_batched
# on gfx950, replacing the CK backend.
# ---------------------------------------------------------------------------
if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    if hasattr(torch.ops.mslk, "f8f8bf16_rowwise_batched"):
        from mslk.utils.flydsl import is_flydsl_available

        if is_flydsl_available():

            @torch.library.impl("mslk::f8f8bf16_rowwise_batched", "CUDA")
            def _f8f8bf16_rowwise_batched_flydsl(
                XQ: Tensor,
                WQ: Tensor,
                x_scale: Tensor,
                w_scale: Tensor,
                bias: Optional[Tensor] = None,
                use_fast_accum: bool = True,
                output: Optional[Tensor] = None,
            ) -> Tensor:
                B = XQ.shape[0]
                WQ_shuf = torch.stack(
                    [flydsl_preshuffle(WQ[i]) for i in range(B)]
                )
                return flydsl_preshuffle_batched_gemm(
                    XQ, WQ_shuf, x_scale, w_scale, out=output,
                )
