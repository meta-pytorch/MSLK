# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL preshuffle GEMM for FP8 rowwise scaling (gfx950).

Provides single and batched FP8 preshuffle GEMM via FlyDSL, with optional
HIP graph acceleration for the batched path. Also registers as the ROCm
implementation of the ``mslk`` rowwise FP8 ops on gfx950.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Tile configuration for the preshuffle GEMM kernel
# ---------------------------------------------------------------------------


@dataclass
class KernelConfig:
    """Tile and launch parameters for one FlyDSL preshuffle GEMM variant."""

    tile_m: int  # M-dimension tile size (rows of activation per workgroup)
    tile_n: int  # N-dimension tile size (columns of weight per workgroup)
    tile_k: int  # K-dimension tile size (reduction loop step)
    lds_stage: int  # number of LDS pipeline stages (software pipelining depth)
    use_cshuffle_epilog: int = 0  # 1 to use cross-lane shuffle in the epilogue
    use_async_copy: int = 0  # 1 to use async global→LDS copy instructions
    waves_per_eu: int = 0  # occupancy hint; 0 = let the compiler decide
    xcd_swizzle: int = 0  # XCD (cross-chiplet die) swizzle pattern index


DEFAULT_CONFIGS_GFX950: dict[int, KernelConfig] = {
    -1: KernelConfig(128, 256, 256, 2, 0, 0, 2, 0),
    -2: KernelConfig(16, 64, 512, 2, 0, 0, 2, 0),
    -3: KernelConfig(32, 64, 512, 2, 0, 0, 2, 0),
    -4: KernelConfig(128, 128, 128, 2, 0, 0, 2, 0),
}


def select_default_config(m: int, n: int, k: int) -> KernelConfig:
    """Select a default FlyDSL tile config based on shape heuristics."""
    configs = DEFAULT_CONFIGS_GFX950
    fits = [c for c in configs.values() if n % c.tile_n == 0 and k % c.tile_k == 0]
    if not fits:
        raise RuntimeError(
            f"No FlyDSL preshuffle config fits shape ({m}, {n}, {k}). "
            f"N must be divisible by tile_n and K by tile_k."
        )
    want_tm = min(256, max(16, 1 << (m - 1).bit_length())) if m > 0 else 16
    return min(fits, key=lambda c: (abs(c.tile_m - want_tm), -c.tile_n, -c.tile_k))


# ---------------------------------------------------------------------------
# Weight preshuffle
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Kernel compiler (lazy-loaded)
# ---------------------------------------------------------------------------

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


def _as_i8(t: Tensor) -> Tensor:
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


# ---------------------------------------------------------------------------
# Single GEMM
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batched GEMM with HIP graph acceleration
# ---------------------------------------------------------------------------

_batched_graph_cache: dict = {}


def _batched_dispatch_loop(
    exe,
    out,
    XQ_i8,
    WQ_i8,
    xs,
    ws,
    dummy_bias_ptr,
    B,
    M,
    N,
    stream,
    ptr_arg,
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
            dummy_bias_ptr,
            M,
            N,
            stream,
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

    Args:
        XQ: FP8 activation tensor (B, M, K).
        WQ: FP8 weight tensor (B, N, K), pre-shuffled via ``flydsl_preshuffle``.
        x_scale: Per-token activation scale (B, M) or (B, M, 1), float32.
        w_scale: Per-channel weight scale (B, N) or (B, 1, N), float32.
        out: Optional pre-allocated output tensor (B, M, N).
        dtype: Output dtype (bfloat16 or float16).
        use_hip_graph: If True, capture and replay via HIP graph.
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
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )

    import flydsl.expr as fx  # pyre-ignore[21]

    from mslk.gemm.flydsl._kernels.tensor_shim import _run_compiled, ptr_arg

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
            exe,
            out,
            XQ_i8,
            WQ_i8,
            xs_contig,
            ws_contig,
            dummy_bias_ptr,
            B,
            M,
            N,
            stream,
            ptr_arg,
            _run_compiled,
        )
        return out

    s_xq = XQ_i8.clone()
    s_wq = WQ_i8.clone()
    s_xs = xs_contig.clone()
    s_ws = ws_contig.clone()
    s_out = torch.empty_like(out)

    _batched_dispatch_loop(
        exe,
        s_out,
        s_xq,
        s_wq,
        s_xs,
        s_ws,
        dummy_bias_ptr,
        B,
        M,
        N,
        stream,
        ptr_arg,
        _run_compiled,
    )
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _batched_dispatch_loop(
            exe,
            s_out,
            s_xq,
            s_wq,
            s_xs,
            s_ws,
            dummy_bias_ptr,
            B,
            M,
            N,
            stream,
            ptr_arg,
            _run_compiled,
        )

    _batched_graph_cache[cache_key] = {
        "graph": g,
        "s_xq": s_xq,
        "s_wq": s_wq,
        "s_xs": s_xs,
        "s_ws": s_ws,
        "s_out": s_out,
    }

    s_xq.copy_(XQ_i8)
    s_wq.copy_(WQ_i8)
    s_xs.copy_(xs_contig)
    s_ws.copy_(ws_contig)
    g.replay()
    out.copy_(s_out)
    return out


# ---------------------------------------------------------------------------
# Register FlyDSL as the ROCm implementation of mslk rowwise FP8 ops on gfx950
# ---------------------------------------------------------------------------
if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    from mslk.utils.flydsl import is_flydsl_available

    if is_flydsl_available():

        _preshuffle_cache: dict = {}

        def _get_preshuffled(WQ: Tensor) -> Tensor:
            key = WQ.data_ptr()
            cached = _preshuffle_cache.get(key)
            if cached is not None and cached.shape == WQ.shape:
                return cached
            shuf = flydsl_preshuffle(WQ)
            _preshuffle_cache[key] = shuf
            return shuf

        def _flydsl_rowwise_impl(
            XQ: Tensor,
            WQ: Tensor,
            x_scale: Tensor,
            w_scale: Tensor,
            bias: Optional[Tensor] = None,
            use_fast_accum: bool = True,
            dtype: torch.dtype = torch.bfloat16,
            output: Optional[Tensor] = None,
        ) -> Tensor:
            WQ_shuf = _get_preshuffled(WQ)
            return flydsl_preshuffle_gemm(
                XQ,
                WQ_shuf,
                x_scale,
                w_scale,
                out=output,
                dtype=dtype,
            )

        if hasattr(torch.ops.mslk, "f8f8bf16_rowwise"):

            @torch.library.impl("mslk::f8f8bf16_rowwise", "CUDA")
            def _f8f8bf16_rowwise_flydsl(
                XQ: Tensor,
                WQ: Tensor,
                x_scale: Tensor,
                w_scale: Tensor,
                bias: Optional[Tensor] = None,
                use_fast_accum: bool = True,
            ) -> Tensor:
                return _flydsl_rowwise_impl(
                    XQ,
                    WQ,
                    x_scale,
                    w_scale,
                    bias,
                    use_fast_accum,
                    dtype=torch.bfloat16,
                )

        if hasattr(torch.ops.mslk, "f8f8bf16_rowwise_out"):

            @torch.library.impl("mslk::f8f8bf16_rowwise_out", "CUDA")
            def _f8f8bf16_rowwise_out_flydsl(
                XQ: Tensor,
                WQ: Tensor,
                x_scale: Tensor,
                w_scale: Tensor,
                output: Tensor,
                bias: Optional[Tensor] = None,
                use_fast_accum: bool = True,
            ) -> None:
                _flydsl_rowwise_impl(
                    XQ,
                    WQ,
                    x_scale,
                    w_scale,
                    bias,
                    use_fast_accum,
                    dtype=output.dtype,
                    output=output,
                )

        if hasattr(torch.ops.mslk, "f8f8f16_rowwise"):

            @torch.library.impl("mslk::f8f8f16_rowwise", "CUDA")
            def _f8f8f16_rowwise_flydsl(
                XQ: Tensor,
                WQ: Tensor,
                x_scale: Tensor,
                w_scale: Tensor,
                bias: Optional[Tensor] = None,
                use_fast_accum: bool = True,
            ) -> Tensor:
                return _flydsl_rowwise_impl(
                    XQ,
                    WQ,
                    x_scale,
                    w_scale,
                    bias,
                    use_fast_accum,
                    dtype=torch.float16,
                )

    # --- batched op ---
    if hasattr(torch.ops.mslk, "f8f8bf16_rowwise_batched"):
        from mslk.utils.flydsl import is_flydsl_available as _is_flydsl_batched

        if _is_flydsl_batched():

            _batched_preshuffle_cache: dict = {}

            def _get_batched_preshuffled(WQ: Tensor) -> Tensor:
                key = WQ.data_ptr()
                cached = _batched_preshuffle_cache.get(key)
                if cached is not None and cached.shape == WQ.shape:
                    return cached
                B = WQ.shape[0]
                shuf = torch.stack([flydsl_preshuffle(WQ[i]) for i in range(B)])
                _batched_preshuffle_cache[key] = shuf
                return shuf

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
                WQ_shuf = _get_batched_preshuffled(WQ)
                return flydsl_preshuffle_batched_gemm(
                    XQ,
                    WQ_shuf,
                    x_scale,
                    w_scale,
                    out=output,
                )
