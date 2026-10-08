# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL decode dispatcher — public entry point for the decoder ops.

Targets gfx942 (CDNA3/MI300) and gfx950 (CDNA4/MI355), wave64. Compute lives in
pa_decode_gfx950 (head-packed fast path, GQA ratio 1..16), pa_decode_gfx950_coop
(per-head fallback), and pa_decode_generic (arch-generic fallback, off-gfx950).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .utils import WARP_SIZE

NUM_WARPS = 4
BLOCK_SIZE = NUM_WARPS * WARP_SIZE  # 256

_CU_COUNT: Optional[int] = None


def _get_cu_count() -> int:
    global _CU_COUNT
    if _CU_COUNT is None:
        try:
            prop = torch.cuda.get_device_properties(0)
            _CU_COUNT = prop.multi_processor_count
        except Exception:
            _CU_COUNT = 120  # conservative default
    return _CU_COUNT


def auto_split_k(
    B: int, G: int, H_q: int, KV_MAX: int, num_warps: int = NUM_WARPS
) -> int:
    """Default split_k for the generic fallback: target ~4 waves to hide memory latency."""
    n_cus = _get_cu_count()
    target_ctas = n_cus * 4  # 4 waves
    base_ctas = B * G * H_q
    if base_ctas >= target_ctas:
        return 1
    needed = (target_ctas + base_ctas - 1) // base_ctas
    sk = 1
    while sk < needed:
        sk *= 2
    min_toks_per_part = 64
    max_sk = max(1, KV_MAX // min_toks_per_part)
    sk = min(sk, max_sk, 64)
    return max(1, sk)


def auto_split_k_coop(B: int, G: int, H_q: int, KV_MAX: int) -> int:
    """split_k for the gfx950 coop-DMA kernel: latency-bound, wants ~8 waves, cap 64."""
    n_cus = _get_cu_count()
    target_ctas = n_cus * 8  # 8 waves
    base_ctas = B * G * H_q
    needed = max(1, (target_ctas + base_ctas - 1) // base_ctas)
    sk = 1
    while sk < needed:
        sk *= 2
    MIN_CHUNK_TOKENS = 64
    max_sk = max(1, KV_MAX // MIN_CHUNK_TOKENS)
    sk = min(sk, max_sk, 64)
    return max(1, sk)


# One CTA is one warp here, so split_k multiplies the warp count directly.
# Above 256 the combine pass dominates: it stages per-partition stats in LDS and
# its accumulation loop is unrolled over partitions, so cost grows linearly while
# the extra parallelism has already saturated. Measured on MI350X at B=1, split
# 512 was slower than 256 on every shape tried (ctx 32k..512k, D 64/128).
_MAX_SPLIT_K_HP = 256

# Tokens a split must own for its share of the combine pass to be worth it. The
# measured knee sits between 256 and 2000 tokens/split and rises with context;
# 256 puts the heuristic within ~9% of the per-shape optimum across that range.
_MIN_CHUNK_TOKENS_HP = 256


def auto_split_k_hp(B: int, G: int, H_q: int, H_kv: int, KV_MAX: int) -> int:
    """split_k for the head-packed gfx950 kernel: ~8 waves counted in warps (B*G*H_kv).

    Capped by `_MAX_SPLIT_K_HP` and by the requirement that each split own at
    least `_MIN_CHUNK_TOKENS_HP` tokens. Always returns a power of two so the
    number of compiled (kernel, reduce) variants stays small.
    """
    n_cus = _get_cu_count()
    target_warps = n_cus * 8
    base_warps = B * G * H_kv
    needed = max(1, (target_warps + base_warps - 1) // base_warps)
    sk = 1
    while sk < needed:
        sk *= 2
    max_sk = max(1, KV_MAX // _MIN_CHUNK_TOKENS_HP)
    sk = min(sk, max_sk, _MAX_SPLIT_K_HP)
    # min() can land off a power of two; round back down.
    p = 1
    while p * 2 <= sk:
        p *= 2
    return max(1, p)


# ── Host launcher ─────────────────────────────────────────────────────────────


def pa_decode_launch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    seq_positions: Optional[torch.Tensor],
    softmax_scale: float,
    split_k: int = 0,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Paged-attention decode (public entry point). Dispatches to gfx950 (head-packed,
    ratio 1..16) or gfx950_coop, both falling back to generic off-gfx950."""
    _, _, _, H_q, _ = Q.shape
    H_kv = K.shape[3]
    ratio = H_q // H_kv if H_kv > 0 else 0
    use_hp = H_kv > 0 and H_q % H_kv == 0 and 1 <= ratio <= 16
    if use_hp:
        from .pa_decode_gfx950 import pa_decode_gfx950_launch

        return pa_decode_gfx950_launch(
            Q, K, V, seq_positions, softmax_scale, split_k, output_dtype
        )
    from .pa_decode_gfx950_coop import pa_decode_gfx950_coop_launch

    return pa_decode_gfx950_coop_launch(
        Q, K, V, seq_positions, softmax_scale, split_k, output_dtype
    )


def query_group_seqlen(seqlen_kv: int, seqlen_q: int, groups: int, group: int) -> int:
    """KV length to pass for one query group, preserving bottom-right causality.

    In the full block, query token ``i`` attends to ``[0, L - Sq + i + 1)``.
    Group ``g`` holds tokens ``g*S .. g*S+S-1`` where ``S = Sq/groups``, and is
    launched as its own block of ``S`` tokens. Inside that launch, token ``j``
    attends to ``[0, L_g - S + j + 1)``. Setting

        L_g = L - (groups - 1 - g) * S

    makes those identical, since ``L_g - S + j + 1 == L - Sq + (g*S + j) + 1``.
    The last group therefore sees the full context and earlier groups a
    correspondingly shorter prefix -- no masking change is needed in the kernel.
    """
    span = seqlen_q // groups
    return seqlen_kv - (groups - 1 - group) * span


def pa_decode_paged_launch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen_k: Optional[torch.Tensor],
    softmax_scale: float,
    *,
    page_size: int,
    max_seqlen_kv: int,
    split_k: int = 0,
    output_dtype: Optional[torch.dtype] = None,
    query_groups: int = 1,
) -> torch.Tensor:
    """Paged-KV head-packed decode (short query blocks).

    The head-packed gfx950 kernel maps the MFMA M-axis to (query-token, head)
    pairs, so a decode shape fills the matrix core -- unlike the dualwave prefill
    kernel, whose M-axis is query rows and which therefore runs at 1/32 MFMA
    utilisation at Sq=1. Causal masking is applied per query token inside the
    kernel (bottom-right alignment).

    Shapes:
      q            [B, Sq, G, H_q, D]  (Sq bounded by the M-tile budget; see
                   MAX_M_TILES_BY_HEAD_DIM in pa_decode_gfx950)
      k/v_cache    [num_pages, page_size, H_kv, D]   (MSLK "linear" layout)
      block_table  [B, max_pages_per_seq] int32
      seqlen_k     [B] int32, or None for "all of max_seqlen_kv"

    ``max_seqlen_kv`` is required: deriving it from ``seqlen_k`` would be a
    device->host sync and is illegal under CUDA-graph capture.

    ``query_groups`` > 1 covers the query block in several shallower passes
    instead of one deep one. Each M-tile adds loop-carried accumulator, so depth
    costs occupancy (measured 9 waves/SIMD at one tile down to 2 at eight) and at
    D=128 spills past 6 tiles. Splitting into G passes uses ``Sq/G`` tokens each
    -- ``G`` times the KV traffic, but shallower tiles. Measured 1.5x-2.0x for
    G=2 where a single pass would spill. See `query_group_seqlen` for why this is exact.

    Raises ValueError for configurations the kernel cannot serve (GQA ratio > 16,
    D % 32, page_size % 32, non-gfx950) -- there is no generic paged fallback, so
    callers must gate before calling.
    """
    from .pa_decode_gfx950 import pa_decode_gfx950_launch

    Sq = q.shape[1]
    if query_groups <= 1 or Sq % query_groups != 0:
        # Unchanged path. Kept as an early return so grouping cannot perturb the
        # shapes that do not use it.
        return pa_decode_gfx950_launch(
            q,
            k_cache,
            v_cache,
            seqlen_k,
            softmax_scale,
            split_k,
            output_dtype,
            block_table=block_table,
            page_size=page_size,
            kv_max=max_seqlen_kv,
        )

    span = Sq // query_groups
    parts = []
    for g in range(query_groups):
        kv_g = query_group_seqlen(max_seqlen_kv, Sq, query_groups, g)
        # `.contiguous()` is load-bearing, not hygiene. The kernel's non-split
        # epilogue addresses the output with Q's strides, so it requires the two
        # to share a layout. A slice of the query block keeps the parent's
        # stride(0) (Sq*...) while the freshly allocated output has span*...,
        # and the mismatch silently corrupts every batch above the first.
        parts.append(
            pa_decode_gfx950_launch(
                q[:, g * span : (g + 1) * span].contiguous(),
                k_cache,
                v_cache,
                None if seqlen_k is None else seqlen_k - (Sq - (g + 1) * span),
                softmax_scale,
                split_k,
                output_dtype,
                block_table=block_table,
                page_size=page_size,
                kv_max=kv_g,
            )
        )
    return torch.cat(parts, dim=1)


# ── AOT interface ─────────────────────────────────────────────────────────────


AOT_ARCHS: List[str] = ["gfx942", "gfx950"]

# KV is f16/bf16 only; split-K path writes f32 partials, so out="f32" for sk>1.
_HEAD_SIZES = (64, 128, 256)
_KV_DTYPES = ("f16", "bf16")
_SPLIT_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# Paged decode is reached only from flydsl_flash_attn_func, whose native paged
# path is fixed at these (see _PAGED_PAGE_SIZE in flash_attn_interface.py).
_PAGED_HEAD_SIZES = (64, 128)
_PAGED_PAGE_SIZE = 64
# The head-packed gfx950 kernel compiles the GQA ratio in. Cover the common
# power-of-two ratios at a single query token; anything else JITs on first use.
_GFX950_AOT_GQA_RATIOS = (1, 2, 4, 8, 16)

AOT_CONFIGS: List[Dict[str, Any]] = [
    {
        "head_size": hs,
        "kv_dtype_str": kv,
        "output_dtype_str": ("f32" if sk > 1 else kv),
        "split_k": sk,
    }
    for hs in _HEAD_SIZES
    for kv in _KV_DTYPES
    for sk in _SPLIT_KS
]


def compile_aot_config(config: Dict[str, Any], arch: str) -> None:
    """Precompile one config. generic on every arch; gfx950 + coop only on gfx950."""
    from .pa_decode_generic import compile_pa_decode_generic

    hs = config["head_size"]
    kv = config["kv_dtype_str"]
    od = config["output_dtype_str"]
    sk = config["split_k"]

    compile_pa_decode_generic(
        head_size=hs,
        kv_dtype_str=kv,
        output_dtype_str=od,
        split_k=sk,
        arch=arch,
    )

    if arch.startswith("gfx950"):
        from .pa_decode_gfx950 import compile_pa_decode_gfx950
        from .pa_decode_gfx950_coop import compile_pa_decode_gfx950_coop

        compile_pa_decode_gfx950_coop(
            head_size=hs,
            kv_dtype_str=kv,
            output_dtype_str=od,
            split_k=sk,
            arch=arch,
        )
        for ratio in _GFX950_AOT_GQA_RATIOS:
            compile_pa_decode_gfx950(
                head_size=hs,
                kv_dtype_str=kv,
                output_dtype_str=od,
                split_k=sk,
                arch=arch,
                gqa_ratio=ratio,
            )
            # Paged variant (block-table KV), used by the flydsl_flash_attn_func
            # paged decode fast path. Only D=64/128 reach it -- 256 is dense-only.
            if hs in _PAGED_HEAD_SIZES:
                compile_pa_decode_gfx950(
                    head_size=hs,
                    kv_dtype_str=kv,
                    output_dtype_str=od,
                    split_k=sk,
                    arch=arch,
                    paged=True,
                    page_size=_PAGED_PAGE_SIZE,
                    gqa_ratio=ratio,
                )
