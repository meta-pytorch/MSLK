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


def auto_split_k_hp(B: int, G: int, H_q: int, H_kv: int, KV_MAX: int) -> int:
    """split_k for the head-packed gfx950 kernel: ~8 waves counted in warps (B*G*H_kv), cap 64."""
    n_cus = _get_cu_count()
    target_warps = n_cus * 8
    base_warps = B * G * H_kv
    needed = max(1, (target_warps + base_warps - 1) // base_warps)
    sk = 1
    while sk < needed:
        sk *= 2
    MIN_CHUNK_TOKENS = 64
    max_sk = max(1, KV_MAX // MIN_CHUNK_TOKENS)
    sk = min(sk, max_sk, 64)
    return max(1, sk)


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


# ── AOT interface ─────────────────────────────────────────────────────────────


AOT_ARCHS: List[str] = ["gfx942", "gfx950"]

# KV is f16/bf16 only; split-K path writes f32 partials, so out="f32" for sk>1.
_HEAD_SIZES = (64, 128, 256)
_KV_DTYPES = ("f16", "bf16")
_SPLIT_KS = (1, 2, 4, 8, 16, 32, 64)

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
        compile_pa_decode_gfx950(
            head_size=hs,
            kv_dtype_str=kv,
            output_dtype_str=od,
            split_k=sk,
            arch=arch,
        )
