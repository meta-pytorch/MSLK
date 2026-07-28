# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL decode dispatcher — public entry point for the decoder ops.

Targets gfx942 (CDNA3/MI300) and gfx950 (CDNA4/MI355), wave64. Compute lives in:
  * pa_decode_gfx950 — primary fast path (head-packed MFMA + double-buffered wide
    V load). gfx950, GQA ratio in [1,16].
  * pa_decode_gfx950_coop — per-head coop-DMA for ratios that can't head-pack.
  * pa_decode_generic — arch-generic fallback, off-gfx950.
Holds split_k heuristics, the launcher (dispatches to gfx950/coop, both self-fall
back to generic), and the AOT interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .utils import WARP_SIZE

NUM_WARPS  = 4
BLOCK_SIZE = NUM_WARPS * WARP_SIZE   # 256

# Approximate CU count for auto split_k. Will be updated at first launch.
_CU_COUNT: Optional[int] = None


def _get_cu_count() -> int:
    global _CU_COUNT
    if _CU_COUNT is None:
        try:
            prop = torch.cuda.get_device_properties(0)
            # multi_processor_count is exposed for both CUDA and ROCm
            _CU_COUNT = prop.multi_processor_count
        except Exception:
            _CU_COUNT = 120  # conservative default
    return _CU_COUNT


def auto_split_k(B: int, G: int, H_q: int, KV_MAX: int, num_warps: int = NUM_WARPS) -> int:
    """Default split_k: target ~4 waves (4× CU count CTAs) to hide memory latency.

    Total CTAs = B*G*Hq*sk. Tuned for the low-register generic fallback; the coop
    path uses auto_split_k_coop() which oversubscribes harder (latency-bound).
    """
    n_cus = _get_cu_count()
    target_ctas = n_cus * 4  # 4 waves
    base_ctas = B * G * H_q
    if base_ctas >= target_ctas:
        return 1
    needed = (target_ctas + base_ctas - 1) // base_ctas
    # Round up to power of 2, cap at 64
    sk = 1
    while sk < needed:
        sk *= 2
    # Ensure each partition has enough tokens to be meaningful
    min_toks_per_part = 64
    max_sk = max(1, KV_MAX // min_toks_per_part)
    sk = min(sk, max_sk, 64)
    return max(1, sk)


def auto_split_k_coop(B: int, G: int, H_q: int, KV_MAX: int) -> int:
    """split_k for the gfx950 coop-DMA kernel: deeper than the generic default.

    Its large per-lane PV accumulator (_D_CHUNKS×16 f32 regs) makes it latency-bound
    under high VGPR pressure, so it wants far deeper oversubscription. Fit to a
    rocprof GPU-kernel-time sweep (NOT wall-clock — masked by ~20-30us dispatch on
    these small kernels): ~8 waves, keep splitting to ~64-token partitions, cap 64.
    Within 2% of per-shape optimum (avg 1.002x, worst 1.02x).
    """
    n_cus = _get_cu_count()
    target_ctas = n_cus * 8  # 8 waves
    base_ctas = B * G * H_q
    needed = max(1, (target_ctas + base_ctas - 1) // base_ctas)
    sk = 1
    while sk < needed:
        sk *= 2
    # Keep splitting to ~64-token partitions (coop stays latency-bound past CU
    # saturation), cap 64. Reduce pass is cheap (~3us).
    MIN_CHUNK_TOKENS = 64
    max_sk = max(1, KV_MAX // MIN_CHUNK_TOKENS)
    sk = min(sk, max_sk, 64)
    return max(1, sk)


def auto_split_k_hp(B: int, G: int, H_q: int, H_kv: int, KV_MAX: int) -> int:
    """split_k for the head-packed gfx950 kernel.

    Head-packing puts a whole GQA group in ONE warp/CTA, launching only B*G*H_kv
    CTAs (ratio× fewer than coop), so it must lean harder on split_k: target ~8 waves
    counted in WARPS (B*G*H_kv*sk), not coop's B*G*H_q*sk CTAs. Rocprof-fit hits the
    per-shape optimum (1.00x) at sk=32-64 on B=8 shapes.
    """
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
    B = Q.shape[0]
    KV = K.shape[1]
    ratio = H_q // H_kv if H_kv > 0 else 0
    use_hp = (H_kv > 0 and H_q % H_kv == 0 and 1 <= ratio <= 16)
    if use_hp:
        from .pa_decode_gfx950 import pa_decode_gfx950_launch
        return pa_decode_gfx950_launch(Q, K, V, seq_positions, softmax_scale, split_k, output_dtype)
    from .pa_decode_gfx950_coop import pa_decode_gfx950_coop_launch
    return pa_decode_gfx950_coop_launch(Q, K, V, seq_positions, softmax_scale, split_k, output_dtype)


# ── AOT interface ─────────────────────────────────────────────────────────────


AOT_ARCHS: List[str] = ["gfx942", "gfx950"]

# Precompiled cache grid. KV is f16/bf16 only (no f32 KV support); the split-K path
# writes f32 partials, so out="f32" is used for sk>1.
_HEAD_SIZES = (64, 128, 256)
_KV_DTYPES = ("f16", "bf16")
_SPLIT_KS = (1, 2, 4, 8, 16, 32, 64)

AOT_CONFIGS: List[Dict[str, Any]] = [
    {
        "head_size":        hs,
        "kv_dtype_str":     kv,
        "output_dtype_str": ("f32" if sk > 1 else kv),
        "split_k":          sk,
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
        head_size=hs, kv_dtype_str=kv, output_dtype_str=od, split_k=sk, arch=arch,
    )

    if arch.startswith("gfx950"):
        from .pa_decode_gfx950_coop import compile_pa_decode_gfx950_coop
        from .pa_decode_gfx950 import compile_pa_decode_gfx950

        # coop = small-shape fallback; gfx950 = primary head-packed fast path.
        compile_pa_decode_gfx950_coop(
            head_size=hs, kv_dtype_str=kv, output_dtype_str=od, split_k=sk, arch=arch,
        )
        compile_pa_decode_gfx950(
            head_size=hs, kv_dtype_str=kv, output_dtype_str=od, split_k=sk, arch=arch,
        )
