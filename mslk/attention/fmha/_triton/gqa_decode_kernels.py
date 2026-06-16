# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

"""Triton kernels for GQA decode attention (ROCm-first, CUDA-capable)."""

from typing import List

import triton
import triton.language as tl

AUTOTUNER_KEY = [
    "Z",
    "G",
    "Hq",
    "Mq",
    "N_CTX_K",
    "BLOCK_DMODEL",
    "PAGE_SIZE",
]


@triton.jit
def _fwd_gqa_decode_kernel(
    Q,
    K,
    V,
    Out,
    Seq_len,
    block_tables,
    sm_scale,
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qk,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kk,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vk,
    stride_oz,
    stride_om,
    stride_og,
    stride_oh,
    stride_ok,
    stride_bt_batch,
    stride_bt_page,
    Z: tl.constexpr,
    G: tl.constexpr,
    Hq: tl.constexpr,
    Mq: tl.constexpr,
    N_CTX_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_PAGED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HQ_BLOCK: tl.constexpr,
):
    """
    Decode-phase causal attention for BMGHK layout.

    Launch grid: (batch Z, kv head group G). Each program handles all Hq query
    heads that share one KV head (typical GQA: H_kv=1, stored at head index 0).
    """
    tl.static_assert(
        not USE_PAGED or BLOCK_N <= PAGE_SIZE,
        "BLOCK_N must be <= PAGE_SIZE for paged attention: each tile must fit within one page",
    )
    off_b = tl.program_id(0)
    off_g = tl.program_id(1)

    kv_len = tl.load(Seq_len + off_b)
    if kv_len <= 0:
        return

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HQ_BLOCK)

    # KV head index 0: one logical KV head per group (standard GQA/MQA decode layout).
    k_base = K + off_g * stride_kg
    v_base = V + off_g * stride_vg
    if not USE_PAGED:
        k_base += off_b * stride_kz
        v_base += off_b * stride_vz

    num_kv_blocks = tl.cdiv(kv_len, BLOCK_N)

    for off_mq in tl.static_range(Mq):
        # Load all Hq query heads as a 2D tile [HQ_BLOCK, D].
        # Rows offs_h >= Hq are masked to 0 — they never influence real outputs.
        q_all = (
            tl.load(
                Q
                + off_b * stride_qz
                + off_mq * stride_qm
                + off_g * stride_qg
                + offs_h[:, None] * stride_qh
                + offs_d[None, :] * stride_qk,
                mask=(offs_h[:, None] < Hq) & (offs_d[None, :] < BLOCK_DMODEL),
                other=0.0,
            ).to(tl.float32)
            * sm_scale
        )  # [HQ_BLOCK, D]

        m_is = tl.full([HQ_BLOCK], -float("inf"), dtype=tl.float32)
        l_is = tl.zeros([HQ_BLOCK], dtype=tl.float32)
        accs = tl.zeros([HQ_BLOCK, BLOCK_DMODEL], dtype=tl.float32)

        for block_idx in range(num_kv_blocks):
            start_n = block_idx * BLOCK_N
            n_mask = offs_n < kv_len - start_n

            # KV loaded once per tile and reused across all Hq heads via tl.dot.
            if USE_PAGED:
                n_idx = start_n + offs_n
                page_idx = n_idx // PAGE_SIZE
                page_offset = n_idx % PAGE_SIZE
                physical_page = tl.load(
                    block_tables + off_b * stride_bt_batch + page_idx * stride_bt_page
                ).to(tl.int64)
                k = tl.load(
                    k_base
                    + physical_page[:, None] * PAGE_SIZE * stride_kn
                    + page_offset[:, None] * stride_kn
                    + offs_d[None, :] * stride_kk,
                    mask=n_mask[:, None] & (offs_d[None, :] < BLOCK_DMODEL),
                    other=0.0,
                )  # [BLOCK_N, D]
                v = tl.load(
                    v_base
                    + physical_page[:, None] * PAGE_SIZE * stride_vn
                    + page_offset[:, None] * stride_vn
                    + offs_d[None, :] * stride_vk,
                    mask=n_mask[:, None] & (offs_d[None, :] < BLOCK_DMODEL),
                    other=0.0,
                )  # [BLOCK_N, D]
            else:
                k_offs_n = start_n + offs_n
                k = tl.load(
                    k_base
                    + k_offs_n[:, None] * stride_kn
                    + offs_d[None, :] * stride_kk,
                    mask=n_mask[:, None] & (offs_d[None, :] < BLOCK_DMODEL),
                    other=0.0,
                )  # [BLOCK_N, D]
                v = tl.load(
                    v_base
                    + k_offs_n[:, None] * stride_vn
                    + offs_d[None, :] * stride_vk,
                    mask=n_mask[:, None] & (offs_d[None, :] < BLOCK_DMODEL),
                    other=0.0,
                )  # [BLOCK_N, D]

            # QK: [HQ_BLOCK, D] x [D, BLOCK_N] -> [HQ_BLOCK, BLOCK_N]
            # MFMA dims on ROCm: M=HQ_BLOCK>=16, K=D>=64, N=BLOCK_N>=16
            qk = tl.dot(
                q_all.to(Q.dtype.element_ty),
                tl.trans(k).to(Q.dtype.element_ty),
            ).to(
                tl.float32
            )  # [HQ_BLOCK, BLOCK_N]

            # Mask padded head rows and out-of-bounds KV positions.
            qk = tl.where(offs_h[:, None] < Hq, qk, float("-inf"))
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # Online softmax — 2D state, each row independent.
            m_new = tl.maximum(m_is, tl.max(qk, axis=1))  # [HQ_BLOCK]
            alpha = tl.exp(m_is - m_new)  # [HQ_BLOCK]
            p = tl.exp(qk - m_new[:, None])  # [HQ_BLOCK, BLOCK_N]
            l_is = alpha * l_is + tl.sum(p, axis=1)  # [HQ_BLOCK]

            # PV: [HQ_BLOCK, BLOCK_N] x [BLOCK_N, D] -> [HQ_BLOCK, D]
            # MFMA dims on ROCm: M=HQ_BLOCK>=16, K=BLOCK_N>=16, N=D>=64
            accs = accs * alpha[:, None] + tl.dot(
                p.to(Q.dtype.element_ty),
                v.to(Q.dtype.element_ty),
            )  # [HQ_BLOCK, D]
            m_is = m_new

        # Finalise: only real head rows (offs_h < Hq) are written.
        accs = accs / tl.maximum(l_is[:, None], 1e-6)
        tl.store(
            Out
            + off_b * stride_oz
            + off_mq * stride_om
            + off_g * stride_og
            + offs_h[:, None] * stride_oh
            + offs_d[None, :] * stride_ok,
            accs.to(Out.dtype.element_ty),
            mask=(offs_h[:, None] < Hq) & (offs_d[None, :] < BLOCK_DMODEL),
        )


def _build_autotune_configs() -> List[triton.Config]:
    configs: List[triton.Config] = []
    for block_n in (16, 32, 64, 128):
        for num_warps in (1, 2, 4, 8):
            for num_stages in (1, 2):
                configs.append(
                    triton.Config(
                        {"BLOCK_N": block_n},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


def _early_config_prune(configs, named_args, **kwargs):
    if kwargs.get("USE_PAGED", False):
        page_size = kwargs["PAGE_SIZE"]
        return [c for c in configs if page_size % c.kwargs["BLOCK_N"] == 0]
    return configs


_fwd_gqa_decode_autotune = triton.autotune(
    configs=_build_autotune_configs(),
    key=AUTOTUNER_KEY,
    prune_configs_by={"early_config_prune": _early_config_prune},
)(_fwd_gqa_decode_kernel)


def get_gqa_decode_kernel():
    return _fwd_gqa_decode_autotune
