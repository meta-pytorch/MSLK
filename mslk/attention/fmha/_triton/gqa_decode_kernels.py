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
    KV_BLOCKS_PER_ROW: tl.constexpr,
):
    """
    Decode-phase causal attention for BMGHK layout.

    Launch grid: (batch Z, kv head group G). Each program handles all Hq query
    heads that share one KV head (typical GQA: H_kv=1, stored at head index 0).
    """
    off_b = tl.program_id(0)
    off_g = tl.program_id(1)

    kv_len = tl.load(Seq_len + off_b)
    if kv_len <= 0:
        return

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    # KV head index 0: one logical KV head per group (standard GQA/MQA decode layout).
    k_base = K + off_g * stride_kg
    v_base = V + off_g * stride_vg
    if not USE_PAGED:
        k_base += off_b * stride_kz
        v_base += off_b * stride_vz

    num_kv_blocks = tl.cdiv(kv_len, BLOCK_N)

    for off_hq in tl.static_range(Hq):
        for off_mq in tl.static_range(Mq):
            q_ptr = (
                Q
                + off_b * stride_qz
                + off_mq * stride_qm
                + off_g * stride_qg
                + off_hq * stride_qh
            )
            q = tl.load(
                q_ptr + offs_d * stride_qk,
                mask=offs_d < BLOCK_DMODEL,
                other=0.0,
            ).to(tl.float32)
            q = q * sm_scale

            m_i = tl.full([], -float("inf"), dtype=tl.float32)
            l_i = tl.full([], 0.0, dtype=tl.float32)
            acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

            for block_idx in range(num_kv_blocks):
                start_n = block_idx * BLOCK_N
                n_mask = (offs_n < kv_len - start_n) & (offs_n >= 0)

                if USE_PAGED:
                    n_idx = start_n + offs_n
                    page_idx = n_idx // PAGE_SIZE
                    page_offset = n_idx % PAGE_SIZE
                    physical_page = tl.load(
                        block_tables
                        + off_b * stride_bt_batch
                        + page_idx * stride_bt_page
                    ).to(tl.int64)
                    k = tl.load(
                        k_base
                        + physical_page * PAGE_SIZE * stride_kn
                        + page_offset[:, None] * stride_kn
                        + offs_d[None, :] * stride_kk,
                        mask=n_mask[:, None] & (offs_d[None, :] < BLOCK_DMODEL),
                        other=0.0,
                    ).to(tl.float32)
                    v = tl.load(
                        v_base
                        + physical_page * PAGE_SIZE * stride_vn
                        + page_offset[:, None] * stride_vn
                        + offs_d[None, :] * stride_vk,
                        mask=n_mask[:, None] & (offs_d[None, :] < BLOCK_DMODEL),
                        other=0.0,
                    ).to(tl.float32)
                else:
                    k_offs_n = start_n + offs_n
                    k = tl.load(
                        k_base
                        + k_offs_n[:, None] * stride_kn
                        + offs_d[None, :] * stride_kk,
                        mask=n_mask[:, None] & (offs_d[None, :] < BLOCK_DMODEL),
                        other=0.0,
                    ).to(tl.float32)
                    v = tl.load(
                        v_base
                        + k_offs_n[:, None] * stride_vn
                        + offs_d[None, :] * stride_vk,
                        mask=n_mask[:, None] & (offs_d[None, :] < BLOCK_DMODEL),
                        other=0.0,
                    ).to(tl.float32)

                qk = tl.sum(q[None, :] * k, axis=1)
                qk = tl.where(n_mask, qk, float("-inf"))

                m_ij = tl.max(qk, axis=0)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new)
                l_new = alpha * l_i + tl.sum(p, axis=0)

                acc = acc * alpha
                acc += tl.sum(p[:, None] * v, axis=0)

                l_i = l_new
                m_i = m_new

            acc = acc / tl.maximum(l_i, 1e-6)

            out_ptr = (
                Out
                + off_b * stride_oz
                + off_mq * stride_om
                + off_g * stride_og
                + off_hq * stride_oh
            )
            tl.store(
                out_ptr + offs_d * stride_ok,
                acc.to(Out.dtype.element_ty),
                mask=offs_d < BLOCK_DMODEL,
            )


def _build_autotune_configs() -> List[triton.Config]:
    configs: List[triton.Config] = []
    for block_n in (32, 64, 128):
        for num_warps in (2, 4):
            for num_stages in (1, 2):
                configs.append(
                    triton.Config(
                        {"BLOCK_N": block_n},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


_fwd_gqa_decode_autotune = triton.autotune(
    configs=_build_autotune_configs(),
    key=AUTOTUNER_KEY,
)(_fwd_gqa_decode_kernel)


def get_gqa_decode_kernel():
    return _fwd_gqa_decode_autotune
