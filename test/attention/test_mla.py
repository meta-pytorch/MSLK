# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness tests for Triton MLA (Multi-head Latent Attention) kernels.

Tests MLA decode (split-K) and prefill (single-stage) against a PyTorch
CPU/GPU reference. DeepSeek-V3 dimensions: num_heads=128, num_kv_heads=1,
kv_lora_rank=512, qk_rope_head_dim=64, qk_head_dim=576, v_head_dim=512.

Hardware: AMD MI300X (gfx942) and MI350X (gfx950).
"""

import math

import pytest
import torch

# Skip entire module if not on ROCm
pytestmark = pytest.mark.skipif(
    torch.version.hip is None, reason="MLA Triton kernels require ROCm"
)

# MLA architecture constants (DeepSeek-V3, post weight-absorption)
NUM_HEADS = 128
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK  # 512

COSINE_TOL = 3e-2


def _detect_fp8_dtype() -> torch.dtype:
    """Return the native FP8 dtype for the current device."""
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx942" in arch:
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _ref_mla_decode(
    q_bf16: torch.Tensor,
    kv_master_bf16: torch.Tensor,
    sm_scale: float,
    seqused_k: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference: MLA decode in FP32, supports variable seq lengths.

    Args:
        q_bf16: (batch, num_heads, qk_head_dim) bf16
        kv_master_bf16: (batch, max_ctx_len, qk_head_dim) bf16
        seqused_k: (batch,) int — actual KV length per sequence
    Returns:
        (batch, num_heads, v_head_dim) bf16
    """
    batch = q_bf16.shape[0]
    outs = []
    for i in range(batch):
        seq_len = int(seqused_k[i].item())
        q = q_bf16[i].float().unsqueeze(0)  # (1, H, D)
        k = kv_master_bf16[i, :seq_len].float().unsqueeze(0)  # (1, S, D)
        v = k[..., :V_HEAD_DIM]

        scores = torch.einsum("bhd,bkd->bhk", q, k) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhk,bkd->bhd", probs, v)
        outs.append(out)
    return torch.cat(outs, dim=0).to(torch.bfloat16)


def _ref_mla_prefill(
    q_bf16: torch.Tensor,
    kv_master_bf16: torch.Tensor,
    sm_scale: float,
    qlen: int,
) -> torch.Tensor:
    """PyTorch reference: MLA prefill with causal masking in FP32.

    Args:
        q_bf16: (batch * qlen, num_heads, qk_head_dim) bf16
        kv_master_bf16: (batch, ctx_len, qk_head_dim) bf16
    Returns:
        (batch * qlen, num_heads, v_head_dim) bf16
    """
    batch, ctx_len, _ = kv_master_bf16.shape
    q = q_bf16.view(batch, qlen, NUM_HEADS, QK_HEAD_DIM).float()
    k = kv_master_bf16.float()
    v = k[..., :V_HEAD_DIM]

    scores = torch.einsum("bqhd,bkd->bqhk", q, k) * sm_scale
    mask = torch.ones(qlen, ctx_len, dtype=torch.bool, device=q.device).tril(
        diagonal=ctx_len - qlen
    )
    scores.masked_fill_(mask[None, :, None, :].logical_not(), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bqhk,bkd->bqhd", probs, v)
    return out.reshape(batch * qlen, NUM_HEADS, V_HEAD_DIM).to(torch.bfloat16)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    return float(
        1
        - 2
        * (a_f * b_f).sum().item()
        / max((a_f * a_f + b_f * b_f).sum().item(), 1e-12)
    )


def _build_decode_inputs(
    batch: int,
    ctx_len: int,
    page_size: int,
    seed: int = 42,
    variable_seqlens: bool = False,
):
    """Build decode inputs with paged KV cache.

    If variable_seqlens is True, each sequence in the batch gets a different
    KV length (multiples of page_size, between page_size and ctx_len).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    blocks_per_seq = ctx_len // page_size

    if variable_seqlens and batch > 1:
        seq_lens = []
        for i in range(batch):
            n_blocks = max(1, (i + 1) * blocks_per_seq // batch)
            seq_lens.append(n_blocks * page_size)
        seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        max_blocks = blocks_per_seq
    else:
        seqused_k = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")
        max_blocks = blocks_per_seq

    num_pages = batch * max_blocks

    kv_master_bf16 = (
        torch.randn(batch, ctx_len, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.1
    )
    q_bf16 = (
        torch.randn(batch, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.1
    )

    page_perm = torch.randperm(num_pages, dtype=torch.int32, device="cuda")
    block_tables = page_perm.view(batch, max_blocks)

    kv_buffer_bf16 = torch.zeros(
        num_pages,
        page_size,
        NUM_KV_HEADS,
        QK_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for i in range(batch):
        seq_len = int(seqused_k[i].item())
        n_blocks = seq_len // page_size
        for b in range(n_blocks):
            phys = int(block_tables[i, b].item())
            start = b * page_size
            end = start + page_size
            kv_buffer_bf16[phys, :, 0, :] = kv_master_bf16[i, start:end, :]

    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / math.sqrt(QK_HEAD_DIM)

    return {
        "q_bf16": q_bf16,
        "kv_buffer_bf16": kv_buffer_bf16,
        "kv_master_bf16": kv_master_bf16,
        "block_tables": block_tables,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "sm_scale": sm_scale,
    }


def _build_prefill_inputs(
    batch: int, ctx_len: int, qlen: int, page_size: int, seed: int = 42
):
    """Build prefill inputs with paged KV cache."""
    blocks_per_seq = ctx_len // page_size
    num_pages = batch * blocks_per_seq

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    kv_master_bf16 = (
        torch.randn(batch, ctx_len, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.1
    )
    q_bf16 = (
        torch.randn(
            batch * qlen, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        * 0.1
    )

    page_perm = torch.randperm(num_pages, dtype=torch.int32, device="cuda")
    block_tables = page_perm.view(batch, blocks_per_seq)

    kv_pages_logical = kv_master_bf16.view(
        batch, blocks_per_seq, page_size, QK_HEAD_DIM
    ).reshape(num_pages, page_size, QK_HEAD_DIM)

    kv_buffer_bf16 = torch.empty(
        num_pages,
        page_size,
        NUM_KV_HEADS,
        QK_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    flat_block_tables = block_tables.reshape(-1).long()
    kv_buffer_bf16[flat_block_tables, :, 0, :] = kv_pages_logical

    cu_seqlens_q = torch.arange(
        0, batch * qlen + 1, qlen, dtype=torch.int32, device="cuda"
    )
    seqused_k = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")

    sm_scale = 1.0 / math.sqrt(QK_HEAD_DIM)

    return {
        "q_bf16": q_bf16,
        "kv_buffer_bf16": kv_buffer_bf16,
        "kv_master_bf16": kv_master_bf16,
        "block_tables": block_tables,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "sm_scale": sm_scale,
        "qlen": qlen,
    }


# ===================================================================
#  DECODE TESTS
# ===================================================================


class TestMLADecode:
    @pytest.mark.parametrize("batch", [1, 16, 64])
    @pytest.mark.parametrize("ctx_len", [1024, 4096])
    def test_decode_bf16(self, batch: int, ctx_len: int) -> None:
        """Test MLA decode with BF16 inputs against reference."""
        from mslk.attention.mla import mla_decode_fwd

        page_size = 1
        inp = _build_decode_inputs(batch, ctx_len, page_size)

        out = mla_decode_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_decode(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], inp["seqused_k"]
        )

        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Decode BF16 cosine distance {cos_dist:.6f} >= {COSINE_TOL} "
            f"(batch={batch}, ctx_len={ctx_len})"
        )

    @pytest.mark.parametrize("batch", [1, 16])
    @pytest.mark.parametrize("ctx_len", [1024, 4096])
    def test_decode_fp8(self, batch: int, ctx_len: int) -> None:
        """Test MLA decode with FP8 quantized inputs."""
        from mslk.attention.mla import mla_decode_fwd

        page_size = 1
        inp = _build_decode_inputs(batch, ctx_len, page_size)

        fp8_dtype = _detect_fp8_dtype()
        q_fp8 = inp["q_bf16"].to(fp8_dtype)
        kv_fp8 = inp["kv_buffer_bf16"].to(fp8_dtype)

        out = mla_decode_fwd(
            query=q_fp8,
            kv_buffer=kv_fp8,
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_decode(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], inp["seqused_k"]
        )

        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Decode FP8 cosine distance {cos_dist:.6f} >= {COSINE_TOL} "
            f"(batch={batch}, ctx_len={ctx_len})"
        )

    @pytest.mark.parametrize("ctx_len", [64, 128, 256])
    def test_decode_short_context(self, ctx_len: int) -> None:
        """Test decode with short contexts near tile boundaries."""
        from mslk.attention.mla import mla_decode_fwd

        batch, page_size = 4, 1
        inp = _build_decode_inputs(batch, ctx_len, page_size)

        out = mla_decode_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_decode(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], inp["seqused_k"]
        )

        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Decode short-ctx cosine distance {cos_dist:.6f} >= {COSINE_TOL} "
            f"(ctx_len={ctx_len})"
        )

    def test_decode_long_context(self) -> None:
        """Test decode at ctx_len=8192 (backlog spec shape)."""
        from mslk.attention.mla import mla_decode_fwd

        batch, ctx_len, page_size = 4, 8192, 1
        inp = _build_decode_inputs(batch, ctx_len, page_size)

        out = mla_decode_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_decode(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], inp["seqused_k"]
        )

        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Decode long-ctx cosine distance {cos_dist:.6f} >= {COSINE_TOL}"
        )

    def test_decode_variable_seqlens(self) -> None:
        """Test decode where each batch element has a different KV length."""
        from mslk.attention.mla import mla_decode_fwd

        batch, ctx_len, page_size = 8, 1024, 1
        inp = _build_decode_inputs(batch, ctx_len, page_size, variable_seqlens=True)

        out = mla_decode_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_decode(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], inp["seqused_k"]
        )

        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Decode variable-seqlen cosine distance {cos_dist:.6f} >= {COSINE_TOL}"
        )

    def test_decode_output_shape(self) -> None:
        """Verify output tensor shape matches (batch, num_heads, v_head_dim)."""
        from mslk.attention.mla import mla_decode_fwd

        batch, ctx_len, page_size = 8, 512, 1
        inp = _build_decode_inputs(batch, ctx_len, page_size)

        out = mla_decode_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        assert out.shape == (
            batch,
            NUM_HEADS,
            V_HEAD_DIM,
        ), f"Expected shape ({batch}, {NUM_HEADS}, {V_HEAD_DIM}), got {out.shape}"
        assert out.dtype == torch.bfloat16

    def test_decode_reproducibility(self) -> None:
        """Verify decode gives the same output across two runs with same seed."""
        from mslk.attention.mla import mla_decode_fwd

        batch, ctx_len, page_size = 4, 1024, 1

        inp1 = _build_decode_inputs(batch, ctx_len, page_size, seed=99)
        out1 = mla_decode_fwd(
            inp1["q_bf16"],
            inp1["kv_buffer_bf16"],
            inp1["block_tables"],
            inp1["cu_seqlens_q"],
            inp1["seqused_k"],
            inp1["sm_scale"],
        )

        inp2 = _build_decode_inputs(batch, ctx_len, page_size, seed=99)
        out2 = mla_decode_fwd(
            inp2["q_bf16"],
            inp2["kv_buffer_bf16"],
            inp2["block_tables"],
            inp2["cu_seqlens_q"],
            inp2["seqused_k"],
            inp2["sm_scale"],
        )

        assert torch.equal(out1, out2), "Decode is not deterministic across runs"


# ===================================================================
#  PREFILL TESTS
# ===================================================================


class TestMLAPrefill:
    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("ctx_len", [1024, 4096])
    def test_prefill_bf16(self, batch: int, ctx_len: int) -> None:
        """Test MLA prefill with BF16 inputs against reference."""
        from mslk.attention.mla import mla_prefill_fwd

        page_size = 16
        qlen = ctx_len
        inp = _build_prefill_inputs(batch, ctx_len, qlen, page_size)

        out = mla_prefill_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_prefill(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], qlen
        )

        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Prefill BF16 cosine distance {cos_dist:.6f} >= {COSINE_TOL} "
            f"(batch={batch}, ctx_len={ctx_len})"
        )

    @pytest.mark.parametrize("ctx_len", [1024, 4096])
    def test_prefill_fp8(self, ctx_len: int) -> None:
        """Test MLA prefill with FP8 quantized inputs."""
        from mslk.attention.mla import mla_prefill_fwd

        batch, page_size = 1, 16
        qlen = ctx_len
        inp = _build_prefill_inputs(batch, ctx_len, qlen, page_size)

        fp8_dtype = _detect_fp8_dtype()
        q_fp8 = inp["q_bf16"].to(fp8_dtype)
        kv_fp8 = inp["kv_buffer_bf16"].to(fp8_dtype)

        out = mla_prefill_fwd(
            query=q_fp8,
            kv_buffer=kv_fp8,
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_prefill(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], qlen
        )

        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Prefill FP8 cosine distance {cos_dist:.6f} >= {COSINE_TOL} "
            f"(ctx_len={ctx_len})"
        )

    @pytest.mark.parametrize("ctx_len", [64, 128, 256])
    def test_prefill_short_context(self, ctx_len: int) -> None:
        """Test prefill with short contexts near tile boundaries."""
        from mslk.attention.mla import mla_prefill_fwd

        batch, page_size = 1, 16
        qlen = ctx_len
        inp = _build_prefill_inputs(batch, ctx_len, qlen, page_size)

        out = mla_prefill_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_prefill(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], qlen
        )

        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Prefill short-ctx cosine distance {cos_dist:.6f} >= {COSINE_TOL} "
            f"(ctx_len={ctx_len})"
        )

    def test_prefill_output_shape(self) -> None:
        """Verify output tensor shape matches (total_tokens, num_heads, v_head_dim)."""
        from mslk.attention.mla import mla_prefill_fwd

        batch, ctx_len, page_size = 1, 512, 16
        qlen = ctx_len
        inp = _build_prefill_inputs(batch, ctx_len, qlen, page_size)

        out = mla_prefill_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        assert out.shape == (
            batch * qlen,
            NUM_HEADS,
            V_HEAD_DIM,
        ), (
            f"Expected shape ({batch * qlen}, {NUM_HEADS}, {V_HEAD_DIM}), got {out.shape}"
        )
        assert out.dtype == torch.bfloat16


# ===================================================================
#  CROSS-KERNEL CONSISTENCY
# ===================================================================


class TestMLACrossKernel:
    def test_decode_prefill_consistency(self) -> None:
        """For qlen=1, prefill should produce the same output as decode."""
        from mslk.attention.mla import mla_decode_fwd, mla_prefill_fwd

        batch, ctx_len = 4, 512
        page_size_decode = 1
        page_size_prefill = 16

        torch.manual_seed(77)
        torch.cuda.manual_seed(77)

        kv_master = (
            torch.randn(
                batch, ctx_len, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
            )
            * 0.1
        )
        q = (
            torch.randn(
                batch, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
            )
            * 0.1
        )
        sm_scale = 1.0 / math.sqrt(QK_HEAD_DIM)
        seqused_k = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")

        # Build decode inputs (page_size=1)
        num_pages_d = batch * ctx_len
        kv_buf_d = torch.empty(
            num_pages_d,
            page_size_decode,
            NUM_KV_HEADS,
            QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        bt_d = torch.arange(num_pages_d, dtype=torch.int32, device="cuda").view(
            batch, ctx_len
        )
        for i in range(batch):
            for t in range(ctx_len):
                kv_buf_d[i * ctx_len + t, 0, 0, :] = kv_master[i, t, :]
        cu_d = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda")

        out_decode = mla_decode_fwd(q, kv_buf_d, bt_d, cu_d, seqused_k, sm_scale)

        # Build prefill inputs (page_size=16, qlen=1)
        blocks_per_seq = ctx_len // page_size_prefill
        num_pages_p = batch * blocks_per_seq
        kv_buf_p = torch.empty(
            num_pages_p,
            page_size_prefill,
            NUM_KV_HEADS,
            QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        bt_p = torch.arange(num_pages_p, dtype=torch.int32, device="cuda").view(
            batch, blocks_per_seq
        )
        for i in range(batch):
            for b_idx in range(blocks_per_seq):
                start = b_idx * page_size_prefill
                kv_buf_p[i * blocks_per_seq + b_idx, :, 0, :] = kv_master[
                    i, start : start + page_size_prefill, :
                ]
        q_prefill = q.view(batch, NUM_HEADS, QK_HEAD_DIM)
        cu_p = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda")

        out_prefill = mla_prefill_fwd(
            q_prefill,
            kv_buf_p,
            bt_p,
            cu_p,
            seqused_k,
            sm_scale,
        )

        # Both should agree with the reference
        ref = _ref_mla_decode(q, kv_master, sm_scale, seqused_k)

        cos_decode = _cosine_distance(out_decode, ref)
        cos_prefill = _cosine_distance(out_prefill, ref)
        assert cos_decode < COSINE_TOL, f"Decode vs ref: {cos_decode:.6f}"
        assert cos_prefill < COSINE_TOL, f"Prefill vs ref: {cos_prefill:.6f}"


# ===================================================================
#  EDGE CASES AND PRODUCTION SCENARIOS
# ===================================================================


class TestMLAEdgeCases:
    # --- Odd batch sizes (not powers of 2) ---

    @pytest.mark.parametrize("batch", [3, 7, 13])
    def test_decode_odd_batch(self, batch: int) -> None:
        """Odd batch sizes can expose grid/tiling alignment bugs."""
        from mslk.attention.mla import mla_decode_fwd

        ctx_len, page_size = 512, 1
        inp = _build_decode_inputs(batch, ctx_len, page_size)

        out = mla_decode_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_decode(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], inp["seqused_k"]
        )
        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Odd batch={batch} cosine distance {cos_dist:.6f} >= {COSINE_TOL}"
        )

    @pytest.mark.parametrize("batch", [3, 7])
    def test_prefill_odd_batch(self, batch: int) -> None:
        """Odd batch sizes for prefill."""
        from mslk.attention.mla import mla_prefill_fwd

        ctx_len, page_size = 256, 16
        qlen = ctx_len
        inp = _build_prefill_inputs(batch, ctx_len, qlen, page_size)

        out = mla_prefill_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_prefill(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], qlen
        )
        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Odd batch={batch} prefill cosine distance {cos_dist:.6f} >= {COSINE_TOL}"
        )

    # --- Non-aligned context lengths ---

    @pytest.mark.parametrize("ctx_len", [100, 500, 1000, 1023])
    def test_decode_nonaligned_ctx(self, ctx_len: int) -> None:
        """Context lengths that don't divide evenly by BLOCK_N or page_size."""
        from mslk.attention.mla import mla_decode_fwd

        batch, page_size = 4, 1
        inp = _build_decode_inputs(batch, ctx_len, page_size)

        out = mla_decode_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_decode(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], inp["seqused_k"]
        )
        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Non-aligned ctx_len={ctx_len} cosine distance {cos_dist:.6f} >= {COSINE_TOL}"
        )

    # --- Large batch decode ---

    @pytest.mark.parametrize("batch", [128, 256])
    def test_decode_large_batch(self, batch: int) -> None:
        """Large batches typical of high-throughput DeepSeek serving."""
        from mslk.attention.mla import mla_decode_fwd

        ctx_len, page_size = 512, 1
        inp = _build_decode_inputs(batch, ctx_len, page_size)

        out = mla_decode_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_decode(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], inp["seqused_k"]
        )
        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Large batch={batch} cosine distance {cos_dist:.6f} >= {COSINE_TOL}"
        )

    # --- Chunked prefill (qlen < ctx_len) ---

    @pytest.mark.parametrize(
        "ctx_len,qlen",
        [
            (1024, 256),  # 768 cached + 256 new
            (1024, 128),  # 896 cached + 128 new
            (2048, 512),  # 1536 cached + 512 new
            (512, 16),  # 496 cached + 16 new (minimal chunk)
        ],
    )
    def test_prefill_chunked(self, ctx_len: int, qlen: int) -> None:
        """Chunked prefill: qlen < ctx_len, prior context is cached.

        In real serving, a prompt may be processed in chunks. The kernel
        sees the full KV cache (ctx_len tokens) but only qlen query tokens
        at the tail end. The causal mask ensures query token qi attends to
        KV positions 0..(ctx_len - qlen + qi).
        """
        from mslk.attention.mla import mla_prefill_fwd

        batch, page_size = 1, 16
        inp = _build_prefill_inputs(batch, ctx_len, qlen, page_size)

        out = mla_prefill_fwd(
            query=inp["q_bf16"],
            kv_buffer=inp["kv_buffer_bf16"],
            block_tables=inp["block_tables"],
            cu_seqlens_q=inp["cu_seqlens_q"],
            seqused_k=inp["seqused_k"],
            softmax_scale=inp["sm_scale"],
        )

        ref = _ref_mla_prefill(
            inp["q_bf16"], inp["kv_master_bf16"], inp["sm_scale"], qlen
        )

        cos_dist = _cosine_distance(out, ref)
        assert cos_dist < COSINE_TOL, (
            f"Chunked prefill (ctx={ctx_len}, qlen={qlen}) "
            f"cosine distance {cos_dist:.6f} >= {COSINE_TOL}"
        )
