# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compress_kv_basic():
    """Mean-pool compression produces correct output shape and values."""
    from mslk.attention.sparse_attn.compress import compress_kv

    B, N, H_kv, D = 2, 1024, 4, 128
    block_size = 64
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

    K_cmp, V_cmp = compress_kv(K, V, block_size)

    N_cmp = N // block_size
    assert K_cmp.shape == (B, N_cmp, H_kv, D)
    assert V_cmp.shape == (B, N_cmp, H_kv, D)

    # Verify mean-pool: first block should be mean of first block_size positions
    expected_k0 = K[:, :block_size].float().mean(dim=1)  # (B, H_kv, D)
    actual_k0 = K_cmp[:, 0].float()
    torch.testing.assert_close(actual_k0, expected_k0, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compress_kv_with_projection():
    """Compression with W_k, W_v learned projections."""
    from mslk.attention.sparse_attn.compress import compress_kv

    B, N, H_kv, D = 1, 512, 2, 64
    block_size = 64
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    W_k = torch.randn(H_kv, D, D, device="cuda", dtype=torch.bfloat16)
    W_v = torch.randn(H_kv, D, D, device="cuda", dtype=torch.bfloat16)

    K_cmp, V_cmp = compress_kv(K, V, block_size, W_k, W_v)

    N_cmp = N // block_size
    assert K_cmp.shape == (B, N_cmp, H_kv, D)
    assert V_cmp.shape == (B, N_cmp, H_kv, D)

    # With projection, output should differ from simple mean
    K_cmp_no_proj, _ = compress_kv(K, V, block_size)
    assert not torch.allclose(K_cmp.float(), K_cmp_no_proj.float(), atol=1e-2)
