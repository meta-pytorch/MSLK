# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nsa_forward_basic():
    """NSA forward produces valid output."""
    from mslk.attention.sparse_attn.nsa_forward import nsa_forward

    B, N, H, D = 1, 1024, 4, 128
    H_kv = 2
    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

    O = nsa_forward(Q, K, V, compress_block_size=64, num_selected_blocks=8)
    assert O.shape == Q.shape
    assert O.dtype == Q.dtype
    assert not torch.isnan(O).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nsa_forward_matches_reference():
    """NSA forward output is close to pure PyTorch reference."""
    from mslk.attention.sparse_attn.nsa_forward import nsa_forward
    from mslk.attention.sparse_attn.reference import nsa_forward_reference

    B, N, H, D = 1, 512, 4, 128
    H_kv = 2
    torch.manual_seed(42)
    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

    O_opt = nsa_forward(Q, K, V, compress_block_size=64, num_selected_blocks=8)
    O_ref = nsa_forward_reference(
        Q, K, V, compress_block_size=64, num_selected_blocks=8
    )

    # Tolerance is loose because FA4 and reference use different attention implementations
    torch.testing.assert_close(O_opt.float(), O_ref.float(), atol=0.1, rtol=0.1)
