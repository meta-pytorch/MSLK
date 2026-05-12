# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nsa_forward_reference_basic():
    """Reference forward produces valid output shape."""
    from mslk.attention.sparse_attn.reference import nsa_forward_reference

    B, N, H, D = 1, 512, 4, 128
    H_kv = 2
    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

    O = nsa_forward_reference(Q, K, V, compress_block_size=64, num_selected_blocks=4)
    assert O.shape == Q.shape
    assert O.dtype == Q.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nsa_forward_reference_with_weights():
    """Reference forward with learned compression + gating weights."""
    from mslk.attention.sparse_attn.reference import nsa_forward_reference

    B, N, H, D = 1, 512, 4, 128
    H_kv = 2
    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    W_k = torch.randn(H_kv, D, D, device="cuda", dtype=torch.bfloat16)
    W_v = torch.randn(H_kv, D, D, device="cuda", dtype=torch.bfloat16)
    gate_w = torch.randn(H, 3, D, device="cuda", dtype=torch.bfloat16)

    O = nsa_forward_reference(
        Q,
        K,
        V,
        W_k_compress=W_k,
        W_v_compress=W_v,
        gate_proj_weight=gate_w,
    )
    assert O.shape == Q.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nsa_backward_reference():
    """Reference backward produces gradients for all inputs."""
    from mslk.attention.sparse_attn.reference import (
        compute_block_indices_reference,
        nsa_backward_reference,
    )

    B, N, H, D = 1, 512, 4, 128
    H_kv = 2
    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.float32)
    dO = torch.randn(B, N, H, D, device="cuda", dtype=torch.float32)
    W_k = torch.randn(H_kv, D, D, device="cuda", dtype=torch.float32)
    gate_w = torch.randn(H, 3, D, device="cuda", dtype=torch.float32)

    block_indices = compute_block_indices_reference(Q, K, W_k_compress=W_k)
    grads = nsa_backward_reference(
        Q,
        K,
        V,
        dO,
        W_k_compress=W_k,
        gate_proj_weight=gate_w,
        _block_indices=block_indices,
    )

    assert grads["dQ"].shape == Q.shape
    assert grads["dK"].shape == K.shape
    assert grads["dV"].shape == V.shape
    assert grads["dW_k_compress"].shape == W_k.shape
    assert grads["dgate_proj_weight"].shape == gate_w.shape
