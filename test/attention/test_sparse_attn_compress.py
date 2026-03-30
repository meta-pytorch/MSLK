# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for KV compression."""

import pytest
import torch


class TestCompressKV:
    """Test KV compression with mean-pooling and projection."""

    def test_basic_shape(self) -> None:
        """Compressed output has correct shape."""
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H_kv, D = 2, 1024, 4, 64
        block_size = 64
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, V_cmp = compress_kv(K, V, block_size)
        assert K_cmp.shape == (B, N // block_size, H_kv, D)
        assert V_cmp.shape == (B, N // block_size, H_kv, D)

    def test_mean_pool_correctness(self) -> None:
        """Mean-pool matches manual reshape + mean."""
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H_kv, D = 2, 256, 2, 32
        block_size = 64
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.float32)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.float32)

        K_cmp, V_cmp = compress_kv(K, V, block_size)

        # Manual mean pool
        K_expected = K.reshape(B, N // block_size, block_size, H_kv, D).mean(dim=2)
        V_expected = V.reshape(B, N // block_size, block_size, H_kv, D).mean(dim=2)

        torch.testing.assert_close(K_cmp, K_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(V_cmp, V_expected, atol=1e-5, rtol=1e-5)

    def test_with_projection(self) -> None:
        """Projection applies correctly after mean-pooling."""
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H_kv, D = 2, 256, 2, 32
        block_size = 64
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.float32)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.float32)
        W_k = torch.randn(H_kv, D, D, device="cuda", dtype=torch.float32)
        W_v = torch.randn(H_kv, D, D, device="cuda", dtype=torch.float32)

        K_cmp, V_cmp = compress_kv(K, V, block_size, W_k, W_v)

        # Manual: mean pool then project
        K_pooled = K.reshape(B, N // block_size, block_size, H_kv, D).mean(dim=2)
        K_expected = torch.einsum("bnhd,hde->bnhe", K_pooled, W_k)

        torch.testing.assert_close(K_cmp, K_expected, atol=1e-5, rtol=1e-5)

    def test_bf16_tolerance(self) -> None:
        """BF16 compression is within tolerance of FP32 compression."""
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H_kv, D = 2, 512, 4, 64
        block_size = 64

        K_fp32 = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.float32)
        V_fp32 = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.float32)

        K_cmp_fp32, V_cmp_fp32 = compress_kv(K_fp32, V_fp32, block_size)
        K_cmp_bf16, V_cmp_bf16 = compress_kv(
            K_fp32.bfloat16(), V_fp32.bfloat16(), block_size
        )

        torch.testing.assert_close(
            K_cmp_bf16.float(), K_cmp_fp32, atol=1e-2, rtol=1e-2
        )
