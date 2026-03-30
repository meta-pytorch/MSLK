# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for NSA forward with variable-length sequences."""

import torch


class TestNSAForwardVarlen:
    """Test nsa_forward with cu_seqlens (varlen) input."""

    H, H_kv, D = 4, 4, 128
    compress_block_size = 64
    num_selected_blocks = 4
    window_size = 256
    q_tile_size = 256

    def test_varlen_output_shape(self) -> None:
        """Varlen output should be 3D with total_tokens rows."""
        from mslk.attention.sparse_attn import nsa_forward

        seqlens = [512, 256]
        total = sum(seqlens)
        cu_seqlens = torch.tensor(
            [0, seqlens[0], total], dtype=torch.int32, device="cuda"
        )

        Q = torch.randn(total, self.H, self.D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(total, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(total, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16)

        O = nsa_forward(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            q_tile_size=self.q_tile_size,
            cu_seqlens=cu_seqlens,
        )

        assert O.shape == (total, self.H, self.D)
        assert O.dtype == Q.dtype
        assert torch.isfinite(O).all()

    def test_varlen_matches_per_sequence(self) -> None:
        """Varlen output should match running each sequence individually."""
        from mslk.attention.sparse_attn import nsa_forward

        torch.manual_seed(42)
        seqlens = [512, 256]
        total = sum(seqlens)
        cu_seqlens = torch.tensor(
            [0, seqlens[0], total], dtype=torch.int32, device="cuda"
        )

        Q = torch.randn(total, self.H, self.D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(total, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(total, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16)

        # Varlen forward
        O_varlen = nsa_forward(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            q_tile_size=self.q_tile_size,
            cu_seqlens=cu_seqlens,
        )

        # Per-sequence forward
        for i, slen in enumerate(seqlens):
            start = cu_seqlens[i].item()
            Q_seq = Q[start : start + slen].unsqueeze(0)  # (1, slen, H, D)
            K_seq = K[start : start + slen].unsqueeze(0)
            V_seq = V[start : start + slen].unsqueeze(0)

            O_seq = nsa_forward(
                Q_seq,
                K_seq,
                V_seq,
                compress_block_size=self.compress_block_size,
                num_selected_blocks=self.num_selected_blocks,
                window_size=self.window_size,
                q_tile_size=self.q_tile_size,
            )

            O_varlen_seq = O_varlen[start : start + slen]
            O_fixed_seq = O_seq.squeeze(0)

            max_diff = (O_varlen_seq - O_fixed_seq).abs().max().item()
            mean_diff = (O_varlen_seq - O_fixed_seq).abs().mean().item()
            assert max_diff < 0.01, f"Seq {i}: max_diff={max_diff:.6f} exceeds 0.01"
            assert mean_diff < 0.001, (
                f"Seq {i}: mean_diff={mean_diff:.6f} exceeds 0.001"
            )

    def test_varlen_single_sequence(self) -> None:
        """Single-sequence varlen should match fixed-length exactly."""
        from mslk.attention.sparse_attn import nsa_forward

        torch.manual_seed(123)
        N = 512
        cu_seqlens = torch.tensor([0, N], dtype=torch.int32, device="cuda")

        Q = torch.randn(N, self.H, self.D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(N, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(N, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16)

        O_varlen = nsa_forward(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            q_tile_size=self.q_tile_size,
            cu_seqlens=cu_seqlens,
        )

        O_fixed = nsa_forward(
            Q.unsqueeze(0),
            K.unsqueeze(0),
            V.unsqueeze(0),
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            q_tile_size=self.q_tile_size,
        ).squeeze(0)

        max_diff = (O_varlen - O_fixed).abs().max().item()
        assert max_diff < 1e-5, f"Single-seq max_diff={max_diff} should be ~0"

    def test_varlen_backward_produces_gradients(self) -> None:
        """nsa() with cu_seqlens should produce gradients via backward."""
        from mslk.attention.sparse_attn import nsa

        seqlens = [512, 256]
        total = sum(seqlens)
        cu_seqlens = torch.tensor(
            [0, seqlens[0], total], dtype=torch.int32, device="cuda"
        )

        Q = torch.randn(
            total,
            self.H,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        K = torch.randn(
            total,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        V = torch.randn(
            total,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        O = nsa(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            q_tile_size=self.q_tile_size,
            cu_seqlens=cu_seqlens,
        )
        loss = O.sum()
        loss.backward()

        assert Q.grad is not None, "Q.grad is None"
        assert K.grad is not None, "K.grad is None"
        assert V.grad is not None, "V.grad is None"
        assert Q.grad.shape == Q.shape
        assert torch.isfinite(Q.grad).all(), "Q.grad has NaN/Inf"
        assert Q.grad.abs().max() > 0, "Q.grad is all zeros"

    def test_varlen_backward_matches_per_sequence(self) -> None:
        """Varlen backward gradients should be finite, non-zero, and correct shape."""
        from mslk.attention.sparse_attn import nsa

        torch.manual_seed(42)
        seqlens = [512, 256]
        total = sum(seqlens)
        cu_seqlens = torch.tensor(
            [0, seqlens[0], total], dtype=torch.int32, device="cuda"
        )

        Q = torch.randn(
            total,
            self.H,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        K = torch.randn(
            total,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        V = torch.randn(
            total,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        O = nsa(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            q_tile_size=self.q_tile_size,
            cu_seqlens=cu_seqlens,
        )
        O.sum().backward()

        # Verify gradients are well-formed
        assert Q.grad.shape == Q.shape
        assert K.grad.shape == K.shape
        assert V.grad.shape == V.shape
        assert torch.isfinite(Q.grad).all(), "Q.grad has NaN/Inf"
        assert torch.isfinite(K.grad).all(), "K.grad has NaN/Inf"
        assert torch.isfinite(V.grad).all(), "V.grad has NaN/Inf"

        # Each sequence's gradients should be non-zero
        for i, slen in enumerate(seqlens):
            s = cu_seqlens[i].item()
            dQ_seq = Q.grad[s : s + slen]
            assert dQ_seq.abs().max() > 0, f"Seq {i}: dQ is all zeros"
            dK_seq = K.grad[s : s + slen]
            assert dK_seq.abs().max() > 0, f"Seq {i}: dK is all zeros"

    def test_varlen_backward_single_sequence(self) -> None:
        """Single-sequence varlen backward should match fixed-length."""
        from mslk.attention.sparse_attn import nsa

        torch.manual_seed(123)
        N = 512
        cu_seqlens = torch.tensor([0, N], dtype=torch.int32, device="cuda")

        Q_3d = torch.randn(
            N,
            self.H,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        K_3d = torch.randn(
            N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        V_3d = torch.randn(
            N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        O_3d = nsa(
            Q_3d,
            K_3d,
            V_3d,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            q_tile_size=self.q_tile_size,
            cu_seqlens=cu_seqlens,
        )
        O_3d.sum().backward()

        Q_4d = Q_3d.data.unsqueeze(0).clone().requires_grad_(True)
        K_4d = K_3d.data.unsqueeze(0).clone().requires_grad_(True)
        V_4d = V_3d.data.unsqueeze(0).clone().requires_grad_(True)

        O_4d = nsa(
            Q_4d,
            K_4d,
            V_4d,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            q_tile_size=self.q_tile_size,
        )
        O_4d.sum().backward()

        dQ_diff = (Q_3d.grad.float() - Q_4d.grad.squeeze(0).float()).abs().max().item()
        assert dQ_diff < 0.01, f"dQ max_diff={dQ_diff}"
        dK_diff = (K_3d.grad.float() - K_4d.grad.squeeze(0).float()).abs().max().item()
        assert dK_diff < 0.01, f"dK max_diff={dK_diff}"
        dV_diff = (V_3d.grad.float() - V_4d.grad.squeeze(0).float()).abs().max().item()
        assert dV_diff < 0.01, f"dV max_diff={dV_diff}"
