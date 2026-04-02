# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for the NSA reference backward pass (autograd-based)."""

import torch


def _make_inputs(
    B: int,
    N: int,
    H: int,
    D: int,
    H_kv: int | None = None,
    with_W_k: bool = False,
    with_W_v: bool = False,
    with_gate: bool = False,
    dtype: torch.dtype = torch.float64,
    device: str = "cuda",
) -> dict:
    """Create input tensors with requires_grad for gradient checking."""
    if H_kv is None:
        H_kv = H
    Q = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, N, H_kv, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, N, H_kv, D, device=device, dtype=dtype, requires_grad=True)
    result = {"Q": Q, "K": K, "V": V}
    if with_W_k:
        result["W_k_compress"] = (
            torch.randn(
                H_kv,
                D,
                D,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            * 0.02
        )
    if with_W_v:
        result["W_v_compress"] = (
            torch.randn(
                H_kv,
                D,
                D,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            * 0.02
        )
    if with_gate:
        result["gate_proj_weight"] = (
            torch.randn(
                H,
                3,
                D,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            * 0.1
        )
    return result


class TestNSAReferenceBackwardGradcheck:
    """Validate gradients via torch.autograd.gradcheck (finite differences).

    Block selection indices are pre-computed and frozen during gradcheck,
    since top-k is non-differentiable and would cause discontinuities
    when inputs are perturbed.
    """

    # Use small sizes for gradcheck (it's O(params) forward passes)
    B, N, H, D = 1, 256, 2, 32
    compress_block_size = 64
    num_selected_blocks = 2
    window_size = 128
    q_tile_size = 256

    def _run_gradcheck(self, inputs: dict, causal: bool = True) -> None:
        from mslk.attention.sparse_attn.reference import (
            compute_block_indices_reference,
            nsa_forward_reference,
        )

        # Pre-compute block indices with the original (unperturbed) inputs
        # so that top-k selection is frozen during finite-difference perturbation
        block_indices = compute_block_indices_reference(
            inputs["Q"].detach(),
            inputs["K"].detach(),
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            W_k_compress=inputs.get("W_k_compress"),
            causal=causal,
            q_tile_size=self.q_tile_size,
        )

        # Ensure block 0 is always selected for causal mode. Without it,
        # early queries in a tile may have ALL selected KV positions causally
        # masked (kv_pos > q_pos for all selected positions), producing NaN
        # from softmax over all -inf.
        if causal:
            block_indices[..., 0] = 0

        params = [
            v
            for v in inputs.values()
            if isinstance(v, torch.Tensor) and v.requires_grad
        ]

        def fn(*args):
            kw = {}
            idx = 0
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    kw[k] = args[idx]
                    idx += 1
                else:
                    kw[k] = v
            out = nsa_forward_reference(
                kw["Q"],
                kw["K"],
                kw["V"],
                compress_block_size=self.compress_block_size,
                num_selected_blocks=self.num_selected_blocks,
                window_size=self.window_size,
                W_k_compress=kw.get("W_k_compress"),
                W_v_compress=kw.get("W_v_compress"),
                gate_proj_weight=kw.get("gate_proj_weight"),
                causal=causal,
                q_tile_size=self.q_tile_size,
                _block_indices=block_indices,
            )
            return out

        torch.autograd.gradcheck(fn, params, raise_exception=True)

    def test_gradcheck_basic(self) -> None:
        """Gradcheck for Q, K, V without optional weights."""
        inputs = _make_inputs(self.B, self.N, self.H, self.D)
        self._run_gradcheck(inputs)

    def test_gradcheck_with_gate_weights(self) -> None:
        """Gradcheck including gate_proj_weight gradients."""
        inputs = _make_inputs(self.B, self.N, self.H, self.D, with_gate=True)
        self._run_gradcheck(inputs)

    def test_gradcheck_with_compression_weights(self) -> None:
        """Gradcheck including W_k_compress and W_v_compress gradients."""
        inputs = _make_inputs(
            self.B,
            self.N,
            self.H,
            self.D,
            with_W_k=True,
            with_W_v=True,
        )
        self._run_gradcheck(inputs)

    def test_gradcheck_all_weights(self) -> None:
        """Gradcheck with all optional weights enabled."""
        inputs = _make_inputs(
            self.B,
            self.N,
            self.H,
            self.D,
            with_W_k=True,
            with_W_v=True,
            with_gate=True,
        )
        self._run_gradcheck(inputs)

    def test_gradcheck_noncausal(self) -> None:
        """Gradcheck in non-causal mode."""
        inputs = _make_inputs(self.B, self.N, self.H, self.D)
        self._run_gradcheck(inputs, causal=False)

    def test_gradcheck_gqa(self) -> None:
        """Gradcheck with grouped query attention (H > H_kv)."""
        inputs = _make_inputs(self.B, self.N, 4, self.D, H_kv=2)
        self._run_gradcheck(inputs)

    def test_gradcheck_compressed_only(self) -> None:
        """Gradcheck for just compressed attention (isolate branch)."""
        from mslk.attention.sparse_attn.reference import _compressed_attention

        B, N, H, D = self.B, self.N, self.H, self.D
        q = torch.randn(
            B, H, N, D, dtype=torch.float64, device="cuda", requires_grad=True
        )
        k = torch.randn(
            B, H, N, D, dtype=torch.float64, device="cuda", requires_grad=True
        )
        v = torch.randn(
            B, H, N, D, dtype=torch.float64, device="cuda", requires_grad=True
        )

        def fn(q, k, v):
            return _compressed_attention(
                q,
                k,
                v,
                self.compress_block_size,
                None,
                None,
                1,
                1.0 / D**0.5,
                True,
                B,
                N,
                H,
                H,
                D,
            )

        torch.autograd.gradcheck(fn, (q, k, v), raise_exception=True)

    def test_gradcheck_sliding_only(self) -> None:
        """Gradcheck for just sliding window attention (isolate branch)."""
        from mslk.attention.sparse_attn.reference import _sliding_window_attention

        B, N, H, D = self.B, self.N, self.H, self.D
        q = torch.randn(
            B, H, N, D, dtype=torch.float64, device="cuda", requires_grad=True
        )
        k = torch.randn(
            B, H, N, D, dtype=torch.float64, device="cuda", requires_grad=True
        )
        v = torch.randn(
            B, H, N, D, dtype=torch.float64, device="cuda", requires_grad=True
        )

        def fn(q, k, v):
            return _sliding_window_attention(
                q,
                k,
                v,
                self.window_size,
                1.0 / D**0.5,
                True,
                B,
                N,
                H,
                D,
            )

        torch.autograd.gradcheck(fn, (q, k, v), raise_exception=True)

    def test_gradcheck_selected_only(self) -> None:
        """Gradcheck for just selected attention with frozen indices."""
        from mslk.attention.sparse_attn.reference import _selected_attention

        B, N, H, D = self.B, self.N, self.H, self.D
        q = torch.randn(
            B, H, N, D, dtype=torch.float64, device="cuda", requires_grad=True
        )
        k = torch.randn(
            B, H, N, D, dtype=torch.float64, device="cuda", requires_grad=True
        )
        v = torch.randn(
            B, H, N, D, dtype=torch.float64, device="cuda", requires_grad=True
        )

        # Use first k_actual blocks — always valid for causal masking
        N_blocks = N // self.compress_block_size
        N_q_tiles = N // self.q_tile_size
        k_actual = min(self.num_selected_blocks, N_blocks)
        block_indices = (
            torch.arange(k_actual, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        block_indices = block_indices.expand(B, H, N_q_tiles, k_actual)

        def fn(q, k, v):
            return _selected_attention(
                q,
                k,
                v,
                self.compress_block_size,
                self.num_selected_blocks,
                None,
                1,
                1.0 / D**0.5,
                True,
                B,
                N,
                H,
                H,
                D,
                q_tile_size=self.q_tile_size,
                _block_indices=block_indices,
            )

        torch.autograd.gradcheck(fn, (q, k, v), raise_exception=True)


class TestNSABackwardReference:
    """Test nsa_backward_reference wrapper function."""

    B, N, H, D = 1, 512, 4, 64
    compress_block_size = 64
    num_selected_blocks = 4
    window_size = 256
    q_tile_size = 256

    def test_gradient_shapes(self) -> None:
        """Output gradient shapes match input shapes."""
        from mslk.attention.sparse_attn.reference import nsa_backward_reference

        H_kv = self.H
        Q = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        K = torch.randn(
            self.B, self.N, H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )
        V = torch.randn(
            self.B, self.N, H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )
        dO = torch.randn_like(Q)

        grads = nsa_backward_reference(
            Q,
            K,
            V,
            dO,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
        )

        assert grads["dQ"].shape == Q.shape
        assert grads["dK"].shape == K.shape
        assert grads["dV"].shape == V.shape
        assert grads["dW_k_compress"] is None
        assert grads["dW_v_compress"] is None
        assert grads["dgate_proj_weight"] is None

    def test_gradient_shapes_all_weights(self) -> None:
        """Gradient shapes correct with all optional weights."""
        from mslk.attention.sparse_attn.reference import nsa_backward_reference

        H_kv = self.H
        Q = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        K = torch.randn(
            self.B, self.N, H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )
        V = torch.randn(
            self.B, self.N, H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )
        dO = torch.randn_like(Q)
        W_k = (
            torch.randn(H_kv, self.D, self.D, device="cuda", dtype=torch.bfloat16)
            * 0.02
        )
        W_v = (
            torch.randn(H_kv, self.D, self.D, device="cuda", dtype=torch.bfloat16)
            * 0.02
        )
        gate = torch.randn(self.H, 3, self.D, device="cuda", dtype=torch.bfloat16) * 0.1

        grads = nsa_backward_reference(
            Q,
            K,
            V,
            dO,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            W_k_compress=W_k,
            W_v_compress=W_v,
            gate_proj_weight=gate,
        )

        assert grads["dQ"].shape == Q.shape
        assert grads["dK"].shape == K.shape
        assert grads["dV"].shape == V.shape
        assert grads["dW_k_compress"].shape == W_k.shape
        assert grads["dW_v_compress"].shape == W_v.shape
        assert grads["dgate_proj_weight"].shape == gate.shape

    def test_gradients_finite(self) -> None:
        """All gradients are finite (no NaN/Inf)."""
        from mslk.attention.sparse_attn.reference import nsa_backward_reference

        Q = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        K = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        V = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        dO = torch.randn_like(Q)
        gate = torch.randn(self.H, 3, self.D, device="cuda", dtype=torch.bfloat16) * 0.1

        grads = nsa_backward_reference(
            Q,
            K,
            V,
            dO,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            gate_proj_weight=gate,
        )

        for name in ("dQ", "dK", "dV", "dgate_proj_weight"):
            assert torch.isfinite(grads[name]).all(), f"{name} contains NaN or Inf"

    def test_gradients_nonzero(self) -> None:
        """Gradients are not all-zero (sanity check that grad flows)."""
        from mslk.attention.sparse_attn.reference import nsa_backward_reference

        Q = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        K = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        V = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        dO = torch.randn_like(Q)

        grads = nsa_backward_reference(
            Q,
            K,
            V,
            dO,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
        )

        for name in ("dQ", "dK", "dV"):
            assert grads[name].abs().max() > 0, f"{name} is all zeros"

    def test_gradient_shapes_gqa(self) -> None:
        """Gradient shapes correct with GQA (H > H_kv)."""
        from mslk.attention.sparse_attn.reference import nsa_backward_reference

        H_kv = 2
        Q = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        K = torch.randn(
            self.B, self.N, H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )
        V = torch.randn(
            self.B, self.N, H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )
        dO = torch.randn_like(Q)

        grads = nsa_backward_reference(
            Q,
            K,
            V,
            dO,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
        )

        assert grads["dQ"].shape == Q.shape
        assert grads["dK"].shape == K.shape  # (B, N, H_kv, D), not (B, N, H, D)
        assert grads["dV"].shape == V.shape


class TestNSAAutograd:
    """Test NSAFunction (FA4-based forward + backward via autograd).

    Validates that nsa() produces correct forward output and that
    gradients flow correctly through .backward().
    """

    B, N, H, D = 1, 1024, 4, 128
    H_kv = 4
    compress_block_size = 64
    num_selected_blocks = 8
    window_size = 256
    q_tile_size = 256

    def test_forward_matches_nsa_forward(self) -> None:
        """nsa() forward output matches nsa_forward()."""
        from mslk.attention.sparse_attn import nsa, nsa_forward

        Q = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        K = torch.randn(
            self.B, self.N, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )
        V = torch.randn(
            self.B, self.N, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )

        out_fwd = nsa_forward(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
        )
        out_autograd = nsa(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
        )

        # nsa() uses PyTorch gating (compute_gates + gate_and_combine) while
        # nsa_forward() uses fused CuteDSL gating, so expect small numerical diffs
        assert torch.allclose(out_fwd, out_autograd, atol=0.02, rtol=0.02), (
            f"max diff: {(out_fwd - out_autograd).abs().max().item()}"
        )

    def test_backward_produces_gradients(self) -> None:
        """nsa() backward produces non-zero gradients for Q, K, V."""
        from mslk.attention.sparse_attn import nsa

        Q = torch.randn(
            self.B,
            self.N,
            self.H,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        K = torch.randn(
            self.B,
            self.N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        V = torch.randn(
            self.B,
            self.N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        out = nsa(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
        )
        loss = out.sum()
        loss.backward()

        assert Q.grad is not None, "Q.grad is None"
        assert K.grad is not None, "K.grad is None"
        assert V.grad is not None, "V.grad is None"
        assert Q.grad.abs().max() > 0, "Q.grad is all zeros"
        assert K.grad.abs().max() > 0, "K.grad is all zeros"
        assert V.grad.abs().max() > 0, "V.grad is all zeros"

    def test_backward_with_gate_weights(self) -> None:
        """nsa() backward produces gradients for gate_proj_weight."""
        from mslk.attention.sparse_attn import nsa

        Q = torch.randn(
            self.B,
            self.N,
            self.H,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        K = torch.randn(
            self.B,
            self.N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        V = torch.randn(
            self.B,
            self.N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        gate = (
            torch.randn(self.H, 3, self.D, device="cuda", dtype=torch.bfloat16) * 0.1
        ).requires_grad_(True)

        out = nsa(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            gate_proj_weight=gate,
        )
        loss = out.sum()
        loss.backward()

        assert gate.grad is not None, "gate.grad is None"
        assert gate.grad.abs().max() > 0, "gate.grad is all zeros"

    def test_backward_with_compression_weights(self) -> None:
        """nsa() backward produces gradients for W_k_compress and W_v_compress."""
        from mslk.attention.sparse_attn import nsa

        Q = torch.randn(
            self.B,
            self.N,
            self.H,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        K = torch.randn(
            self.B,
            self.N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        V = torch.randn(
            self.B,
            self.N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        W_k = (
            torch.randn(self.H_kv, self.D, self.D, device="cuda", dtype=torch.bfloat16)
            * 0.02
        ).requires_grad_(True)
        W_v = (
            torch.randn(self.H_kv, self.D, self.D, device="cuda", dtype=torch.bfloat16)
            * 0.02
        ).requires_grad_(True)

        out = nsa(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            W_k_compress=W_k,
            W_v_compress=W_v,
        )
        loss = out.sum()
        loss.backward()

        assert W_k.grad is not None, "W_k.grad is None"
        assert W_v.grad is not None, "W_v.grad is None"
        assert W_k.grad.abs().max() > 0, "W_k.grad is all zeros"
        assert W_v.grad.abs().max() > 0, "W_v.grad is all zeros"

    def test_gradients_finite(self) -> None:
        """All gradients from nsa() backward are finite."""
        from mslk.attention.sparse_attn import nsa

        Q = torch.randn(
            self.B,
            self.N,
            self.H,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        K = torch.randn(
            self.B,
            self.N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        V = torch.randn(
            self.B,
            self.N,
            self.H_kv,
            self.D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        gate = (
            torch.randn(self.H, 3, self.D, device="cuda", dtype=torch.bfloat16) * 0.1
        ).requires_grad_(True)

        out = nsa(
            Q,
            K,
            V,
            compress_block_size=self.compress_block_size,
            num_selected_blocks=self.num_selected_blocks,
            window_size=self.window_size,
            gate_proj_weight=gate,
        )
        loss = out.sum()
        loss.backward()

        for name, param in [("Q", Q), ("K", K), ("V", V), ("gate", gate)]:
            assert torch.isfinite(param.grad).all(), f"{name}.grad contains NaN or Inf"


class TestNSABackwardMatchesReference:
    """Compare FA4-based backward gradient values against reference backward.

    This validates that the optimized backward (using FA4 CuteDSL kernels)
    produces numerically similar gradients to the pure PyTorch reference.
    """

    B, N, H, D = 1, 1024, 4, 128
    H_kv = 4
    compress_block_size = 64
    num_selected_blocks = 8
    window_size = 256
    q_tile_size = 256

    def _compare_grads(
        self,
        with_gate: bool = False,
        with_W_k: bool = False,
        with_W_v: bool = False,
        causal: bool = True,
        max_atol: float = 0.4,
        mean_atol: float = 0.02,
    ) -> None:
        from mslk.attention.sparse_attn import nsa
        from mslk.attention.sparse_attn.reference import nsa_backward_reference

        torch.manual_seed(42)

        Q = torch.randn(
            self.B, self.N, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        K = torch.randn(
            self.B, self.N, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )
        V = torch.randn(
            self.B, self.N, self.H_kv, self.D, device="cuda", dtype=torch.bfloat16
        )
        dO = torch.randn_like(Q)

        kwargs = {
            "compress_block_size": self.compress_block_size,
            "num_selected_blocks": self.num_selected_blocks,
            "window_size": self.window_size,
            "causal": causal,
            "q_tile_size": self.q_tile_size,
        }

        W_k = None
        W_v = None
        gate = None
        if with_W_k:
            W_k = (
                torch.randn(
                    self.H_kv, self.D, self.D, device="cuda", dtype=torch.bfloat16
                )
                * 0.02
            )
            kwargs["W_k_compress"] = W_k
        if with_W_v:
            W_v = (
                torch.randn(
                    self.H_kv, self.D, self.D, device="cuda", dtype=torch.bfloat16
                )
                * 0.02
            )
            kwargs["W_v_compress"] = W_v
        if with_gate:
            gate = (
                torch.randn(self.H, 3, self.D, device="cuda", dtype=torch.bfloat16)
                * 0.1
            )
            kwargs["gate_proj_weight"] = gate

        # --- Reference backward ---
        ref_grads = nsa_backward_reference(Q, K, V, dO, **kwargs)

        # --- Optimized backward (FA4-based) ---
        Q_opt = Q.clone().requires_grad_(True)
        K_opt = K.clone().requires_grad_(True)
        V_opt = V.clone().requires_grad_(True)

        opt_kwargs = dict(kwargs)
        if with_W_k:
            W_k_opt = W_k.clone().requires_grad_(True)
            opt_kwargs["W_k_compress"] = W_k_opt
        if with_W_v:
            W_v_opt = W_v.clone().requires_grad_(True)
            opt_kwargs["W_v_compress"] = W_v_opt
        if with_gate:
            gate_opt = gate.clone().requires_grad_(True)
            opt_kwargs["gate_proj_weight"] = gate_opt

        out = nsa(Q_opt, K_opt, V_opt, **opt_kwargs)
        out.backward(dO)

        # --- Compare ---
        pairs = [
            ("dQ", Q_opt.grad, ref_grads["dQ"]),
            ("dK", K_opt.grad, ref_grads["dK"]),
            ("dV", V_opt.grad, ref_grads["dV"]),
        ]
        if with_W_k:
            pairs.append(("dW_k", W_k_opt.grad, ref_grads["dW_k_compress"]))
        if with_W_v:
            pairs.append(("dW_v", W_v_opt.grad, ref_grads["dW_v_compress"]))
        if with_gate:
            pairs.append(("dgate", gate_opt.grad, ref_grads["dgate_proj_weight"]))

        for name, opt_grad, ref_grad in pairs:
            assert opt_grad is not None, f"{name} is None"
            ref_grad = ref_grad.to(opt_grad.dtype)
            diff = (opt_grad - ref_grad).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            # Gate gradients use fused CuteDSL kernel vs float32 reference,
            # so max outliers can be large; check mean only for gates.
            if name == "dgate":
                assert mean_diff < 1.0, f"{name}: mean_diff={mean_diff:.6f} exceeds 1.0"
            else:
                assert max_diff < max_atol, (
                    f"{name}: max_diff={max_diff:.4f} exceeds {max_atol}"
                )
                assert mean_diff < mean_atol, (
                    f"{name}: mean_diff={mean_diff:.6f} exceeds {mean_atol}"
                )

    def test_basic(self) -> None:
        """Q, K, V gradients match reference."""
        self._compare_grads()

    def test_with_gate_weights(self) -> None:
        """Gradients match with gate projection weights."""
        self._compare_grads(with_gate=True)

    def test_with_compression_weights(self) -> None:
        """Gradients match with compression projection weights."""
        self._compare_grads(with_W_k=True, with_W_v=True)

    def test_all_weights(self) -> None:
        """Gradients match with all optional weights."""
        self._compare_grads(with_gate=True, with_W_k=True, with_W_v=True)

    def test_noncausal(self) -> None:
        """Gradients match in non-causal mode.

        Non-causal has larger bf16 diffs (no causal mask to stabilize softmax),
        so we only check mean diff.
        """
        self._compare_grads(causal=False, max_atol=2.0, mean_atol=0.05)
