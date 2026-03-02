# @nolint # fbcode
"""Tests for CuteDSL BF16 Grouped GEMM kernel."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell (SM100+)"
)
class TestBF16GroupedGemm:
    """Tests for CuteDSL BF16 grouped GEMM on Blackwell."""

    def _run_bf16_grouped_gemm(
        self,
        G: int,
        M_per_group: int,
        N: int,
        K: int,
        atol: float = 1e-1,
        rtol: float = 1e-1,
    ):
        """Helper to run BF16 grouped GEMM and validate against reference."""
        from mslk.gemm.cutedsl import bf16_grouped_gemm

        torch.manual_seed(42)
        GM = G * M_per_group
        split_sizes = torch.full(
            (G,), M_per_group, dtype=torch.int32, device="cuda"
        )
        x = torch.randn(GM, K, dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(G, N, K, dtype=torch.bfloat16, device="cuda") * 0.1

        result = bf16_grouped_gemm(x, w, split_sizes)

        # Reference: split x into groups, do individual matmuls
        ref_parts = []
        offset = 0
        for g in range(G):
            m = split_sizes[g].item()
            x_g = x[offset : offset + m, :]
            w_g = w[g]
            ref_parts.append(torch.mm(x_g, w_g.t()))
            offset += m
        ref = torch.cat(ref_parts, dim=0)

        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "G,M_per_group,N,K",
        [
            pytest.param(2, 256, 256, 256, id="2g_small_square"),
            pytest.param(4, 256, 256, 512, id="4g_medium"),
            pytest.param(2, 512, 512, 1024, id="2g_large"),
            pytest.param(8, 256, 512, 256, id="8g_rectangular"),
        ],
    )
    def test_basic_shapes(self, G, M_per_group, N, K):
        """Test basic shapes for BF16 grouped GEMM."""
        self._run_bf16_grouped_gemm(G, M_per_group, N, K)

    @pytest.mark.parametrize(
        "G,M_per_group,N,K",
        [
            pytest.param(8, 256, 7168, 8192, id="llama4_qkv_proj"),
            pytest.param(8, 256, 8192, 7168, id="llama4_out_proj"),
            pytest.param(8, 256, 16384, 8192, id="llama4_ffn_up"),
            pytest.param(8, 256, 8192, 16384, id="llama4_ffn_down"),
        ],
    )
    def test_llama4_shapes(self, G, M_per_group, N, K):
        """Test Llama4-relevant shapes for BF16 grouped GEMM."""
        self._run_bf16_grouped_gemm(G, M_per_group, N, K)

    def test_preallocated_output(self):
        """Test BF16 grouped GEMM with pre-allocated output tensor."""
        from mslk.gemm.cutedsl import bf16_grouped_gemm

        torch.manual_seed(42)
        G, M_per_group, N, K = 2, 256, 256, 256
        GM = G * M_per_group
        split_sizes = torch.full(
            (G,), M_per_group, dtype=torch.int32, device="cuda"
        )
        x = torch.randn(GM, K, dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(G, N, K, dtype=torch.bfloat16, device="cuda") * 0.1
        y = torch.empty(GM, N, dtype=torch.bfloat16, device="cuda")

        result = bf16_grouped_gemm(x, w, split_sizes, y=y)
        assert result.data_ptr() == y.data_ptr()

        ref_parts = []
        offset = 0
        for g in range(G):
            m = split_sizes[g].item()
            x_g = x[offset : offset + m, :]
            w_g = w[g]
            ref_parts.append(torch.mm(x_g, w_g.t()))
            offset += m
        ref = torch.cat(ref_parts, dim=0)

        torch.testing.assert_close(result, ref, atol=1e-1, rtol=1e-1)
