# @nolint # fbcode
"""Tests for Mixed-Input GEMM CuteDSL kernel."""

import pytest
import torch

from mslk.gemm.blackwell_mixed_input_gemm import (
    MixedInputGemmKernel,
    create_tensors,
    compare,
)
import cutlass
import cutlass.cute as cute


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell (SM100+)"
)
class TestMixedInputGemmNumeric:
    """Numeric validation tests using compare function directly."""

    def _run_and_compare(
        self,
        mnkl: tuple[int, int, int, int],
        scale_granularity_m: int,
        scale_granularity_k: int,
        a_dtype: type,
        b_dtype: type,
        c_dtype: type,
        acc_dtype: type,
        a_major: str,
        b_major: str,
        c_major: str,
        mma_tiler_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        use_2cta_instrs: bool,
        use_tma_store: bool,
        tolerance: float,
    ) -> None:
        """Helper to run kernel and compare results."""
        import cutlass.utils as utils
        import cuda.bindings.driver as cuda_bindings

        m, n, k, l = mnkl  # noqa: E741

        if not MixedInputGemmKernel.can_implement(
            mnkl,
            a_dtype,
            b_dtype,
            c_dtype,
            a_major,
            b_major,
            c_major,
            scale_granularity_m,
            scale_granularity_k,
            mma_tiler_mnk,
            cluster_shape_mn,
            use_2cta_instrs,
            use_tma_store,
        ):
            pytest.skip("GEMM configuration not supported")

        torch_stream = torch.cuda.current_stream()
        current_stream = cuda_bindings.CUstream(torch_stream.cuda_stream)

        (
            a_tensor,
            a_scale_tensor,
            b_tensor,
            c_tensor,
            a_torch_cpu,
            a_scale_torch_cpu,
            b_torch_cpu,
            c_torch_gpu,
        ) = create_tensors(
            l,
            m,
            n,
            k,
            a_major,
            b_major,
            c_major,
            a_dtype,
            b_dtype,
            c_dtype,
            scale_granularity_m,
            scale_granularity_k,
        )

        mixed_input_gemm = MixedInputGemmKernel(
            scale_granularity_m,
            scale_granularity_k,
            acc_dtype,
            use_2cta_instrs,
            mma_tiler_mnk,
            cluster_shape_mn,
            use_tma_store,
        )

        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1],
        )
        compiled_kernel = cute.compile(
            mixed_input_gemm,
            a_tensor,
            a_scale_tensor,
            b_tensor,
            c_tensor,
            max_active_clusters,
            current_stream,
        )

        compiled_kernel(
            a_tensor,
            a_scale_tensor,
            b_tensor,
            c_tensor,
            current_stream,
        )

        compare(
            a_torch_cpu, b_torch_cpu, a_scale_torch_cpu, c_torch_gpu, c_dtype, tolerance
        )

    @pytest.mark.parametrize(
        "mnkl,scale_granularity_m,scale_granularity_k,a_dtype,b_dtype,c_dtype,"
        "acc_dtype,a_major,b_major,c_major,mma_tiler_mnk,cluster_shape_mn,"
        "use_2cta_instrs,use_tma_store,tolerance",
        [
            pytest.param(
                (128, 128, 256, 1), 0, 0, cutlass.Int8, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (128, 128, 64), (1, 1), False, False, 0.1,
                id="int8_bf16_small_matrices",
            ),
            pytest.param(
                (128, 256, 16384, 1), 0, 0, cutlass.Int8, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (128, 128, 64), (1, 1), False, False, 0.1,
                id="int8_bf16_large_k",
            ),
            pytest.param(
                (128, 256, 512, 4), 0, 0, cutlass.Int8, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (128, 128, 64), (1, 1), False, False, 0.1,
                id="int8_bf16_batched",
            ),
            pytest.param(
                (128, 256, 512, 1), 1, 128, cutlass.Int4, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (128, 128, 128), (1, 1), False, False, 0.1,
                id="int4_bf16_small_scale_granularity",
            ),
            pytest.param(
                (512, 1024, 4096, 1), 1, 256, cutlass.Int4, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (128, 128, 128), (1, 1), False, False, 0.1,
                id="int4_bf16_large_matrices",
            ),
            pytest.param(
                (256, 512, 2048, 2), 1, 256, cutlass.Int4, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (128, 128, 128), (1, 1), False, False, 0.1,
                id="int4_bf16_batched_with_scale",
            ),
            pytest.param(
                (256, 256, 512, 1), 0, 0, cutlass.Int8, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "m",
                (128, 128, 64), (1, 1), False, False, 0.1,
                id="int8_bf16_c_major_m",
            ),
            pytest.param(
                (256, 256, 512, 1), 0, 0, cutlass.Int8, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "n", "n",
                (128, 128, 64), (1, 1), False, False, 0.1,
                id="int8_bf16_b_major_n",
            ),
            pytest.param(
                (256, 256, 512, 1), 0, 0, cutlass.Int8, cutlass.BFloat16,
                cutlass.Float16, cutlass.Float32, "m", "k", "n",
                (128, 128, 64), (1, 1), False, False, 0.1,
                id="int8_f16_output",
            ),
            pytest.param(
                (256, 256, 1024, 1), 1, 128, cutlass.Int4, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (64, 128, 128), (1, 1), False, False, 0.1,
                id="int4_bf16_different_mma_tiler",
            ),
            pytest.param(
                (128, 1024, 512, 1), 0, 0, cutlass.Int8, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (128, 128, 64), (1, 1), False, False, 0.1,
                id="int8_bf16_wide_n",
            ),
            pytest.param(
                (1024, 128, 512, 1), 0, 0, cutlass.Int8, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (128, 128, 64), (1, 1), False, False, 0.1,
                id="int8_bf16_tall_m",
            ),
            pytest.param(
                (256, 256, 2048, 1), 1, 512, cutlass.Int4, cutlass.BFloat16,
                cutlass.BFloat16, cutlass.Float32, "m", "k", "n",
                (128, 128, 128), (1, 1), False, False, 0.1,
                id="int4_bf16_large_scale_k",
            ),
        ],
    )
    def test_mixed_input_gemm(
        self,
        mnkl,
        scale_granularity_m,
        scale_granularity_k,
        a_dtype,
        b_dtype,
        c_dtype,
        acc_dtype,
        a_major,
        b_major,
        c_major,
        mma_tiler_mnk,
        cluster_shape_mn,
        use_2cta_instrs,
        use_tma_store,
        tolerance,
    ):
        """Test mixed-input GEMM with various configurations."""
        self._run_and_compare(
            mnkl=mnkl,
            scale_granularity_m=scale_granularity_m,
            scale_granularity_k=scale_granularity_k,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            acc_dtype=acc_dtype,
            a_major=a_major,
            b_major=b_major,
            c_major=c_major,
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
            use_tma_store=use_tma_store,
            tolerance=tolerance,
        )
