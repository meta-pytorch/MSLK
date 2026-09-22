# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Optional
from unittest import mock

import torch
from mslk.gemm.triton.fp8_gemm import (
    MATMUL_CONFIGS,
    MINIMUM_BLOCK_K,
    MINIMUM_BLOCK_M,
    MINIMUM_BLOCK_N,
    prune_configs_h100_static,
)

if torch.cuda.is_available():
    from mslk.gemm.triton.fp8_gemm import matmul_fp8_block, matmul_fp8_row
    from mslk.gemm.triton.matmul_perf_model import estimate_matmul_time
    from mslk.quantize.triton.fp8_quantize import quantize_fp8_block, quantize_fp8_row
    from mslk.utils.triton.fp8_utils import get_fp8_constants


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp8Matmul(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_matmul_fp8_row(self) -> None:
        def _test_matmul_fp8_row(
            shape: tuple[int, int, int],
            device: torch.device,
            fp8_fast_accum: bool,
            use_bias: bool = False,
            transpose_input: bool = False,
            compile: bool = False,
        ) -> None:
            M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            # Make a non-contiguous tensor and check that we still get proper results.
            if transpose_input:
                a = a.t()
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
            bias = (
                torch.randn(N, dtype=torch.float32, device=device) if use_bias else None
            )

            # Test that we can compile the full fp8 matmul operation.
            if compile:

                @torch.compile(fullgraph=True)
                def _quantize_matmul_fp8(
                    a: torch.Tensor,
                    b: torch.Tensor,
                    bias: Optional[torch.Tensor],
                    fp8_fast_accum: bool,
                ) -> torch.Tensor:
                    a_fp8, a_scale = quantize_fp8_row(a)
                    b_fp8, b_scale = quantize_fp8_row(b)
                    return matmul_fp8_row(
                        a_fp8,
                        b_fp8,
                        a_scale,
                        b_scale,
                        bias=bias,
                        fp8_fast_accum=fp8_fast_accum,
                    )

                result = _quantize_matmul_fp8(a, b, bias, fp8_fast_accum)
            # Otherwise run normally.
            else:
                # Quantize inputs.
                a_fp8, a_scale = quantize_fp8_row(a)
                b_fp8, b_scale = quantize_fp8_row(b)

                result = matmul_fp8_row(
                    a_fp8,
                    b_fp8,
                    a_scale,
                    b_scale,
                    bias=bias,
                    fp8_fast_accum=fp8_fast_accum,
                )
            self.assertTrue(result.shape == (M, N))

            expected_result = a @ b.T
            if use_bias:
                # pyre-fixme[6]: For 1st argument expected `Union[bool, complex,
                #  float, int, Tensor]` but got `Optional[Tensor]`.
                expected_result += bias
            self.assertTrue(
                torch.allclose(result, expected_result, atol=2e-1, rtol=5e-2)
            )

        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True, compile=True)
        _test_matmul_fp8_row(
            (5, 4, 5), torch.device("cuda"), True, transpose_input=True
        )
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True, True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), False)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), False, True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cpu"), False)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cpu"), False, True)

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.version.hip is not None
        or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
        "Device-side TMA persistent path is CUDA-only (Hopper+)",
    )
    def test_matmul_fp8_row_tma_persistent(self) -> None:
        # Aligned shapes (K % 16 == 0, N % 8 == 0) take the device-side TMA path;
        # the tiny shapes in test_matmul_fp8_row only hit the fallback. Compare to
        # a dequantized-fp8 reference to isolate the kernel from quant noise.
        for M, N, K, fast_accum, use_bias in [
            (128, 256, 128, True, False),
            (256, 256, 256, False, False),
            (512, 512, 256, True, True),
        ]:
            a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
            bias = (
                torch.randn(N, dtype=torch.float32, device="cuda") if use_bias else None
            )
            a_fp8, a_scale = quantize_fp8_row(a)
            b_fp8, b_scale = quantize_fp8_row(b)
            ref = (a_fp8.to(torch.float32) * a_scale[:, None]) @ (
                b_fp8.to(torch.float32) * b_scale[:, None]
            ).T
            if use_bias:
                ref = ref + bias
            out = matmul_fp8_row(
                a_fp8,
                b_fp8,
                a_scale,
                b_scale,
                bias=bias,
                fp8_fast_accum=fast_accum,
                tma_persistent=True,
            )
            torch.testing.assert_close(out.to(torch.float32), ref, atol=1e-1, rtol=2e-2)

    def test_matmul_fp8_row_skip_scaling(self) -> None:
        def _fp8_clamp(x: torch.Tensor) -> torch.Tensor:
            # Use the platform-correct fp8 dtype (e4m3fnuz on AMD MI300/MI350,
            # e4m3fn on Nvidia) so matmul_fp8_row's dtype assertion is satisfied.
            fp8_dtype, _, fp8_max, _ = get_fp8_constants()
            xq = torch.clamp(x, min=-1 * fp8_max, max=fp8_max).to(fp8_dtype)
            return xq

        def _test_matmul_fp8_row_skip_scaling(
            shape: tuple[int, int, int],
            device: torch.device,
            use_bias: bool = True,
            transpose_input: bool = False,
            compile: bool = False,
        ) -> None:
            M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            # Make a non-contiguous tensor and check that we still get proper results.
            if transpose_input:
                a = a.t()
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
            bias = (
                torch.randn(N, dtype=torch.float32, device=device) if use_bias else None
            )

            # Test that we can compile the full fp8 matmul operation.
            if compile:

                @torch.compile(fullgraph=True)
                def _quantize_matmul_fp8(
                    a: torch.Tensor,
                    b: torch.Tensor,
                    bias: Optional[torch.Tensor],
                ) -> torch.Tensor:
                    a_fp8, a_scale = _fp8_clamp(a), None
                    b_fp8, b_scale = quantize_fp8_row(b)
                    return matmul_fp8_row(
                        a_fp8,
                        b_fp8,
                        a_scale,
                        b_scale,
                        bias=bias,
                        fp8_fast_accum=True,
                        imprecise_acc=False,
                        tma_persistent=False,
                        no_use_persistent=False,
                        use_warp_specialization=False,
                    )

                result = _quantize_matmul_fp8(a, b, bias)
            # Otherwise run normally.
            else:
                # Quantize inputs.
                a_fp8, a_scale = _fp8_clamp(a), None
                b_fp8, b_scale = quantize_fp8_row(b)

                result = matmul_fp8_row(
                    a_fp8,
                    b_fp8,
                    a_scale,
                    b_scale,
                    bias=bias,
                    fp8_fast_accum=True,
                    imprecise_acc=False,
                    tma_persistent=False,
                    no_use_persistent=False,
                    use_warp_specialization=False,
                )
            self.assertTrue(result.shape == (M, N))

            expected_result = a @ b.T
            if use_bias:
                # pyre-fixme[6]: For 1st argument expected `Union[bool, complex,
                #  float, int, Tensor]` but got `Optional[Tensor]`.
                expected_result += bias
            self.assertTrue(
                torch.allclose(result, expected_result, atol=2e-1, rtol=5e-2)
            )

        _test_matmul_fp8_row_skip_scaling((3, 4, 5), torch.device("cuda"))
        # The compile / transposed / no-bias variants each re-trigger the
        # persistent-path Triton autotune, which is very slow to compile on
        # ROCm (hipcc). One representative case above already guards the
        # skip-scaling path on ROCm; run the full matrix only on CUDA to keep
        # the ROCm CI within its time budget.
        if torch.version.hip is None:
            _test_matmul_fp8_row_skip_scaling(
                (3, 4, 5), torch.device("cuda"), compile=True
            )
            _test_matmul_fp8_row_skip_scaling(
                (5, 4, 5), torch.device("cuda"), transpose_input=True
            )
            _test_matmul_fp8_row_skip_scaling(
                (3, 4, 5), torch.device("cuda"), use_bias=False
            )

    def test_matmul_fp8_block(self) -> None:
        def _test_matmul_fp8_block(
            shape: tuple[int, int, int],
            block_shape: tuple[int, int, int],
            fp8_fast_accum: bool,
            transpose_input: bool = False,
            device: str = "cuda",
        ) -> None:
            M, N, K = shape
            BLOCK_M, BLOCK_N, BLOCK_K = block_shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            # Make a non-contiguous tensor and check that we still get proper results.
            if transpose_input:
                a = a.t()
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)

            # Quantize inputs.
            a_fp8, a_scale = quantize_fp8_block(
                a, BLOCK_M, BLOCK_K, output_device=torch.device("cuda")
            )
            b_fp8, b_scale = quantize_fp8_block(
                b, BLOCK_N, BLOCK_K, output_device=torch.device("cuda")
            )

            result = matmul_fp8_block(
                a_fp8,
                b_fp8,
                a_scale,
                b_scale,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                fp8_fast_accum=fp8_fast_accum,
            )
            self.assertTrue(result.shape == (M, N))

            expected_result = (a @ b.T).to("cuda")

            self.assertTrue(
                torch.allclose(result, expected_result, atol=1e2, rtol=5e-2)
            )

        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), True)
        _test_matmul_fp8_block((5, 4, 5), (256, 256, 256), True, transpose_input=True)
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), True)
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), False)
        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), False)
        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), True, device="cpu")
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), True, device="cpu")
        # K values that leave a partial trailing scale block while still making
        # EVEN_K true, so the last iteration is reached with
        # k_remaining == BLOCK_K * SPLIT_K. The K=5 shapes above make EVEN_K
        # false, which is why they already took the tail branch and never
        # covered this case. scale_block_m/n are kept at M/N so that no tile can
        # span two M/N scale blocks whichever BLOCK_M/BLOCK_N autotuning picks.
        _test_matmul_fp8_block((256, 256, 640), (256, 256, 256), True)
        _test_matmul_fp8_block((256, 256, 64), (256, 256, 256), True)
        _test_matmul_fp8_block((256, 256, 192), (256, 256, 128), True)


class _FakeConfig:
    """Minimal stand-in for triton.Config (only the fields the prune reads)."""

    def __init__(self, bm: int, bn: int, bk: int, ns: int, nw: int) -> None:
        self.kwargs = {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "SPLIT_K": 1,
        }
        self.num_stages = ns
        self.num_warps = nw


class TestPruneConfigsH100Static(unittest.TestCase):
    """CPU-runnable coverage for the H100 static autotune prune.

    Uses lightweight fake configs so no GPU or Triton launch is needed;
    the Hopper gate is faked via mock.
    """

    def _make_configs(self) -> list[_FakeConfig]:
        configs = []
        for bm, bn, bk, ns, nw in [
            (128, 256, 32, 3, 8),
            (256, 128, 32, 3, 8),
            (64, 128, 256, 4, 4),
            (256, 256, 32, 4, 4),
            (64, 32, 32, 5, 2),
            (64, 256, 32, 4, 4),
        ]:
            configs.append(_FakeConfig(bm, bn, bk, ns, nw))
        for ns in [2, 3, 4, 5, 6]:
            configs.append(_FakeConfig(16, 64, 32, ns, 2))
        return configs

    def _prune(
        self,
        m: int,
        n: int,
        k: int,
        hopper: bool = True,
        configs: Optional[list[_FakeConfig]] = None,
    ) -> list[_FakeConfig]:
        with mock.patch(
            "mslk.gemm.triton.fp8_gemm.is_hopper",
            return_value=hopper,
        ):
            return prune_configs_h100_static(
                configs if configs is not None else self._make_configs(),
                {"M": m, "N": n, "K": k},
            )

    @staticmethod
    def _keys(kept: list[_FakeConfig]) -> set[tuple[int, int, int]]:
        return {
            (c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"])
            for c in kept
        }

    def test_pool_minima_match_config_space(self) -> None:
        self.assertEqual(
            MINIMUM_BLOCK_M, min(c.kwargs["BLOCK_M"] for c in MATMUL_CONFIGS)
        )
        self.assertEqual(
            MINIMUM_BLOCK_N, min(c.kwargs["BLOCK_N"] for c in MATMUL_CONFIGS)
        )
        self.assertEqual(
            MINIMUM_BLOCK_K, min(c.kwargs["BLOCK_K"] for c in MATMUL_CONFIGS)
        )

    def test_small_m_drops_large_block_m(self) -> None:
        kept = self._prune(1, 1024, 1024)
        bms = {c.kwargs["BLOCK_M"] for c in kept}
        self.assertTrue(bms <= {16, 32, 64})

    def test_small_n_drops_large_block_n(self) -> None:
        kept = self._prune(512, 32, 256)
        keys = self._keys(kept)
        self.assertIn((64, 32, 32), keys)
        self.assertNotIn((128, 256, 32), keys)
        self.assertNotIn((64, 256, 32), keys)

    def test_small_k_drops_large_block_k(self) -> None:
        kept = self._prune(512, 1024, 32)
        keys = self._keys(kept)
        self.assertNotIn((64, 128, 256), keys)
        self.assertIn((64, 256, 32), keys)

    def test_large_k_keeps_large_block_k(self) -> None:
        kept = self._prune(512, 1024, 19712)
        self.assertIn((64, 128, 256), self._keys(kept))

    def test_bk_floor_keeps_32_below_threshold(self) -> None:
        configs = [
            _FakeConfig(64, 64, 32, 4, 4),
            _FakeConfig(64, 64, 64, 4, 4),
        ]
        kept = self._prune(1024, 1024, 256, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 32), (64, 64, 64)})

    def test_bk_floor_drops_32_at_large_k(self) -> None:
        configs = [
            _FakeConfig(64, 64, 32, 4, 4),
            _FakeConfig(64, 64, 64, 4, 4),
        ]
        kept = self._prune(1024, 1024, 1024, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 64)})

    def test_bk_floor_drops_64_at_huge_k(self) -> None:
        configs = [
            _FakeConfig(64, 64, 64, 4, 4),
            _FakeConfig(64, 64, 128, 4, 4),
        ]
        kept = self._prune(1024, 1024, 6656, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 128)})

    def test_bm_floor_drops_small_bm_at_big_m(self) -> None:
        configs = [
            _FakeConfig(16, 64, 64, 4, 4),
            _FakeConfig(32, 64, 64, 4, 4),
            _FakeConfig(64, 64, 64, 4, 4),
        ]
        kept = self._prune(512, 1024, 1024, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 64)})

    def test_bm_floor_keeps_small_bm_at_small_k(self) -> None:
        configs = [
            _FakeConfig(16, 64, 32, 4, 4),
            _FakeConfig(32, 64, 32, 4, 4),
            _FakeConfig(64, 64, 32, 4, 4),
        ]
        kept = self._prune(512, 1024, 16, configs=configs)
        self.assertEqual(self._keys(kept), {(16, 64, 32), (32, 64, 32), (64, 64, 32)})

    def test_bn_floor_drops_32_at_big_n(self) -> None:
        configs = [
            _FakeConfig(64, 32, 64, 4, 4),
            _FakeConfig(64, 64, 64, 4, 4),
        ]
        kept = self._prune(512, 1024, 1024, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 64)})

    def test_bn_floor_drops_64_at_huge_n(self) -> None:
        configs = [
            _FakeConfig(64, 64, 128, 4, 4),
            _FakeConfig(64, 128, 128, 4, 4),
        ]
        kept = self._prune(512, 8192, 2048, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 128, 128)})

    def test_bmn_floor_keeps_small_tiles_at_small_problem(self) -> None:
        configs = [
            _FakeConfig(16, 32, 32, 4, 2),
            _FakeConfig(32, 32, 32, 4, 2),
        ]
        kept = self._prune(64, 256, 256, configs=configs)
        self.assertEqual(self._keys(kept), {(16, 32, 32), (32, 32, 32)})

    def test_register_cap_drops_spiller(self) -> None:
        configs = [
            _FakeConfig(256, 256, 64, 4, 4),
            _FakeConfig(128, 256, 128, 3, 8),
        ]
        kept = self._prune(8192, 8192, 2048, configs=configs)
        self.assertEqual(self._keys(kept), {(128, 256, 128)})

    def test_register_cap_boundary(self) -> None:
        configs = [
            _FakeConfig(255, 128, 64, 4, 4),
            _FakeConfig(256, 128, 64, 4, 4),
        ]
        kept = self._prune(8192, 8192, 2048, configs=configs)
        self.assertEqual(self._keys(kept), {(255, 128, 64)})

    def test_wgmma_floors_keep_minimum_tiles(self) -> None:
        configs = [
            _FakeConfig(64, 32, 32, 4, 4),
            _FakeConfig(128, 32, 32, 4, 4),
            _FakeConfig(64, 64, 32, 4, 4),
            _FakeConfig(64, 32, 64, 4, 4),
        ]
        kept = self._prune(1, 1, 1, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 32, 32)})

    def test_tiny_n_keeps_bn32_winner(self) -> None:
        configs = [
            _FakeConfig(16, 32, 32, 3, 2),
            _FakeConfig(64, 64, 32, 4, 4),
            _FakeConfig(16, 128, 32, 3, 2),
        ]
        kept = self._prune(3, 4, 5, configs=configs)
        keys = self._keys(kept)
        self.assertIn((16, 32, 32), keys)
        self.assertNotIn((16, 128, 32), keys)
        self.assertLess(len(kept), len(configs))

    def test_mid_m_drops_block_m_256(self) -> None:
        kept = self._prune(64, 1024, 1024)
        bms = {c.kwargs["BLOCK_M"] for c in kept}
        self.assertNotIn(256, bms)

    def test_large_m_keeps_block_m_256(self) -> None:
        kept = self._prune(8192, 1024, 256)
        keys = self._keys(kept)
        self.assertIn((256, 128, 32), keys)
        self.assertNotIn((256, 256, 32), keys)

    def test_stages_kept_when_pipeline_saturates(self) -> None:
        for m in (1, 8192):
            kept = self._prune(m, 1024, 256)
            stages = sorted(c.num_stages for c in kept if c.kwargs["BLOCK_M"] == 16)
            self.assertEqual(stages, [2, 3, 4, 5, 6])

    def test_smem_fit_keeps_tightest_config(self) -> None:
        kept = self._prune(512, 1024, 19712)
        self.assertIn((64, 128, 256), self._keys(kept))

    def test_smem_drops_infeasible_config(self) -> None:
        configs = [
            _FakeConfig(128, 256, 256, 4, 8),
            _FakeConfig(64, 64, 128, 5, 2),
        ]
        kept = self._prune(512, 1024, 19712, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 128)})

    def test_pipeline_drops_unsaturable_stages(self) -> None:
        configs = [_FakeConfig(16, 32, 64, ns, 2) for ns in (2, 3, 4, 5, 6)]
        kept = self._prune(64, 256, 256, configs=configs)
        stages = sorted(c.num_stages for c in kept)
        self.assertEqual(stages, [2, 3, 4, 5])

    def test_pipeline_keeps_all_at_single_k_tile(self) -> None:
        configs = [_FakeConfig(16, 32, 32, ns, 2) for ns in (2, 3, 4, 5, 6)]
        kept = self._prune(512, 1024, 16, configs=configs)
        stages = sorted(c.num_stages for c in kept)
        self.assertEqual(stages, [2, 3, 4, 5, 6])

    def test_pipeline_ignores_compute_configs(self) -> None:
        configs = [_FakeConfig(64, 32, 32, 5, 2)]
        kept = self._prune(512, 1024, 64, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 32, 32)})

    def test_fallback_returns_full_space_when_all_pruned(self) -> None:
        configs = [c for c in self._make_configs() if c.kwargs["BLOCK_M"] >= 128]
        self.assertTrue(len(configs) > 0)
        kept = self._prune(1, 1024, 1024, configs=configs)
        self.assertEqual(kept, configs)

    def test_non_hopper_is_noop(self) -> None:
        configs = self._make_configs()
        kept = self._prune(1, 1024, 1024, hopper=False, configs=configs)
        self.assertEqual(kept, configs)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class TestPerfModelFp8(unittest.TestCase):
    """estimate_matmul_time must accept torch fp8 dtypes.

    Regression test: Triton's peak-flops table has no torch fp8 entries, so
    unmapped dtypes crash with "dtype not supported" (this also fixes a latent
    crash for block kernels whenever early pruning keeps more than top_k).
    """

    def test_fp8_dtypes_accepted(self) -> None:
        device = torch.device("cuda")
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            a = torch.empty(64, 256, dtype=dtype, device=device)
            b = torch.empty(256, 256, dtype=dtype, device=device)
            c = torch.empty(64, 256, dtype=torch.bfloat16, device=device)
            est = estimate_matmul_time(4, 4, a, b, c, 64, 256, 256, 64, 128, 32, 1)
            self.assertTrue(est > 0)
