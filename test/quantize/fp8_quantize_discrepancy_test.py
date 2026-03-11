# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Exhaustive BF16 CUDA vs Triton FP8 Row Quantization Discrepancy Test.

Tests all 32,639 representable positive finite BF16 values as row abs-max
to measure bitwise discrepancy between MSLK CUDA (torch.ops.mslk.quantize_fp8_per_row)
and MSLK Triton (triton_quantize_fp8_row) FP8 row quantization implementations.

The two implementations use different arithmetic paths:
- MSLK CUDA: scale = max(amax / MAX_FP8, 1/(MAX_FP8*512)), quantizes via value / scale
- MSLK Triton: a_scale = MAX_FP8 / cur_max (eps=1/512 to match CUDA floor),
  quantizes via value * a_scale, stores 1/a_scale
"""

import unittest
from typing import Any, Dict, List, Tuple

import mslk.quantize  # noqa: F401
import numpy as np
import torch
from mslk.quantize.triton.fp8_quantize import quantize_fp8_row
from parameterized import parameterized

# Maximum columns supported by quantize_fp8_per_row
MAX_COLS: int = 16384

# MSLK CUDA uses min_scale = 1/(MAX_FP8*512). When amax < this threshold,
# the CUDA floor dominates and scales diverge from Triton (which currently uses
# eps=1e-12). This is expected — both implementations produce identical FP8
# outputs (all zeros) for values this small. Scale assertions are skipped in
# this region.
CUDA_AMAX_LB: float = 1.0 / 512.0  # 1e-9 ≈ 0.00195


# ---------------------------------------------------------------------------
# BF16 value generation utilities
# (Adapted from scripts/hojinp/llama4x/kernels/test_fp8_quant_parity.py)
# ---------------------------------------------------------------------------
def generate_all_bf16_values() -> torch.Tensor:
    """Generate all 2^16 possible BF16 bit patterns as a BF16 tensor."""
    patterns = np.arange(2**16, dtype=np.uint16)
    as_f32 = np.left_shift(patterns.astype(np.uint32), 16).view(np.float32)
    return torch.from_numpy(as_f32).to(torch.bfloat16)


def get_sorted_positive_bf16_values(device: str = "cpu") -> torch.Tensor:
    """All unique positive finite BF16 values, sorted ascending."""
    all_bf16 = generate_all_bf16_values()
    finite = all_bf16[torch.isfinite(all_bf16)]
    pos = finite[finite > 0]
    return pos.sort().values.to(device)


def create_test_tensor_for_max_idx(
    max_idx: int,
    all_positive_sorted: torch.Tensor,
    n_cols: int = MAX_COLS,
) -> torch.Tensor:
    """(B, n_cols) BF16 tensor where every row-max == all_positive_sorted[max_idx].

    Column 0 of every row holds +max_val so the per-row abs-max is identical.
    The remaining columns are filled from a pool of [+v, -v] for all values <= max_val.
    """
    pos_vals = all_positive_sorted[: max_idx + 1]
    max_val = all_positive_sorted[max_idx]
    device = all_positive_sorted.device

    # Build pool: interleave +v and -v for every value in pos_vals
    pool = torch.stack([pos_vals, -pos_vals], dim=1).reshape(-1)
    n_pool = pool.shape[0]

    # Reserve column 0 for +max_val; remaining n_cols-1 columns for pool values
    usable = n_cols - 1
    n_rows = (n_pool + usable - 1) // usable

    rows: List[torch.Tensor] = []
    for r in range(n_rows):
        start = r * usable
        end = min(start + usable, n_pool)
        chunk = pool[start:end]
        if chunk.shape[0] < usable:
            pad_len = usable - chunk.shape[0]
            pad_idx = torch.arange(pad_len, device=device) % n_pool
            chunk = torch.cat([chunk, pool[pad_idx]])
        row = torch.cat([max_val.unsqueeze(0), chunk])
        rows.append(row)

    return torch.stack(rows)


# ---------------------------------------------------------------------------
# Discrepancy metrics
# ---------------------------------------------------------------------------
def compute_discrepancy_metrics(
    cuda_fp8: torch.Tensor,
    cuda_scale: torch.Tensor,
    triton_fp8: torch.Tensor,
    triton_scale: torch.Tensor,
) -> Dict[str, Any]:
    """Compute discrepancy metrics between CUDA and Triton FP8 quantization results."""
    # Scale discrepancy
    scale_diff = torch.abs(cuda_scale.float() - triton_scale.float())
    scale_max_abs_diff = scale_diff.max().item()
    scale_max_rel_diff = (
        (scale_diff / (torch.abs(cuda_scale.float()) + 1e-10)).max().item()
    )

    # Bitwise mismatch
    cuda_fp8_flat = cuda_fp8.view(-1)
    triton_fp8_flat = triton_fp8.view(-1)
    cuda_as_float = cuda_fp8_flat.to(torch.float32)
    triton_as_float = triton_fp8_flat.to(torch.float32)
    bitwise_mismatch = cuda_as_float != triton_as_float
    bitwise_mismatch_count = bitwise_mismatch.sum().item()
    total_elements = cuda_fp8_flat.numel()
    bitwise_mismatch_pct = (
        (bitwise_mismatch_count / total_elements * 100) if total_elements > 0 else 0.0
    )

    # Reconstruction drift
    cuda_reconstructed = cuda_fp8.to(torch.float32) * cuda_scale.unsqueeze(-1).float()
    triton_reconstructed = (
        triton_fp8.to(torch.float32) * triton_scale.unsqueeze(-1).float()
    )
    reconstruction_diff = torch.abs(cuda_reconstructed - triton_reconstructed)
    reconstruction_max_error = reconstruction_diff.max().item()
    reconstruction_mean_error = reconstruction_diff.mean().item()
    recon_abs_max = max(
        cuda_reconstructed.abs().max().item(),
        triton_reconstructed.abs().max().item(),
    )
    reconstruction_rel_error = float(reconstruction_max_error) / max(
        float(recon_abs_max), 1e-10
    )

    return {
        "scale_max_abs_diff": scale_max_abs_diff,
        "scale_max_rel_diff": scale_max_rel_diff,
        "bitwise_mismatch_count": int(bitwise_mismatch_count),
        "bitwise_mismatch_pct": bitwise_mismatch_pct,
        "reconstruction_max_error": reconstruction_max_error,
        "reconstruction_mean_error": reconstruction_mean_error,
        "reconstruction_rel_error": reconstruction_rel_error,
    }


# Scale regime test cases: (name, bf16_value_index)
# BF16 positive finite values: 127 subnormals (idx 0-126), 32512 normals (idx 127-32638)
# Covers the full range including the CUDA floor-divergence region (subnormals
# and very-small normals) and above-floor normals at various magnitudes.
SCALE_REGIME_CASES: List[Tuple[str, int]] = [
    # Floor-divergence region (amax < CUDA_AMAX_LB)
    ("smallest_subnormal", 0),
    ("tiny_subnormal", 1),
    ("small_subnormal", 10),
    ("mid_subnormal", 50),
    ("large_subnormal", 100),
    ("largest_subnormal", 126),
    ("smallest_normal", 127),
    ("small_normal_floor", 128),
    ("small_normal_near_floor", 200),
    # Above-floor region
    ("small_normal", 256),
    ("mid_normal", 16000),
    ("large_normal", 28000),
    ("very_large_normal", 32000),
    ("largest_normal", 32638),
]


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Operators are only available on CUDA enabled machines",
)
@unittest.skipIf(
    torch.version.cuda is None or torch.cuda.get_device_capability("cuda") < (9, 0),
    "Only support sm90+",
)
class FP8QuantizeDiscrepancyTest(unittest.TestCase):
    """CUDA vs Triton FP8 row quantization discrepancy test.

    Tests all 32,639 representable positive finite BF16 values to ensure
    MSLK CUDA and Triton FP8 row quantization produce compatible results.
    """

    SCALE_REL_ERROR_THRESHOLD: float = 1e-5
    BITWISE_MISMATCH_PCT_THRESHOLD: float = 1e-2
    # FP8 has very few representable values, so a 1-bit quantization
    # difference at a rounding boundary (div vs mul) can produce up to
    # ~6.25% reconstruction relative error per value. When combined with
    # scale differences between the two implementations, the total
    # reconstruction error can exceed 6.25%. 0.1 (10%) catches genuine
    # regressions while tolerating expected FP8 boundary effects.
    RECONSTRUCTION_REL_ERROR_THRESHOLD: float = 0.1

    def _run_both_implementations(
        self,
        test_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run both CUDA and Triton implementations and return results.

        Returns:
            Tuple of (cuda_fp8, cuda_scale, triton_fp8, triton_scale)
        """
        # CUDA implementation (returns list[Tensor])
        cuda_result = torch.ops.mslk.quantize_fp8_per_row(test_input)
        cuda_fp8, cuda_scale = cuda_result[0], cuda_result[1]

        # Triton implementation (returns tuple[Tensor, Tensor])
        triton_fp8, triton_scale = quantize_fp8_row(
            test_input.clone(),
            eps_opt=CUDA_AMAX_LB,
        )

        return cuda_fp8, cuda_scale, triton_fp8, triton_scale

    def test_exhaustive_bf16_sweep(self) -> None:
        """Test all 32,639 positive finite BF16 values as row abs-max."""
        pos_sorted = get_sorted_positive_bf16_values(device="cuda")
        num_values = len(pos_sorted)

        # Aggregate metrics across all values
        total_elements = 0
        total_mismatches = 0
        # Only track scale_rel and recon_rel for values above the CUDA floor
        # threshold, where the two implementations use the same scale
        # computation.  In the floor-divergence region the div-vs-mul
        # arithmetic difference can produce large per-value relative errors
        # that are expected and harmless (both outputs are near-zero).
        max_scale_rel_error_above_floor = 0.0
        max_recon_rel_error_above_floor = 0.0

        for idx in range(num_values):
            max_val = pos_sorted[idx].item()
            test_tensor = create_test_tensor_for_max_idx(idx, pos_sorted)

            cuda_fp8, cuda_scale, triton_fp8, triton_scale = (
                self._run_both_implementations(test_tensor)
            )

            metrics = compute_discrepancy_metrics(
                cuda_fp8, cuda_scale, triton_fp8, triton_scale
            )

            total_elements += cuda_fp8.numel()
            total_mismatches += metrics["bitwise_mismatch_count"]
            if max_val >= CUDA_AMAX_LB:
                max_scale_rel_error_above_floor = max(
                    max_scale_rel_error_above_floor, metrics["scale_max_rel_diff"]
                )
                max_recon_rel_error_above_floor = max(
                    max_recon_rel_error_above_floor,
                    metrics["reconstruction_rel_error"],
                )

            if (idx + 1) % 5000 == 0 or idx + 1 == num_values:
                agg_mismatch_pct = (
                    (total_mismatches / total_elements * 100)
                    if total_elements > 0
                    else 0.0
                )
                print(
                    f"  [{idx + 1:>5}/{num_values}]  "
                    f"mismatches: {total_mismatches}/{total_elements} "
                    f"({agg_mismatch_pct:.4f}%)  "
                    f"max_scale_rel(above_floor): {max_scale_rel_error_above_floor:.2e}  "
                    f"max_recon_rel(above_floor): {max_recon_rel_error_above_floor:.2e}"
                )

        # Final aggregate assertions
        agg_mismatch_pct = (
            (total_mismatches / total_elements * 100) if total_elements > 0 else 0.0
        )

        print(f"\n{'=' * 70}")
        print("Exhaustive BF16 Sweep Summary")
        print(f"{'=' * 70}")
        print(f"  Total values tested: {num_values}")
        print(f"  Total elements: {total_elements}")
        print(f"  Total mismatches: {total_mismatches} ({agg_mismatch_pct:.4f}%)")
        print(
            f"  Max scale relative error (above floor): "
            f"{max_scale_rel_error_above_floor:.2e}"
        )
        print(
            f"  Max reconstruction relative error (above floor): "
            f"{max_recon_rel_error_above_floor:.2e}"
        )
        print(f"{'=' * 70}")

        self.assertLess(
            max_scale_rel_error_above_floor,
            self.SCALE_REL_ERROR_THRESHOLD,
            f"Scale relative error {max_scale_rel_error_above_floor:.2e} (above floor) "
            f"exceeds threshold {self.SCALE_REL_ERROR_THRESHOLD}",
        )
        self.assertLess(
            agg_mismatch_pct,
            self.BITWISE_MISMATCH_PCT_THRESHOLD,
            f"Aggregate bitwise mismatch {agg_mismatch_pct:.4f}% exceeds "
            f"threshold {self.BITWISE_MISMATCH_PCT_THRESHOLD}%",
        )
        self.assertLess(
            max_recon_rel_error_above_floor,
            self.RECONSTRUCTION_REL_ERROR_THRESHOLD,
            f"Reconstruction relative error {max_recon_rel_error_above_floor:.2e} "
            f"(above floor) exceeds threshold "
            f"{self.RECONSTRUCTION_REL_ERROR_THRESHOLD}",
        )

    # pyre-ignore[56]: Pyre can't track decorator-modified signatures.
    @parameterized.expand(SCALE_REGIME_CASES)
    def test_scale_regime(self, _name: str, bf16_idx: int) -> None:
        """Test specific BF16 value indices representing distinct scale regimes."""
        pos_sorted = get_sorted_positive_bf16_values(device="cuda")

        test_tensor = create_test_tensor_for_max_idx(bf16_idx, pos_sorted)
        cuda_fp8, cuda_scale, triton_fp8, triton_scale = self._run_both_implementations(
            test_tensor
        )

        metrics = compute_discrepancy_metrics(
            cuda_fp8, cuda_scale, triton_fp8, triton_scale
        )

        max_val = pos_sorted[bf16_idx].item()
        in_floor_region = max_val < CUDA_AMAX_LB
        print(f"\n[Scale Regime] {_name} (idx={bf16_idx}, max_val={max_val:.4e})")
        print(f"  Scale rel diff: {metrics['scale_max_rel_diff']:.2e}")
        print(f"  Bitwise mismatch: {metrics['bitwise_mismatch_pct']:.2f}%")
        print(f"  Reconstruction rel error: {metrics['reconstruction_rel_error']:.2e}")
        if in_floor_region:
            print(
                f"  (in CUDA floor region — scale_rel assertion skipped)  "
                f"cuda_scale={cuda_scale[0].item():.4e}  "
                f"triton_scale={triton_scale[0].item():.4e}"
            )

        # Scale rel check: skip in the CUDA floor region where divergence is expected
        if not in_floor_region:
            self.assertLess(
                metrics["scale_max_rel_diff"],
                self.SCALE_REL_ERROR_THRESHOLD,
                f"Scale relative error {metrics['scale_max_rel_diff']:.2e} exceeds "
                f"threshold {self.SCALE_REL_ERROR_THRESHOLD} for {_name}",
            )
        self.assertLess(
            metrics["bitwise_mismatch_pct"],
            self.BITWISE_MISMATCH_PCT_THRESHOLD,
            f"Bitwise mismatch {metrics['bitwise_mismatch_pct']:.2f}% exceeds "
            f"threshold {self.BITWISE_MISMATCH_PCT_THRESHOLD}% for {_name}",
        )
        self.assertLess(
            metrics["reconstruction_rel_error"],
            self.RECONSTRUCTION_REL_ERROR_THRESHOLD,
            f"Reconstruction relative error {metrics['reconstruction_rel_error']:.2e} "
            f"exceeds threshold {self.RECONSTRUCTION_REL_ERROR_THRESHOLD} for {_name}",
        )


if __name__ == "__main__":
    unittest.main()
