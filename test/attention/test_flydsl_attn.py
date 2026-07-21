# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

"""Correctness tests for the FlyDSL flash-attention backend (ROCm / gfx950).

Validates the forward output and the optional per-row log-sum-exp (LSE) emitted
by ``mslk.attention.flydsl.flydsl_flash_attn_func`` against a float32 PyTorch
reference. The LSE interface contract is natural-log with the softmax scale
folded in::

    LSE_i = ln( sum_j exp( sm_scale * (q_i . k_j) ) )   (masked keys excluded)

with ``sm_scale = 1 / sqrt(head_dim)`` and fully-masked rows storing ``-inf``.
Causal masking is bottom-right aligned (key ``j`` is visible to query ``i`` iff
``j <= i + (Skv - Sq)``), matching the kernel. The suite is skipped unless CUDA
is available and FlyDSL supports the current GPU arch.
"""

import math
import unittest

import torch
from mslk.utils.flydsl import is_flydsl_available
from parameterized import parameterized

if torch.cuda.is_available() and is_flydsl_available():
    from mslk.attention.flydsl import flydsl_flash_attn_func

# bf16/f16 dot-products accumulate with a slightly different order than the fp32
# reference, so allow a small absolute tolerance (output values are O(1)).
_ATOL_OUT = {torch.bfloat16: 3e-2, torch.float16: 1.5e-2}
_ATOL_LSE = {torch.bfloat16: 8e-3, torch.float16: 4e-3}


def _sm_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference (out, lse) for dense inputs.

    q: [B, Sq, H, D], k/v: [B, Skv, Hkv, D] -> out: [B, Sq, H, D], lse: [B, H, Sq].
    GQA/MQA is handled by repeating KV heads. Bottom-right aligned causal mask.
    """
    B, Sq, H, D = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    sm = _sm_scale(D)
    qf = q.float().permute(0, 2, 1, 3)  # B, H, Sq, D
    kf = (
        k.float().permute(0, 2, 1, 3).repeat_interleave(H // Hkv, dim=1)
    )  # B, H, Skv, D
    vf = (
        v.float().permute(0, 2, 1, 3).repeat_interleave(H // Hkv, dim=1)
    )  # B, H, Skv, D
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * sm  # B, H, Sq, Skv
    if causal:
        qi = torch.arange(Sq, device=q.device).view(Sq, 1)
        ki = torch.arange(Skv, device=q.device).view(1, Skv)
        mask = (ki > (qi + (Skv - Sq))).view(1, 1, Sq, Skv)
        scores = scores.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # B, H, Sq
    # softmax over visible keys; fully-masked rows (all -inf) -> 0 weights.
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    out = torch.einsum("bhqk,bhkd->bhqd", p, vf).permute(0, 2, 1, 3)  # B, Sq, H, D
    return out, lse


def _assert_out_matches(out: torch.Tensor, out_ref: torch.Tensor, atol: float) -> None:
    diff = (out.float() - out_ref.float()).abs().max().item()
    assert diff <= atol, f"output max abs diff {diff:.3e} exceeds atol {atol:.3e}"


def _assert_lse_matches(lse: torch.Tensor, lse_ref: torch.Tensor, atol: float) -> None:
    """Finite rows match within atol; -inf (fully-masked) rows match exactly."""
    lse = lse.float()
    lse_ref = lse_ref.float()
    finite = torch.isfinite(lse_ref)
    if (~finite).any():
        assert bool(((~torch.isfinite(lse)) == (~finite)).all()), (
            "fully-masked (-inf) rows mismatch"
        )
    if finite.any():
        diff = (lse[finite] - lse_ref[finite]).abs().max().item()
        assert diff <= atol, f"LSE max abs diff {diff:.3e} exceeds atol {atol:.3e}"


def _rand(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


# (name, dtype, causal, B, S, H, Hkv, D)
_DENSE_CASES = []
for _dtype, _tag in ((torch.bfloat16, "bf16"), (torch.float16, "f16")):
    _shapes = [
        ("mha", 2, 256, 8, 8, 128),
        ("gqa", 2, 256, 8, 2, 128),
        ("d64", 2, 192, 4, 4, 64),
    ]
    # Keep f16 coverage focused on MHA to limit JIT-compile count.
    if _dtype == torch.float16:
        _shapes = _shapes[:1]
    for _sname, _B, _S, _H, _Hkv, _D in _shapes:
        for _causal in (False, True):
            _DENSE_CASES.append(
                (
                    f"{_tag}_{_sname}_{'causal' if _causal else 'full'}",
                    _dtype,
                    _causal,
                    _B,
                    _S,
                    _H,
                    _Hkv,
                    _D,
                )
            )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA/ROCm not available")
@unittest.skipUnless(
    torch.cuda.is_available() and is_flydsl_available(),
    "FlyDSL not available for this GPU arch",
)
class FlyDSLFlashAttnTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    @parameterized.expand(_DENSE_CASES)
    def test_dense_out_and_lse(
        self,
        _name: str,
        dtype: torch.dtype,
        causal: bool,
        B: int,
        S: int,
        H: int,
        Hkv: int,
        D: int,
    ) -> None:
        q = _rand(B, S, H, D, dtype=dtype)
        k = _rand(B, S, Hkv, D, dtype=dtype)
        v = _rand(B, S, Hkv, D, dtype=dtype)
        out, lse = flydsl_flash_attn_func(
            q, k, v, causal=causal, num_kv_heads=Hkv, return_lse=True
        )
        torch.cuda.synchronize()
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(lse.shape, (B, H, S))
        self.assertEqual(lse.dtype, torch.float32)
        out_ref, lse_ref = _reference(q, k, v, causal)
        _assert_out_matches(out, out_ref, _ATOL_OUT[dtype])
        _assert_lse_matches(lse, lse_ref, _ATOL_LSE[dtype])

    @parameterized.expand(
        [
            ("d128_split2_full", 128, 2, False),
            ("d128_split2_causal", 128, 2, True),
            ("d64_split3_causal", 64, 3, True),
        ]
    )
    def test_splitk_out_and_lse(
        self, _name: str, D: int, num_kv_splits: int, causal: bool
    ) -> None:
        # Split-K requires seq_len >= 384, D in {64,128}, bf16/f16.
        B, S, H = 1, 512, 8
        dtype = torch.bfloat16
        q = _rand(B, S, H, D, dtype=dtype)
        k = _rand(B, S, H, D, dtype=dtype)
        v = _rand(B, S, H, D, dtype=dtype)
        out, lse = flydsl_flash_attn_func(
            q, k, v, causal=causal, num_kv_splits=num_kv_splits, return_lse=True
        )
        torch.cuda.synchronize()
        self.assertEqual(lse.shape, (B, H, S))
        out_ref, lse_ref = _reference(q, k, v, causal)
        _assert_out_matches(out, out_ref, _ATOL_OUT[dtype])
        _assert_lse_matches(lse, lse_ref, _ATOL_LSE[dtype])

    @parameterized.expand([("full", False), ("causal", True)])
    def test_varlen_lse(self, _name: str, causal: bool) -> None:
        dtype = torch.bfloat16
        D, H, Hkv = 128, 4, 4
        seqs = [100, 200, 60]
        cu = torch.tensor([0, 100, 300, 360], dtype=torch.int32, device="cuda")
        total = int(cu[-1].item())
        max_seqlen_q = max(seqs)
        q = _rand(total, H, D, dtype=dtype)
        k = _rand(total, Hkv, D, dtype=dtype)
        v = _rand(total, Hkv, D, dtype=dtype)
        out, lse = flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max_seqlen_q,
            cross_seqlen=False,
            return_lse=True,
        )
        torch.cuda.synchronize()
        self.assertEqual(lse.shape, (len(seqs), H, max_seqlen_q))
        for b, (s0, s1) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
            n = s1 - s0
            qb = q[s0:s1].unsqueeze(0)  # 1, n, H, D
            kb = k[s0:s1].unsqueeze(0)
            vb = v[s0:s1].unsqueeze(0)
            out_ref, lse_ref = _reference(qb, kb, vb, causal)
            # Only the valid [:n] region is defined; padded rows are ignored.
            _assert_out_matches(out[s0:s1], out_ref[0], _ATOL_OUT[dtype])
            _assert_lse_matches(lse[b, :, :n], lse_ref[0], _ATOL_LSE[dtype])

    def test_fully_masked_rows_lse(self) -> None:
        """Cross-attn causal with Skv < Sq: leading query rows see no keys -> -inf."""
        dtype = torch.bfloat16
        B, Sq, Skv, H, D = 2, 128, 32, 4, 128
        q = _rand(B, Sq, H, D, dtype=dtype)
        k = _rand(B, Skv, H, D, dtype=dtype)
        v = _rand(B, Skv, H, D, dtype=dtype)
        _out, lse = flydsl_flash_attn_func(q, k, v, causal=True, return_lse=True)
        torch.cuda.synchronize()
        _out_ref, lse_ref = _reference(q, k, v, True)
        self.assertTrue(
            bool((~torch.isfinite(lse_ref)).any()),
            "test setup should produce fully-masked rows",
        )
        _assert_lse_matches(lse, lse_ref, _ATOL_LSE[dtype])

    def test_return_lse_false_returns_tensor(self) -> None:
        """Default return_lse=False returns a bare output tensor (not a tuple)."""
        dtype = torch.bfloat16
        q = _rand(2, 128, 4, 128, dtype=dtype)
        k = _rand(2, 128, 4, 128, dtype=dtype)
        v = _rand(2, 128, 4, 128, dtype=dtype)
        out = flydsl_flash_attn_func(q, k, v, causal=False)
        torch.cuda.synchronize()
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, q.shape)

    def test_return_lse_rejects_fp8(self) -> None:
        q = _rand(2, 128, 4, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        with self.assertRaises(NotImplementedError):
            flydsl_flash_attn_func(
                q,
                q,
                q,
                causal=False,
                q_descale=torch.ones(1, device="cuda"),
                k_descale=torch.ones(1, device="cuda"),
                v_descale=torch.ones(1, device="cuda"),
                return_lse=True,
            )


if __name__ == "__main__":
    unittest.main()
