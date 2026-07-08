# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""MX mixed-dtype quantization + packing utilities.

Currently:
    - pack_fp6_e2m3:            4 unpacked uint8 -> 3 packed bytes (LSB-first)
    - quantize_bf16_to_mx6_e2m3: BF16 -> unpacked MX6 (E2M3) bytes + E8M0 scales

Used by the mxf8f6f4-family CUTLASS GEMM kernels (mx8mx6bf16 etc), whose
WQ operand is read as a bit-contiguous 6-bit stream. Callers that produce
MX6 weights (quantizers, LUT ops) must feed packed bytes into these
kernels; the mx8mx6bf16 op TORCH_CHECKs on the packed shape.
"""

from __future__ import annotations

import torch


__all__ = [
    "E2M3_DECODE",
    "E2M3_MAX",
    "pack_fp6_e2m3",
    "quantize_bf16_to_mx6_e2m3",
]


# E2M3 magnitude decode table — 32 unsigned values. Copied from
# fbcode/ai_codesign/numerics/quantization/experimental/lut_upconvert/lut.py
# (D99909748); also baked into the LUT kernel (nvfp4_to_mx6.cuh) and
# matches cutlass's float_e2m3_t decode.
E2M3_DECODE: tuple[float, ...] = (
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.375,
    1.5,
    1.625,
    1.75,
    1.875,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
)
E2M3_MAX: float = 7.5


def pack_fp6_e2m3(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack 6-bit FP6 (E2M3) values stored as bytes into bit-packed bytes.

    4 unpacked values (24 bits) -> 3 packed bytes, LSB-first.
    Input shape (..., K), output shape (..., K * 6 // 8).
    cutlass mx_float6_t<float_e2m3_t> has sizeof_bits == 6 and reads WQ as
    a bit-contiguous 6-bit stream. Pass the output to
    torch.ops.mslk.mx8mx6bf16 (or any mxf8f6f4-family kernel with an MX6
    operand).
    """
    assert unpacked.dtype == torch.uint8
    assert unpacked.shape[-1] % 4 == 0
    leading = unpacked.shape[:-1]
    K = unpacked.shape[-1]
    # uint16 has no CUDA bitwise ops; widen to int32 for the pack arithmetic.
    x = unpacked.reshape(*leading, K // 4, 4).to(torch.int32)
    b0 = (x[..., 0] & 0x3F) | ((x[..., 1] & 0x03) << 6)
    b1 = ((x[..., 1] >> 2) & 0x0F) | ((x[..., 2] & 0x0F) << 4)
    b2 = ((x[..., 2] >> 4) & 0x03) | ((x[..., 3] & 0x3F) << 2)
    packed = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8)
    return packed.reshape(*leading, K * 6 // 8).contiguous()


def quantize_bf16_to_mx6_e2m3(
    x: torch.Tensor, block_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 -> MX6 (E2M3 + E8M0 BS32 scale) quantizer.

    Vectorized reimplementation of the BF16->MX6 path from
    nv4_to_mx6_baseline_dequant() in D99909748's test_real_weights.py:
        amax/block -> shared_exp = ceil(log2(amax / 7.5)) + 127
                   -> snap each element to nearest E2M3 code

    Returns:
        bytes:  [..., K] uint8, byte = [zero:2 | sign:1 | mag_idx:5]
        scales: [..., K // block_size] uint8 E8M0 (biased exponent)
    """
    assert x.shape[-1] % block_size == 0
    leading = x.shape[:-1]
    K = x.shape[-1]
    nb = K // block_size

    codes = torch.tensor(E2M3_DECODE, dtype=torch.float32, device=x.device)
    blocks = x.float().reshape(*leading, nb, block_size)
    sign = (blocks < 0).to(torch.uint8)
    mag = blocks.abs()

    amax = mag.amax(dim=-1, keepdim=True)
    safe_amax = torch.clamp(amax, min=1e-38)
    # Clamp k_exp to the E8M0-representable range [-127, 128], then use the
    # clamped value for BOTH the emitted scale byte AND the normalization
    # factor. Otherwise blocks with amax outside 2**(-127..128) would emit
    # a clamped scale byte but normalize against an unclamped 2**k_exp,
    # producing a quantizer/dequantizer mismatch on extreme inputs.
    k_exp = (
        torch.ceil(torch.log2(safe_amax / E2M3_MAX)).clamp(-127, 128).to(torch.int32)
    )
    scale_byte = (k_exp + 127).to(torch.uint8)
    scale = torch.pow(torch.tensor(2.0, device=x.device), k_exp.float())

    normalized = mag / scale
    # Snap each value to nearest of the 32 E2M3 codes via argmin distance.
    e2m3_idx = (normalized.unsqueeze(-1) - codes).abs().argmin(dim=-1).to(torch.uint8)

    out = ((sign << 5) | (e2m3_idx & 0x1F)).reshape(*leading, K)
    return out, scale_byte.squeeze(-1)
