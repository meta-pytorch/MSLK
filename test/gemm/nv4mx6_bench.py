# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import time

import mslk.gemm  # noqa: F401
import mslk.quantize  # noqa: F401
import torch
from mslk.gemm.triton.fp8_gemm import to_mxfp8
from mslk.quantize.triton.fp4_quantize import _to_blocked


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        print("SKIP: requires SM100+ (Blackwell)")
        return 0

    shapes = [
        (16, 8192, 16384),
        (128, 8192, 16384),
        (256, 8192, 16384),
        (512, 8192, 16384),
        (1024, 8192, 16384),
        (2048, 8192, 16384),
        (4096, 8192, 16384),
    ]

    print(
        f"{'M':>6} {'N':>6} {'K':>6} | "
        f"{'mx8mx6 (ms)':>12} {'nv4mx6_fused (ms)':>18} {'speedup%':>9} "
        f"{'SQNR_fused':>11} {'SQNR_mx6':>9}"
    )
    print("-" * 95)

    for M, N, K in shapes:
        a_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 0.1
        b_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device="cuda") * 0.01

        # BF16 reference
        out_bf16 = (a_bf16 @ b_bf16.t()).float()

        # MX8 activations (shared)
        (a_scale_raw, aq) = to_mxfp8(a_bf16)
        a_scale = _to_blocked(a_scale_raw.view(torch.int8).reshape(M, -1)).view(
            torch.uint8
        )

        # MX6 weights for standard kernel [N, K]
        (b_scale_raw, bq_fp8) = to_mxfp8(b_bf16)
        b_scale_mx6 = _to_blocked(b_scale_raw.view(torch.int8).reshape(N, -1)).view(
            torch.uint8
        )
        bq_mx6 = bq_fp8.view(torch.uint8) & 0x3F

        # MX4 weights for fused kernel [N, K/2]
        from mslk.quantize.triton.fp4_quantize import triton_quantize_mx4_unpack

        bq_mx4, b_scale_mx4 = triton_quantize_mx4_unpack(b_bf16)
        wq_packed = bq_mx4.view(torch.uint8)
        w_scale_packed = b_scale_mx4.view(torch.uint8).flatten()

        # Performance
        t_mx6 = bench(
            lambda _aq=aq,
            _bq=bq_mx6,
            _as=a_scale,
            _bs=b_scale_mx6: torch.ops.mslk.mx8mx6bf16(_aq, _bq, _as, _bs)
        )
        t_fused = bench(
            lambda _aq=aq,
            _wq=wq_packed,
            _as=a_scale,
            _ws=w_scale_packed: torch.ops.mslk.nv4mx6bf16_fused(_aq, _wq, _as, _ws)
        )

        # SQNR: fused (MX4→MX6) vs bf16
        out_fused = torch.ops.mslk.nv4mx6bf16_fused(
            aq, wq_packed, a_scale, w_scale_packed
        ).float()
        noise_fused = out_fused - out_bf16
        sqnr_fused = (
            10 * torch.log10((out_bf16**2).mean() / (noise_fused**2).mean()).item()
        )

        # SQNR: standard mx8mx6 vs bf16
        out_mx6 = torch.ops.mslk.mx8mx6bf16(aq, bq_mx6, a_scale, b_scale_mx6).float()
        noise_mx6 = out_mx6 - out_bf16
        sqnr_mx6 = 10 * torch.log10((out_bf16**2).mean() / (noise_mx6**2).mean()).item()

        speedup = (t_mx6 - t_fused) / t_mx6 * 100
        print(
            f"{M:>6} {N:>6} {K:>6} | "
            f"{t_mx6:>12.4f} {t_fused:>18.4f} {speedup:>+8.1f}% "
            f"{sqnr_fused:>10.2f} {sqnr_mx6:>9.2f}"
        )

    # Verification: run fused kernel twice with same inputs, confirm deterministic
    print("\n--- Verification ---")
    M, N, K = 256, 8192, 16384
    a_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 0.1
    (a_scale_raw, aq) = to_mxfp8(a_bf16)
    a_scale = _to_blocked(a_scale_raw.view(torch.int8).reshape(M, -1)).view(torch.uint8)
    wq_nv4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    w_scale_nv4 = torch.randint(
        0, 128, (N * K // 16,), dtype=torch.uint8, device="cuda"
    )

    out1 = torch.ops.mslk.nv4mx6bf16_fused(aq, wq_nv4, a_scale, w_scale_nv4)
    out2 = torch.ops.mslk.nv4mx6bf16_fused(aq, wq_nv4, a_scale, w_scale_nv4)
    torch.cuda.synchronize()

    if torch.equal(out1, out2):
        print("PASS: nv4mx6bf16_fused is deterministic (same input → same output)")
    else:
        print("FAIL: outputs differ between runs!")
        return 1

    if not out1.isnan().any() and not out1.isinf().any():
        print(
            f"PASS: no NaN/Inf in output "
            f"(shape={tuple(out1.shape)}, dtype={out1.dtype})"
        )
    else:
        print("FAIL: output contains NaN or Inf")
        return 1

    print(
        f"Output stats: min={out1.min().item():.4f}, max={out1.max().item():.4f}, "
        f"mean={out1.float().mean().item():.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
