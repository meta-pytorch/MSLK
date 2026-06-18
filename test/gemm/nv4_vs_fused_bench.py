# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Full comparison: f4f4 (NV4×NV4), mx8×mx4, mx8×mx6, nv4→mx6 fused.
Separate tables for Performance and SQNR.
All kernels use weights downcasted from same BF16 source.
"""

import time

import mslk.gemm  # noqa: F401
import mslk.quantize  # noqa: F401
import torch
from mslk.gemm.triton.fp8_gemm import to_mxfp8
from mslk.quantize.triton.fp4_quantize import _to_blocked, triton_quantize_mx4_unpack


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms


def sqnr_db(ref, approx):
    noise = approx - ref
    return 10 * torch.log10((ref**2).mean() / (noise**2).mean()).item()


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
        (64, 8192, 16384),
        (128, 8192, 16384),
        (256, 8192, 16384),
        (512, 8192, 16384),
        (1024, 8192, 16384),
        (2048, 8192, 16384),
        (4096, 8192, 16384),
    ]

    perf_results = []
    sqnr_results = []

    for M, N, K in shapes:
        A_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 0.1
        B_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device="cuda") * 0.01

        # BF16 reference
        out_ref = (A_bf16 @ B_bf16.t()).float()

        # --- Downcast activations ---
        # MX8 (for mx8×mx4, mx8×mx6, fused)
        (a_scale_raw, aq_mx8) = to_mxfp8(A_bf16)
        a_scale_mx8 = _to_blocked(a_scale_raw.view(torch.int8).reshape(M, -1)).view(
            torch.uint8
        )

        # MX4 activations (for f4f4)
        aq_mx4_raw, a_scale_mx4 = triton_quantize_mx4_unpack(A_bf16)
        aq_mx4 = aq_mx4_raw.view(torch.float4_e2m1fn_x2)

        # --- Downcast weights ---
        # MX4/NV4 weights (for f4f4, mx8×mx4, fused)
        bq_mx4_raw, b_scale_mx4 = triton_quantize_mx4_unpack(B_bf16)
        bq_mx4 = bq_mx4_raw.view(torch.float4_e2m1fn_x2)

        # NV4 packed (for fused kernel)
        wq_nv4 = bq_mx4_raw.view(torch.uint8)
        w_scale_nv4 = b_scale_mx4.view(torch.uint8).flatten()

        # MX6 weights (for mx8×mx6) — truncate FP8 to 6 bits
        (b_scale_raw_6, bq_fp8) = to_mxfp8(B_bf16)
        b_scale_mx6 = _to_blocked(b_scale_raw_6.view(torch.int8).reshape(N, -1)).view(
            torch.uint8
        )
        bq_mx6 = bq_fp8.view(torch.uint8) & 0x3F

        # --- Performance ---
        # f4f4 (NV4×NV4)
        try:
            t_f4f4 = bench(
                lambda _aq=aq_mx4,
                _bq=bq_mx4,
                _as=a_scale_mx4,
                _bs=b_scale_mx4: torch.ops.mslk.f4f4bf16(_aq, _bq, _as, _bs)
            )
        except RuntimeError:
            t_f4f4 = float("nan")

        # mx8×mx4
        try:
            t_mx8mx4 = bench(
                lambda _aq=aq_mx8,
                _bq=bq_mx4,
                _as=a_scale_mx8,
                _bs=b_scale_mx4: torch.ops.mslk.mx8mx4bf16(_aq, _bq, _as, _bs)
            )
        except RuntimeError:
            t_mx8mx4 = float("nan")

        # mx8×mx6
        t_mx8mx6 = bench(
            lambda _aq=aq_mx8,
            _bq=bq_mx6,
            _as=a_scale_mx8,
            _bs=b_scale_mx6: torch.ops.mslk.mx8mx6bf16(_aq, _bq, _as, _bs)
        )

        # nv4→mx6 fused
        t_fused = bench(
            lambda _aq=aq_mx8,
            _wq=wq_nv4,
            _as=a_scale_mx8,
            _ws=w_scale_nv4: torch.ops.mslk.nv4mx6bf16_fused(_aq, _wq, _as, _ws)
        )

        perf_results.append((M, N, K, t_f4f4, t_mx8mx4, t_mx8mx6, t_fused))

        # --- SQNR ---
        try:
            out_f4f4 = torch.ops.mslk.f4f4bf16(
                aq_mx4, bq_mx4, a_scale_mx4, b_scale_mx4
            ).float()
            s_f4f4 = sqnr_db(out_ref, out_f4f4)
        except RuntimeError:
            s_f4f4 = float("nan")

        try:
            out_mx8mx4 = torch.ops.mslk.mx8mx4bf16(
                aq_mx8, bq_mx4, a_scale_mx8, b_scale_mx4
            ).float()
            s_mx8mx4 = sqnr_db(out_ref, out_mx8mx4)
        except RuntimeError:
            s_mx8mx4 = float("nan")

        out_mx8mx6 = torch.ops.mslk.mx8mx6bf16(
            aq_mx8, bq_mx6, a_scale_mx8, b_scale_mx6
        ).float()
        s_mx8mx6 = sqnr_db(out_ref, out_mx8mx6)

        out_fused = torch.ops.mslk.nv4mx6bf16_fused(
            aq_mx8, wq_nv4, a_scale_mx8, w_scale_nv4
        ).float()
        s_fused = sqnr_db(out_ref, out_fused)

        sqnr_results.append((M, N, K, s_f4f4, s_mx8mx4, s_mx8mx6, s_fused))

    # --- Print Performance Table ---
    print("=" * 90)
    print(
        f"PERFORMANCE (ms) — {torch.cuda.get_device_name()}, N=8192, K=16384, iters=100"
    )
    print("=" * 90)
    print(
        f"{'M':>6} | {'f4f4':>8} {'mx8×mx4':>9} "
        f"{'mx8×mx6':>9} {'nv4→mx6':>9} | "
        f"{'fused vs mx6':>13}"
    )
    print("-" * 90)
    for M, N, K, t_f4f4, t_mx8mx4, t_mx8mx6, t_fused in perf_results:
        sp_vs_mx6 = (t_mx8mx6 - t_fused) / t_mx8mx6 * 100
        print(
            f"{M:>6} | {t_f4f4:>8.4f} {t_mx8mx4:>9.4f} "
            f"{t_mx8mx6:>9.4f} {t_fused:>9.4f} | "
            f"{sp_vs_mx6:>+12.1f}%"
        )

    # --- Print SQNR Table ---
    print()
    print("=" * 90)
    print("SQNR (dB vs BF16 reference) — higher is better")
    print("=" * 90)
    print(
        f"{'M':>6} | {'f4f4':>8} {'mx8×mx4':>9} "
        f"{'mx8×mx6':>9} {'nv4→mx6':>9} | "
        f"{'fused - f4f4':>13}"
    )
    print("-" * 90)
    for M, N, K, s_f4f4, s_mx8mx4, s_mx8mx6, s_fused in sqnr_results:
        gain = s_fused - s_f4f4 if s_f4f4 == s_f4f4 else float("nan")
        print(
            f"{M:>6} | {s_f4f4:>8.2f} {s_mx8mx4:>9.2f} "
            f"{s_mx8mx6:>9.2f} {s_fused:>9.2f} | "
            f"{gain:>+12.2f} dB"
        )

    print()
    print("=" * 90)
    print("KEY:")
    print("  f4f4     = MX4 acts × MX4 weights (4b×4b, fastest)")
    print("  mx8×mx4  = MX8 acts × MX4 weights (8b×4b)")
    print("  mx8×mx6  = MX8 acts × MX6 weights (8b×6b, best quality)")
    print(
        "  nv4→mx6  = MX8 acts × NV4 weights → MX6 compute (8b×4b stored, 6b compute)"
    )
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
