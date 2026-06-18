# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
End-to-end benchmark: quantization + GEMM latency.

Compares two pipelines:
  1. NVFP4 pipeline: triton_quantize_nvfp4(A) + f4f4bf16 GEMM
  2. Fused pipeline:  to_mxfp8(A) + nv4mx6bf16_fused GEMM

Weights are pre-quantized (stored on device) — only activation quantization
is on the critical path. This simulates inference where weights are static.

Measures:
  - Activation quantization latency alone
  - GEMM latency alone
  - End-to-end (quant + GEMM) latency
  - SQNR vs BF16 reference
"""

import time

import mslk.gemm  # noqa: F401
import mslk.quantize  # noqa: F401
import torch
from mslk.quantize.triton.fp4_quantize import (
    triton_quantize_mx4_unpack,
    triton_quantize_nvfp4,
)
from mslk.quantize.triton.fp4_utils import global_scale_nvfp4
from mslk.test.gemm.fused_mxfp8_quant import fused_mxfp8_quant


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
        (1, 8192, 16384),
        (16, 8192, 16384),
        (64, 8192, 16384),
        (128, 8192, 16384),
        (256, 8192, 16384),
        (512, 8192, 16384),
        (1024, 8192, 16384),
        (2048, 8192, 16384),
        (4096, 8192, 16384),
    ]

    print("=" * 110)
    print(f"END-TO-END: Quantize Activations + GEMM — {torch.cuda.get_device_name()}")
    print("=" * 110)
    print("  NVFP4 pipeline: triton_quantize_nvfp4(A) + f4f4bf16(aq, wq, ...)")
    print("  Fused pipeline: to_mxfp8(A) + nv4mx6bf16_fused(aq, wq, ...)")
    print("  Weights pre-quantized (not on critical path)")
    print("=" * 110)

    # === Table 1: Quantization Latency ===
    print(f"\n{'':=<110}\nTABLE 1: ACTIVATION QUANTIZATION LATENCY (ms)\n{'':=<110}")
    print(
        f"{'M':>6} {'N':>6} {'K':>6} | "
        f"{'nvfp4_quant':>12} {'mx8_quant':>10} {'speedup':>8}"
    )
    print("-" * 70)

    quant_results = []
    for M, N, K in shapes:
        A_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 0.1

        # Pre-compute global scale for NVFP4
        gs = global_scale_nvfp4(A_bf16)

        # Bench NVFP4 quantization
        t_nvfp4_q = bench(lambda _a=A_bf16, _gs=gs: triton_quantize_nvfp4(_a, _gs))

        # Bench MX8 quantization (fused: quant + blocked scale layout in one kernel)
        t_mx8_q = bench(lambda _a=A_bf16: fused_mxfp8_quant(_a))

        speedup = (t_nvfp4_q - t_mx8_q) / t_nvfp4_q * 100
        quant_results.append((M, N, K, t_nvfp4_q, t_mx8_q))
        print(
            f"{M:>6} {N:>6} {K:>6} | "
            f"{t_nvfp4_q:>12.4f} {t_mx8_q:>10.4f} {speedup:>+7.1f}%"
        )

    # === Table 2: GEMM-only Latency ===
    print(
        f"\n{'':=<110}\n"
        f"TABLE 2: GEMM-ONLY LATENCY (ms) — weights pre-quantized\n"
        f"{'':=<110}"
    )
    print(
        f"{'M':>6} {'N':>6} {'K':>6} | "
        f"{'f4f4_gemm':>10} {'fused_gemm':>11} {'speedup':>8}"
    )
    print("-" * 70)

    gemm_results = []
    for M, N, K in shapes:
        A_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 0.1
        B_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device="cuda") * 0.01

        # NVFP4 weights: quantize with proper NVFP4 format
        gs_w = global_scale_nvfp4(B_bf16)
        wq_nvfp4, w_scale_nvfp4 = triton_quantize_nvfp4(B_bf16, gs_w)

        # MX4 weights for fused kernel (E8M0 block scales)
        bq_mx4_raw, b_scale_mx4 = triton_quantize_mx4_unpack(B_bf16)
        wq_nv4 = bq_mx4_raw.view(torch.uint8)
        w_scale_nv4 = b_scale_mx4.view(torch.uint8).flatten()

        # Pre-quantize activations for GEMM-only measurement
        gs_a = global_scale_nvfp4(A_bf16)
        aq_nvfp4, a_scale_nvfp4 = triton_quantize_nvfp4(A_bf16, gs_a)

        aq_mx8, a_scale_mx8 = fused_mxfp8_quant(A_bf16)

        # Combined global scale for f4f4 NVFP4 GEMM (reciprocal of product)
        gs_combined = 1.0 / (gs_a * gs_w)

        # Bench f4f4 GEMM only
        try:
            t_f4f4 = bench(
                lambda _aq=aq_nvfp4,
                _bq=wq_nvfp4,
                _as=a_scale_nvfp4,
                _bs=w_scale_nvfp4,
                _gs=gs_combined: torch.ops.mslk.f4f4bf16(
                    _aq, _bq, _as, _bs, global_scale=_gs
                )
            )
        except RuntimeError:
            t_f4f4 = float("nan")

        # Bench fused GEMM only
        t_fused = bench(
            lambda _aq=aq_mx8,
            _wq=wq_nv4,
            _as=a_scale_mx8,
            _ws=w_scale_nv4: torch.ops.mslk.nv4mx6bf16_fused(_aq, _wq, _as, _ws)
        )

        speedup = (
            (t_f4f4 - t_fused) / t_f4f4 * 100 if t_f4f4 == t_f4f4 else float("nan")
        )
        gemm_results.append((M, N, K, t_f4f4, t_fused))
        print(
            f"{M:>6} {N:>6} {K:>6} | {t_f4f4:>10.4f} {t_fused:>11.4f} {speedup:>+7.1f}%"
        )

    # === Table 3: End-to-End (Quant + GEMM) ===
    print(
        f"\n{'':=<110}\nTABLE 3: END-TO-END (quant activations + GEMM) — ms\n{'':=<110}"
    )
    print(
        f"{'M':>6} {'N':>6} {'K':>6} | "
        f"{'nvfp4_e2e':>10} {'fused_e2e':>10} {'speedup':>8} | "
        f"{'nvfp4 SQNR':>11} {'fused SQNR':>11} {'SQNR gain':>10}"
    )
    print("-" * 110)

    for M, N, K in shapes:
        A_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 0.1
        B_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device="cuda") * 0.01

        # BF16 reference
        out_ref = (A_bf16 @ B_bf16.t()).float()

        # NVFP4 weights
        gs_w = global_scale_nvfp4(B_bf16)
        wq_nvfp4, w_scale_nvfp4 = triton_quantize_nvfp4(B_bf16, gs_w)

        # MX4 weights for fused kernel
        bq_mx4_raw, b_scale_mx4 = triton_quantize_mx4_unpack(B_bf16)
        wq_nv4 = bq_mx4_raw.view(torch.uint8)
        w_scale_nv4 = b_scale_mx4.view(torch.uint8).flatten()

        # --- End-to-end NVFP4 pipeline ---
        gs_a = global_scale_nvfp4(A_bf16)
        gs_combined = 1.0 / (gs_a * gs_w)

        def nvfp4_pipeline(
            _a=A_bf16,
            _bq=wq_nvfp4,
            _bs=w_scale_nvfp4,
            _gs_a=gs_a,
            _gs=gs_combined,
        ):
            aq, a_sc = triton_quantize_nvfp4(_a, _gs_a)
            return torch.ops.mslk.f4f4bf16(aq, _bq, a_sc, _bs, global_scale=_gs)

        # --- End-to-end Fused pipeline ---
        def fused_pipeline(_a=A_bf16, _wq=wq_nv4, _ws=w_scale_nv4):
            aq, a_sc = fused_mxfp8_quant(_a)
            return torch.ops.mslk.nv4mx6bf16_fused(aq, _wq, a_sc, _ws)

        try:
            t_nvfp4_e2e = bench(nvfp4_pipeline)
        except RuntimeError:
            t_nvfp4_e2e = float("nan")

        t_fused_e2e = bench(fused_pipeline)

        # SQNR
        try:
            out_nvfp4 = nvfp4_pipeline().float()
            s_nvfp4 = sqnr_db(out_ref, out_nvfp4)
        except RuntimeError:
            s_nvfp4 = float("nan")

        out_fused = fused_pipeline().float()
        s_fused = sqnr_db(out_ref, out_fused)

        speedup = (
            (t_nvfp4_e2e - t_fused_e2e) / t_nvfp4_e2e * 100
            if t_nvfp4_e2e == t_nvfp4_e2e
            else float("nan")
        )
        sqnr_gain = s_fused - s_nvfp4 if s_nvfp4 == s_nvfp4 else float("nan")

        print(
            f"{M:>6} {N:>6} {K:>6} | "
            f"{t_nvfp4_e2e:>10.4f} {t_fused_e2e:>10.4f} {speedup:>+7.1f}% | "
            f"{s_nvfp4:>11.2f} {s_fused:>11.2f} {sqnr_gain:>+9.2f} dB"
        )

    print("\n" + "=" * 110)
    print("KEY:")
    print("  nvfp4_e2e = triton_quantize_nvfp4(A) + f4f4bf16(aq, wq, gs)")
    print("  fused_e2e = to_mxfp8(A) + nv4mx6bf16_fused(aq, wq)")
    print(
        "  Hypothesis: MX8 quant is cheaper than NVFP4 quant, "
        "offsetting the GEMM latency gap"
    )
    print("=" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
