#!/usr/bin/env python3
# T8: Evaluate SDPA (W1-W3) on MI350X (gfx950)
#
# Measures forward and backward latency across providers for the actual
# W1-W3 workload shapes from the epic.
#
# Usage:
#   MSLK_PYTHON_ONLY=1 python3 bench/attn/t8_sdpa_eval.py [--csv results.csv]

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import click
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test", "attention"))

from mslk.attention import fmha
from mslk.attention.fmha import flydsl, triton_splitk

try:
    from mslk.attention.fmha import ck as ck_mod
    CK_AVAILABLE = ck_mod.FwOp.is_available()
except Exception:
    ck_mod = None
    CK_AVAILABLE = False

try:
    from aiter.ops.mha import flash_attn_func
    AITER_AVAILABLE = True
except Exception:
    AITER_AVAILABLE = False


# ---------------------------------------------------------------------------
# W1-W3 shapes from the epic (exact workload definitions)
# ---------------------------------------------------------------------------
# W1: SDPA flash, B 683-1172, H 1-2, Lq 16/24/32 (cross and self),
#     Lkv 24-1984, D 80/96/128/256, fwd+bwd
# W2: SDPA flash, B 1024, H 1, Lq 32/224 (cross) and 2200 (self),
#     Lkv 200/400/2200, D 256, fwd+bwd
# W3: SDPA efficient, B 1172, H 1, Lq 16, Lkv 25, D 512, fwd+bwd
#
# A5: training, non-causal, no dropout, K may be same tensor as V
# W2's 2200x2200 supported both causal and non-causal

# (workload, B, H, Lq, Lkv, D, causal, k_is_v)
W_SHAPES: List[Tuple[str, int, int, int, int, int, bool, bool]] = []

# --- W1 shapes ---
W1_BS = [683, 1172]
W1_HS = [1, 2]
W1_LQS = [16, 24, 32]
W1_LKVS = [24, 128, 512, 1024, 1984]
W1_DS = [80, 96, 128, 256]

for D in W1_DS:
    for B in W1_BS:
        for H in W1_HS:
            for Lq in W1_LQS:
                for Lkv in W1_LKVS:
                    causal = (Lq == Lkv)
                    for k_is_v in ([True, False] if not causal else [False]):
                        label = f"W1_B{B}_H{H}_Lq{Lq}_Lkv{Lkv}_D{D}"
                        if k_is_v:
                            label += "_KisV"
                        W_SHAPES.append((label, B, H, Lq, Lkv, D, causal, k_is_v))

# W1 self-attention shapes (Lq == Lkv)
for D in W1_DS:
    for B in W1_BS:
        for H in W1_HS:
            for L in W1_LQS:  # self-attn: Lq == Lkv
                W_SHAPES.append((f"W1_B{B}_H{H}_self{L}_D{D}", B, H, L, L, D, True, False))

# --- W2 shapes ---
# Cross: Lq 32/224, Lkv 200/400
for Lq in [32, 224]:
    for Lkv in [200, 400]:
        for k_is_v in [True, False]:
            label = f"W2_B1024_H1_Lq{Lq}_Lkv{Lkv}_D256"
            if k_is_v:
                label += "_KisV"
            W_SHAPES.append((label, 1024, 1, Lq, Lkv, 256, False, k_is_v))

# Self: 2200x2200, causal and non-causal
W_SHAPES.append(("W2_B1024_H1_self2200_D256_causal", 1024, 1, 2200, 2200, 256, True, False))
W_SHAPES.append(("W2_B1024_H1_self2200_D256_noncausal", 1024, 1, 2200, 2200, 256, False, False))

# --- W3 shape ---
W_SHAPES.append(("W3_B1172_H1_Lq16_Lkv25_D512", 1172, 1, 16, 25, 512, False, False))
W_SHAPES.append(("W3_B1172_H1_Lq16_Lkv25_D512_KisV", 1172, 1, 16, 25, 512, False, True))


# ---------------------------------------------------------------------------
# Timing utility (plain loop, uniform for all providers)
# ---------------------------------------------------------------------------

def time_us(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1e3


# ---------------------------------------------------------------------------
# Provider wrappers
# ---------------------------------------------------------------------------

def _make_tensors(B, H, Lq, Lkv, D, dtype, k_is_v=False):
    Q = torch.randn(B, Lq, H, D, device="cuda", dtype=dtype)
    K = torch.randn(B, Lkv, H, D, device="cuda", dtype=dtype)
    V = K.clone() if k_is_v else torch.randn(B, Lkv, H, D, device="cuda", dtype=dtype)
    return Q, K, V


def _try_bench(name, fn):
    try:
        t = fn()
        return f"{t:.1f}", ""
    except Exception as e:
        return "ERR", str(e)[:80]


def bench_fw(Q, K, V, causal, op):
    attn_bias = fmha.attn_bias.LowerTriangularMask() if causal else None
    inp = fmha.Inputs(query=Q, key=K, value=V, attn_bias=attn_bias)
    reasons = op.not_supported_reasons(inp)
    if reasons:
        return "N/A", "; ".join(reasons)[:80]
    fn = lambda: fmha.memory_efficient_attention_forward(Q, K, V, attn_bias=attn_bias, op=op)
    return _try_bench(op.NAME, lambda: time_us(fn))


def bench_sdpa_fw(Q, K, V, causal):
    Qp = Q.transpose(1, 2).contiguous()
    Kp = K.transpose(1, 2).contiguous()
    Vp = V.transpose(1, 2).contiguous()
    fn = lambda: torch.nn.functional.scaled_dot_product_attention(Qp, Kp, Vp, is_causal=causal)
    try:
        fn()
    except Exception as e:
        return "N/A", str(e)[:80]
    return _try_bench("sdpa", lambda: time_us(fn))


def bench_aiter_fw(Q, K, V, causal):
    if not AITER_AVAILABLE:
        return "N/A", "AITER not available"
    try:
        fn = lambda: flash_attn_func(Q, K, V, causal=causal)
        fn()
    except Exception as e:
        return "N/A", str(e)[:80]
    return _try_bench("aiter", lambda: time_us(fn))


def bench_sdpa_bw(Q, K, V, causal):
    Qp = Q.transpose(1, 2).contiguous().requires_grad_(True)
    Kp = K.transpose(1, 2).contiguous().requires_grad_(True)
    Vp = V.transpose(1, 2).contiguous().requires_grad_(True)
    try:
        out = torch.nn.functional.scaled_dot_product_attention(Qp, Kp, Vp, is_causal=causal)
    except Exception as e:
        return "N/A", str(e)[:80]
    grad_out = torch.randn_like(out)
    def bw():
        Qp.grad = Kp.grad = Vp.grad = None
        out.backward(grad_out, retain_graph=True)
    return _try_bench("sdpa_bw", lambda: time_us(bw))


def bench_aiter_bw(Q, K, V, causal):
    if not AITER_AVAILABLE:
        return "N/A", "AITER not available"
    Q_g = Q.clone().requires_grad_(True)
    K_g = K.clone().requires_grad_(True)
    V_g = V.clone().requires_grad_(True)
    try:
        result = flash_attn_func(Q_g, K_g, V_g, causal=causal, return_lse=True)
        out = result[0]
    except Exception as e:
        return "N/A", str(e)[:80]
    grad_out = torch.randn_like(out)
    def bw():
        Q_g.grad = K_g.grad = V_g.grad = None
        out.backward(grad_out, retain_graph=True)
    return _try_bench("aiter_bw", lambda: time_us(bw))


def bench_flydsl_bw(Q, K, V, causal):
    attn_bias = fmha.attn_bias.LowerTriangularMask() if causal else None
    Q_g = Q.clone().requires_grad_(True)
    K_g = K.clone().requires_grad_(True)
    V_g = V.clone().requires_grad_(True)
    inp = fmha.Inputs(query=Q_g, key=K_g, value=V_g, attn_bias=attn_bias)
    reasons = flydsl.BwOp.not_supported_reasons(inp)
    if reasons:
        return "N/A", "; ".join(reasons)[:80]
    # Get LSE from any available forward
    try:
        if not flydsl.FwOp.not_supported_reasons(inp):
            out, lse = fmha.memory_efficient_attention_forward_requires_grad(
                Q_g, K_g, V_g, attn_bias=attn_bias, op=flydsl.FwOp)
        elif AITER_AVAILABLE:
            result = flash_attn_func(Q_g, K_g, V_g, causal=causal, return_lse=True)
            out, lse = result[0], result[1]
        elif not triton_splitk.FwOp.not_supported_reasons(inp):
            out, lse = fmha.memory_efficient_attention_forward_requires_grad(
                Q_g, K_g, V_g, attn_bias=attn_bias, op=triton_splitk.FwOp)
        else:
            return "N/A", "no forward provider for LSE"
    except Exception as e:
        return "N/A", f"fwd for LSE: {e}"[:80]
    grad_out = torch.randn_like(out)
    ctx = fmha.Context(out=out, lse=lse)
    def bw():
        flydsl.BwOp.apply(ctx, inp, grad_out)
    return _try_bench("flydsl_bw", lambda: time_us(bw))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--csv-file", "csv_path", default=None, help="Export results to CSV")
@click.option("--dtype", default="bf16", type=click.Choice(["bf16", "f16"]))
@click.option("--workload", default=None, help="Filter: W1, W2, W3, or all")
@click.option("--head-dims", "head_dims_str", default=None, help="Comma-separated D filter")
@click.option("--forward-only", is_flag=True, help="Skip backward benchmarks")
@click.option("--iters", default=50, type=int)
@click.option("--warmup", default=10, type=int)
def main(csv_path, dtype, workload, head_dims_str, forward_only, iters, warmup):
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"T8 SDPA Evaluation (W1-W3 workload shapes)")
    print(f"Device: {torch.cuda.get_device_name(0)} ({arch})")
    print(f"dtype={dtype}  iters={iters}  warmup={warmup}")
    print(f"CK: {CK_AVAILABLE}  AITER: {AITER_AVAILABLE}")
    print()

    head_dims_filter = None
    if head_dims_str:
        head_dims_filter = [int(d) for d in head_dims_str.split(",")]

    shapes = W_SHAPES
    if workload:
        shapes = [s for s in shapes if s[0].startswith(workload)]
    if head_dims_filter:
        shapes = [s for s in shapes if s[5] in head_dims_filter]

    print(f"Total shapes to benchmark: {len(shapes)}")
    print()

    rows = []
    for idx, (label, B, H, Lq, Lkv, D, causal, k_is_v) in enumerate(shapes):
        print(f"[{idx+1}/{len(shapes)}] {label} ...", flush=True)

        try:
            Q, K, V = _make_tensors(B, H, Lq, Lkv, D, dt, k_is_v)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM - skipping")
            rows.append({"label": label, "B": B, "H": H, "Lq": Lq, "Lkv": Lkv,
                          "D": D, "causal": causal, "k_is_v": k_is_v, "note": "OOM"})
            torch.cuda.empty_cache()
            continue

        row = {"label": label, "B": B, "H": H, "Lq": Lq, "Lkv": Lkv,
               "D": D, "causal": causal, "k_is_v": k_is_v, "dtype": dtype}

        # Forward
        row["fw_flydsl"], row["fw_flydsl_note"] = bench_fw(Q, K, V, causal, flydsl.FwOp)
        row["fw_tsk"], row["fw_tsk_note"] = bench_fw(Q, K, V, causal, triton_splitk.FwOp)
        row["fw_sdpa"], row["fw_sdpa_note"] = bench_sdpa_fw(Q, K, V, causal)
        row["fw_aiter"], row["fw_aiter_note"] = bench_aiter_fw(Q, K, V, causal)

        fw_parts = [f"fly={row['fw_flydsl']}", f"tsk={row['fw_tsk']}",
                     f"sdpa={row['fw_sdpa']}", f"aiter={row['fw_aiter']}"]
        print(f"  FW: {' | '.join(fw_parts)}")

        # Backward
        if not forward_only:
            Q2, K2, V2 = _make_tensors(B, H, Lq, Lkv, D, dt, k_is_v)
            row["bw_flydsl"], row["bw_flydsl_note"] = bench_flydsl_bw(Q2, K2, V2, causal)
            row["bw_sdpa"], row["bw_sdpa_note"] = bench_sdpa_bw(Q2, K2, V2, causal)
            row["bw_aiter"], row["bw_aiter_note"] = bench_aiter_bw(Q2, K2, V2, causal)

            bw_parts = [f"fly={row['bw_flydsl']}", f"sdpa={row['bw_sdpa']}",
                         f"aiter={row['bw_aiter']}"]
            print(f"  BW: {' | '.join(bw_parts)}")

            # dK+dV sum cost for K-is-V
            if k_is_v:
                dK = torch.randn_like(K2)
                dV = torch.randn_like(V2)
                sum_t = time_us(lambda: torch.add(dK, dV))
                row["dkdv_sum_us"] = f"{sum_t:.1f}"

        rows.append(row)
        torch.cuda.empty_cache()

    # CSV export
    if csv_path and rows:
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults exported to {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
