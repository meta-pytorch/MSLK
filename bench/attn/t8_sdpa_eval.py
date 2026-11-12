#!/usr/bin/env python3
# T8: Evaluate SDPA (W1-W3) on MI350X (gfx950)
#
# Measures forward and backward latency across providers for the actual
# W1-W3 workload shapes from the epic.
#
# Usage:
#   MSLK_PYTHON_ONLY=1 python3 bench/attn/t8_sdpa_eval.py [--csv-file results.csv]

import csv
import math
import os
from functools import partial
from itertools import product

import click
import torch
from mslk.aiter.common import import_aiter_module, is_aiter_enabled
from mslk.attention import fmha
from mslk.attention.fmha import ck as ck_mod, flydsl, triton_splitk

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

W1_BS = [683, 1172]
W1_HS = [1, 2]
W1_LQS = [16, 24, 32]
W1_LKVS = [24, 128, 512, 1024, 1984]
W1_DS = [80, 96, 128, 256]


def build_shapes() -> list[tuple[str, int, int, int, int, int, bool, bool]]:
    shapes = []
    for D, B, H, Lq, Lkv, k_is_v in product(
        W1_DS, W1_BS, W1_HS, W1_LQS, W1_LKVS, (True, False)
    ):
        if Lq == Lkv:
            continue
        label = f"W1_B{B}_H{H}_Lq{Lq}_Lkv{Lkv}_D{D}"
        if k_is_v:
            label += "_KisV"
        shapes.append((label, B, H, Lq, Lkv, D, False, k_is_v))
    for D, B, H, L in product(W1_DS, W1_BS, W1_HS, W1_LQS):
        shapes.append((f"W1_B{B}_H{H}_self{L}_D{D}", B, H, L, L, D, True, False))
    for Lq, Lkv, k_is_v in product((32, 224), (200, 400), (True, False)):
        label = f"W2_B1024_H1_Lq{Lq}_Lkv{Lkv}_D256"
        if k_is_v:
            label += "_KisV"
        shapes.append((label, 1024, 1, Lq, Lkv, 256, False, k_is_v))
    shapes.extend(
        [
            ("W2_B1024_H1_self2200_D256_causal", 1024, 1, 2200, 2200, 256, True, False),
            (
                "W2_B1024_H1_self2200_D256_noncausal",
                1024,
                1,
                2200,
                2200,
                256,
                False,
                False,
            ),
            ("W3_B1172_H1_Lq16_Lkv25_D512", 1172, 1, 16, 25, 512, False, False),
            ("W3_B1172_H1_Lq16_Lkv25_D512_KisV", 1172, 1, 16, 25, 512, False, True),
        ]
    )
    return shapes


# ---------------------------------------------------------------------------
# Timing utility (plain loop, uniform for all providers)
# ---------------------------------------------------------------------------


def time_us(fn, iters=50, warmup=10):
    if iters <= 0 or warmup < 0:
        raise ValueError("iters must be positive and warmup nonnegative")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1e3


# ---------------------------------------------------------------------------
# Provider wrappers
# ---------------------------------------------------------------------------


def _make_tensors(B, H, Lq, Lkv, D, dtype, k_is_v=False, device="cuda"):
    Q = torch.randn(B, Lq, H, D, device=device, dtype=dtype)
    K = torch.randn(B, Lkv, H, D, device=device, dtype=dtype)
    V = K if k_is_v else torch.randn(B, Lkv, H, D, device=device, dtype=dtype)
    return Q, K, V


def _try_bench(name, fn):
    try:
        t = fn()
        if not math.isfinite(t) or t <= 0:
            raise ValueError(f"{name} returned an invalid duration: {t}")
        return f"{t:.1f}", ""
    except Exception as e:
        return "ERR", str(e)[:80]


def _run_provider(fn):
    try:
        return fn()
    except Exception as e:
        return "ERR", str(e)[:80]


def _time_gradient_sum(K, V, *, iters, warmup):
    dK, dV = torch.randn_like(K), torch.randn_like(V)
    return time_us(lambda: torch.add(dK, dV), iters=iters, warmup=warmup)


def bench_fw(Q, K, V, causal, op, *, iters=50, warmup=10):
    attn_bias = fmha.attn_bias.LowerTriangularMask() if causal else None
    inp = fmha.Inputs(query=Q, key=K, value=V, attn_bias=attn_bias)
    reasons = op.not_supported_reasons(inp)
    if reasons:
        return "N/A", "; ".join(reasons)[:80]

    def fn():
        return fmha.memory_efficient_attention_forward(
            Q, K, V, attn_bias=attn_bias, op=op
        )

    return _try_bench(op.NAME, lambda: time_us(fn, iters=iters, warmup=warmup))


def bench_sdpa_fw(Q, K, V, causal, *, iters=50, warmup=10):
    Qp = Q.transpose(1, 2).contiguous()
    Kp = K.transpose(1, 2).contiguous()
    Vp = Kp if K is V else V.transpose(1, 2).contiguous()

    def fn():
        return torch.nn.functional.scaled_dot_product_attention(
            Qp, Kp, Vp, is_causal=causal
        )

    try:
        fn()
    except Exception as e:
        return "ERR", str(e)[:80]
    return _try_bench("sdpa", lambda: time_us(fn, iters=iters, warmup=warmup))


def bench_aiter_fw(Q, K, V, causal, *, iters=50, warmup=10):
    if not is_aiter_enabled("fmha"):
        return "N/A", "AITER not available"
    if Q.shape[-1] > 256:
        return "N/A", "AITER head dimension must be at most 256"
    try:
        flash_attn_func = import_aiter_module("aiter.ops.mha").flash_attn_func

        def fn():
            return flash_attn_func(Q, K, V, causal=causal)

        fn()
    except Exception as e:
        return "ERR", str(e)[:80]
    return _try_bench("aiter", lambda: time_us(fn, iters=iters, warmup=warmup))


def bench_sdpa_bw(Q, K, V, causal, *, iters=50, warmup=10):
    Qp = Q.transpose(1, 2).contiguous().requires_grad_(True)
    Kp = K.transpose(1, 2).contiguous().requires_grad_(True)
    Vp = Kp if K is V else V.transpose(1, 2).contiguous().requires_grad_(True)
    try:
        out = torch.nn.functional.scaled_dot_product_attention(
            Qp, Kp, Vp, is_causal=causal
        )
    except Exception as e:
        return "ERR", str(e)[:80]
    grad_out = torch.randn_like(out)

    def bw():
        Qp.grad = Kp.grad = Vp.grad = None
        out.backward(grad_out, retain_graph=True)

    return _try_bench("sdpa_bw", lambda: time_us(bw, iters=iters, warmup=warmup))


def bench_aiter_bw(Q, K, V, causal, *, iters=50, warmup=10):
    if not is_aiter_enabled("fmha"):
        return "N/A", "AITER not available"
    if Q.shape[-1] > 256:
        return "N/A", "AITER head dimension must be at most 256"
    Q_g = Q.clone().requires_grad_(True)
    K_g = K.clone().requires_grad_(True)
    V_g = K_g if K is V else V.clone().requires_grad_(True)
    try:
        flash_attn_func = import_aiter_module("aiter.ops.mha").flash_attn_func
        result = flash_attn_func(Q_g, K_g, V_g, causal=causal, return_lse=True)
        out = result[0]
    except Exception as e:
        return "ERR", str(e)[:80]
    grad_out = torch.randn_like(out)

    def bw():
        Q_g.grad = K_g.grad = V_g.grad = None
        out.backward(grad_out, retain_graph=True)

    return _try_bench("aiter_bw", lambda: time_us(bw, iters=iters, warmup=warmup))


def bench_flydsl_bw(Q, K, V, causal, *, iters=50, warmup=10):
    attn_bias = fmha.attn_bias.LowerTriangularMask() if causal else None
    Q_g = Q.clone().requires_grad_(True)
    K_g = K.clone().requires_grad_(True)
    V_g = K_g if K is V else V.clone().requires_grad_(True)
    inp = fmha.Inputs(query=Q_g, key=K_g, value=V_g, attn_bias=attn_bias)
    reasons = flydsl.BwOp.not_supported_reasons(inp)
    if reasons:
        return "N/A", "; ".join(reasons)[:80]
    # Get LSE from any available forward
    try:
        if not flydsl.FwOp.not_supported_reasons(inp):
            out, lse = fmha.memory_efficient_attention_forward_requires_grad(
                Q_g, K_g, V_g, attn_bias=attn_bias, op=flydsl.FwOp
            )
        elif is_aiter_enabled("fmha"):
            flash_attn_func = import_aiter_module("aiter.ops.mha").flash_attn_func
            result = flash_attn_func(Q_g, K_g, V_g, causal=causal, return_lse=True)
            out, lse = result[0], result[1]
        elif not triton_splitk.FwOp.not_supported_reasons(inp):
            out, lse = fmha.memory_efficient_attention_forward_requires_grad(
                Q_g, K_g, V_g, attn_bias=attn_bias, op=triton_splitk.FwOp
            )
        else:
            return "N/A", "no forward provider for LSE"
    except Exception as e:
        return "ERR", f"fwd for LSE: {e}"[:80]
    grad_out = torch.randn_like(out)
    ctx = fmha.Context(out=out, lse=lse)

    def bw():
        flydsl.BwOp.apply(ctx, inp, grad_out)

    return _try_bench("flydsl_bw", lambda: time_us(bw, iters=iters, warmup=warmup))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--csv-file", "--csv", "csv_path", default=None, help="Export results to CSV"
)
@click.option("--dtype", default="bf16", type=click.Choice(["bf16", "f16"]))
@click.option("--workload", default=None, type=click.Choice(["W1", "W2", "W3"]))
@click.option(
    "--head-dims", "head_dims_str", default=None, help="Comma-separated D filter"
)
@click.option("--forward-only", is_flag=True, help="Skip backward benchmarks")
@click.option("--iters", default=50, type=click.IntRange(min=1))
@click.option("--warmup", default=10, type=click.IntRange(min=0))
@click.option(
    "--providers", default="flydsl,tsk,sdpa,aiter", help="Comma-separated providers"
)
@click.option(
    "--strict", is_flag=True, help="Exit nonzero for provider errors after writing CSV"
)
@click.option(
    "--list-shapes", is_flag=True, help="List selected shapes without using a GPU"
)
@click.option(
    "--limit", default=None, type=click.IntRange(min=1), help="Limit selected shapes"
)
def main(
    csv_path,
    dtype,
    workload,
    head_dims_str,
    forward_only,
    iters,
    warmup,
    providers,
    strict,
    list_shapes,
    limit,
):
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    head_dims_filter = None
    if head_dims_str:
        try:
            head_dims_filter = [int(d) for d in head_dims_str.split(",")]
            if any(d <= 0 for d in head_dims_filter):
                raise ValueError
        except ValueError as exc:
            raise click.BadParameter(
                "expected positive comma-separated dimensions", param_hint="--head-dims"
            ) from exc

    shapes = build_shapes()
    if workload:
        shapes = [s for s in shapes if s[0].startswith(workload)]
    if head_dims_filter:
        shapes = [s for s in shapes if s[5] in head_dims_filter]

    if limit is not None:
        shapes = shapes[:limit]
    selected = set(providers.split(","))
    if not selected or selected - {"flydsl", "tsk", "sdpa", "aiter"}:
        raise click.BadParameter(
            "expected flydsl,tsk,sdpa,aiter", param_hint="--providers"
        )
    if not shapes:
        raise click.ClickException("No shapes match the filters")
    print(f"Total shapes to benchmark: {len(shapes)}")
    if list_shapes:
        for shape in shapes:
            print(shape)
        return
    if not torch.cuda.is_available():
        raise click.ClickException("A CUDA or ROCm GPU is required")
    props = torch.cuda.get_device_properties()
    arch = getattr(props, "gcnArchName", "CUDA").lower()
    if arch.startswith("gfx"):
        detected_arch = arch.split(":", 1)[0]
        configured_arch = (os.environ.get("FLYDSL_GPU_ARCH") or detected_arch).lower()
        if configured_arch.split(":", 1)[0] != detected_arch:
            raise click.ClickException(
                f"FLYDSL_GPU_ARCH={configured_arch} disagrees with GPU {detected_arch}"
            )
        os.environ["FLYDSL_GPU_ARCH"] = detected_arch
    print("T8 SDPA Evaluation (W1-W3 workload shapes)")
    print(f"Device: {props.name} ({arch})")
    print(f"dtype={dtype}  iters={iters}  warmup={warmup}")
    print(f"CK: {ck_mod.FwOp.is_available()}  AITER: {is_aiter_enabled('fmha')}")
    if "gfx950" not in arch:
        print(
            "INFO: historical evaluation used gfx950; "
            "this run uses another architecture"
        )
    print()
    timing = {"iters": iters, "warmup": warmup}

    rows = []
    for idx, (label, B, H, Lq, Lkv, D, causal, k_is_v) in enumerate(shapes):
        print(f"[{idx + 1}/{len(shapes)}] {label} ...", flush=True)

        try:
            Q, K, V = _make_tensors(B, H, Lq, Lkv, D, dt, k_is_v)
        except torch.cuda.OutOfMemoryError:
            print("  OOM - skipping")
            rows.append(
                {
                    "label": label,
                    "B": B,
                    "H": H,
                    "Lq": Lq,
                    "Lkv": Lkv,
                    "D": D,
                    "causal": causal,
                    "k_is_v": k_is_v,
                    "note": "OOM",
                }
            )
            torch.cuda.empty_cache()
            continue

        row = {
            "label": label,
            "B": B,
            "H": H,
            "Lq": Lq,
            "Lkv": Lkv,
            "D": D,
            "causal": causal,
            "k_is_v": k_is_v,
            "dtype": dtype,
        }

        forward = {
            "flydsl": partial(bench_fw, Q, K, V, causal, flydsl.FwOp, **timing),
            "tsk": partial(bench_fw, Q, K, V, causal, triton_splitk.FwOp, **timing),
            "sdpa": partial(bench_sdpa_fw, Q, K, V, causal, **timing),
            "aiter": partial(bench_aiter_fw, Q, K, V, causal, **timing),
        }
        for provider in sorted(selected):
            row[f"fw_{provider}"], row[f"fw_{provider}_note"] = _run_provider(
                forward[provider]
            )
            print(f"  FW {provider}: {row[f'fw_{provider}']}")

        if not forward_only:
            backward = {
                "flydsl": partial(bench_flydsl_bw, Q, K, V, causal, **timing),
                "sdpa": partial(bench_sdpa_bw, Q, K, V, causal, **timing),
                "aiter": partial(bench_aiter_bw, Q, K, V, causal, **timing),
            }
            for provider in sorted(selected & backward.keys()):
                row[f"bw_{provider}"], row[f"bw_{provider}_note"] = _run_provider(
                    backward[provider]
                )
                print(f"  BW {provider}: {row[f'bw_{provider}']}")
            if k_is_v:
                row["dkdv_sum_us"], row["dkdv_sum_note"] = _try_bench(
                    "dkdv_sum", partial(_time_gradient_sum, K, V, **timing)
                )
            del backward
        del forward, Q, K, V

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

    errors = sum(value == "ERR" for row in rows for value in row.values())
    errors += sum(row.get("note") == "OOM" for row in rows)
    if strict and errors:
        raise click.ClickException(
            f"{errors} provider errors or allocation failures; see CSV notes"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
