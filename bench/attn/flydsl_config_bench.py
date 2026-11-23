# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Warmed GPU traces for FlyDSL attention's supported and newly supported routes."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import flydsl.compiler as flyc
import torch
from mslk.attention.flydsl import flydsl_flash_attn_func
from mslk.attention.flydsl.dropout_philox import draw_dropout_seed, reference_keep_mask
from mslk.attention.flydsl.fmha_bwd_mfma_gfx950 import (
    compile_fmha_bwd_dqdkdv_mfma_gfx950,
    gfx950_tile_defaults,
)
from mslk.attention.flydsl.fmha_bwd_preprocess import compile_fmha_bwd_preprocess


BenchmarkOutput = torch.Tensor | tuple[torch.Tensor, ...]
BenchmarkCallable = Callable[[], BenchmarkOutput]

# Drop probability for the dropout configs.
_DROPOUT_P = 0.5


def config(name: str, **overrides: Any) -> dict[str, Any]:
    result = {
        "workload_family": "synthetic",
        "name": name,
        "operation": "forward",
        "layout": "dense",
        "batch": 1,
        "query_length": 512,
        "kv_length": 512,
        "heads": 8,
        "kv_heads": 8,
        "head_dim_qk": 128,
        "head_dim_v": 128,
        "dtype": "bf16",
        "causal": True,
        "window_left": -1,
        "splits": 0,
        "scale": None,
        "logit_scale": 1.0,
        "bias": False,
        "dropout": False,
    }
    result.update(overrides)
    return result


def get_configs() -> list[dict[str, Any]]:
    return [
        config("torch_add_control", operation="control", causal=False),
        config(
            "dense_representative",
            workload_family="prod",
            query_length=4096,
            kv_length=4096,
            heads=16,
            kv_heads=16,
            causal=False,
        ),
        config(
            "causal_representative",
            workload_family="prod",
            query_length=4096,
            kv_length=4096,
            heads=16,
            kv_heads=16,
        ),
        config(
            "window_representative",
            workload_family="prod",
            query_length=2048,
            kv_length=2048,
            heads=16,
            kv_heads=16,
            window_left=256,
        ),
        config(
            "decode_head_packed",
            workload_family="prod",
            layout="paged",
            batch=16,
            query_length=32,
            kv_length=4096,
            heads=8,
            kv_heads=8,
        ),
        config(
            "decode_native_control",
            workload_family="prod",
            layout="paged",
            batch=4,
            query_length=1,
            kv_length=4096,
            heads=32,
            kv_heads=4,
        ),
        config(
            "backward_representative",
            workload_family="prod",
            operation="backward",
            query_length=1024,
            kv_length=1024,
        ),
        config("dense_masked_96", head_dim_qk=96, head_dim_v=96),
        config("dense_masked_40", head_dim_qk=40, head_dim_v=40, dtype="f16"),
        config("dense_masked_200_160", head_dim_qk=200, head_dim_v=160),
        config("dense_mismatched_value", head_dim_qk=64, head_dim_v=128),
        config(
            "varlen_masked_96",
            layout="varlen",
            batch=2,
            query_length=256,
            kv_length=256,
            head_dim_qk=96,
            head_dim_v=96,
        ),
        config(
            "varlen_noncausal_window",
            layout="varlen",
            batch=2,
            query_length=256,
            kv_length=256,
            causal=False,
            window_left=65,
        ),
        config(
            "noncausal_cross_window",
            query_length=384,
            kv_length=513,
            causal=False,
            window_left=65,
        ),
        config("forced_split_window", splits=4, window_left=65),
        config("forced_split_scale", splits=4, scale=0.05),
        config("forced_split_bias", splits=4, bias=True),
        config("forced_split_dropout", splits=4, dropout=True),
        config("forced_split_cross", splits=4, kv_length=768),
        config(
            "dense_cross_kv_tail",
            query_length=128,
            kv_length=193,
            head_dim_qk=64,
            head_dim_v=64,
        ),
        config("dropout_short_query", query_length=64, kv_length=1024, dropout=True),
        config(
            "paged_masked_fused_combine",
            layout="paged",
            batch=8,
            query_length=4,
            kv_length=8192,
            heads=16,
            kv_heads=16,
            head_dim_qk=72,
            head_dim_v=72,
            splits=4,
        ),
        config(
            "paged_masked_gappy",
            layout="paged_gappy",
            batch=2,
            query_length=16,
            kv_length=1024,
            heads=16,
            kv_heads=4,
            head_dim_qk=96,
            head_dim_v=96,
        ),
        config("dense_large_logits", heads=16, kv_heads=4, logit_scale=32.0),
        config(
            "paged_large_logits",
            layout="paged",
            kv_length=4096,
            heads=16,
            kv_heads=4,
            causal=False,
            logit_scale=32.0,
            splits=1,
        ),
        config("fp8_large_logits", heads=16, kv_heads=4, dtype="fp8", logit_scale=32.0),
        config("dualwave_split_combine", splits=4),
        config(
            "backward_masked_96",
            operation="backward",
            query_length=256,
            kv_length=256,
            head_dim_qk=96,
            head_dim_v=96,
        ),
        config(
            "backward_mismatched_value",
            operation="backward",
            query_length=256,
            kv_length=256,
            head_dim_v=64,
        ),
        config(
            "backward_native_160",
            operation="backward",
            query_length=256,
            kv_length=256,
            head_dim_qk=160,
            head_dim_v=160,
            dtype="f16",
        ),
    ]


def rand(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=torch.float32).to(dtype)


def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: dict[str, Any],
    bias: torch.Tensor | None = None,
    dropout: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    group = cfg["heads"] // cfg["kv_heads"]
    k = k.float().repeat_interleave(group, dim=2)
    v = v.float().repeat_interleave(group, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k)
    scores *= cfg["scale"] if cfg["scale"] is not None else cfg["head_dim_qk"] ** -0.5
    sq, sk = q.shape[1], k.shape[1]
    row = torch.arange(sq, device="cuda")[:, None] + sk - sq
    col = torch.arange(sk, device="cuda")[None, :]
    keep = torch.ones((sq, sk), device="cuda", dtype=torch.bool)
    if cfg["causal"]:
        keep &= col <= row
    if cfg["window_left"] >= 0:
        keep &= col >= row - cfg["window_left"]
    scores = scores.masked_fill(~keep, float("-inf"))
    if bias is not None:
        scores += bias.float()
    lse = scores.logsumexp(dim=-1)
    weights = scores.softmax(dim=-1).nan_to_num()
    if dropout is not None:
        weights *= dropout.float()
    return torch.einsum("bhqk,bkhd->bqhd", weights, v), lse


def dense_inputs(
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "fp8": torch.float8_e4m3fn}[
        cfg["dtype"]
    ]
    b, sq, sk = cfg["batch"], cfg["query_length"], cfg["kv_length"]
    q = rand((b, sq, cfg["heads"], cfg["head_dim_qk"]), dtype)
    k = rand((b, sk, cfg["kv_heads"], cfg["head_dim_qk"]), dtype)
    v = rand((b, sk, cfg["kv_heads"], cfg["head_dim_v"]), dtype)
    if cfg["dtype"] != "fp8":
        q = (q.float() * cfg["logit_scale"]).to(dtype)
    return q, k, v


def prepare_paged_inputs(
    cfg: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    expected: torch.Tensor,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    page = 64
    b, sk = cfg["batch"], cfg["kv_length"]
    if sk % page:
        raise ValueError(f"paged kv_length must be divisible by {page}, got {sk}")
    k, v = (
        t.reshape(b * (sk // page), page, cfg["kv_heads"], t.shape[-1]) for t in (k, v)
    )
    kwargs.update(
        block_table=torch.arange(
            b * (sk // page), device="cuda", dtype=torch.int32
        ).reshape(b, sk // page),
        seqlen_k=torch.full((b,), sk, device="cuda", dtype=torch.int32),
        kv_cache_layout="linear",
        max_seqlen_kv=sk,
    )
    if cfg["layout"] == "paged_gappy":
        offset = 7
        kwargs.update(
            kv_seqstart=torch.full((b,), offset, device="cuda", dtype=torch.int32),
            seqlen_k=torch.full((b,), sk - offset, device="cuda", dtype=torch.int32),
        )
        expected, _ = reference(
            q,
            k.reshape(b, sk, cfg["kv_heads"], cfg["head_dim_qk"])[:, offset:],
            v.reshape(b, sk, cfg["kv_heads"], cfg["head_dim_v"])[:, offset:],
            cfg,
        )
        kwargs.update(
            cu_seqlens_q=torch.arange(b + 1, device="cuda", dtype=torch.int32)
            * cfg["query_length"],
            cu_seqlens_kv=torch.arange(b + 1, device="cuda", dtype=torch.int32)
            * (sk - offset),
            max_seqlen_q=cfg["query_length"],
            cross_seqlen=True,
        )
        q = q.flatten(0, 1)
        expected = expected.flatten(0, 1)
    return q, k, v, expected


def forward_case(
    cfg: dict[str, Any],
) -> tuple[BenchmarkCallable, torch.Tensor]:
    q, k, v = dense_inputs(cfg)
    kwargs: dict[str, Any] = {
        "causal": cfg["causal"],
        "num_kv_heads": cfg["kv_heads"],
        "window_left": cfg["window_left"],
        "num_kv_splits": cfg["splits"],
        "sm_scale": cfg["scale"],
    }
    bias = (
        rand(
            (cfg["batch"], cfg["heads"], cfg["query_length"], cfg["kv_length"]), q.dtype
        )
        * 0.1
        if cfg["bias"]
        else None
    )
    dropout = None
    if cfg["dropout"]:
        # The kernel draws keep bits from the seed; the reference rebuilds them.
        seed = draw_dropout_seed(torch.device("cuda"))
        dropout = reference_keep_mask(
            int(seed.item()),
            _DROPOUT_P,
            cfg["batch"],
            cfg["heads"],
            cfg["query_length"],
            cfg["kv_length"],
            torch.device("cuda"),
        )
        kwargs.update(dropout_p=_DROPOUT_P, dropout_seed=seed)
    kwargs.update(bias=bias)
    q_ref = q.float() * cfg["logit_scale"] if cfg["dtype"] == "fp8" else q
    expected, _ = reference(q_ref, k, v, cfg, bias, dropout)
    if cfg["dtype"] == "fp8":
        kwargs.update(
            q_descale=torch.full((1,), cfg["logit_scale"], device="cuda"),
            k_descale=torch.ones(1, device="cuda"),
            v_descale=torch.ones(1, device="cuda"),
        )
    if cfg["layout"] == "varlen":
        starts = (
            torch.arange(cfg["batch"] + 1, device="cuda", dtype=torch.int32)
            * cfg["query_length"]
        )
        kwargs.update(
            cu_seqlens_q=starts,
            cu_seqlens_kv=starts,
            max_seqlen_q=cfg["query_length"],
            max_seqlen_kv=cfg["kv_length"],
            cross_seqlen=False,
        )
        q, k, v = (t.flatten(0, 1) for t in (q, k, v))
        expected = expected.flatten(0, 1)
    elif cfg["layout"].startswith("paged"):
        q, k, v, expected = prepare_paged_inputs(cfg, q, k, v, expected, kwargs)
    return lambda: flydsl_flash_attn_func(q, k, v, **kwargs), expected


def backward_case(
    cfg: dict[str, Any],
) -> tuple[BenchmarkCallable, None]:
    q, k, v = dense_inputs(cfg)
    out, lse = reference(q, k, v, cfg)
    out = out.to(q.dtype)
    grad = torch.randn_like(out)
    dqk, dv = cfg["head_dim_qk"], cfg["head_dim_v"]
    dim = next(d for d in (32, 64, 128, 160, 256) if d >= max(dqk, dv))
    parameters = inspect.signature(compile_fmha_bwd_dqdkdv_mfma_gfx950).parameters
    dimensions = {"D_QK": dqk, "D_V": dv} if "D_QK" in parameters else {}
    if not dimensions and (dqk, dv) != (dim, dim):
        raise NotImplementedError(
            "baseline backward builder has no stored head-dimension masking"
        )
    b, sq, sk, h = cfg["batch"], cfg["query_length"], cfg["kv_length"], cfg["heads"]
    rows = b * sq * h
    dvec = torch.empty((rows, 1), device="cuda", dtype=torch.float32)
    preprocess = compile_fmha_bwd_preprocess(D=dv, dtype_str=cfg["dtype"])
    stream = torch.cuda.current_stream()
    preprocess_args = (
        grad.view(torch.int16).reshape(rows, dv),
        out.view(torch.int16).reshape(rows, dv),
        dvec,
        rows,
        stream,
    )
    compiled_preprocess = flyc.compile(preprocess, *preprocess_args)
    block_m, block_n = gfx950_tile_defaults(dim, sk, "gfx950")
    builder = compile_fmha_bwd_dqdkdv_mfma_gfx950(
        D=dim,
        dtype_str=cfg["dtype"],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        scale=dqk**-0.5,
        causal_mode=1 if cfg["causal"] else 0,
        ck_scope_dvdk=True,
        gpu_arch="gfx950",
        **dimensions,
    )
    gradients = tuple(
        torch.zeros((b * n * h * dim, 1), device="cuda", dtype=torch.float32)
        for n in (sk, sk, sq)
    )
    dummy_seq = torch.zeros(1, device="cuda", dtype=torch.int32)
    dummy_bias = torch.zeros(1, device="cuda", dtype=q.dtype)
    dummy_dbias = torch.zeros(1, device="cuda", dtype=torch.float32)
    args = (
        q.view(torch.int16),
        k.view(torch.int16),
        v.view(torch.int16),
        grad.view(torch.int16),
        *gradients,
        lse.contiguous().reshape(rows, 1),
        dvec,
        b,
        sq,
        sk,
        h,
        -(-sq // block_m),
        h,
        h,
        h,
        dummy_seq,
        dummy_seq,
        sq,
        dummy_bias,
        0,
        0,
        0,
        dummy_dbias,
        0,
        stream,
    )
    compiled = flyc.compile(builder, *args)

    def call() -> tuple[torch.Tensor, ...]:
        for gradient in gradients:
            gradient.zero_()
        current = torch.cuda.current_stream()
        compiled_preprocess(*preprocess_args[:-1], current)
        compiled(*args[:-1], current)
        return gradients

    return call, None


def control_case(
    cfg: dict[str, Any],
) -> tuple[BenchmarkCallable, torch.Tensor]:
    left, right, _ = dense_inputs(cfg)
    expected = left.float() + right.float()
    return lambda: torch.add(left, right), expected


def profile(
    call: BenchmarkCallable, trace_path: Path, warmup: int, iterations: int
) -> BenchmarkOutput:
    result = None
    for _ in range(warmup):
        result = call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        result = call()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    return result


def run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    started = time.monotonic()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "config": cfg,
        "status": "failed",
    }
    trace = output / f"{cfg['name']}.json"
    try:
        torch.manual_seed(0)
        status.update(
            hardware=torch.cuda.get_device_name(),
            arch=torch.cuda.get_device_properties(0).gcnArchName,
        )
        factory = {"backward": backward_case, "control": control_case}.get(
            cfg["operation"], forward_case
        )
        call, expected = factory(cfg)
        first = call()
        if expected is not None:
            primary = first[0] if isinstance(first, tuple) else first
            torch.testing.assert_close(primary.float(), expected, atol=0.04, rtol=0.02)
        result = profile(call, trace, args.warmup, args.iterations)
        if expected is not None:
            primary = result[0] if isinstance(result, tuple) else result
            torch.testing.assert_close(primary.float(), expected, atol=0.04, rtol=0.02)
        tensors = result if isinstance(result, tuple) else (result,)
        status["finite"] = all(bool(torch.isfinite(t).all()) for t in tensors)
        if not status["finite"]:
            raise AssertionError("non-finite kernel output")
        status.update(status="passed", trace_file=trace.name)
    except (ValueError, NotImplementedError, AssertionError) as exc:
        status.update(status="unsupported_or_incorrect", error=str(exc))
        if not args.allow_unsupported:
            raise
    except RuntimeError as exc:
        status["error"] = str(exc)
        if not args.allow_unsupported or not str(exc).startswith(
            "flash_attn_dualwave_swp supports D=64 or D=128 only, got head_dim="
        ):
            raise
        status.update(status="unsupported_or_incorrect", error=str(exc))
    except Exception as exc:
        status["error"] = str(exc)
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        (output / f"{cfg['name']}.status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status))
    return 0


def main() -> int:
    configs = get_configs()
    config_columns = list(config(""))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--config", choices=[c["name"] for c in configs])
    parser.add_argument("--output-dir")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Record baseline support/correctness gaps without stopping the sweep.",
    )
    args = parser.parse_args()
    if args.list_configs:
        print(
            json.dumps({"config_columns": config_columns, "configs": configs}, indent=2)
        )
        return 0
    if not args.config or not args.output_dir:
        parser.error("--config and --output-dir are required")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        parser.error("this full suite requires ROCm gfx950 (MI350)")
    return run(next(c for c in configs if c["name"] == args.config), args)


if __name__ == "__main__":
    raise SystemExit(main())
