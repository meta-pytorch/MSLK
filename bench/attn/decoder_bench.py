# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Paged-attention decode benchmark: FlyDSL vs Triton.

Backends (``--backends``): flydsl, triton (dense f16/bf16); flydsl_fp8 (native
e4m3fn per-token symmetric), triton_fp8 (int32-packed asymmetric scale/shift). The
two fp8 schemes aren't bit-compatible but decode the same KV (quantized once, outside
the timed region), so latencies are comparable.

Two timing modes:
  * eager (``--no-cuda-graph``, default): shared ``mslk.bench.common.utils.do_bench``.
  * graph (``--cuda-graph``): HIP graph capture + replay (removes launch overhead).

gfx950 gotchas (see the runners / _bench_ms_graph for detail):
  * fp8 runners cache the ``flyc.compile`` CompiledFunction so timing is kernel-only
    and scales with KV (calling the public dispatcher directly pays ~0.38ms/call of
    JIT dispatch that hides the ~0.02ms kernel — flat, meaningless numbers).
  * flydsl_fp8 IS graph-capturable (its kernels thread the capture stream); dense
    flydsl is NOT (launches on default stream → empty graph, caught by EmptyGraphError
    → reported skip); triton/triton_fp8 skipped in graph mode (HSA_INVALID_PACKET).
  * triton/*fp8/flydsl_fp8 are timed in a subprocess per shape (allocator scratch
    faults / cross-kernel symbol clashes would otherwise crash the sweep).

Usage:
    python bench/attn/decoder_bench.py
    python bench/attn/decoder_bench.py --shapes decode_llm --dtype bf16
    python bench/attn/decoder_bench.py --backends flydsl_fp8,triton_fp8 --cuda-graph
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import click
import torch

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

from mslk.bench.common.utils import BenchOptions, do_bench


def _bench_ms_eager(fn: Callable, rep_ms: int = 200) -> float:
    """Eager GPU time via the shared do_bench (consistent with gemm/conv/quantize benches).

    do_bench's cuda_graph/rotating_buffer are NOT used: cuda_graph unrolls thousands of
    fn() into one capture (segfaults the fp8 kernel, no empty-graph guard) so graph
    timing stays local; rotating_buffer needs tensors passed as args but our runners
    close over them (zero-arg thunk).
    """
    return do_bench(fn, (), BenchOptions(cuda_graph=False, rep_ms=rep_ms))


def _bench_ms_eager_events(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    """Fixed-rep raw-event eager timing — internal probe only, used as the
    non-empty-graph baseline in _bench_ms_graph (do_bench self-tunes its rep count)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(rep):
        fn()
    end_ev.record()
    end_ev.synchronize()
    return start_ev.elapsed_time(end_ev) / rep  # ms


class EmptyGraphError(RuntimeError):
    """CUDA-graph capture recorded no work — fn launched off the capture stream (e.g.
    FlyDSL's default-stream launch). Replay is a no-op, so surface it instead of a
    bogus sub-microsecond time."""


def _bench_ms_graph(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    """GPU kernel time via CUDA-graph capture + replay (removes per-launch dispatch).

    Requires fn to launch onto the current (capture) stream and reuse the same buffers.
    Default-stream launches capture empty -> raised as EmptyGraphError. Kept local (not
    do_bench_cudagraph, which segfaults the fp8 kernel and lacks an empty-graph guard).
    """
    # Eager baseline (small, cheap) — used only to sanity-check that the graph is
    # non-empty.  A real graph replays in ~the eager kernel time; an empty graph
    # replays in a fraction of it.  Uses a plain event loop (not do_bench) so this
    # stays a lightweight internal probe.
    eager_ref = _bench_ms_eager_events(fn, warmup=warmup, rep=min(rep, 50))

    # Warm up on a side stream first so lazy allocations / autotune happen before
    # capture (capture forbids new allocations and synchronizations).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    # A few replays to settle before timing.
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    for _ in range(rep):
        graph.replay()
    end_ev.record()
    end_ev.synchronize()

    ms = start_ev.elapsed_time(end_ev) / rep

    # Empty-graph guard: a genuine capture replays in at least a good fraction of
    # the eager kernel time.  ≤15% means nothing was recorded (off-capture-stream
    # launch) — the sub-µs "time" is noise, not a real speedup.
    if ms < 0.15 * eager_ref:
        raise EmptyGraphError(
            f"graph replay {ms*1e3:.1f}us << eager {eager_ref*1e3:.1f}us — "
            "kernel launched off the capture stream (nothing captured)"
        )
    return ms


def _bench_ms(fn: Callable, rep_ms: int = 200, use_cuda_graph: bool = False) -> float:
    """Dispatch to graph (local capture) or eager (shared do_bench) timing.

    ``rep_ms`` is the target duration passed through to both timers.  The graph
    path converts it to a fixed replay count (~10 reps/ms, capped) since it times a
    single captured launch; the eager path hands ``rep_ms`` straight to do_bench.
    """
    if use_cuda_graph:
        rep = min(500, max(10, rep_ms * 10))
        return _bench_ms_graph(fn, warmup=25, rep=rep)
    return _bench_ms_eager(fn, rep_ms=rep_ms)


def _bytes_read_write(B: int, Hq: int, Hkv: int, kv_seqlen: int, D: int,
                      dtype: torch.dtype) -> int:
    """Approximate HBM traffic for one decode step (bytes)."""
    elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    # Q read: B * 1 * Hq * D
    q_read = B * Hq * D * elem
    # K read: B * kv_seqlen * Hkv * D
    kv_read = B * kv_seqlen * Hkv * D * elem * 2
    # Output write: B * 1 * Hq * D
    out_write = B * Hq * D * elem
    return q_read + kv_read + out_write


# --------------------------------------------------------------------------- #
# Shape definitions
# --------------------------------------------------------------------------- #

# Shape = (B, Hq, Hkv, kv_seqlen, D)
ShapeList = List[Tuple[int, int, int, int, int]]
_shape_registry: Dict[str, Callable[[], ShapeList]] = {}


def register_shapes(name: str):
    def deco(fn: Callable[[], ShapeList]) -> Callable[[], ShapeList]:
        _shape_registry[name] = fn
        return fn
    return deco


@register_shapes("default")
def _shapes_default() -> ShapeList:
    """Representative decode shapes: small/medium batch × popular LLM head configs."""
    shapes = []
    # (B, Hq, Hkv, kv_seqlen, D)
    for kv_len in [512, 2048, 4096, 8192]:
        shapes.append((1,  32, 8,  kv_len, 128))   # Llama3-8B  MQA-like
        shapes.append((8,  32, 8,  kv_len, 128))
        shapes.append((1,  64, 8,  kv_len, 128))   # Llama3-70B GQA
        shapes.append((8,  64, 8,  kv_len, 128))
    return shapes


@register_shapes("decode_llm")
def _shapes_decode_llm() -> ShapeList:
    """Common LLM decode shapes with various GQA ratios."""
    return [
        # (B,  Hq,   Hkv, kv_len, D)
        (1,   32,   8,    512,   128),
        (1,   32,   8,   2048,   128),
        (1,   32,   8,   4096,   128),
        (8,   32,   8,   2048,   128),
        (16,  32,   8,   2048,   128),
        (1,   64,   8,   2048,   128),
        (1,   64,  16,   2048,   128),
        (1,  128,  16,   2048,   128),  # large model
        (1,   32,   4,   2048,   256),  # D=256
        (8,   32,   4,   2048,   256),
    ]


@register_shapes("sweep_kv")
def _shapes_sweep_kv() -> ShapeList:
    """Sweep KV sequence length."""
    shapes = []
    for kv_len in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        shapes.append((1, 32, 8, kv_len, 128))
    return shapes


@register_shapes("sweep_batch")
def _shapes_sweep_batch() -> ShapeList:
    """Sweep batch size."""
    shapes = []
    for B in [1, 2, 4, 8, 16, 32, 64]:
        shapes.append((B, 32, 8, 2048, 128))
    return shapes


@register_shapes("ck_test")
def _shapes_ck_test() -> ShapeList:
    """Shapes from test_ck_splitk_decoder in the test suite."""
    shapes = []
    for d in [128, 256]:
        for padding, bsz in [(32, 8), (4096, 1), (32, 1), (4096, 8)]:
            shapes.append((bsz, 16, 16, padding, d))
    return shapes


# --------------------------------------------------------------------------- #
# Backend runners
# --------------------------------------------------------------------------- #

def _make_tensors(B: int, Hq: int, Hkv: int, kv_seqlen: int, D: int,
                  dtype: torch.dtype, device: str = "cuda"):
    """Allocate Q/K/V tensors in the canonical 5D BMGHK layout."""
    q = torch.randn(B, 1, 1, Hq, D, dtype=dtype, device=device)
    k = torch.randn(B, kv_seqlen, 1, Hkv, D, dtype=dtype, device=device)
    v = torch.randn(B, kv_seqlen, 1, Hkv, D, dtype=dtype, device=device)
    seq = torch.full((B,), kv_seqlen, dtype=torch.int32, device=device)
    scale = float(D ** -0.5)
    return q, k, v, seq, scale


def _run_flydsl(q, k, v, seq, scale) -> Optional[Callable]:
    """FlyDSL MFMA decode (primary kernel)."""
    try:
        from mslk.attention.fmha.flydsl.pa_decode_dense import pa_decode_launch
        from mslk.flydsl.common import is_flydsl_available
        if not is_flydsl_available():
            return None
        B, _, G, H_q, D = q.shape
        _, KV_MAX, _, _, _ = k.shape
        if D % 16 != 0 or q.dtype not in (torch.float16, torch.bfloat16):
            return None
        # split_k=0 lets the launcher pick via the per-kernel heuristic.
        pa_decode_launch(q, k, v, seq, scale, split_k=0)
        return lambda: pa_decode_launch(q, k, v, seq, scale, split_k=0)
    except Exception:
        return None


def _run_triton(q, k, v, seq, scale,
                disable_autotune: bool = False) -> Optional[Callable]:
    """Build a callable that runs the Triton split-K kernel for one shape.

    On ROCm/gfx950, Triton's intermediate buffers (o_splitk, lse_splitk) can be
    freed by PyTorch's caching allocator before the GPU kernel finishes when
    shapes change within one process, causing a GPU memory fault.  The main loop
    therefore times Triton in a subprocess per shape (see ``_bench_triton_subproc``);
    this in-process runner is only safe for a single shape.
    ``disable_autotune=True`` uses FwOp_S1 (split_k=1) to skip autotuning.
    """
    try:
        from mslk.attention.fmha.triton_splitk import FwOp, FwOp_S1
        from mslk.attention.fmha.common import Inputs
        from mslk.attention.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
        op = FwOp_S1 if disable_autotune else FwOp
        if not op.is_available():
            return None

        B, _, G, Hq, D = q.shape
        _, KV, _, Hkv, _ = k.shape
        kv_seqlen_list = seq.cpu().tolist()

        attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
            q_seqlen=[1] * B,
            kv_seqlen=[int(s) for s in kv_seqlen_list],
            kv_padding=KV,
        )
        k_flat = k.reshape(1, B * KV, 1, Hkv, D).contiguous()
        v_flat = v.reshape(1, B * KV, 1, Hkv, D).contiguous()
        q_flat = q.reshape(1, B, 1, Hq, D).contiguous()
        attn_bias.k_seqinfo.to(k.device)
        attn_bias.q_seqinfo.to(q.device)

        inp = Inputs(q_flat, k_flat, v_flat, attn_bias=attn_bias, scale=scale)
        reasons = op.not_supported_reasons(inp)
        if reasons:
            return None
        op.apply(inp, False)
        torch.cuda.synchronize()
        return lambda: op.apply(inp, False)
    except Exception:
        return None


def _make_attn_bias(B: int, KV: int, seq):
    """Padded-decode attention bias (1 query token per sequence, KV padded to KV)."""
    from mslk.attention.fmha.attn_bias import (
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    )
    ab = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[1] * B,
        kv_seqlen=[int(s) for s in seq.cpu().tolist()],
        kv_padding=KV,
    )
    ab.k_seqinfo.to(seq.device)
    ab.q_seqinfo.to(seq.device)
    return ab


def _run_flydsl_fp8(q, k, v, seq, scale) -> Optional[Callable]:
    """FlyDSL native-fp8 paged decode over a pre-quantized fp8 KV cache (gfx950).

    KV is quantized + paged ONCE outside the timed region (like _run_triton_fp8, and
    like a real fp8-resident cache) so the timed callable is kernel-only. The per-call
    quantize path (Inputs.quantize_kv_to_fp8) would fold quant cost into every call.
    """
    try:
        import flydsl.compiler as flyc
        from mslk.attention.fmha.flydsl.pa_decode_fp8_dispatch import (
            is_fp8_paged_decode_available,
        )
        from mslk.attention.fmha.flydsl.fp8_paged_adapter import dense_kv_to_fp8_paged
        from mslk.attention.fmha.flydsl.pa_decode_fp8 import (
            get_recommended_splits, KV_COMPUTE_BLOCK,
            compile_pa_decode_ps, compile_pa_decode_ps_reduce,
            _get_query_input_dtype, _get_output_dtype_str,
        )
        if not is_fp8_paged_decode_available():
            return None
        B, _, G, Hq, D = q.shape          # bench tensors: G == 1, Hkv in the H slot
        _, _, _, Hkv, _ = k.shape

        # One-time quant + paging (realistic fp8-resident KV cache).
        key_cache, value_cache, key_scale, value_scale, block_tables = (
            dense_kv_to_fp8_paged(k, v, block_size=16)
        )
        BG = B * G
        context_lengths = (
            seq.to(torch.int32).view(B, 1).expand(B, G).reshape(BG).contiguous()
        )
        q_flat = q.reshape(BG, Hq, D).contiguous()
        out = torch.zeros(BG, Hq, D, dtype=q.dtype, device=q.device)

        num_kv_heads = key_cache.shape[1]
        query_group_size = Hq // num_kv_heads
        eqgs = query_group_size          # query_length == 1 for decode
        block_size = key_cache.shape[-2]
        trans_v = len(value_cache.shape) == 5
        per_token_kv = key_scale.ndim > 1
        mcpn = get_recommended_splits(
            BG, num_kv_heads, split_kv_blocks=KV_COMPUTE_BLOCK // block_size
        )
        dev = q.device
        # Preallocate the partition scratch ONCE (the launcher otherwise allocates
        # exp_sums/max_logits/temporary_output on every call).
        exp_sums = torch.zeros(BG, num_kv_heads, mcpn, eqgs, device=dev, dtype=torch.float32)
        max_logits = torch.full((BG, num_kv_heads, mcpn, eqgs), float("-inf"),
                                device=dev, dtype=torch.float32)
        tmp_out = torch.zeros(BG, num_kv_heads, mcpn, eqgs, D, device=dev, dtype=torch.bfloat16)
        out_5d = out.reshape(BG, 1, num_kv_heads, query_group_size, D)

        # Build the compute + reduce kernels once (lru_cached), then cache their
        # FlyDSL CompiledFunction so the timed region skips the per-call JIT dispatch
        # path (arg-binding + Protocol isinstance cache-key rebuild) that otherwise
        # dominates — ~0.38ms/call of pure Python, hiding the real ~0.02ms GPU time.
        # This mirrors what mslk.flydsl.jit.run_compiled does for the dense path.
        compute = compile_pa_decode_ps(
            block_size=block_size, max_context_partition_num=mcpn, softmax_scale=scale,
            trans_v=trans_v, query_group_size=query_group_size, per_token_kv=per_token_kv,
            query_length=1, query_input_dtype=_get_query_input_dtype(q_flat), head_dim=D,
        )
        reduce = compile_pa_decode_ps_reduce(
            head_dim=D, eqgs=eqgs, max_parts=mcpn,
            output_dtype_str=_get_output_dtype_str(out),
        )
        # Everything except the trailing stream slot is fixed per shape.  The stream
        # is appended at CALL time (not baked in here): CUDA-graph capture runs on a
        # side stream, and both kernels must launch onto THAT stream to be recorded —
        # a build-time snapshot of the default stream would make capture see nothing.
        compute_head = (
            exp_sums, max_logits, tmp_out, q_flat, key_cache, value_cache,
            block_tables, context_lengths, key_scale, value_scale,
            q_flat.stride(0), q_flat.stride(1),
            key_cache.stride(0), key_cache.stride(1),
            value_cache.stride(0), value_cache.stride(1),
            exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
            tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2), tmp_out.stride(3),
            block_tables.stride(0),
            key_scale.stride(0) if per_token_kv else 0,
            key_scale.stride(1) if per_token_kv else 0,
            BG, num_kv_heads, mcpn,
        )
        reduce_head = (
            out_5d, exp_sums, max_logits, tmp_out,
            num_kv_heads * eqgs * D, eqgs * D,
            exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
            tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2), tmp_out.stride(3),
            num_kv_heads, BG, num_kv_heads,
        )
        # Compile once against a representative stream (the default stream is fine;
        # the CompiledFunction is keyed on arg TYPES, not the stream pointer value).
        s0 = torch.cuda.current_stream()
        cf_compute = flyc.compile(compute["launch"], *compute_head, s0)
        cf_reduce = flyc.compile(reduce["launch"], *reduce_head, s0)

        def _call():
            s = torch.cuda.current_stream()  # live stream — the capture stream during graph capture
            cf_compute(*compute_head, s)
            cf_reduce(*reduce_head, s)
        _call()
        torch.cuda.synchronize()
        return _call
    except Exception:
        return None


def _quant_pack_triton_fp8(x: torch.Tensor):
    """Quantize dense KV to Triton's int32-packed asymmetric fp8 format.

    Returns ``(packed_int32, scale_shift_int32)`` where the last-dim fp8 bytes are
    reinterpreted as int32 and the per-token (scale, shift) pair is packed as two
    f16 values into one int32 — the layout ``triton_splitk.InputsFp8`` expects.
    Mirrors the reference harness; uses the arch-correct fp8 dtype from
    ``get_fp8_constants`` (e4m3fn on gfx950).
    """
    from mslk.utils.triton.fp8_utils import get_fp8_constants
    fp8_dtype = get_fp8_constants()[0]
    fmax = torch.finfo(fp8_dtype).max

    Bx, M, G, H, Dx = x.shape
    xr = x.reshape(-1, Dx).float()
    shift = xr.mean(-1)
    xc = xr - shift[..., None]
    s = torch.nan_to_num(xc.abs().max(-1)[0] / fmax, posinf=1)
    xq = (xc / s[..., None]).to(fp8_dtype)
    packed = xq.view(torch.uint8).reshape(Bx, M, G, H, Dx).view(torch.int32)
    ss = torch.concat(
        [s.reshape(Bx, M, G, H, 1).half(), shift.reshape(Bx, M, G, H, 1).half()],
        dim=-1,
    ).flatten(-2).view(torch.int32)
    return packed, ss


def _run_triton_fp8(q, k, v, seq, scale) -> Optional[Callable]:
    """Triton split-K decode over int32-packed asymmetric fp8 KV (``InputsFp8``).

    KV is pre-quantized once (outside the timed region) into Triton's packed
    format; the timed callable only runs the kernel.  Like the dense Triton path
    this is timed in a subprocess per shape (allocator scratch-freeing fault).
    """
    try:
        from mslk.attention.fmha.triton_splitk import FwOp
        from mslk.attention.fmha.common import InputsFp8
        if not FwOp.is_available():
            return None
        B, _, G, Hq, D = q.shape
        _, KV, _, Hkv, _ = k.shape
        attn_bias = _make_attn_bias(B, KV, seq)
        q_flat = q.reshape(1, B, 1, Hq, D).contiguous()
        k_flat = k.reshape(1, B * KV, 1, Hkv, D).contiguous()
        v_flat = v.reshape(1, B * KV, 1, Hkv, D).contiguous()
        ki, ks = _quant_pack_triton_fp8(k_flat)
        vi, vs = _quant_pack_triton_fp8(v_flat)
        inp = InputsFp8(q_flat, ki, vi, attn_bias=attn_bias, scale=scale,
                        k_fp8_scale_shift=ks, v_fp8_scale_shift=vs)
        reasons = FwOp.not_supported_reasons(inp)
        if reasons:
            return None
        FwOp.apply(inp, False)
        torch.cuda.synchronize()
        return lambda: FwOp.apply(inp, False)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Subprocess isolation (Triton eager multi-shape)
# --------------------------------------------------------------------------- #

# In-process runners keyed by backend name.  Only backends listed here can be
# timed via the subprocess worker (Triton needs it; FlyDSL does not but is
# included so the worker is backend-agnostic).
_RUNNERS: Dict[str, Callable] = {
    "flydsl": _run_flydsl,
    "triton": _run_triton,
    "flydsl_fp8": _run_flydsl_fp8,
    "triton_fp8": _run_triton_fp8,
}


def _bench_subproc(
    backend: str,
    B: int, Hq: int, Hkv: int, kv_seqlen: int, D: int, dtype: str,
    rep_ms: int, disable_autotune: bool, use_graph: bool = False,
) -> Tuple[float, str]:
    """Time one backend+shape in a fresh subprocess; return ``(ms, status)``.

    A crashing child (GPU fault, non-zero exit) is reported as ``("err")`` /
    ``("skip")`` without taking down the parent sweep — that isolation is the
    whole point (Triton's allocator frees scratch across shapes → GPU fault; the
    FlyDSL fp8 artifact collides with the dense path's GPU-module symbols).  With
    ``use_graph`` the child times via CUDA-graph capture (and reports ``skip`` if
    the graph comes back empty).
    """
    import json
    import subprocess

    payload = json.dumps({
        "backend": backend, "B": B, "Hq": Hq, "Hkv": Hkv,
        "kv_seqlen": kv_seqlen, "D": D, "dtype": dtype,
        "rep_ms": rep_ms, "disable_autotune": disable_autotune,
        "use_graph": use_graph,
    })
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", payload],
        capture_output=True, text=True,
    )
    # The worker prints exactly one line: ``RESULT <json>`` on success.
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            try:
                res = json.loads(line[len("RESULT "):])
                return float(res["ms"]), res["status"]
            except Exception:
                break
    # No RESULT line → the child faulted/crashed (core dump, OOM, GPU fault).
    return 0.0, "err"


def _worker_main(payload: str) -> None:
    """Subprocess entry: time ONE backend on ONE shape, print ``RESULT <json>``.

    Runs in its own process so a Triton GPU fault (freed o_splitk/lse_splitk
    scratch across shapes) cannot corrupt the parent's CUDA context.
    """
    import json

    spec = json.loads(payload)
    torch_dtype = {
        "f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32
    }[spec["dtype"]]
    q, k, v, seq, scale = _make_tensors(
        spec["B"], spec["Hq"], spec["Hkv"], spec["kv_seqlen"], spec["D"], torch_dtype
    )
    runner = _RUNNERS[spec["backend"]]
    if spec["backend"] == "triton":
        fn = runner(q, k, v, seq, scale, disable_autotune=spec["disable_autotune"])
    else:
        fn = runner(q, k, v, seq, scale)  # fp8 runners take no extra kwargs
    if fn is None:
        print(f"RESULT {json.dumps({'ms': 0.0, 'status': 'skip'})}")
        return
    try:
        ms = _bench_ms(
            fn, rep_ms=spec["rep_ms"], use_cuda_graph=spec.get("use_graph", False),
        )
    except EmptyGraphError:
        print(f"RESULT {json.dumps({'ms': 0.0, 'status': 'skip'})}")
        return
    print(f"RESULT {json.dumps({'ms': ms, 'status': 'ok'})}")


# --------------------------------------------------------------------------- #
# Metrics and formatting
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    B: int
    Hq: int
    Hkv: int
    kv_seqlen: int
    D: int
    dtype: str
    backend: str
    ms: float
    bw_gbs: float
    status: str  # "ok" | "skip" | "err"


# Short column labels per backend.
_BACKEND_LABEL = {
    "flydsl": "FlyDSL", "triton": "Triton",
    "flydsl_fp8": "FlyDSL-f8", "triton_fp8": "Triton-f8",
}


def _header(run_backends: List[str]) -> str:
    cols = f"{'B':>4} {'Hq':>4} {'Hkv':>4} {'KV':>6} {'D':>4} {'dtype':>7}  "
    cols += " ".join(f"{_BACKEND_LABEL.get(b, b):>10}" for b in run_backends)
    # Speedup vs Triton for whichever FlyDSL variant(s) ran.
    if "flydsl" in run_backends and "triton" in run_backends:
        cols += f"  {'Fly/Tri':>9}"
    if "flydsl_fp8" in run_backends and "triton_fp8" in run_backends:
        cols += f"  {'Fly8/Tri8':>9}"
    return cols


def _result_row(
    B: int, Hq: int, Hkv: int, kv_seqlen: int, D: int, dtype: str,
    results: Dict[str, Optional[Result]], run_backends: List[str],
) -> str:
    def fmt_ms(r):
        if r is None or r.status != "ok":
            return f"{'N/A':>10}"
        return f"{r.ms:>9.3f}ms" if r.ms >= 0.001 else f"{'<0.001':>10}"

    def speedup(a: Optional[Result], b: Optional[Result]) -> str:
        if a is None or b is None or a.status != "ok" or b.status != "ok":
            return f"{'N/A':>9}"
        return f"{b.ms / a.ms:>8.2f}x"

    row = f"{B:>4} {Hq:>4} {Hkv:>4} {kv_seqlen:>6} {D:>4} {dtype:>7}  "
    row += " ".join(fmt_ms(results.get(b)) for b in run_backends)
    if "flydsl" in run_backends and "triton" in run_backends:
        row += f"  {speedup(results.get('flydsl'), results.get('triton'))}"
    if "flydsl_fp8" in run_backends and "triton_fp8" in run_backends:
        row += f"  {speedup(results.get('flydsl_fp8'), results.get('triton_fp8'))}"
    return row


# --------------------------------------------------------------------------- #
# Main benchmark
# --------------------------------------------------------------------------- #

@click.command()
@click.option("--shapes", default="default", type=click.Choice(list(_shape_registry)),
              show_default=True, help="Shape set to benchmark.")
@click.option("--dtype", default="f16", type=click.Choice(["f16", "bf16", "f32"]),
              show_default=True, help="KV/Q dtype.")
@click.option("--rep-ms", default=200, show_default=True,
              help="Target benchmark duration per shape (ms).")
@click.option("--cuda-graph/--no-cuda-graph", default=False, show_default=True,
              help="Time via real CUDA-graph replay (removes launch overhead). "
                   "Triton is skipped in this mode (un-graphable on gfx950); use "
                   "--both-graph-modes to get graphed FlyDSL + eager Triton together.")
@click.option("--both-graph-modes", is_flag=True, default=False,
              help="Run each shape with AND without CUDA graph, writing both to CSV.")
@click.option("--backends", default="flydsl,triton",
              help="Comma-separated backends: flydsl, triton, flydsl_fp8, triton_fp8.")
@click.option("--output", default=None,
              help="Write CSV results to this path. Defaults to bench/attn/results/<shapes>_<dtype>_<device>_<timestamp>.csv")
@click.option("--disable-triton-autotune", is_flag=True, default=False,
              help="Pin Triton to split_k=1 (avoids GPU hang during autotuning on some configs).")
@click.option("--worker", default=None, hidden=True,
              help="Internal: JSON spec to time one backend+shape in this subprocess.")
def invoke_main(shapes: str, dtype: str, rep_ms: int, cuda_graph: bool, both_graph_modes: bool,
         backends: str, output: Optional[str], disable_triton_autotune: bool,
         worker: Optional[str]) -> None:
    """Decode attention benchmark: FlyDSL vs Triton."""
    if worker is not None:
        _worker_main(worker)
        return

    import csv as _csv
    import os

    torch_dtype = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[dtype]
    run_backends = [b.strip() for b in backends.split(",")]
    shape_list = _shape_registry[shapes]()

    device_name = torch.cuda.get_device_name(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output is None:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        dev_slug = device_name.replace(" ", "_").replace("/", "_")[:30]
        output = os.path.join(results_dir, f"{shapes}_{dtype}_{dev_slug}_{timestamp}.csv")

    graph_modes = [True, False] if both_graph_modes else [cuda_graph]

    print(f"Decoder attention benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Device: {device_name}")
    print(f"Shapes: {shapes} ({len(shape_list)} configs), dtype={dtype}")
    print(f"Backends: {', '.join(run_backends)}")
    print(f"Graph modes: {graph_modes}")
    print(f"Output CSV: {output}")

    runner_map = {
        "flydsl":     _run_flydsl,
        "flydsl_fp8": _run_flydsl_fp8,
        "triton":     lambda q, k, v, seq, scale: _run_triton(
            q, k, v, seq, scale, disable_autotune=disable_triton_autotune
        ),
        "triton_fp8": _run_triton_fp8,
    }

    # Backends that cannot be timed under CUDA-graph capture: the Triton paths raise
    # HSA_STATUS_ERROR_INVALID_PACKET_FORMAT during HIP graph capture on gfx950, so
    # they are skipped in graph mode.  (flydsl_fp8 IS graph-capturable — the fp8
    # kernels thread the capture stream through both compute + reduce launches.)
    _NO_GRAPH_BACKENDS = {"triton", "triton_fp8"}
    # Backends timed in a subprocess per shape (BOTH modes).  Independent reasons:
    #   * triton / triton_fp8 — o_splitk/lse_splitk scratch is freed by the caching
    #     allocator across shape changes within one process, faulting the GPU; some
    #     fp8 GQA shapes also fault the kernel outright.  (Eager only — graph-skipped.)
    #   * flydsl_fp8 — its compiled artifact shares GPU-module global symbols with
    #     the dense FlyDSL path; running both in one process collides and faults.
    #     Each is clean alone, so a fresh process per shape sidesteps the clash — in
    #     graph mode too (the child captures + replays, reporting real kernel time).
    _SUBPROC_BACKENDS = {"triton", "triton_fp8", "flydsl_fp8"}

    all_csv_rows: List[dict] = []

    for use_graph in graph_modes:
        graph_label = "cuda_graph" if use_graph else "no_graph"
        timing_note = ("CUDA-graph replay (launch overhead removed)"
                       if use_graph else
                       "shared do_bench eager launch (per-call dispatch included)")
        print(f"\n{'='*80}")
        print(f"  Mode: {graph_label}")
        print(f"  Timing: {timing_note}.")
        if use_graph and any(b in _NO_GRAPH_BACKENDS for b in run_backends):
            skipped = [b for b in run_backends if b in _NO_GRAPH_BACKENDS]
            print(f"  Note: {', '.join(skipped)} skipped in graph mode (un-graphable on gfx950).")
        print(f"{'='*80}")
        print(_header(run_backends))
        print("-" * len(_header(run_backends)))

        for B, Hq, Hkv, kv_seqlen, D in shape_list:
            q, k, v, seq, scale = _make_tensors(B, Hq, Hkv, kv_seqlen, D, torch_dtype)
            nbytes = _bytes_read_write(B, Hq, Hkv, kv_seqlen, D, torch_dtype)
            row_results: Dict[str, Optional[Result]] = {}

            for backend in run_backends:
                runner = runner_map.get(backend)
                if runner is None:
                    row_results[backend] = None
                    continue

                # Graph mode: skip un-graphable backends entirely.
                if use_graph and backend in _NO_GRAPH_BACKENDS:
                    row_results[backend] = Result(
                        B, Hq, Hkv, kv_seqlen, D, dtype, backend, 0.0, 0.0, "skip"
                    )
                    continue

                # Subprocess-isolated backends: route out of process so a GPU fault on
                # one shape can't kill the sweep.  The child honors use_graph, so
                # flydsl_fp8 is captured+replayed there (Triton only reaches this path
                # in eager mode — it is graph-skipped above).
                if backend in _SUBPROC_BACKENDS:
                    ms, status = _bench_subproc(
                        backend, B, Hq, Hkv, kv_seqlen, D, dtype,
                        rep_ms=rep_ms,
                        disable_autotune=disable_triton_autotune,
                        use_graph=use_graph,
                    )
                    bw = nbytes / ms / 1e6 if status == "ok" else 0.0
                    row_results[backend] = Result(
                        B, Hq, Hkv, kv_seqlen, D, dtype, backend, ms, bw, status
                    )
                    if status == "err":
                        click.echo(f"  [{backend}] B={B} Hq={Hq} KV={kv_seqlen} D={D}: "
                                   f"subprocess crashed", err=True)
                    continue

                try:
                    fn = runner(q, k, v, seq, scale)
                    if fn is None:
                        row_results[backend] = Result(
                            B, Hq, Hkv, kv_seqlen, D, dtype, backend, 0.0, 0.0, "skip"
                        )
                        continue
                    ms = _bench_ms(fn, rep_ms=rep_ms, use_cuda_graph=use_graph)
                    bw = nbytes / ms / 1e6  # GB/s
                    row_results[backend] = Result(
                        B, Hq, Hkv, kv_seqlen, D, dtype, backend, ms, bw, "ok"
                    )
                except EmptyGraphError:
                    # Backend launches off the capture stream — can't be graphed on
                    # this stack.  Report as skip (not a bogus fast number).
                    row_results[backend] = Result(
                        B, Hq, Hkv, kv_seqlen, D, dtype, backend, 0.0, 0.0, "skip"
                    )
                    if not getattr(invoke_main, "_warned_empty_graph", False):
                        click.echo(
                            f"  [{backend}] not graph-capturable on this stack "
                            "(launches on default stream); reported as skip in graph mode.",
                            err=True)
                        invoke_main._warned_empty_graph = True
                except Exception as e:
                    row_results[backend] = Result(
                        B, Hq, Hkv, kv_seqlen, D, dtype, backend, 0.0, 0.0, "err"
                    )
                    click.echo(f"  [{backend}] B={B} Hq={Hq} KV={kv_seqlen} D={D}: {e}",
                               err=True)

            print(_result_row(B, Hq, Hkv, kv_seqlen, D, dtype, row_results, run_backends))

            for bk, r in row_results.items():
                if r is not None:
                    all_csv_rows.append({
                        "device":    device_name,
                        "timestamp": timestamp,
                        "shapes":    shapes,
                        "dtype":     dtype,
                        "cuda_graph": use_graph,
                        "B":         B,
                        "Hq":        Hq,
                        "Hkv":       Hkv,
                        "kv_seqlen": kv_seqlen,
                        "D":         D,
                        "GQA_ratio": Hq // Hkv if Hkv > 0 else 1,
                        "backend":   bk,
                        "ms":        r.ms if r.status == "ok" else "",
                        "bw_gbs":    r.bw_gbs if r.status == "ok" else "",
                        "status":    r.status,
                    })

    with open(output, "w", newline="") as f:
        if all_csv_rows:
            writer = _csv.DictWriter(f, fieldnames=all_csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_csv_rows)
    print(f"\nResults written to {output}")


if __name__ == "__main__":
    invoke_main()
