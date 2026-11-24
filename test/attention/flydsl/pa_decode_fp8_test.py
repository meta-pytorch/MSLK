# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlyDSL native-fp8 paged decode (pa_decode_fp8) with more than 16 query heads
per KV head. The kernel then sets up Q in groups of 16 heads ("MTP groups"), and
every group goes through the same LDS slots."""

import math
import re

import pytest
import torch
from mslk.flydsl.common import is_flydsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flydsl_available(),
    reason="requires a ROCm GPU with FlyDSL",
)

PARTITION = 256  # keys per sub-partition of the kernel


def _gfx950() -> bool:
    from flydsl.runtime.device import get_rocm_arch

    return get_rocm_arch().startswith("gfx950")


def _inputs(B, H_kv, qgs, D, kv_len, block_size):
    """Random bf16 queries and K/V, and the fp8 paged KV cache (per-token scales)
    that dense_kv_to_fp8_paged builds from K/V, with its pages shuffled.

    Context lengths cycle through full, more than half, any, and at most 3
    sub-partitions, so the workgroups of one call finish at different times."""
    from mslk.attention.flydsl.decode.fp8_paged_cache import dense_kv_to_fp8_paged

    gen = torch.Generator(device="cuda").manual_seed(B * H_kv * qgs + D)
    shape = (B, kv_len, 1, H_kv, D)
    k = torch.randn(shape, generator=gen, device="cuda").to(torch.bfloat16)
    v = torch.randn(shape, generator=gen, device="cuda").to(torch.bfloat16)
    q = torch.randn(B, H_kv * qgs, D, generator=gen, device="cuda")
    u = torch.rand(B, generator=gen, device="cuda").tolist()
    lengths = []
    for b in range(B):
        n = [kv_len, kv_len // 2 + 1 + int(u[b] * (kv_len // 2))]
        n += [1 + int(u[b] * kv_len), 1 + int(u[b] * 3 * PARTITION)]
        lengths.append(min(n[b % 4], kv_len))
    *pages, block_tables = dense_kv_to_fp8_paged(k, v, block_size=block_size)
    # Physical page i moves to slot perm[i], as in a cache with scattered pages.
    perm = torch.randperm(block_tables.numel(), generator=gen, device="cuda")
    shuffled = [torch.empty_like(p) for p in pages]
    for dst, src in zip(shuffled, pages):
        dst[perm] = src
    block_tables = perm[block_tables.long()].to(torch.int32)
    lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return q.to(torch.bfloat16), k, v, lengths, (*shuffled, block_tables)


def _decode(q, cache, lengths):
    from mslk.attention.flydsl.decode.pa_decode_fp8_dispatch import (
        paged_attention_decode_fp8,
    )

    key_cache, value_cache, key_scale, value_scale, block_tables = cache
    out = torch.empty_like(q)
    paged_attention_decode_fp8(
        out,
        q,
        key_cache,
        value_cache,
        lengths,
        block_tables,
        q.shape[-1] ** -0.5,
        key_scale,
        value_scale,
    )
    return out


def _reference(q, k, v, lengths):
    """fp32 softmax(q k^T / sqrt(D)) v over each sequence's first lengths[b] keys."""
    B, kv_len, _, H_kv, D = k.shape
    qf = q.float().view(B, H_kv, -1, D)
    kf, vf = k[:, :, 0].float(), v[:, :, 0].float()
    scores = torch.einsum("bgrd,bkgd->bgrk", qf, kf) / math.sqrt(D)
    past = torch.arange(kv_len, device=q.device)[None, :] >= lengths[:, None]
    scores = scores.masked_fill(past[:, None, None, :], -math.inf)
    out = torch.einsum("bgrk,bkgd->bgrd", scores.softmax(-1), vf)
    return out.reshape(q.shape)


def _q_setup_lds_order(isa: str, staging_end: int) -> tuple[int, int]:
    """Scans the kernel's assembly up to its first MFMA, that is the Q setup
    before the KV loop. Returns the number of LDS writes into the Q and query
    scale slots (immediate offsets below staging_end; the block table and the
    per-token K/V scales are above it), and how many of those writes come after
    an LDS read with no s_barrier in between."""
    writes = unordered = reads = 0
    for line in isa.splitlines():
        instruction = line.split(";", 1)[0].strip()
        if instruction.startswith("v_mfma"):
            break
        if instruction == "s_barrier":
            reads = 0
        elif instruction.startswith("ds_read"):
            reads += 1
        elif instruction.startswith("ds_write"):
            offset = re.search(r"\boffset:(\d+)", instruction)
            if offset and int(offset.group(1)) >= staging_end:
                continue
            writes += 1
            unordered += reads > 0
    return writes, unordered


@pytest.mark.parametrize(
    "qgs,block_size,D",
    [
        (32, 16, 128),  # two full MTP groups
        (24, 64, 256),  # the second group has 8 heads
    ],
)
def test_mtp_group_q_setup_waits_for_reads_of_previous_group(
    monkeypatch, tmp_path, qgs, block_size, D
):
    """Each MTP group writes its Q and query scales to the same LDS slots, and
    every wave reads all 16 heads' slots. A barrier must keep any wave from
    writing the next group's values while another wave may still be reading
    the previous group's. Checked on the generated code, so it does not depend
    on timing."""
    if not _gfx950():
        pytest.skip("pa_decode_fp8 is gfx950-only")
    from mslk.attention.flydsl.decode import pa_decode_fp8 as mod

    # Compile the kernel again in this test, with FlyDSL's IR dump on.
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    mod.compile_pa_decode_ps.cache_clear()
    q, _, _, lengths, cache = _inputs(1, 1, qgs, D, PARTITION, block_size)
    _decode(q, cache, lengths)
    torch.cuda.synchronize()
    (isa_file,) = tmp_path.glob("pa_decode_ps_kernel*/*final_isa.s")

    writes, unordered = _q_setup_lds_order(
        isa_file.read_text(), mod.LDS_LOGITS_BYTES + mod.LDS_SOFTMAX_BYTES
    )

    # At least a Q store and a query-scale store per group.
    assert writes >= 2 * math.ceil(qgs / 16), writes
    assert unordered == 0, (
        f"{unordered} of the {writes} Q setup writes to LDS can overwrite values "
        "that another wave has not read yet (no s_barrier after the previous "
        "MTP group's LDS reads)"
    )


def test_qgs32_batch_with_two_workgroups_per_cu_is_deterministic():
    """32 sequences x 8 KV heads x 8 partitions = 2048 workgroups; at 213 VGPRs
    two of them run on each CU of an MI350X at a time. Without the barrier
    between MTP groups, about 1 call in 8 on these inputs returned a different
    output on an otherwise idle GPU. The kernels have no atomics and a fixed
    reduction order, so calls on the same inputs must give the same bits."""
    if not _gfx950():
        pytest.skip("pa_decode_fp8 is gfx950-only")
    calls = 100
    q, k, v, lengths, cache = _inputs(32, 8, 32, 128, 4096, 16)

    outs = [_decode(q, cache, lengths) for _ in range(calls)]

    torch.testing.assert_close(
        outs[0].float(), _reference(q, k, v, lengths), atol=0.2, rtol=0.15
    )
    differ = sum(not torch.equal(out, outs[0]) for out in outs[1:])
    assert differ == 0, f"{differ} of {calls - 1} calls differ from the first call"
