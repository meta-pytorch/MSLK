# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers shared by the FlyDSL paged attention tests: an fp32 reference over a
paged KV cache and a call-recording spy."""

import math

import torch


def paged_reference(q, k, v, block_table, lengths, causal, window_left=-1, starts=None):
    """fp32 attention of ``q`` [B, Sq, H, D] over a paged ``k``/``v`` cache
    [pages, page_size, num_kv_heads, D], returned as [B, Sq, H, D].

    Request ``b`` attends to ``lengths[b]`` keys starting at ``starts[b]``
    (default 0) of the row its ``block_table`` pages spell out; a zero-length
    request returns zeros. Masks are bottom-right aligned: query row ``i`` sits
    at ``pos = n - Sq + i``, ``causal`` keeps keys ``<= pos`` and
    ``window_left >= 0`` keeps keys ``>= pos - window_left``.
    """
    batch, seqlen_q, num_heads, head_dim = q.shape
    page_size, num_kv_heads = k.shape[1], k.shape[2]
    group = num_heads // num_kv_heads
    outputs = []
    for b in range(batch):
        n = int(lengths[b])
        if n == 0:
            outputs.append(torch.zeros_like(q[b], dtype=torch.float32))
            continue
        start = 0 if starts is None else int(starts[b])
        pages = block_table[b, : -(-(start + n) // page_size)].long()
        kb = k[pages].reshape(-1, num_kv_heads, head_dim)[start : start + n].float()
        vb = v[pages].reshape(-1, num_kv_heads, head_dim)[start : start + n].float()
        qb = q[b].float().reshape(seqlen_q, num_kv_heads, group, head_dim)
        scores = torch.einsum("tngd,knd->ngtk", qb, kb) / math.sqrt(head_dim)
        q_pos = torch.arange(seqlen_q, device=q.device) + n - seqlen_q
        k_pos = torch.arange(n, device=q.device)
        keep = torch.ones(seqlen_q, n, dtype=torch.bool, device=q.device)
        if causal:
            keep &= k_pos[None, :] <= q_pos[:, None]
        if window_left >= 0:
            keep &= k_pos[None, :] >= q_pos[:, None] - window_left
        probs = scores.masked_fill(~keep, -math.inf).softmax(-1)
        outputs.append(
            torch.einsum("ngtk,knd->tngd", probs, vb).reshape(
                seqlen_q, num_heads, head_dim
            )
        )
    return torch.stack(outputs)


class Calls(list):
    """Keyword arguments of each recorded call; ``results`` holds what each
    call returned."""

    def __init__(self):
        super().__init__()
        self.results = []


def spy(monkeypatch, owner, name):
    """Replace ``owner.name`` with a wrapper that records each call and then
    forwards it to the real function. Returns the :class:`Calls` log."""
    calls = Calls()
    real = getattr(owner, name)

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        result = real(*args, **kwargs)
        calls.results.append(result)
        return result

    monkeypatch.setattr(owner, name, wrapper)
    return calls
