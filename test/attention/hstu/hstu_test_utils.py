# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Seeded input generators and tolerances shared by the HSTU attention tests.

Each ``make_*_inputs`` returns the keyword arguments of the matching reference
in ``mslk.attention.hstu._torch_ref`` (and of the kernel that mirrors it), so a
test can call ``op(**inputs)``.
"""

from typing import Any, Dict, Sequence

import torch

HEAD_DIMS = (64, 128, 256)


def generate_sparse_seq_len(
    size: int, max_seq_len: int, sparsity: float, generator: torch.Generator
) -> torch.Tensor:
    """``[size]`` int64 lengths whose mean is about ``sparsity * max_seq_len``.

    Same scheme as generative-recommenders: sparsity 1 is all ``max_seq_len``,
    above 0.5 draws from ``[(2 * sparsity - 1) * max_seq_len, max_seq_len)``,
    otherwise from ``[0, 2 * sparsity * max_seq_len)``.
    """
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    if sparsity == 1.0:
        return torch.full((size,), max_seq_len, dtype=torch.int64)
    if sparsity >= 0.5:
        low, high = int((2 * sparsity - 1.0) * max_seq_len), max_seq_len
    else:
        low, high = 0, int(2 * sparsity * max_seq_len)
    if high <= low:
        return torch.full((size,), low, dtype=torch.int64)
    return torch.randint(low, high, (size,), generator=generator, dtype=torch.int64)


def generate_targets(
    lengths: torch.Tensor, max_targets: int, generator: torch.Generator
) -> torch.Tensor:
    """``[B]`` int64 target counts, each in ``[0, min(max_targets, length)]``."""
    caps = lengths.clamp(max=max_targets)
    return (
        (torch.rand(lengths.shape, generator=generator) * (caps + 1))
        .floor()
        .to(torch.int64)
    )


def lengths_to_offsets(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths.to(torch.int64), dim=0)
    return offsets.to(device)


def _rand(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return (torch.rand(*shape, generator=generator) * 2 - 1).to(
        device=device, dtype=dtype
    )


def _make_qkv(
    total_q: int,
    total_kv: int,
    heads: int,
    heads_kv: int,
    attn_dim: int,
    hidden_dim: int,
    shared_kv: bool,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    requires_grad: bool,
) -> Dict[str, torch.Tensor]:
    if shared_kv and hidden_dim != attn_dim:
        raise ValueError(
            f"shared_kv needs hidden_dim == attn_dim, got {hidden_dim} and {attn_dim}"
        )
    q = _rand((total_q, heads, attn_dim), dtype, device, generator).requires_grad_(
        requires_grad
    )
    k = _rand((total_kv, heads_kv, attn_dim), dtype, device, generator).requires_grad_(
        requires_grad
    )
    if shared_kv:
        v = k
    else:
        v = _rand(
            (total_kv, heads_kv, hidden_dim), dtype, device, generator
        ).requires_grad_(requires_grad)
    return {"q": q, "k": k, "v": v}


def make_ragged_inputs(
    batch_size: int,
    max_seq_len: int,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    sparsity: float,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    max_targets: int = 0,
    shared_kv: bool = False,
    requires_grad: bool = False,
) -> Dict[str, Any]:
    """Inputs of ``ragged_hstu_attention_ref``.

    ``num_targets`` is set when ``max_targets > 0``.
    """
    generator = torch.Generator().manual_seed(seed)
    lengths = generate_sparse_seq_len(batch_size, max_seq_len, sparsity, generator)
    total = int(lengths.sum())
    inputs: Dict[str, Any] = {"N": max_seq_len, "alpha": attn_dim**-0.5}
    inputs.update(
        _make_qkv(
            total,
            total,
            heads,
            heads,
            attn_dim,
            hidden_dim,
            shared_kv,
            dtype,
            device,
            generator,
            requires_grad,
        )
    )
    inputs["seq_offsets"] = lengths_to_offsets(lengths, device)
    if max_targets > 0:
        inputs["num_targets"] = generate_targets(lengths, max_targets, generator).to(
            device
        )
    inputs["shared_kv"] = shared_kv
    return inputs


def make_cross_inputs(
    batch_size: int,
    max_kv_len: int,
    max_q_len: int,
    heads: int,
    G: int,
    attn_dim: int,
    hidden_dim: int,
    sparsity: float,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    max_targets: int = 0,
    shared_kv: bool = False,
    requires_grad: bool = False,
) -> Dict[str, Any]:
    """Inputs of ``cross_hstu_attention_ref``.

    KV lengths follow ``sparsity``; query lengths are uniform in ``[1, max_q_len]``.
    ``attn_scale`` is ``None`` (``1 / max_kv_len``); ``num_targets`` is set when
    ``max_targets > 0``.
    """
    generator = torch.Generator().manual_seed(seed)
    lengths_kv = generate_sparse_seq_len(batch_size, max_kv_len, sparsity, generator)
    lengths_q = torch.randint(
        1, max_q_len + 1, (batch_size,), generator=generator, dtype=torch.int64
    )
    inputs: Dict[str, Any] = {"max_seq_len": max_kv_len, "alpha": attn_dim**-0.5}
    inputs.update(
        _make_qkv(
            int(lengths_q.sum()),
            int(lengths_kv.sum()),
            heads,
            heads // G,
            attn_dim,
            hidden_dim,
            shared_kv,
            dtype,
            device,
            generator,
            requires_grad,
        )
    )
    inputs.update(
        seq_offsets=lengths_to_offsets(lengths_kv, device),
        seq_offsets_q=lengths_to_offsets(lengths_q, device),
        max_q_len=max_q_len,
        attn_scale=None,
        G=G,
        shared_kv=shared_kv,
    )
    if max_targets > 0:
        inputs["num_targets"] = generate_targets(lengths_q, max_targets, generator).to(
            device
        )
    return inputs


def make_grouped_inputs(
    group_size: int,
    max_seq_lens: Sequence[int],
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    sparsity: float,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    max_targets: int = 0,
    requires_grad: bool = False,
) -> Dict[str, Any]:
    """Inputs of ``grouped_ragged_hstu_attention_ref``.

    ``len(max_seq_lens)`` groups of ``group_size`` sequences; the per-group mask
    options are all 0.
    """
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.cat(
        [
            generate_sparse_seq_len(group_size, n, sparsity, generator)
            for n in max_seq_lens
        ]
    )
    total = int(lengths.sum())
    seq_offsets = lengths_to_offsets(lengths, device)
    zeros = torch.zeros(len(max_seq_lens), dtype=torch.int64, device=device)
    inputs: Dict[str, Any] = {"Z": group_size}
    inputs.update(
        _make_qkv(
            total,
            total,
            heads,
            heads,
            attn_dim,
            hidden_dim,
            False,
            dtype,
            device,
            generator,
            requires_grad,
        )
    )
    num_targets = None
    if max_targets > 0:
        num_targets = generate_targets(lengths, max_targets, generator).to(device)
    inputs.update(
        seq_offsets=seq_offsets,
        group_offsets=seq_offsets[::group_size].clone(),
        num_targets=num_targets,
        max_seq_len_tensor=torch.tensor(max_seq_lens, device=device),
        max_attn_len_tensor=zeros,
        full_attn_size_tensor=zeros.clone(),
        contextual_seq_len_tensor=zeros.clone(),
        alpha=attn_dim**-0.5,
        invalid_attn_mask_type="lower_triangular",
    )
    return inputs


def assert_hstu_close(
    out: torch.Tensor, ref: torch.Tensor, mode: str, name: str = "out"
) -> None:
    """Compare against the reference with the public TLX test tolerances.

    ``silu``: atol = 8e-3 * max|ref|, rtol = 8e-3 (outputs are unnormalized, so
    the scale follows the reference). ``softmax``: atol = 8e-4, rtol = 1.6e-2.
    Apply per tensor: outputs and each gradient are checked on their own.
    """
    if mode == "silu":
        ref_max = ref.abs().max().item() if ref.numel() else 0.0
        atol, rtol = 8e-3 * ref_max, 8e-3
    elif mode == "softmax":
        atol, rtol = 8e-4, 1.6e-2
    else:
        raise ValueError(f"mode must be 'silu' or 'softmax', got {mode!r}")
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=atol,
        rtol=rtol,
        msg=lambda formatted: f"{name} ({mode}): {formatted}",
    )
