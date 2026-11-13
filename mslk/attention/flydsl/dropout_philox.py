# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

"""In-kernel Philox dropout shared by the FlyDSL forward and backward kernels.

The mask is never stored; both kernels derive the keep bit of P[b, h, m, n]:
Philox4x32-10 (as triton.language.random) with key = seed and counter =
(n // 4, m, b * H + h, 0), H = query heads. Word n % 4 is the random value; keep
if it is >= round(p * 2**32). The seed is a device int64[1] tensor, so neither
side syncs with the host and CUDA graph replays draw fresh masks.
"""

import flydsl.expr as fx
import torch
from mslk.flydsl import buffer_ops

PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85
PHILOX_ROUND_A = 0xD2511F53
PHILOX_ROUND_B = 0xCD9E8D57
PHILOX_ROUNDS = 10
# Consecutive columns of a row that share one Philox call.
DROPOUT_COLS_PER_CALL = 4


def dropout_drop_threshold(p: float) -> int:
    """Random words below this are dropped: round(p * 2**32), as an i32 bit pattern."""
    thr = min(int(round(p * 2**32)), 2**32 - 1)
    return thr - 2**32 if thr >= 2**31 else thr


def draw_dropout_seed(device: torch.device) -> torch.Tensor:
    """Fresh per-call Philox seed as a device int64[1] tensor (graph-capture safe)."""
    return torch.randint(0, 2**62, (1,), dtype=torch.int64, device=device)


def _mulhilo(a, b):
    prod = a.to(fx.Uint64) * b.to(fx.Uint64)
    return (prod >> 32).to(fx.Uint32), prod.to(fx.Uint32)


def philox_4x32(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 on Uint32 values (matches triton.language.random.philox)."""
    for i in range(PHILOX_ROUNDS):
        hi0, lo0 = _mulhilo(fx.Uint32(PHILOX_ROUND_A), c0)
        hi1, lo1 = _mulhilo(fx.Uint32(PHILOX_ROUND_B), c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if i < PHILOX_ROUNDS - 1:
            k0 = k0 + fx.Uint32(PHILOX_KEY_A)
            k1 = k1 + fx.Uint32(PHILOX_KEY_B)
    return c0, c1, c2, c3


def load_dropout_key(seed_tensor):
    """In-kernel: the (lo, hi) Uint32 Philox key from the int64[1] seed tensor."""
    rsrc = buffer_ops.create_buffer_resource(seed_tensor)
    lo = buffer_ops.buffer_load(rsrc, 0, vec_width=1, dtype=fx.Int32)
    hi = buffer_ops.buffer_load(rsrc, 1, vec_width=1, dtype=fx.Int32)
    return fx.Uint32(lo), fx.Uint32(hi)


def _u32(index):
    return fx.Uint32(fx.Int32(index))


def dropout_random_words(key, bh, row, col):
    """In-kernel: random words for columns 4*(col//4) .. 4*(col//4)+3 of a row."""
    return philox_4x32(
        _u32(col // DROPOUT_COLS_PER_CALL),
        _u32(row),
        _u32(bh),
        fx.Uint32(0),
        key[0],
        key[1],
    )


def keep_flag(word, threshold):
    """In-kernel: 1.0 if the element survives dropout, else 0.0 (Float32)."""
    keep = word >= fx.Uint32(threshold)
    return keep.select(fx.Float32(1.0), fx.Float32(0.0))


_U32 = 0xFFFFFFFF


def _mulhilo32_ref(const: int, x: torch.Tensor):
    """(hi, lo) 32-bit halves of const * x for 0 <= x < 2**32, exact in int64."""
    x_lo = x & 0xFFFF
    x_hi = x >> 16
    p0 = const * x_lo
    p1 = const * x_hi
    t = p0 + ((p1 & 0xFFFF) << 16)
    return (p1 >> 16) + (t >> 32), t & _U32


def reference_keep_mask(
    seed: int,
    p: float,
    B: int,
    H: int,
    Mq: int,
    Mkv: int,
    device: torch.device,
) -> torch.Tensor:
    """Host-side reference of the kernels' keep bits: uint8 [B, H, Mq, Mkv], 1 = kept.

    For tests and benchmarks; memory is O(B * H * Mq * Mkv).
    """
    groups = -(-Mkv // DROPOUT_COLS_PER_CALL)
    shape = (B * H, Mq, groups)
    c0 = torch.arange(groups, device=device, dtype=torch.int64).expand(shape)
    c1 = torch.arange(Mq, device=device, dtype=torch.int64).view(1, Mq, 1).expand(shape)
    c2 = (
        torch.arange(B * H, device=device, dtype=torch.int64)
        .view(-1, 1, 1)
        .expand(shape)
    )
    c3 = torch.zeros(shape, device=device, dtype=torch.int64)
    k0, k1 = seed & _U32, (seed >> 32) & _U32
    for i in range(PHILOX_ROUNDS):
        hi0, lo0 = _mulhilo32_ref(PHILOX_ROUND_A, c0)
        hi1, lo1 = _mulhilo32_ref(PHILOX_ROUND_B, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if i < PHILOX_ROUNDS - 1:
            k0 = (k0 + PHILOX_KEY_A) & _U32
            k1 = (k1 + PHILOX_KEY_B) & _U32
    words = torch.stack((c0, c1, c2, c3), dim=-1).reshape(B * H, Mq, -1)[..., :Mkv]
    threshold = min(int(round(p * 2**32)), 2**32 - 1)
    return (words >= threshold).to(torch.uint8).reshape(B, H, Mq, Mkv)
