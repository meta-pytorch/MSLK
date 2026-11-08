# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dropout keep-mask that is bit-identical to CK's FMHA dropout.

CK draws the mask with a 7-round Philox keyed on (seed, offset) and counters
laid out per 32x32 tile of the [Mq, Mkv] score matrix of each (batch, head):

    philox(seed, offset + (batch * H + head) * 64 + lane)
        .get_philox_4x32(subsequence = (tile_row, tile_col))

Each of the 64 lanes of a tile owns 16 random bytes, and byte r of lane l lands
at row (r // 8) * 16 + (l // 32) * 8 + r % 8, column l % 32 of the tile. An
element is kept iff its byte is <= floor((1 - p) * 255).

Because the mask is a pure function of (seed, offset, shape, p), the FlyDSL
forward and backward can both regenerate it and stay interchangeable with CK's
forward and backward, which exchange the same (seed, offset) through
`Context.rng_state`.

Reference: ck_tile/core/utility/philox_rand.hpp, ck_tile/ops/fmha/block/
block_dropout.hpp and ck_tile/host/reference/reference_batched_dropout_randval.hpp.
"""

import torch

_PHILOX_KEY_A = 0x9E3779B9
_PHILOX_KEY_B = 0xBB67AE85
_PHILOX_MUL_A = 0xD2511F53
_PHILOX_MUL_B = 0xCD9E8D57
_U32 = 0xFFFFFFFF

_TILE = 32  # CK draws random numbers per 32x32 tile of the score matrix
_PHILOX_PER_TILE = 64  # Philox calls per tile: 64 lanes x 16 bytes = 32 x 32
# Bound on live Philox results per chunk; each costs a few int64 temporaries.
_MAX_CALLS_PER_CHUNK = 1 << 22


def keep_threshold(p: float) -> int:
    """Largest random byte that is kept (CK's `p_undrop_in_uint8_t`)."""
    return int((1.0 - p) * 255.0)


def philox_offset_increment(B: int, H: int, Mq: int, Mkv: int) -> int:
    """Generator offset to reserve per call, as CK's forward does."""
    return (B + 3) * (H + 1) * (Mq + 1) * (Mkv + 1)


def _mulhilo32(const: int, x: torch.Tensor):
    """(lo, hi) 32-bit halves of const * x for x < 2**32, in int64 arithmetic."""
    x_lo = x & 0xFFFF
    x_hi = x >> 16
    p0 = const * x_lo
    p1 = const * x_hi
    t = p0 + ((p1 & 0xFFFF) << 16)
    return t & _U32, (p1 >> 16) + (t >> 32)


def _philox_4x32(seed: int, ctr_x, ctr_y, ctr_z, ctr_w):
    key_x, key_y = seed & _U32, (seed >> 32) & _U32
    for _ in range(7):
        lo0, hi0 = _mulhilo32(_PHILOX_MUL_A, ctr_x)
        lo1, hi1 = _mulhilo32(_PHILOX_MUL_B, ctr_z)
        ctr_x, ctr_y, ctr_z, ctr_w = (
            hi1 ^ ctr_y ^ key_x,
            lo1,
            hi0 ^ ctr_w ^ key_y,
            lo0,
        )
        key_x = (key_x + _PHILOX_KEY_A) & _U32
        key_y = (key_y + _PHILOX_KEY_B) & _U32
    return ctr_x, ctr_y, ctr_z, ctr_w


def philox_randval(
    seed: int,
    offset: int,
    B: int,
    H: int,
    Mq: int,
    Mkv: int,
    device: torch.device,
) -> torch.Tensor:
    """CK's uint8 random values, shape [B, H, Mq, Mkv]."""
    rows = -(-Mq // _TILE)
    cols = -(-Mkv // _TILE)
    out = torch.empty(
        B, H, rows * _TILE, cols * _TILE, dtype=torch.uint8, device=device
    )
    lane = torch.arange(_PHILOX_PER_TILE, device=device, dtype=torch.int64)
    tile_row = torch.arange(rows, device=device, dtype=torch.int64)
    tile_col = torch.arange(cols, device=device, dtype=torch.int64)
    calls_per_head = rows * cols * _PHILOX_PER_TILE
    heads_per_chunk = max(1, _MAX_CALLS_PER_CHUNK // calls_per_head)
    bh = B * H
    for start in range(0, bh, heads_per_chunk):
        idx = torch.arange(start, min(bh, start + heads_per_chunk), device=device)
        n = idx.numel()
        shape = (n, rows, cols, _PHILOX_PER_TILE)
        # 64-bit counter offset + (batch * H + head) * 64 + lane
        head_off = offset + idx.view(n, 1, 1, 1) * _PHILOX_PER_TILE + lane
        words = _philox_4x32(
            seed,
            (head_off & _U32).expand(shape),
            (head_off >> 32).expand(shape),
            tile_row.view(1, rows, 1, 1).expand(shape),
            tile_col.view(1, 1, cols, 1).expand(shape),
        )
        # byte r of the lane's 16 bytes: word r // 4, byte r % 4 (little endian)
        vals = torch.stack(
            [
                ((words[r // 4] >> (8 * (r % 4))) & 0xFF).to(torch.uint8)
                for r in range(16)
            ],
            dim=-1,
        )
        # [n, rows, cols, half, 32 j, rh, rl] with r = rh * 8 + rl and
        # i = rh * 16 + half * 8 + rl, j = lane % 32.
        vals = vals.view(n, rows, cols, 2, _TILE, 2, 8)
        vals = vals.permute(0, 1, 5, 3, 6, 2, 4)  # n, rows, rh, half, rl, cols, j
        chunk = vals.reshape(n, rows * _TILE, cols * _TILE)
        out.view(bh, rows * _TILE, cols * _TILE)[idx] = chunk
    return out[:, :, :Mq, :Mkv]


def philox_keep_mask(
    seed: int,
    offset: int,
    p: float,
    B: int,
    H: int,
    Mq: int,
    Mkv: int,
    device: torch.device,
) -> torch.Tensor:
    """uint8 keep-mask (1 = kept), shape [B, H, Mq, Mkv], contiguous."""
    randval = philox_randval(seed, offset, B, H, Mq, Mkv, device)
    return (randval <= keep_threshold(p)).to(torch.uint8).contiguous()
