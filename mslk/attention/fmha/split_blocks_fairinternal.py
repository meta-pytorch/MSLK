# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

from typing import List, Optional, Tuple, Union

import torch

from .attn_bias import (
    _GappySeqInfo,
    _PaddedSeqLenInfo,
    _SeqLenInfo,
    AttentionBias,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)


def split_blocks_for_decoding_gpu_part(
    input_bias: Union[
        BlockDiagonalPaddedKeysMask, BlockDiagonalCausalWithOffsetPaddedKeysMask
    ],
    batchify_len: Optional[int],
    block_tables: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    This is the gpu part of split_blocks_for_decoding,
    which can be called in advance.
    """
    if batchify_len is None:
        return None
    assert batchify_len > 0
    assert input_bias.q_seqinfo.min_seqlen == input_bias.q_seqinfo.max_seqlen

    seqstart = input_bias.k_seqinfo.seqstart  # (B+1,)
    seqlen = input_bias.k_seqinfo.seqlen  # (B,)

    # compute raw block boundaries
    k_ends = seqstart[:-1] + seqlen  # (B,)
    # For non-speculative decoding, we have a causal bias here,
    # which will always be from-bottom-right style.
    # Q and K are aligned so that their last tokens are at the same position.
    # If seqlen == batchify_len, the first token of the query is at position batchify_len - 1,
    # and it can attend to all keys from the previous iRoPE chunk.
    # The diagram shows that when seqlen == batchify_len == N and the bias is causal,
    # Q can still attend to K from the previous chunk.
    # -----------iRoPE chunk 0---------|---------iRoPE chunk 1---------------
    #                             Q[0] |
    # K[0] K[1] K[2] ... K[N-2] K[N-1] |

    # For speculative decoding, we use this function for the prefix bias only.
    # We are called with a non-causal bias.
    # The query is positioned after the keys, and so when seqlen == batchify_len,
    # the first token of the query is at position batchify_len.
    # So it can't attend to any key from the previous chunk,
    # so we want k_starts == k_ends => k_lens == 0.
    # The diagram shows that when seqlen == batchify_len == N and the bias is non-causal,
    # Q is located entirely in the next iRoPE chunk and can't attend to K[0] ... K[N-1].
    # ------------iRoPE chunk 0---------------|---------iRoPE chunk 1---------
    #                                         | Q[0] Q[1] Q[2]
    # K[0] K[1] K[2] ... K[N-3] K[N-2] K[N-1] |

    shift = int(isinstance(input_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask))
    k_starts = (k_ends - shift) // batchify_len * batchify_len
    k_starts = torch.where(seqlen == 0, k_ends, k_starts)
    k_lens = k_ends - k_starts

    if block_tables is None:
        k_seqstarts = torch.cat([k_starts, seqstart[-1:]])
    else:
        k_seqstarts = (k_starts - seqstart[:-1]).clamp(min=0)
        k_lens = k_lens + k_seqstarts

    return k_seqstarts, k_lens


def split_blocks_for_decoding(
    input_bias: Union[
        BlockDiagonalPaddedKeysMask, BlockDiagonalCausalWithOffsetPaddedKeysMask
    ],
    batchify_len: Optional[int],
    block_tables: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    gpu_data: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> Optional[Union[BlockDiagonalGappyKeysMask, PagedBlockDiagonalGappyKeysMask]]:
    """
    For decoding, when query length is 1, we can represent iRoPE-batchified bias as a gappy bias.
    This function can also be applied for speculative decoding, when query length is > 1,
    but same across all batch elements. In this case we assume that query (draft) lies entirely
    in one block/subsequence, not crossing the boundary. Cases when the query crosses the boundary
    need to be handled separately by the caller.
    """
    if batchify_len is None:
        return None
    assert batchify_len > 0
    assert input_bias.q_seqinfo.min_seqlen == input_bias.q_seqinfo.max_seqlen

    if gpu_data is None:
        gpu_data = split_blocks_for_decoding_gpu_part(
            input_bias, batchify_len, block_tables, page_size
        )
        assert gpu_data is not None
    k_seqstarts, k_lens = gpu_data

    k_seqstarts_list = []
    k_seqlens_list = []
    k_seqlens_list_actual = []
    B = len(input_bias.k_seqinfo.seqlen_py)
    # About the shift, see the comment in split_blocks_for_decoding_gpu_part.
    shift = int(isinstance(input_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask))
    for i in range(B):
        input_k_start_ = input_bias.k_seqinfo.seqstart_py[i]
        input_k_len_ = input_bias.k_seqinfo.seqlen_py[i]
        input_k_end_ = input_k_start_ + input_k_len_
        k_seqstart = (input_k_end_ - shift) // batchify_len * batchify_len
        if input_k_len_ == 0:
            k_seqstart = input_k_end_
        k_seqend = min(k_seqstart + batchify_len, input_k_end_)
        k_len = k_seqend - k_seqstart
        # NOTE: With chunked, `k_len` cannot exceed the original length `input_k_len_`, so we clamp it here.
        k_len = min(k_len, input_k_len_)

        if k_seqstart < 0:
            k_len = k_seqstart = 0
        k_seqstart = (
            k_seqstart if block_tables is None else max(k_seqstart - input_k_start_, 0)
        )
        k_seqstarts_list.append(k_seqstart)
        k_seqlens_list_actual.append(k_len)
        k_seqlens_list.append(k_len if block_tables is None else k_len + k_seqstart)

    OutBiasType = (
        BlockDiagonalCausalWithOffsetGappyKeysMask
        if isinstance(input_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
        else BlockDiagonalGappyKeysMask
    )
    PagedOutBiasType = (
        PagedBlockDiagonalCausalWithOffsetGappyKeysMask
        if isinstance(input_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
        else PagedBlockDiagonalGappyKeysMask
    )
    if block_tables is None:
        k_seqstarts_list.append(input_bias.k_seqinfo.seqstart_py[-1])
        return OutBiasType(
            q_seqinfo=input_bias.q_seqinfo,
            k_seqinfo=_GappySeqInfo(
                seqstart_py=k_seqstarts_list,
                seqstart=k_seqstarts,
                seqlen=k_lens,
                seqlen_py=k_seqlens_list,
                min_seqlen=min(k_seqlens_list),
                max_seqlen=max(k_seqlens_list),
            ),
        )
    assert page_size is not None
    return PagedOutBiasType(
        q_seqinfo=input_bias.q_seqinfo,
        k_seqinfo=_GappySeqInfo(
            seqstart_py=k_seqstarts_list,
            seqstart=k_seqstarts,
            seqlen=k_lens,
            seqlen_py=k_seqlens_list,
            min_seqlen=min(k_seqlens_list_actual),
            max_seqlen=max(k_seqlens_list_actual),
        ),
        block_tables=block_tables,
        page_size=page_size,
    )


def split_blocks_for_prefill(
    input_bias: BlockDiagonalPaddedKeysMask, batchify_len: Optional[int]
) -> Optional[BlockDiagonalPaddedKeysMask]:
    """
    From
    https://github.com/fairinternal/llm_inference/blob/11bbb2/llm_inference/models/disagg_transformer.py#L1955
    """
    if batchify_len is None:
        return None
    padding = input_bias.k_seqinfo.padding
    assert padding % batchify_len == 0, f"{padding} % {batchify_len} != 0"
    split_factor = padding // batchify_len
    batch_size = len(input_bias.q_seqinfo.seqstart_py) - 1
    new_batch_size = batch_size * split_factor
    k_seqlen = input_bias.k_seqinfo.seqlen
    q_seqlen = input_bias.q_seqinfo.seqstart[1:] - input_bias.q_seqinfo.seqstart[:-1]
    k_seqlen_each = k_seqlen.repeat_interleave(split_factor, output_size=new_batch_size)
    q_seqlen_each = q_seqlen.repeat_interleave(split_factor, output_size=new_batch_size)
    res_seqlen_each = k_seqlen_each - q_seqlen_each
    seqpos = torch.arange(
        0, padding, batchify_len, device=k_seqlen.device, dtype=k_seqlen.dtype
    )
    seqpos_start = seqpos.repeat(batch_size)
    k_lengths = (k_seqlen_each - seqpos_start).clamp(min=0, max=batchify_len)
    res_lengths = (res_seqlen_each - seqpos_start).clamp(min=0, max=batchify_len)

    k_seqstart = torch.arange(
        0,
        new_batch_size * batchify_len + 1,
        batchify_len,
        device=k_seqlen.device,
        dtype=k_seqlen.dtype,
    )
    k_seqstart_py = list(range(0, new_batch_size * batchify_len + 1, batchify_len))
    q_seqstart = torch.zeros_like(k_seqstart)
    torch.cumsum(k_lengths - res_lengths, 0, out=q_seqstart[1:])

    # start at 2 to avoid reshaping issues with
    # https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp#L602
    max_q_len = 2
    min_q_len = 2
    max_k_len = 0
    q_seqstart_list: List[int] = [0]
    k_seqlen_list: List[int] = []
    for i in range(len(input_bias.k_seqinfo.seqlen)):
        q_seqlen_ = (
            input_bias.q_seqinfo.seqstart_py[i + 1]
            - input_bias.q_seqinfo.seqstart_py[i]
        )
        k_seqlen_ = input_bias.k_seqinfo.seqlen_py[i]
        res_seqlen_ = k_seqlen_ - q_seqlen_
        for seqpos_ in range(0, padding, batchify_len):
            k_chunk_size = max(min(k_seqlen_ - seqpos_, batchify_len), 0)
            res_chunk_size = max(min(res_seqlen_ - seqpos_, batchify_len), 0)
            q_chunk_size = k_chunk_size - res_chunk_size

            q_seqstart_list.append(q_seqstart_list[-1] + q_chunk_size)
            k_seqlen_list.append(k_chunk_size)
            if q_chunk_size > max_q_len:
                max_q_len = q_chunk_size
            if q_chunk_size < min_q_len:
                min_q_len = q_chunk_size
            if k_chunk_size > max_k_len:
                max_k_len = k_chunk_size

    batchify_attn_bias = input_bias.__class__(
        q_seqinfo=_SeqLenInfo(
            seqstart=q_seqstart,
            max_seqlen=max_q_len,
            min_seqlen=min_q_len,
            seqstart_py=q_seqstart_list,
        ),
        k_seqinfo=_PaddedSeqLenInfo(
            seqstart=k_seqstart,
            seqlen_py=k_seqlen_list,
            seqlen=k_lengths,
            padding=batchify_len,
            seqstart_py=k_seqstart_py,
            min_seqlen=0,
            max_seqlen=max_k_len,
        ),
    )
    return batchify_attn_bias


def maybe_make_paged(
    attn_bias: Optional[
        Union[
            BlockDiagonalPaddedKeysMask,
            BlockDiagonalGappyKeysMask,
        ]
    ],
    block_tables: Optional[torch.Tensor],
    page_size: int,
    notional_padding: Optional[int],
) -> Optional[AttentionBias]:
    """
    Convert attention bias into its paged version if block_tables is not None.
    Args:
        attn_bias: input attention bias.
        block_tables: table of shape [batch_size, max_pages_per_lane]
                        redirecting from logical to physical pages.
        page_size: number of tokens per page.
        notional_padding: if input attention bias is gappy, it has
            no notion of padding, sequence starts are arbitrary.
            However, we need to know how to divide logical sequence space
            into lanes corresponding to each row of block tables.
            In other words, where is 0th block in i-th row of block table
            located in the logical space?
            This function assumes that it's located at i * notional_padding.
            The value of notional_padding needs to be consisted which
            padding used when block_tables was created.
            For example, if a gappy bias was created from a padded bias
            using split_blocks* functions, notional padding
            should be equal to the padding of the original bias.
    Returns:
        Paged version of the original attention bias.
    """
    if attn_bias is None:
        return None
    if block_tables is None:
        return attn_bias

    attn_batch_size = len(attn_bias.k_seqinfo.seqlen)
    if attn_batch_size != block_tables.shape[0]:
        # In case of iRoPE each batch lane has been split into smaller chunks,
        # so we need to reshape the block tables accordingly.
        block_tables = block_tables.view(attn_batch_size, -1)
    if isinstance(attn_bias, BlockDiagonalGappyKeysMask):
        assert (
            notional_padding is not None
        ), "Notional padding must be specified to create gappy paged biases."
        return attn_bias.make_paged(
            block_tables=block_tables,
            page_size=page_size,
            notional_padding=notional_padding,
            paged_type=PagedBlockDiagonalGappyKeysMask,
        )
    if isinstance(attn_bias, PagedBlockDiagonalGappyKeysMask):
        return attn_bias
    paged_type = (
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
        else PagedBlockDiagonalPaddedKeysMask
    )
    assert isinstance(attn_bias, BlockDiagonalPaddedKeysMask)
    return attn_bias.make_paged(
        block_tables=block_tables, page_size=page_size, paged_type=paged_type
    )
