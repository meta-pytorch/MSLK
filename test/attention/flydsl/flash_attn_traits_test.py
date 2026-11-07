# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from mslk.attention.flydsl.flash_attn_utils import _make_flash_attn_generic_traits


@pytest.mark.parametrize("gpu_arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("Dqk,Dv", [(6, 64), (64, 6), (6, 6), (64, 64), (72, 40)])
def test_dma_override_requires_whole_lanes(monkeypatch, gpu_arch, Dqk, Dv):
    monkeypatch.setenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_GFX942_DMA", "1")
    monkeypatch.setenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "1")
    monkeypatch.setenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3", "0")
    traits = _make_flash_attn_generic_traits(
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        gpu_arch=gpu_arch,
        block_m=64,
        flat_work_group_size=128,
        path_tag="N32",
        head_dim_qk=Dqk,
        head_dim_v=Dv,
    )
    expected_dma = Dqk % 8 == 0 and Dv % 8 == 0
    assert traits.ENABLE_DMA == expected_dma


@pytest.mark.parametrize("Dqk,Dv", [(0, 64), (64, 0), (0, 0)])
def test_zero_stored_head_dims_rejected(Dqk, Dv):
    with pytest.raises(ValueError, match="stored head dims"):
        _make_flash_attn_generic_traits(
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            gpu_arch="gfx950",
            head_dim_qk=Dqk,
            head_dim_v=Dv,
        )
