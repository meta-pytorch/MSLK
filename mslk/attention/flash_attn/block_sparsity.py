# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    from mslk.fb.mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch
except ImportError:
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

__all__ = ["BlockSparseTensorsTorch"]
