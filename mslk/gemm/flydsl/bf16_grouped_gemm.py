# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""BF16 grouped GEMM via FlyDSL.

Backed by the same kernel as the FP8 siblings in fp8_rowwise_grouped_gemm.py
and fp8_groupwise_grouped_gemm.py, compiled for BF16 operands instead of
quantised ones. BF16 carries its own exponent, so there are no scales: the
kernel folds nothing in the K loop and applies nothing in the epilogue, which
is what ``scaling="none"`` selects.

* ``mslk::bf16bf16bf16_grouped_stacked`` -- groups packed along M with a ``[G]``
  int64 row count per group, and row-major ``[G, N, K]`` weights.

CK served this op on ROCm until now, unlike the FP8 grouped ops, whose C++ slots
were already free. gemm_ops.cpp no longer registers it there and nothing takes
CK's place: FlyDSL is the only ROCm implementation, and calling the op without
the backend raises rather than quietly reaching a slower kernel. That is
deliberate -- CK and Triton are both being deprecated.

CK's own source marks ``bf16bf16bf16_grouped``, ``_cat`` and ``_dynamic``
"UNSUPPORTED AND DEPRECATED -- use _stacked", so they are deliberately not
served here. ``_grad``/``_wgrad`` are backward passes on a different kernel
shape and are left to Triton.

Tensor contract:
  X       : [total_M, K]   BF16  -- all groups concatenated along M
  W       : [G, N, K]      BF16  -- per-group weights, plain row-major
  M_sizes : [G]            int64 -- rows per group (sum to total_M)
  out     : [total_M, N]   BF16  -- optional caller-provided output
  Output  : [total_M, N]   BF16

  out[m, n] = sum_k X[m, k] * W[g, n, k]     for the group g owning row m
"""

from typing import Optional

import torch
from mslk.flydsl.common import require_flydsl
from mslk.gemm.flydsl import grouped_dispatch
from mslk.utils.device import is_gfx942, is_gfx950


def is_supported() -> bool:
    """Whether this module can serve the op on the current GPU.

    The kernel is built on MFMA, which the RDNA parts do not have -- and FlyDSL
    reports itself available on those, so having the backend says nothing about
    whether this op can run. mslk/gemm/__init__.py raises when this is False,
    there being no other ROCm implementation left to fall back to.
    """
    return is_gfx950() or is_gfx942()


def _assert_bf16_operands(X: torch.Tensor, W: torch.Tensor) -> None:
    """Reject operands the kernel would read as the wrong type.

    It passes them through as raw bytes, so a mismatched dtype would be
    contracted with the wrong exponent layout rather than rejected. The FP8
    siblings check the same thing about the FP8 flavour.
    """
    assert X.dtype == torch.bfloat16, f"X must be bfloat16, got {X.dtype}"
    assert W.dtype == torch.bfloat16, f"W must be bfloat16, got {W.dtype}"


def matmul_bf16bf16bf16_grouped_stacked(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    """BF16 grouped GEMM -> BF16, groups packed along M.

    ``num_sms`` is accepted and ignored. CK takes it as a hint for a persistent
    kernel that occupies a fixed number of CUs; this kernel launches one block
    per output tile and lets the scheduler place them, so there is nothing for
    the hint to bind to. It stays in the signature because the op's schema is
    shared with the CUDA implementation.
    """
    # Registration does not probe for FlyDSL, so this is the first point at
    # which it is required.
    require_flydsl()
    assert X.ndim == 2, f"X must be [total_M, K], got {X.shape}"
    assert W.ndim == 3, f"W must be [G, N, K], got {W.shape}"
    assert M_sizes.ndim == 1, f"M_sizes must be [G], got {M_sizes.shape}"
    total_M, K = X.shape
    G, N, Kw = W.shape
    assert Kw == K, f"K mismatch: X K={K}, W K={Kw}"
    assert M_sizes.shape[0] == G, f"M_sizes length {M_sizes.shape[0]} must equal G={G}"
    _assert_bf16_operands(X, W)
    assert M_sizes.dtype == torch.int64, f"M_sizes must be int64, got {M_sizes.dtype}"

    if out is not None:
        # CK's caller passes a flat buffer of total_M * N; the kernel writes a
        # [total_M, N] output, so take whichever shape it arrived in as long as
        # it holds the right number of elements.
        assert out.dtype == torch.bfloat16, f"out must be bfloat16, got {out.dtype}"
        assert out.numel() == total_M * N, (
            f"out must hold {total_M * N} elements, got {out.numel()}"
        )
        out = out.view(total_M, N)

    return grouped_dispatch.dispatch(
        X,
        W,
        None,
        None,
        M_sizes,
        b_preshuffled=False,
        scaling="none",
        in_dtype="bf16",
        out=out,
    )


# This module registers nothing. The op is registered in mslk/gemm/__init__.py,
# whose impl imports this module on the first call.
