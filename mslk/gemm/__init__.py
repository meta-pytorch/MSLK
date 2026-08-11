# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import importlib
from collections.abc import Callable
from types import ModuleType

from mslk.utils.torch.library import load_library_buck

load_library_buck("//mslk/csrc/gemm:gemm_ops")

gemm_ops = [
    "//mslk/csrc/gemm/cutlass:cutlass_bf16bf16bf16_grouped_grad",
    "//mslk/csrc/gemm/cutlass:cutlass_bf16bf16bf16_grouped_wgrad",
]
for op in gemm_ops:
    load_library_buck(op)

# Bypass set_python_module checks for internal; this check is disabled
# by default in OSS PyTorch.
import torch._utils_internal  # noqa: E402

# pyrefly: ignore [missing-attribute]
torch._utils_internal.REQUIRES_SET_PYTHON_MODULE = False

import torch  # noqa: E402

from . import _meta  # noqa: F401, E402

if torch.version.hip is not None:
    # Register Triton implementations for ROCm.  Each import triggers the
    # @torch.library.impl("mslk::...", "CUDA") decoration in the respective
    # module, which overrides the default (non-existent) CUDA impl so that
    # torch.ops.mslk.* dispatches to the Triton kernel on AMD.
    from .triton import (  # noqa: F401
        f8i4bf16_rowwise_gemm as _f8i4bf16_rowwise_gemm,
        fp8_groupwise_gemm,
        fp8_groupwise_grouped_gemm,
        grouped_gemm as _grouped_gemm,
        int4_gemm as _int4_gemm,
        int4_grouped_gemm as _int4_grouped_gemm,
        mx8mx4_gemm,
        mx8mx8_gemm,
    )

    # FlyDSL is an opt-in backend: this package deliberately does NOT depend on
    # //mslk/mslk/gemm:flydsl_ops. That dependency drags the FlyDSL wheel into
    # the binary of everything that reaches mslk.gemm transitively (torchao
    # does), and its mere presence is enough to break the process: FlyDSL ships
    # its own MLIR/LLVM 23, so any consumer that gates a module-scope import on
    # `importlib.util.find_spec("flydsl")` -- torchao's
    # prototype/moe_training/kernels/mxfp8/flydsl_utils.py does -- will load it
    # next to Triton's copy, the two interpose, and the process dies on SIGSEGV
    # inside mlirRegisterAllDialects. No `except` can catch that.
    #
    # Targets that want the FlyDSL kernels depend on flydsl_ops themselves; the
    # lazy imports below then resolve, and Triton serves everyone else.
    @functools.lru_cache(maxsize=1)
    def _flydsl_gemm_module() -> ModuleType | None:
        """The FlyDSL grouped-gemm module, or None when it is not opted into.

        Resolved through ``importlib`` rather than a static ``from .flydsl
        import ...``: this package deliberately does not depend on
        ``//mslk/mslk/gemm:flydsl_ops``, so the module is genuinely absent
        unless the caller opted in, and a static import would not resolve.
        """
        try:
            from mslk.flydsl.common import is_flydsl_available

            if not is_flydsl_available():
                return None
            return importlib.import_module(
                "mslk.gemm.flydsl.fp8_groupwise_grouped_gemm"
            )
        except ImportError:
            return None

    if hasattr(torch.ops, "mslk") and hasattr(
        torch.ops.mslk, "f8f8bf16_groupwise_grouped"
    ):
        # FlyDSL owns this op wherever it is opted into and Triton is the
        # fallback, but only one CUDA implementation can win, so neither kernel
        # module registers it and the choice is arbitrated here, on first call.
        @functools.lru_cache(maxsize=1)
        def _groupwise_grouped_impl() -> Callable[..., torch.Tensor]:
            mod = _flydsl_gemm_module()
            if mod is not None:
                return mod.matmul_f8f8bf16_groupwise_grouped
            from .triton.fp8_groupwise_grouped_gemm import (
                matmul_f8f8bf16_groupwise_grouped as _impl,
            )

            return _impl

        try:

            @torch.library.impl("mslk::f8f8bf16_groupwise_grouped", "CUDA")
            def _f8f8bf16_groupwise_grouped_rocm(
                XQ: torch.Tensor,
                WQ: torch.Tensor,
                x_scale: torch.Tensor,
                w_scale: torch.Tensor,
                M_sizes: torch.Tensor,
            ) -> torch.Tensor:
                return _groupwise_grouped_impl()(XQ, WQ, x_scale, w_scale, M_sizes)

        except RuntimeError:
            pass  # already registered (e.g. module imported more than once)

    if hasattr(torch.ops, "mslk") and hasattr(
        torch.ops.mslk, "f8f8bf16_groupwise_grouped_preshuffle"
    ):
        # ROCm-only, and FlyDSL is its only implementation, so there is nothing
        # to arbitrate -- but registering still must not import FlyDSL, hence
        # the same first-call resolution. Calling it without having opted into
        # flydsl_ops raises rather than silently doing something else.
        try:

            @torch.library.impl("mslk::f8f8bf16_groupwise_grouped_preshuffle", "CUDA")
            def _f8f8bf16_groupwise_grouped_preshuffle_rocm(
                XQ: torch.Tensor,
                WQ: torch.Tensor,
                x_scale: torch.Tensor,
                w_scale: torch.Tensor,
                M_sizes: torch.Tensor,
            ) -> torch.Tensor:
                mod = _flydsl_gemm_module()
                if mod is None:
                    raise RuntimeError(
                        "mslk::f8f8bf16_groupwise_grouped_preshuffle requires the "
                        "FlyDSL backend. Add //mslk/mslk/gemm:flydsl_ops to your "
                        "target's deps."
                    )
                return mod.matmul_f8f8bf16_groupwise_grouped_preshuffle(
                    XQ, WQ, x_scale, w_scale, M_sizes
                )

        except RuntimeError:
            pass  # already registered (e.g. module imported more than once)
