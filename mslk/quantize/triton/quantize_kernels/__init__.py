# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained, single-operator MX4 FP4 quantization Triton kernels.

Each module owns exactly one ``@triton.jit`` kernel, specialized for one
operator (MX4 non-stacked / stacked) with its ``SCALE_FORMAT`` / ``STACKED``
behavior hardcoded, so each operator can be read, perf-tuned, and
regression-tested in isolation. Genuinely-shared pieces (segment mapping,
scale math, pack primitives) live in ``fp4_primitives``.
"""

from typing import Optional

import torch


def _resolve_seed(seed: Optional[int], stochastic: bool, device: torch.device) -> int:
    """Resolve the user-supplied ``seed`` to an int the kernel can consume.

    Behavior matrix:
      - ``stochastic=False``: returns ``0``. The kernel's ``STOCHASTIC=False``
        constexpr DCE's the entire stochastic branch, so the value is
        irrelevant.
      - ``stochastic=True`` and ``seed is not None``: returns
        ``int(seed) & ((1 << 63) - 1)`` (mask to int63 to stay within
        ``tl.int64`` range and avoid Python int overflow on the kernel
        boundary).
      - ``stochastic=True`` and ``seed is None``: derives a seed from
        ``torch.cuda.default_generators[device.index].initial_seed()`` so
        callers respect ``torch.manual_seed``. Falls back to ``0`` when
        the input is on CPU (no CUDA generator available); the kernel
        will only be launched on CUDA in practice.
    """
    if not stochastic:
        return 0
    if seed is not None:
        return int(seed) & ((1 << 63) - 1)
    if device.type == "cuda":
        gen = torch.cuda.default_generators[device.index]
        return int(gen.initial_seed()) & ((1 << 63) - 1)
    return 0
