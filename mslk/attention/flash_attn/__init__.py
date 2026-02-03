# @nolint # fbcode
"""Flash Attention CUTE (CUDA Template Engine) implementation."""

__version__ = "0.1.0"

import cutlass.cute as cute

from .interface import (
    flash_attn_func,
    # TODO: enable this
    # flash_attn_varlen_func,
)
