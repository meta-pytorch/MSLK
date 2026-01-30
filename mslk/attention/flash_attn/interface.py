try:
    from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func
except ImportError:
    from flash_attn.cute import flash_attn_func
