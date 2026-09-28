"""API bridge from SGLang 0.5.6 FA4 calls to pinned official Dao-AILab FA4.

No attention math here: rename the LSE option and adapt the return container.
Official source commit: e9cf2c1651d2303191eb40a739a3c135fda00999.
"""
from flash_attn.cute.interface import flash_attn_varlen_func as official_varlen


def flash_attn_varlen_func(*args, return_softmax_lse=False, **kwargs):
    out, lse = official_varlen(*args, return_lse=return_softmax_lse, **kwargs)
    return (out, lse) if return_softmax_lse else out
