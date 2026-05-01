"""PyTorch reference for FlashAttention-2.

Compared against: flash_attn.flash_attn_func
CUDA source: Dao-AILab/flash-attention csrc/flash_attn/
FlashAttention expects (batch, seqlen, nheads, headdim) layout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        # (B, S, H, D) -> (B, H, S, D) for PyTorch SDPA
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t)
        return out.transpose(1, 2)  # back to (B, S, H, D)


B, S, H, D = 4, 256, 8, 64


def get_inputs():
    q = torch.randn(B, S, H, D, dtype=torch.float16)
    k = torch.randn(B, S, H, D, dtype=torch.float16)
    v = torch.randn(B, S, H, D, dtype=torch.float16)
    return [q, k, v]


def get_init_inputs():
    return []
