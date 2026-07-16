import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.bfloat16

def flash_attention_ref(Q, K, V, causal=True, sm_scale=None):
    if sm_scale is None:
        sm_scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    if causal:
        seq_len_q, seq_len_k = Q.shape[-2], K.shape[-2]
        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return flash_attention_ref(*args)


def get_inputs():
    q = torch.randn(2, 4, 256, 64, device="cuda", dtype=DT)
    k = torch.randn(2, 4, 256, 64, device="cuda", dtype=DT)
    v = torch.randn(2, 4, 256, 64, device="cuda", dtype=DT)
    return [q, k, v]

def get_init_inputs():
    return []
