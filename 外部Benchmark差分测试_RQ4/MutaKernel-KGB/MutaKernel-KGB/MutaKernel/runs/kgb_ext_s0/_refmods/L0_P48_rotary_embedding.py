import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.float16

def rotary_embedding_ref(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return rotary_embedding_ref(*args)


def get_inputs():
    x = torch.randn(1024, 64, device="cuda", dtype=DT)
    cos = torch.randn(1024, 32, device="cuda", dtype=DT)
    sin = torch.randn(1024, 32, device="cuda", dtype=DT)
    return [x, cos, sin]

def get_init_inputs():
    return []
