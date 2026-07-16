import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.float32

def layernorm_ref(x, weight, bias, eps=1e-5):
    normalized_shape = x.shape[-1:]
    return F.layer_norm(x, normalized_shape, weight, bias, eps)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return layernorm_ref(*args)


def get_inputs():
    x = torch.randn(2048, 256, device="cuda", dtype=DT)
    w = torch.randn(256, device="cuda", dtype=DT)
    b = torch.randn(256, device="cuda", dtype=DT)
    return [x, w, b]

def get_init_inputs():
    return []
