import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.bfloat16

def rmsnorm_ref(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return rmsnorm_ref(*args)


def get_inputs():
    x = torch.randn(2048, 256, device="cuda", dtype=DT)
    w = torch.randn(256, device="cuda", dtype=DT)
    return [x, w]

def get_init_inputs():
    return []
