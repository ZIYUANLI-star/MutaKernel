import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.float16

def matmul_ref(A, B):
    return torch.matmul(A, B)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return matmul_ref(*args)


def get_inputs():
    a = torch.randn(512, 256, device="cuda", dtype=DT)
    b = torch.randn(256, 512, device="cuda", dtype=DT)
    return [a, b]

def get_init_inputs():
    return []
