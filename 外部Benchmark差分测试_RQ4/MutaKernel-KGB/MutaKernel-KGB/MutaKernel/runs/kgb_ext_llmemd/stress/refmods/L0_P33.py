import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.float16

def reduce_sum_ref(x, dim=-1):
    return x.sum(dim=dim)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return reduce_sum_ref(*args)


def get_inputs():
    x = torch.randn(1024, 2048, device="cuda", dtype=DT)
    return [x]

def get_init_inputs():
    return []
