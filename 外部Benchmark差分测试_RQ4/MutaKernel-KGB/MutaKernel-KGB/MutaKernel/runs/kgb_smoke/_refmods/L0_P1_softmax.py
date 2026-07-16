import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.bfloat16

def softmax_ref(x, dim=-1):
    return F.softmax(x, dim=dim)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return softmax_ref(*args)


def get_inputs():
    x = torch.randn(128, 512, device="cuda", dtype=DT)
    return [x]

def get_init_inputs():
    return []
