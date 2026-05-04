"""PyTorch reference for softmax (compared against Apex fused softmax).

Compared against: apex scaled_masked_softmax variants
CUDA source: NVIDIA/apex csrc/scaled_masked_softmax_cuda.cu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softmax(x, dim=-1)


B, H, S = 16, 12, 512


def get_inputs():
    return [torch.randn(B, H, S, S)]


def get_init_inputs():
    return []
