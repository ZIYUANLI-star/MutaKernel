"""PyTorch reference for Apex FusedLayerNorm.

Compared against: apex.normalization.FusedLayerNorm
CUDA source: NVIDIA/apex csrc/layer_norm_cuda_kernel.cu
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        return self.ln(x)


B, S, H = 32, 128, 1024


def get_inputs():
    return [torch.randn(B, S, H)]


def get_init_inputs():
    return [H]
