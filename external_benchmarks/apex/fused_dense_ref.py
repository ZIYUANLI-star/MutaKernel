"""PyTorch reference for Apex FusedDense (fused GEMM+bias).

Compared against: apex.fused_dense.FusedDense
CUDA source: NVIDIA/apex csrc/mlp_cuda.cu
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


B, IN_F, OUT_F = 32, 1024, 2048


def get_inputs():
    # FusedDense requires 2D input (batch, features)
    return [torch.randn(B, IN_F)]


def get_init_inputs():
    return [IN_F, OUT_F]
