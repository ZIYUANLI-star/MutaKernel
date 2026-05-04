"""PyTorch reference for Apex FusedDenseGeluDense (Linear+GELU+Linear).

Compared against: apex.fused_dense.FusedDenseGeluDense
CUDA source: NVIDIA/apex csrc/mlp_cuda.cu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, intermediate, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, intermediate)
        self.fc2 = nn.Linear(intermediate, out_features)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


B, IN_F, INTER, OUT_F = 16, 1024, 4096, 1024


def get_inputs():
    # FusedDenseGeluDense requires 2D input (batch, features)
    return [torch.randn(B, IN_F)]


def get_init_inputs():
    return [IN_F, INTER, OUT_F]
