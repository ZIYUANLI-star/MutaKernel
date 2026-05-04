"""PyTorch reference for Apex FusedRMSNorm.

Compared against: apex.normalization.FusedRMSNorm
CUDA source: NVIDIA/apex csrc/layer_norm_cuda_kernel.cu (cuda_rms_norm)
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


B, S, H = 32, 128, 1024


def get_inputs():
    return [torch.randn(B, S, H)]


def get_init_inputs():
    return [H]
