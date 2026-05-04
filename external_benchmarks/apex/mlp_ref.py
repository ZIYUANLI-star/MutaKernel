"""PyTorch reference for Apex MLP (fused multi-layer perceptron).

Compared against: apex.mlp.MLP
CUDA source: NVIDIA/apex csrc/mlp/mlp_cuda.cu
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


B, IN_F, HID_F, OUT_F = 32, 1024, 2048, 512


def get_inputs():
    return [torch.randn(B, IN_F)]


def get_init_inputs():
    return [IN_F, HID_F, OUT_F]
