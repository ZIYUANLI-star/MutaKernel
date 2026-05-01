import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, normalized_shape, weight, eps):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)


def get_inputs():
    return [
            torch.randn(4, 256, 512).cuda(),
    ]

def get_init_inputs():
    return []
