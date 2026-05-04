import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)


def get_inputs():
    return [
        torch.randn(4, 32, 64).cuda(),
        torch.randn(4, 64, 32).cuda(),
    ]

def get_init_inputs():
    return []
