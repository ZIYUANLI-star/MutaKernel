import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xy, out):
        return x * torch.sigmoid(x) * y


def get_inputs():
    return [
            torch.randn(4, 512).cuda(),
            torch.randn(4, 512).cuda(),
    ]

def get_init_inputs():
    return []
