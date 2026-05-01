import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, p, seed):
        return torch.nn.functional.dropout(x, p=0.5, training=False)


def get_inputs():
    return [
            torch.randn(4, 1024).cuda(),
    ]

def get_init_inputs():
    return []
