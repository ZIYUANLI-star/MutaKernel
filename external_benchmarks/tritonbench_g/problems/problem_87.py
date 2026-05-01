import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, grad_output, log_target):
        return torch.nn.functional.kl_div(x.log(), y, reduction='none')


def get_inputs():
    return [
            torch.rand(4, 1024).clamp(min=1e-6).cuda(),
            torch.rand(4, 1024).clamp(min=1e-6).cuda(),
    ]

def get_init_inputs():
    return []
