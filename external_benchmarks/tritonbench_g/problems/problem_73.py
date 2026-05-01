import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dy, x, weight, bias, eps, mean, rstd, dresidual, has_residual, is_rms_norm, x_dtype, recompute_output):
        return torch.nn.functional.layer_norm(x, x.shape[-1:])


def get_inputs():
    return [
            torch.randn(4, 256, 512).cuda(),
    ]

def get_init_inputs():
    return []
