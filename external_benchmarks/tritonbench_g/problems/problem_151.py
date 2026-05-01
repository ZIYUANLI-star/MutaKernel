import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias, eps, residual, x1, weight1, bias1, dropout_p, rowscale, out_dtype, residual_dtype, is_rms_norm, return_dropout_mask):
        return torch.nn.functional.layer_norm(x, x.shape[-1:])


def get_inputs():
    return [
            torch.randn(4, 256, 512).cuda(),
    ]

def get_init_inputs():
    return []
