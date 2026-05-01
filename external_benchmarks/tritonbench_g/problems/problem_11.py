import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, rms_w, use_rbe, start_pos, n_heads, head_dim):
        return torch.matmul(x, y)


def get_inputs():
    return [
            torch.randn(128, 256).cuda(),
            torch.randn(256, 128).cuda(),
    ]

def get_init_inputs():
    return []
