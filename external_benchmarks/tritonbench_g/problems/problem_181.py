import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, start_pos, q_weight, k_weight, v_weight, rms_w, n_heads, head_dim, k, v, eps, theta):
        return torch.matmul(x, y)


def get_inputs():
    return [
            torch.randn(128, 256).cuda(),
            torch.randn(256, 128).cuda(),
    ]

def get_init_inputs():
    return []
