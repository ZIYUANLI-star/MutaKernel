import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_F16):
        return torch.matmul(x, y)


def get_inputs():
    return [
            torch.randn(128, 256).cuda(),
            torch.randn(256, 128).cuda(),
    ]

def get_init_inputs():
    return []
