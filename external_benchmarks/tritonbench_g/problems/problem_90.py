import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
        return torch.nn.functional.softmax(x, dim=-1)


def get_inputs():
    return [
            torch.randn(4, 1024).cuda(),
    ]

def get_init_inputs():
    return []
