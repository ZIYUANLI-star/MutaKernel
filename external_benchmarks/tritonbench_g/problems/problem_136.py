import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, label_smoothing, logit_scale, lse_square_scale, ignored_index, inplace_backward, process_group):
        return torch.nn.functional.cross_entropy(x, y)


def get_inputs():
    return [
            torch.randn(32, 100).cuda(),
            torch.randint(0, 100, (32,)).cuda(),
    ]

def get_init_inputs():
    return []
