import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.float16

def cross_entropy_ref(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return cross_entropy_ref(*args)


def get_inputs():
    logits = torch.randn(256, 512, device="cuda", dtype=DT)
    targets = torch.randint(0, 512, (256,), device="cuda", dtype=torch.long)
    return [logits, targets]

def get_init_inputs():
    return []
