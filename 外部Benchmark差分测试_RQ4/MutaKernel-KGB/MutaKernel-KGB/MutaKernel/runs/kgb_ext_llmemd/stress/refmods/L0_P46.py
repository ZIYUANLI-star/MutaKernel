import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.bfloat16

def cross_entropy_ref(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return cross_entropy_ref(*args)


def get_inputs():
    logits = torch.randn(1024, 1024, device="cuda", dtype=DT)
    targets = torch.randint(0, 1024, (1024,), device="cuda", dtype=torch.long)
    return [logits, targets]

def get_init_inputs():
    return []
