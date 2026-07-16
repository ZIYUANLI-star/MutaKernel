import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.float16

def fused_mlp_ref(x, w_gate, w_up, w_down, activation="silu"):
    gate = x @ w_gate.T
    up = x @ w_up.T
    if activation == "silu":
        gate = F.silu(gate)
    elif activation == "gelu":
        gate = F.gelu(gate)
    return (gate * up) @ w_down.T


class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        return fused_mlp_ref(*args)


def get_inputs():
    x = torch.randn(128, 256, device="cuda", dtype=DT)
    w_gate = torch.randn(512, 256, device="cuda", dtype=DT)
    w_up = torch.randn(512, 256, device="cuda", dtype=DT)
    w_down = torch.randn(256, 512, device="cuda", dtype=DT)
    return [x, w_gate, w_up, w_down]

def get_init_inputs():
    return []
