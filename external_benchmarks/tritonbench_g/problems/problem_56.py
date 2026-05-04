import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in vector_addition_custom.py.

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    """add kernel."""
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs, mask=offs < size)
    b = tl.load(B + offs, mask=offs < size)
    tl.store(C + offs, a + b, mask=offs < size)

def custom_add(a, b):
    """custom add one."""
    c = torch.empty_like(a)
    size = c.size(0)
    BLOCK = 16

    grid = (triton.cdiv(size, BLOCK), )
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    return c






class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return custom_add(a, b)


def get_inputs():
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.float32, device='cuda')
    b = torch.tensor([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device='cuda')
    return [a, b]

def get_init_inputs():
    return []
