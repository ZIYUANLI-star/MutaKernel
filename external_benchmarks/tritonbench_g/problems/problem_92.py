import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in add_value.py.
import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def puzzle1_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, value):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + value
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function to call the kernel
def puzzle1(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    N = output.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=10)
    return output






class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a1):
        return puzzle1(a1)


def get_inputs():
    results = {}
    a1 = torch.Tensor([4, 5, 3, 2]).to(device=torch.device('cuda'))
    return [a1]

def get_init_inputs():
    return []
