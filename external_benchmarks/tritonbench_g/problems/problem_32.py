import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in dropout_triton.py.

import torch
import triton
import triton.language as tl

@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output





class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return dropout(x)


def get_inputs():
    results = {}
    x = torch.randn(size=(10,)).cuda()
    p = 0.5
    x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()
    return [x]

def get_init_inputs():
    return []
