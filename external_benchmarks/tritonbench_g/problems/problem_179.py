import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in quantize_global.py.

import torch
import triton
import triton.language as tl

# global quantize
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_global(
    x_ptr,
    absmax_inv_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)
    output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
    tl.store(output_ptr + offsets, output, mask=mask)

def quantize_global(x: torch.Tensor):
    absmax = x.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    output = torch.empty(*x.shape, device="cuda", dtype=torch.int8)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _quantize_global[grid](x, absmax_inv, output, n_elements)
    return output, absmax






class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        return quantize_global(x1)


def get_inputs():
    results = {}
    x1 = torch.randn(2048, device='cuda', dtype=torch.float32)
    return [x1]

def get_init_inputs():
    return []
