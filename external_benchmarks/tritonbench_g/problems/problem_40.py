import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in masked_add_cuda.py.
import torch
import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(grad_ptr,
                      p_ptr,
                      p_mask_ptr,
                      n_elements,
                      alpha,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad += p * alpha
    tl.store(grad_ptr + offsets, grad, mask=mask)

def masked_add(grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float = 0):
    '''
    equivalent to
    grad.add_(p.data * (1 - p.mask), alpha=decay)
    '''
    assert grad.is_cuda and p_data.is_cuda and p_mask.is_cuda
    assert (grad.layout, p_data.layout, p_mask.layout) == (torch.strided, torch.strided, torch.strided)
    assert grad.stride() == p_data.stride() == p_mask.stride()
    n_elements = grad.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    masked_add_kernel[grid](grad, p_data, p_mask, n_elements, alpha, BLOCK_SIZE=1024)





class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grad_triton, p_data, p_mask):
        return masked_add(grad_triton, p_data, p_mask)


def get_inputs():
    n = 10000
    grad = torch.randn(n, device='cuda')
    p_data = torch.randn(n, device='cuda')
    p_mask = torch.randint(0, 2, (n,), device='cuda')
    results = {}
    grad_triton = grad.clone()
    return [grad_triton, p_data, p_mask]

def get_init_inputs():
    return []
