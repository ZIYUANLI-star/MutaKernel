import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in adam_update_triton.py.
import torch
import triton
import triton.language as tl

@triton.autotune(configs = [
    triton.Config({'BLOCK_SIZE': 128}, num_warps = 4),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps = 8),
], key = ['n_elements'], restore_value=['p_ptr', 'exp_avg_ptr'])

# Triton CUDA kernel

@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Offsetted pointers
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    # Load
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    # Stepweight decay
    p = p * (1 - lr * wd)

    # Diff between momentum running average and grad
    diff = exp_avg - grad

    # Weight update
    update = diff * beta1 + grad

    # torch.sign
    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    p = p + update_sign * can_update

    # Decay the momentum running average coefficient
    exp_avg = diff * beta2 + grad

    # Store new params and momentum running average coefficient
    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)

def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float
):
    assert all([t.is_cuda for t in (p, grad, exp_avg)])
    n_elements = p.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    update_fn_kernel[grid](
        p,
        grad,
        exp_avg,
        lr,
        wd,
        beta1,
        beta2,
        n_elements
    )






class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p1, grad1, exp_avg1, lr, wd, beta1, beta2):
        return update_fn(p1, grad1, exp_avg1, lr, wd, beta1, beta2)


def get_inputs():
    n_elements = 128
    p1 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    grad1 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    exp_avg1 = torch.zeros(n_elements, device='cuda', dtype=torch.float32)
    n_elements = 1024
    p2 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    grad2 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    exp_avg2 = torch.zeros(n_elements, device='cuda', dtype=torch.float32)
    lr = 0.01
    wd = 0.01
    beta1 = 0.9
    beta2 = 0.999
    return [p1, grad1, exp_avg1, lr, wd, beta1, beta2]

def get_init_inputs():
    return []
