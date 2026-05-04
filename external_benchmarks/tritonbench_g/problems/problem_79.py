import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in quant_transpose_kernel.py.

import torch
import triton
import triton.language as tl

# global quantize and transpose
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        # ...
    ],
    key=["M", "N"],
)
@triton.jit
def _quantize_global_transpose(
    A,
    absmax_inv_ptr,
    B,
    stride_am,
    stride_an,
    stride_bn,
    stride_bm,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)

    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))

    tl.store(B, output, mask=mask)

def quantize_global_transpose(input):
    absmax = input.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    M, N = input.shape
    out = torch.empty(N, M, device="cuda", dtype=torch.int8)

    assert out.size(0) == N and out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert out.stride(0) == 1 or out.stride(1) == 1

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _quantize_global_transpose[grid](
        input,
        absmax_inv,
        out,
        input.stride(0),
        input.stride(1),
        out.stride(0),
        out.stride(1),
        M,
        N,
    )
    return out, absmax






class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_1):
        return quantize_global_transpose(input_tensor_1)


def get_inputs():
    results = {}
    input_tensor_1 = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    return [input_tensor_1]

def get_init_inputs():
    return []
