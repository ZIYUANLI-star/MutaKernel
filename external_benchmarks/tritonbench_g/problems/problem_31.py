import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in mul_exponent_compensator.py.
import torch
import triton
import triton.language as tl

# Triton kernel to multiply each element in the source tensor by a constant exponent compensator
@triton.jit
def mul_kernel(src, dst, BLOCK_SIZE: tl.constexpr):
    # Define a constant exponent compensator
    exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)
    # Calculate the indices for the current program ID
    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load elements from the source tensor
    x = tl.load(src + idxs)
    # Multiply each element by the exponent compensator
    y = x * exponent_compensator
    # Store the result in the destination tensor
    tl.store(dst + idxs, y)

# Function to launch the Triton kernel
def launch_mul_kernel(src, BLOCK_SIZE=1):
    # Create an empty tensor for the result
    dst = torch.empty(src.shape, dtype=torch.float32, device='cuda')
    # Launch the Triton kernel
    mul_kernel[(src.shape[0] // BLOCK_SIZE,)](src, dst, BLOCK_SIZE)
    return dst






class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src):
        return launch_mul_kernel(src)


def get_inputs():
    src = torch.tensor([8323072], dtype=torch.int32, device='cuda').view(torch.float32)
    test_cases = {}
    return [src]

def get_init_inputs():
    return []
