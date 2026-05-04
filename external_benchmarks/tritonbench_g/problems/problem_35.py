import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in var_len_copy.py.

import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    a_id = tl.program_id(0)
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + old_start + i + old_offset, mask=old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=new_offset < length)

def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)

    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_location, new_a_start, new_a_location, BLOCK_SIZE)






class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, old_a_start, old_a_len, old_a_location, new_a_start, new_a_location):
        return launch_var_len_copy_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location)


def get_inputs():
    num_arrays = 3
    BLOCK_SIZE = 256
    old_a_start = torch.tensor([0, 100, 300], dtype=torch.int32, device='cuda')
    old_a_len = torch.tensor([50, 150, 200], dtype=torch.int32, device='cuda')
    old_a_location = torch.arange(500, dtype=torch.float32, device='cuda')
    new_a_start = torch.tensor([0, 60, 260], dtype=torch.int32, device='cuda')
    new_a_location = torch.zeros(500, dtype=torch.float32, device='cuda')
    return [old_a_start, old_a_len, old_a_location, new_a_start, new_a_location]

def get_init_inputs():
    return []
