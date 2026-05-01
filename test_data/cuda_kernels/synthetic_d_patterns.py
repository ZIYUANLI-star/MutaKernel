"""Synthetic file with D-category patterns (Python-level tensor operations)."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void simple_kernel(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * 2.0f;
}

torch::Tensor simple_op(torch::Tensor input) {
    auto out = torch::empty_like(input);
    int N = input.numel();
    simple_kernel<<<(N + 255) / 256, 256>>>(
        input.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}
"""

cpp_source = "torch::Tensor simple_op(torch::Tensor input);"

module = load_inline(
    name="test_d_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["simple_op"],
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized model using custom CUDA kernel."""
    def __init__(self, num_heads=8, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.op = module

    def forward(self, x, weight, bias):
        # D2 patterns: .contiguous() calls
        x = x.contiguous()
        weight = weight.contiguous()

        # D1 patterns: shape manipulation
        batch = x.size(0)
        seq_len = x.size(1)

        # .expand() — common in attention masks
        mask = torch.ones(seq_len, seq_len, device=x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch, self.num_heads, seq_len, seq_len)

        # .expand_as() variant
        scale = torch.tensor(1.0 / (self.head_dim ** 0.5), device=x.device)
        scale = scale.expand_as(x)

        # .broadcast_to()
        bias_expanded = bias.broadcast_to(x.shape)

        # Apply kernel
        out = self.op.simple_op(x)

        # More .contiguous() after transpose
        out = out.view(batch, seq_len, self.num_heads, self.head_dim)
        out = out.transpose(1, 2).contiguous()

        # .reshape() and .view() (currently NOT detected by D1)
        out = out.reshape(batch, self.num_heads * self.head_dim, seq_len)
        out = out.view(batch, -1)

        return out
