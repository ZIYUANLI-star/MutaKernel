import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized sum reduction with keepdim=True
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename scalar_t>
__global__ void sum_reduction_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int outer_size,
    const int reduce_size,
    const int inner_size
) {
    // Each thread handles one element in the output
    const int outer_idx = blockIdx.x;
    const int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const scalar_t* input_ptr = input + outer_idx * reduce_size * inner_size + inner_idx;
    scalar_t sum = 0;
    
    // Sequential reduction across reduce dimension
    for (int reduce_idx = 0; reduce_idx < reduce_size; reduce_idx++) {
        sum += input_ptr[reduce_idx * inner_size];
    }
    
    output[outer_idx * inner_size + inner_idx] = sum;
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    // Get tensor dimensions
    auto sizes = input.sizes();
    int ndim = input.dim();
    
    // Validate dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Invalid reduction dimension");
    }
    
    // Calculate sizes for the reduction
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int reduce_size = sizes[dim];
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }
    
    // Prepare output tensor with keepdim=True
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i == dim) {
            output_sizes.push_back(1);
        } else {
            output_sizes.push_back(sizes[i]);
        }
    }
    
    auto output = torch::empty(output_sizes, input.options());
    
    // Launch kernel configuration
    const int threads_per_block = 256;
    dim3 blocks(outer_size, (inner_size + threads_per_block - 1) / threads_per_block);
    
    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sum_reduction_cuda", [&] {
        sum_reduction_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            reduce_size,
            inner_size
        );
    });
    
    return output;
}
"""

sum_reduction_cpp_source = "torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension
    using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        if x.is_cuda:
            return self.sum_reduction.sum_reduction_cuda(x, self.dim)
        else:
            # Fallback to PyTorch for CPU tensors
            return torch.sum(x, dim=self.dim, keepdim=True)