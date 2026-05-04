import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template<typename scalar_t>
__global__ void argmax_kernel(
    const scalar_t* input,
    int64_t* output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t stride_outer,
    const int64_t stride_dim,
    const int64_t stride_inner
) {
    const int64_t outer_idx = blockIdx.x;
    const int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) {
        return;
    }
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int64_t max_idx = 0;
    
    for (int64_t dim_idx = 0; dim_idx < dim_size; ++dim_idx) {
        const int64_t input_idx = outer_idx * stride_outer + 
                                 dim_idx * stride_dim + 
                                 inner_idx * stride_inner;
        scalar_t val = input[input_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = dim_idx;
        }
    }
    
    const int64_t output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = max_idx;
}

torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim) {
    // Get tensor dimensions
    auto sizes = input.sizes().vec();
    const int64_t ndim = input.dim();
    
    // Handle negative dimension
    if (dim < 0) {
        dim = ndim + dim;
    }
    
    // Calculate output shape
    std::vector<int64_t> output_sizes;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_sizes.push_back(sizes[i]);
        }
    }
    
    // Create output tensor
    auto output = torch::empty(
        output_sizes,
        torch::TensorOptions().dtype(torch::kLong).device(input.device())
    );
    
    // Calculate sizes for kernel
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= sizes[i];
    }
    
    int64_t dim_size = sizes[dim];
    
    // Calculate strides
    auto strides = input.strides().vec();
    int64_t stride_outer = (dim > 0) ? strides[dim - 1] : 0;
    int64_t stride_dim = strides[dim];
    int64_t stride_inner = (dim < ndim - 1) ? strides[dim + 1] : 0;
    
    // Launch kernel
    const int64_t block_size = 256;
    dim3 block(block_size);
    dim3 grid(outer_size, (inner_size + block_size - 1) / block_size);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "argmax_cuda",
        ([&] {
            argmax_kernel<scalar_t><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<int64_t>(),
                outer_size,
                dim_size,
                inner_size,
                stride_outer,
                stride_dim,
                stride_inner
            );
        })
    );
    
    return output;
}
"""

argmax_cpp_source = "torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim);"

# Compile the inline CUDA code for argmax
argmax_module = load_inline(
    name="argmax",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_cuda = argmax_module.argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        if x.is_cuda:
            return self.argmax_cuda(x, self.dim)
        else:
            # Fallback to PyTorch implementation for CPU tensors
            return torch.argmax(x, dim=self.dim)