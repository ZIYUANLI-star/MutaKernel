import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized reverse cumulative sum with improved precision
reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void reverse_cumsum_kernel_precise(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int64_t outer_idx = blockIdx.y;
    const int64_t inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Use double precision accumulation for better numerical stability
    // This matches PyTorch's internal behavior more closely for cumulative operations
    double sum = 0.0;
    
    // Process in reverse order for cumulative sum
    for (int64_t dim_idx = dim_size - 1; dim_idx >= 0; --dim_idx) {
        const int64_t input_offset = base_offset + dim_idx * inner_size;
        sum += static_cast<double>(input[input_offset]);
        output[input_offset] = static_cast<scalar_t>(sum);
    }
}

// Specialized kernel for double precision input (no conversion needed)
__global__ void reverse_cumsum_kernel_double(
    const double* __restrict__ input,
    double* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    const int64_t outer_idx = blockIdx.y;
    const int64_t inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    double sum = 0.0;
    
    for (int64_t dim_idx = dim_size - 1; dim_idx >= 0; --dim_idx) {
        const int64_t input_offset = base_offset + dim_idx * inner_size;
        sum += input[input_offset];
        output[input_offset] = sum;
    }
}

torch::Tensor reverse_cumsum_cuda_optimized(torch::Tensor input, int64_t dim) {
    auto input_contiguous = input.contiguous();
    auto output = torch::empty_like(input_contiguous);
    
    const int64_t ndim = input_contiguous.dim();
    
    // Handle empty tensor
    if (input_contiguous.numel() == 0) {
        return output;
    }
    
    // Handle scalar tensor
    if (ndim == 0) {
        output.copy_(input_contiguous);
        return output;
    }
    
    dim = dim < 0 ? ndim + dim : dim;
    
    // Validate dim
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Invalid dimension for reverse cumsum");
    }
    
    // Calculate sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_contiguous.size(i);
    }
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input_contiguous.size(i);
    }
    
    const int64_t dim_size = input_contiguous.size(dim);
    
    // Handle dim_size of 1 (cumsum is identity)
    if (dim_size == 1) {
        output.copy_(input_contiguous);
        return output;
    }
    
    // Configure kernel launch
    const int block_size = 256;
    const int grid_x = (inner_size + block_size - 1) / block_size;
    const dim3 grid(grid_x, outer_size);
    
    if (input_contiguous.scalar_type() == torch::kDouble) {
        reverse_cumsum_kernel_double<<<grid, block_size>>>(
            input_contiguous.data_ptr<double>(),
            output.data_ptr<double>(),
            outer_size,
            dim_size,
            inner_size
        );
    } else if (input_contiguous.scalar_type() == torch::kFloat) {
        reverse_cumsum_kernel_precise<float><<<grid, block_size>>>(
            input_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            outer_size,
            dim_size,
            inner_size
        );
    } else {
        // Fallback to PyTorch implementation for other types
        output = torch::cumsum(input_contiguous.flip(dim), dim).flip(dim);
    }
    
    return output;
}
"""

reverse_cumsum_cpp_source = """
torch::Tensor reverse_cumsum_cuda_optimized(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
reverse_cumsum_module = load_inline(
    name="reverse_cumsum_optimized_v2",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda_optimized"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation 
    using a custom CUDA kernel with improved numerical precision.
    
    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum_module

    def forward(self, x):
        # Use optimized CUDA kernel with double precision accumulation
        return self.reverse_cumsum.reverse_cumsum_cuda_optimized(x, self.dim)