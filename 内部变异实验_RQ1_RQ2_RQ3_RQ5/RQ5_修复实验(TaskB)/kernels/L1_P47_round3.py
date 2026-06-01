import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized sum reduction with keepdim=True
# Using Neumaier summation (improved Kahan) with double precision for maximum numerical stability
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template<typename scalar_t>
__global__ void sum_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer_size,
    const int reduce_size,
    const int inner_size
) {
    const int outer_idx = blockIdx.x;
    const int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const scalar_t* input_ptr = input + outer_idx * reduce_size * inner_size + inner_idx;
    
    // Neumaier summation (improved Kahan) with double precision
    // More robust than Kahan when sum >> x or x >> sum
    double sum = 0.0;
    double c = 0.0;  // Compensation for lost low-order bits
    
    for (int reduce_idx = 0; reduce_idx < reduce_size; reduce_idx++) {
        double x = static_cast<double>(input_ptr[reduce_idx * inner_size]);
        double t = sum + x;
        
        // If sum is bigger, low-order digits of x are lost
        // If x is bigger, low-order digits of sum are lost
        if (fabs(sum) >= fabs(x)) {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    
    // Add compensation at the end
    sum += c;
    
    output[outer_idx * inner_size + inner_idx] = static_cast<scalar_t>(sum);
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int dim) {
    // Ensure input is contiguous for correct memory access pattern
    auto input_contiguous = input.contiguous();
    
    // Get tensor dimensions
    auto sizes = input_contiguous.sizes();
    int ndim = input_contiguous.dim();
    
    // Validate dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Invalid reduction dimension");
    }
    
    // Calculate sizes for the reduction
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t reduce_size = sizes[dim];
    
    int64_t inner_size = 1;
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
    
    auto output = torch::empty(output_sizes, input_contiguous.options());
    
    // Handle edge case of empty reduction
    if (reduce_size == 0) {
        output.fill_(0);
        return output;
    }
    
    // Handle edge case of empty tensor
    if (outer_size == 0 || inner_size == 0) {
        return output;
    }
    
    // Launch kernel configuration
    const int threads_per_block = 256;
    dim3 blocks(static_cast<unsigned int>(outer_size), 
                static_cast<unsigned int>((inner_size + threads_per_block - 1) / threads_per_block));
    
    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contiguous.scalar_type(), "sum_reduction_cuda", [&] {
        sum_reduction_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input_contiguous.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int>(outer_size),
            static_cast<int>(reduce_size),
            static_cast<int>(inner_size)
        );
    });
    
    // Ensure kernel execution completes
    cudaDeviceSynchronize();
    
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
    using custom CUDA kernel with Neumaier summation and double precision accumulation
    for maximum numerical stability.
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