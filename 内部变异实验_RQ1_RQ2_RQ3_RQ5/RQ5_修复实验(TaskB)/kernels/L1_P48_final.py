import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized mean reduction with improved numerical stability
mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void mean_reduction_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int reduction_dim_size,
    const int outer_size,
    const int inner_size
) {
    // Each thread block handles one output position
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Calculate which output position this block handles
    const int outer_idx = bid / inner_size;
    const int inner_idx = bid % inner_size;
    
    // Use double precision for accumulation for better numerical stability
    // This is critical for near-overflow values
    __shared__ double shared_sums[256];
    
    // Each thread accumulates its portion using double precision
    double sum = 0.0;
    for (int i = tid; i < reduction_dim_size; i += blockDim.x) {
        int idx = (outer_idx * reduction_dim_size + i) * inner_size + inner_idx;
        sum += static_cast<double>(input[idx]);
    }
    
    shared_sums[tid] = sum;
    __syncthreads();
    
    // Parallel reduction within block using double precision
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Write final result, converting back to original type
    if (tid == 0) {
        double mean_val = shared_sums[0] / static_cast<double>(reduction_dim_size);
        output[bid] = static_cast<T>(mean_val);
    }
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int64_t dim) {
    // Ensure contiguous tensor for correct memory access pattern
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    auto input_sizes = input.sizes().vec();
    
    // Handle negative dimension
    if (dim < 0) {
        dim = input.dim() + dim;
    }
    
    // Calculate output shape
    std::vector<int64_t> output_sizes;
    for (int64_t i = 0; i < input.dim(); i++) {
        if (i != dim) {
            output_sizes.push_back(input_sizes[i]);
        }
    }
    
    auto output = torch::empty(output_sizes, input.options());
    
    // Calculate dimensions for kernel
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= input_sizes[i];
    }
    
    int64_t reduction_dim_size = input_sizes[dim];
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); i++) {
        inner_size *= input_sizes[i];
    }
    
    int64_t total_positions = outer_size * inner_size;
    
    // Handle empty tensor case
    if (total_positions == 0 || reduction_dim_size == 0) {
        return output;
    }
    
    // Configure kernel launch
    const int block_size = 256;
    int grid_size = total_positions;
    
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "mean_reduction",
        [&] {
            mean_reduction_kernel<scalar_t><<<grid_size, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                reduction_dim_size,
                outer_size,
                inner_size
            );
        }
    );
    
    return output;
}
"""

mean_reduction_cpp_source = """
torch::Tensor mean_reduction_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
mean_reduction_module = load_inline(
    name="mean_reduction_v2",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension
    using custom CUDA kernels with improved numerical stability.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean
        using optimized CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        # Use CUDA kernel if tensor is on GPU
        if x.is_cuda:
            return self.mean_reduction.mean_reduction_cuda(x, self.dim)
        else:
            # Fallback to PyTorch for CPU tensors
            return torch.mean(x, dim=self.dim)