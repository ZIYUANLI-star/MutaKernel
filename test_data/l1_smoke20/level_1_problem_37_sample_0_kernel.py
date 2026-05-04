import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Frobenius norm normalization
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void frobenius_norm_kernel(const T* input, T* output, T* norm, int total_elements) {
    __shared__ T shared_mem[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Parallel reduction for norm calculation
    T local_sum = 0.0;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total_elements; i += stride) {
        T val = input[i];
        local_sum += val * val;
    }
    
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Store block sum to global memory
    if (tid == 0) {
        norm[blockIdx.x] = shared_mem[0];
    }
    __syncthreads();
    
    // Phase 2: Wait for all blocks to complete and compute final norm
    __shared__ T final_norm;
    if (tid == 0) {
        T total_sum = 0.0;
        for (int i = 0; i < gridDim.x; i++) {
            total_sum += norm[i];
        }
        final_norm = sqrt(total_sum + 1e-8);  // Add epsilon for numerical stability
    }
    __syncthreads();
    
    T inv_norm = 1.0 / final_norm;
    
    // Phase 3: Normalize elements
    for (int i = idx; i < total_elements; i += stride) {
        output[i] = input[i] * inv_norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor input) {
    auto total_elements = input.numel();
    auto output = torch::empty_like(input);
    
    // Determine optimal block and grid sizes
    const int block_size = 1024;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    num_blocks = min(num_blocks, 1024);  // Limit number of blocks
    
    // Allocate temporary memory for block sums
    auto block_sums = torch::zeros({num_blocks}, torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "frobenius_norm_kernel", ([&] {
        frobenius_norm_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_sums.data_ptr<scalar_t>(),
            total_elements
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

frobenius_norm_cpp_source = """
torch::Tensor frobenius_norm_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the optimized Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies fused Frobenius norm normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Use custom CUDA kernel for GPU tensors, fallback to PyTorch for CPU
        if x.is_cuda:
            return self.frobenius_norm.frobenius_norm_cuda(x)
        else:
            norm = torch.norm(x, p='fro')
            return x / norm