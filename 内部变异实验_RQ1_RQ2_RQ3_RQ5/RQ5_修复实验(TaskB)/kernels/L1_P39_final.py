import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__global__ void l2_norm_kernel(const T* __restrict__ input, T* __restrict__ output, 
                               int batch_size, int dim) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Use 256 threads per block (power of 2 for reduction)
    const int threads = 256;
    
    // Shared memory for reduction
    __shared__ float sdata[256];
    
    // Each thread processes multiple elements
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        thread_sum += val * val;
    }
    
    // Parallel reduction
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Tree reduction
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute reciprocal of L2 norm (1 / sqrt(sum))
    // No epsilon added to match PyTorch's torch.norm behavior exactly
    float scale = 0.0f;
    if (tid == 0) {
        float sum_sq = sdata[0];
        float norm = sqrtf(sum_sq);
        // Match PyTorch behavior: compute 1/norm directly
        // This matches x / torch.norm(x, p=2, dim=1, keepdim=True)
        scale = 1.0f / norm;
        sdata[0] = scale;
    }
    __syncthreads();
    
    scale = sdata[0];
    
    // Normalize elements
    for (int i = tid; i < dim; i += threads) {
        int idx = batch_idx * dim + i;
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(val * scale);
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_cuda", ([&] {
        l2_norm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor input);"

# Compile the inline CUDA code for L2 normalization
l2_norm_module = load_inline(
    name="l2_norm_v2",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized L2 normalization model with custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            
        Returns:
            torch.Tensor: L2 normalized tensor, same shape as input.
        """
        # Ensure tensor is on CUDA device
        if not x.is_cuda:
            x = x.cuda()
            
        return l2_norm_module.l2_norm_cuda(x)