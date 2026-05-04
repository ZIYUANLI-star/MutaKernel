import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Optimized ELU kernel with minimal branching and vectorization
elu_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__device__ __forceinline__ scalar_t elu_activation(scalar_t x, scalar_t alpha) {
    return x >= scalar_t(0) ? x : alpha * (expf(x) - scalar_t(1));
}

template<typename scalar_t>
__global__ void elu_kernel_fast(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better occupancy
    for (int64_t i = tid; i < size; i += stride) {
        scalar_t val = input[i];
        output[i] = elu_activation<scalar_t>(val, alpha);
    }
}

template<typename scalar_t>
__global__ void elu_kernel_vectorized4(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t alpha,
    int64_t size) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t idx = tid * 4;
    
    if (idx + 3 < size) {
        // Load 4 elements
        scalar_t vals[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            vals[j] = input[idx + j];
        }
        
        // Compute ELU for all 4 elements
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            output[idx + j] = elu_activation<scalar_t>(vals[j], alpha);
        }
    } else {
        // Handle remaining elements
        for (int j = 0; j < 4; ++j) {
            int64_t elem_idx = idx + j;
            if (elem_idx < size) {
                scalar_t val = input[elem_idx];
                output[elem_idx] = elu_activation<scalar_t>(val, alpha);
            }
        }
    }
}

torch::Tensor elu_cuda(torch::Tensor input, float alpha) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    
    int64_t size = input_contig.numel();
    
    if (size == 0) {
        return output;
    }
    
    // Optimal block size for modern GPUs
    const int block_size = 256;
    const int max_blocks = 256;
    int grid_size = (size + block_size - 1) / block_size;
    grid_size = grid_size > max_blocks ? max_blocks : grid_size;
    
    // Use vectorized kernel for large tensors
    if (size >= 4096 && (size % 4) == 0) {
        int vectorized_grid = (size / 4 + block_size - 1) / block_size;
        vectorized_grid = vectorized_grid > max_blocks ? max_blocks : vectorized_grid;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_vectorized4<scalar_t><<<vectorized_grid, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    } else {
        // Use simple grid-stride kernel for other cases
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "elu_cuda", [&] {
            elu_kernel_fast<scalar_t><<<grid_size, block_size>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                size
            );
        });
    }
    
    return output;
}
"""

elu_cpp_source = """
torch::Tensor elu_cuda(torch::Tensor input, float alpha);
"""

# Compile with optimizations
elu_module = load_inline(
    name="elu_module",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=["elu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs ELU activation using custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return elu_module.elu_cuda(x, self.alpha)
        else:
            return F.elu(x, alpha=self.alpha)