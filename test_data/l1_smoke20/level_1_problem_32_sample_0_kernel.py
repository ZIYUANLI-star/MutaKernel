import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for HardTanh
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

template<typename scalar_t>
__device__ __forceinline__ scalar_t hardtanh_activation(scalar_t val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}

template<typename scalar_t, int VEC_SIZE>
__global__ void hardtanh_kernel_vectorized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size,
    float min_val,
    float max_val) {
    
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x * VEC_SIZE + threadIdx.x * VEC_SIZE;
    
    if (idx < size) {
        // Vectorized load and store
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            int64_t pos = idx + i;
            if (pos < size) {
                scalar_t val = input[pos];
                output[pos] = hardtanh_activation(val, min_val, max_val);
            }
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Use 256 threads per block for better occupancy
    const int block_size = 256;
    const int vec_size = 4;  // Process 4 elements per thread
    
    // Calculate grid size
    int64_t grid_size = (size + block_size * vec_size - 1) / (block_size * vec_size);
    // Use std::min to avoid ambiguous overload
    grid_size = grid_size < 65535LL ? grid_size : 65535LL;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "hardtanh_cuda", [&] {
        hardtanh_kernel_vectorized<scalar_t, vec_size><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size,
            min_val,
            max_val
        );
    });
    
    return output;
}
"""

hardtanh_cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);"

# Compile with optimization flags
hardtanh_module = load_inline(
    name="hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-DNDEBUG"],
    extra_ldflags=["-lcudart"],
    extra_include_paths=[],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs HardTanh activation using custom CUDA kernel.
    """
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super(ModelNew, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.hardtanh_cuda = hardtanh_module.hardtanh_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor using optimized CUDA kernel.
        """
        if x.is_cuda:
            return self.hardtanh_cuda(x, self.min_val, self.max_val)
        else:
            return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)