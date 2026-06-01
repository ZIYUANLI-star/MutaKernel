import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Tanh activation using built-in math functions
tanh_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <math.h>

// Use CUDA's built-in tanh for numerical stability
template<typename T>
__device__ __forceinline__ T device_tanh(T x);

template<>
__device__ __forceinline__ float device_tanh<float>(float x) {
    return tanhf(x);
}

template<>
__device__ __forceinline__ double device_tanh<double>(double x) {
    return tanh(x);
}

// Vectorized kernel for float/double (4 elements per thread)
template<typename T>
__global__ void tanh_activation_kernel_vectorized(const T* __restrict__ input, 
                                                   T* __restrict__ output, 
                                                   int64_t size) {
    constexpr int VEC_SIZE = 4;
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    
    if (idx + VEC_SIZE - 1 < size) {
        // Load 4 elements at once
        T vals[VEC_SIZE];
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            vals[i] = input[idx + i];
        }
        
        // Process 4 elements using built-in tanh (numerically stable)
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            vals[i] = device_tanh(vals[i]);
        }
        
        // Store 4 elements at once
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            output[idx + i] = vals[i];
        }
    } else {
        // Handle remainder elements
        for (int i = 0; i < VEC_SIZE; i++) {
            int64_t elem_idx = idx + i;
            if (elem_idx < size) {
                output[elem_idx] = device_tanh(input[elem_idx]);
            }
        }
    }
}

// Specialized kernel for half precision (fp16) with vectorization
__global__ void tanh_activation_kernel_half(const __half* __restrict__ input,
                                            __half* __restrict__ output,
                                            int64_t size) {
    constexpr int VEC_SIZE = 4;
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    
    if (idx + VEC_SIZE - 1 < size) {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            float val = __half2float(input[idx + i]);
            float result = tanhf(val);
            output[idx + i] = __float2half(result);
        }
    } else {
        for (int i = 0; i < VEC_SIZE; i++) {
            int64_t elem_idx = idx + i;
            if (elem_idx < size) {
                float val = __half2float(input[elem_idx]);
                float result = tanhf(val);
                output[elem_idx] = __float2half(result);
            }
        }
    }
}

torch::Tensor tanh_activation_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 512;
    const int vec_size = 4;
    const int num_blocks = (size + block_size * vec_size - 1) / (block_size * vec_size);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "tanh_activation", [&] {
        if constexpr (std::is_same<scalar_t, at::Half>::value) {
            tanh_activation_kernel_half<<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr<scalar_t>()),
                reinterpret_cast<__half*>(output.data_ptr<scalar_t>()),
                size
            );
        } else {
            tanh_activation_kernel_vectorized<scalar_t><<<num_blocks, block_size, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size
            );
        }
    });
    
    return output;
}
"""

tanh_cpp_source = "torch::Tensor tanh_activation_cuda(torch::Tensor input);"

# Compile the inline CUDA code with optimization flags
tanh_activation = load_inline(
    name="tanh_activation",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_kernel_source,
    functions=["tanh_activation_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-march=native"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Tanh activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_op = tanh_activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Tanh activation to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        # Only use custom kernel if tensor is on CUDA
        if x.is_cuda:
            return self.tanh_op.tanh_activation_cuda(x)
        else:
            # Fall back to PyTorch implementation for CPU tensors
            return torch.tanh(x)