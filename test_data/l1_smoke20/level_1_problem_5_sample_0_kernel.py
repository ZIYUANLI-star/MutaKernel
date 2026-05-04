import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-scalar multiplication
matrix_scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_scalar_mul_kernel(const float* A, float scalar, float* C, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        C[idx] = A[idx] * scalar;
    }
}

torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float scalar) {
    auto total_elements = A.numel();
    auto C = torch::empty_like(A);
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    matrix_scalar_mul_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(), 
        scalar, 
        C.data_ptr<float>(), 
        total_elements
    );
    
    return C;
}
"""

matrix_scalar_mul_cpp_source = (
    "torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float scalar);"
)

# Compile the inline CUDA code for matrix-scalar multiplication
matrix_scalar_mul = load_inline(
    name="matrix_scalar_mul",
    cpp_sources=matrix_scalar_mul_cpp_source,
    cuda_sources=matrix_scalar_mul_source,
    functions=["matrix_scalar_mul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-scalar multiplication using custom CUDA kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_scalar_mul = matrix_scalar_mul
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using custom CUDA kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return self.matrix_scalar_mul.matrix_scalar_mul_cuda(A, s)