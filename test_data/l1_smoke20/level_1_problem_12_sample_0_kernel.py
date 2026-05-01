import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for diagonal matrix multiplication with improved memory access
diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(
    const float* diag, 
    const float* matrix, 
    float* output, 
    int N, 
    int M
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N && col < M) {
        int matrix_idx = row * M + col;
        output[matrix_idx] = diag[row] * matrix[matrix_idx];
    }
}

__global__ void diag_matmul_kernel_optimized(
    const float* diag, 
    const float* matrix, 
    float* output, 
    int N, 
    int M
) {
    // Improved memory access pattern: coalesced reads/writes
    int tid = threadIdx.x;
    int row = blockIdx.y;
    int col_start = blockIdx.x * blockDim.x * 4;
    
    // Process 4 elements per thread for better memory throughput
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int col = col_start + tid + i * blockDim.x;
        if (row < N && col < M) {
            int matrix_idx = row * M + col;
            output[matrix_idx] = diag[row] * matrix[matrix_idx];
        }
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor matrix) {
    int N = diag.size(0);
    int M = matrix.size(1);
    
    auto output = torch::empty({N, M}, diag.options());
    
    // Choose kernel based on matrix dimensions for optimal performance
    if (M >= 1024) {
        // Use optimized kernel for large matrices
        dim3 block_size(256, 1);
        dim3 grid_size(
            (M + (block_size.x * 4) - 1) / (block_size.x * 4),
            N
        );
        
        diag_matmul_kernel_optimized<<<grid_size, block_size>>>(
            diag.data_ptr<float>(),
            matrix.data_ptr<float>(),
            output.data_ptr<float>(),
            N,
            M
        );
    } else {
        // Use simple kernel for smaller matrices
        dim3 block_size(16, 16);
        dim3 grid_size(
            (M + block_size.x - 1) / block_size.x,
            (N + block_size.y - 1) / block_size.y
        );
        
        diag_matmul_kernel<<<grid_size, block_size>>>(
            diag.data_ptr<float>(),
            matrix.data_ptr<float>(),
            output.data_ptr<float>(),
            N,
            M
        );
    }
    
    return output;
}
"""

diag_matmul_cpp_source = "torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor matrix);"

# Compile the inline CUDA code
diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    Uses custom CUDA kernel with improved memory access patterns.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using custom CUDA kernel.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return self.diag_matmul.diag_matmul_cuda(A, B)