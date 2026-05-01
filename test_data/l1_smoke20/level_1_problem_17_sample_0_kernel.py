import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Tile size for shared memory optimization
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                              int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B (transposed)
        if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + t * TILE_SIZE + threadIdx.y];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());
    
    // Configure kernel launch parameters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using custom CUDA kernel.
        
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_cuda.matmul_cuda(A, B)