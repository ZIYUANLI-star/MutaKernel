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

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A
        if (row < N && (t * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        if ((t * TILE_SIZE + ty) < N && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor optimized_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_tiled_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
"""

matmul_cpp_source = "torch::Tensor optimized_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
optimized_matmul = load_inline(
    name="optimized_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["optimized_matmul_cuda"],
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
        self.optimized_matmul = optimized_matmul
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using custom CUDA kernel.
        
        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).
            
        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.optimized_matmul.optimized_matmul_cuda(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N).cuda()
    B = torch.randn(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []