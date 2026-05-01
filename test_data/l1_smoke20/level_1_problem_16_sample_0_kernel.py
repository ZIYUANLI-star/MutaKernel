import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication with tiling
matmul_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 32;

__global__ void matmul_transpose_kernel(
    const float* A,  // shape: [K, M]
    const float* B,  // shape: [K, N]
    float* C,        // shape: [M, N]
    int M, int K, int N) {
    
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Row and column of C to compute
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile (transposed access: A[K][M] -> As[M][K])
        int A_row = t * TILE_SIZE + tx;
        int A_col = row;
        if (A_row < K && A_col < M) {
            As[ty][tx] = A[A_row * M + A_col];  // A is stored as [K][M]
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile
        int B_row = t * TILE_SIZE + ty;
        int B_col = col;
        if (B_row < K && B_col < N) {
            Bs[ty][tx] = B[B_row * N + B_col];  // B is stored as [K][N]
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
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // A: [K, M], B: [K, N]
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);
    
    // Output: [M, N]
    auto C = torch::zeros({M, N}, A.options());
    
    // Configure grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matmul_transpose_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}
"""

matmul_transpose_cpp_source = """
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matmul_transpose = load_inline(
    name="matmul_transpose",
    cpp_sources=matmul_transpose_cpp_source,
    cuda_sources=matmul_transpose_source,
    functions=["matmul_transpose_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for matrix multiplication with transpose
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_transpose = matmul_transpose
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication C = A^T * B
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_transpose.matmul_transpose_cuda(A, B)