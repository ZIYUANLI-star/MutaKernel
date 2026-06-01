import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication using shared memory tiling
# with double precision accumulation for better numerical stability
matmul_optimized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matmul_optimized_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, 
    int N, 
    int K
) {
    // Shared memory for tiles with padding to avoid bank conflicts
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within block
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    // Calculate row and column of C element to compute
    int row = blockRow * BLOCK_SIZE + threadRow;
    int col = blockCol * BLOCK_SIZE + threadCol;
    
    // Use double precision for accumulation to improve numerical stability
    double sum = 0.0;
    
    // Loop over tiles of the input matrices
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load A tile
        int A_row = row;
        int A_col = t * BLOCK_SIZE + threadCol;
        if (A_row < M && A_col < K) {
            As[threadRow][threadCol] = A[A_row * K + A_col];
        } else {
            As[threadRow][threadCol] = 0.0f;
        }
        
        // Load B tile
        int B_row = t * BLOCK_SIZE + threadRow;
        int B_col = col;
        if (B_row < K && B_col < N) {
            Bs[threadRow][threadCol] = B[B_row * N + B_col];
        } else {
            Bs[threadRow][threadCol] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile using double precision
        // Determine how many elements are valid in this tile
        int kMax = BLOCK_SIZE;
        int tileStart = t * BLOCK_SIZE;
        if (tileStart + BLOCK_SIZE > K) {
            kMax = K - tileStart;
        }
        
        for (int k = 0; k < kMax; ++k) {
            sum += (double)As[threadRow][k] * (double)Bs[k][threadCol];
        }
        
        __syncthreads();
    }
    
    // Write result to C
    if (row < M && col < N) {
        C[row * N + col] = (float)sum;
    }
}

torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    if (M == 0 || N == 0 || K == 0) {
        return C;
    }
    
    // Configure block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    matmul_optimized_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}
"""

matmul_optimized_cpp_source = "torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for optimized matrix multiplication
matmul_optimized = load_inline(
    name="matmul_optimized_v2",
    cpp_sources=matmul_optimized_cpp_source,
    cuda_sources=matmul_optimized_source,
    functions=["matmul_optimized_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using custom CUDA kernel
    with shared memory tiling and double precision accumulation for numerical stability
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using custom CUDA kernel.
        
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).
            
        Returns:
            Output tensor of shape (M, N).
        """
        # Ensure tensors are on CUDA and contiguous
        A = A.contiguous().cuda()
        B = B.contiguous().cuda()
        
        # Ensure float32 dtype
        if A.dtype != torch.float32:
            A = A.float()
        if B.dtype != torch.float32:
            B = B.float()
        
        return matmul_optimized.matmul_optimized_cuda(A, B)