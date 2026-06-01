import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define corrected CUDA kernel for upper triangular matrix multiplication
triu_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triu_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    const int TILE_SIZE = 32;
    
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    int row = blockRow * TILE_SIZE + ty;
    int col = blockCol * TILE_SIZE + tx;
    
    // Skip entirely lower triangular output blocks
    if ((blockCol + 1) * TILE_SIZE <= blockRow * TILE_SIZE) {
        if (row < N && col < N) {
            C[row * N + col] = 0.0f;
        }
        return;
    }
    
    float sum = 0.0f;
    
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over ALL tiles to compute full matrix product
    for (int tile = 0; tile < numTiles; tile++) {
        int A_row = row;
        int A_col = tile * TILE_SIZE + tx;
        
        if (A_row < N && A_col < N) {
            As[ty][tx] = A[A_row * N + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        int B_row = tile * TILE_SIZE + ty;
        int B_col = col;
        
        if (B_row < N && B_col < N) {
            Bs[ty][tx] = B[B_row * N + B_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result with upper triangular mask
    if (row < N && col < N) {
        if (col >= row) {
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    
    int grid_x = (N + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_size(grid_x, grid_y);
    
    triu_matmul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
"""

triu_matmul_cpp_source = "torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
triu_matmul = load_inline(
    name="triu_matmul_fixed",
    cpp_sources=triu_matmul_cpp_source,
    cuda_sources=triu_matmul_source,
    functions=["triu_matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a custom CUDA kernel with tiling and shared memory.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.triu_matmul = triu_matmul
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using custom CUDA kernel.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return self.triu_matmul.triu_matmul_cuda(A, B)

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []