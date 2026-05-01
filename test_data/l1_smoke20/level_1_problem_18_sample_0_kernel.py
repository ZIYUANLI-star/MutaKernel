import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized matrix multiplication
matmul_optimized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel for transposing matrix A (K, M) -> (M, K)
__global__ void transpose_A_kernel(const float* A, float* A_T, int K, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < M && idy < K) {
        A_T[idx * K + idy] = A[idy * M + idx];
    }
}

// Kernel for transposing matrix B (N, K) -> (K, N)
__global__ void transpose_B_kernel(const float* B, float* B_T, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < K && idy < N) {
        B_T[idx * N + idy] = B[idy * K + idx];
    }
}

torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);
    
    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());
    
    // Create transposed versions
    auto A_T = torch::zeros({M, K}, A.options());
    auto B_T = torch::zeros({K, N}, B.options());
    
    // Transpose A (K, M) -> (M, K)
    dim3 blockDim1(16, 16);
    dim3 gridDim1((M + 15) / 16, (K + 15) / 16);
    transpose_A_kernel<<<gridDim1, blockDim1>>>(
        A.data_ptr<float>(),
        A_T.data_ptr<float>(),
        K, M
    );
    
    // Transpose B (N, K) -> (K, N)
    dim3 gridDim2((K + 15) / 16, (N + 15) / 16);
    transpose_B_kernel<<<gridDim2, blockDim1>>>(
        B.data_ptr<float>(),
        B_T.data_ptr<float>(),
        N, K
    );
    
    // Use cuBLAS for matrix multiplication: C = A_T * B_T
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,  // Note: dimensions are swapped for column-major
                &alpha,
                B_T.data_ptr<float>(), N,
                A_T.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);
    
    cublasDestroy(handle);
    
    return C;
}
"""

matmul_optimized_cpp_source = """
torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matmul_optimized = load_inline(
    name="matmul_optimized",
    cpp_sources=matmul_optimized_cpp_source,
    cuda_sources=matmul_optimized_source,
    functions=["matmul_optimized_cuda"],
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
        self.matmul_optimized = matmul_optimized
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication A.T * B.T
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_optimized.matmul_optimized_cuda(A, B)