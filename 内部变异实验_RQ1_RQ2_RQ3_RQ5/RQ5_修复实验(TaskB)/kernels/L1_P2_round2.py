import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Use cuBLAS directly to match PyTorch's numerical behavior exactly
matmul_optimized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    
    auto C = torch::empty({M, N}, A.options());
    
    if (M == 0 || N == 0 || K == 0) {
        return torch::zeros({M, N}, A.options());
    }
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS handle creation failed");
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // cuBLAS uses column-major format, but PyTorch uses row-major
    // For row-major A (M x K) and B (K x N), C = A * B
    // In cuBLAS column-major view:
    // A becomes A^T of size (K x M)
    // B becomes B^T of size (N x K)  
    // C becomes C^T of size (N x M)
    // We want C^T = B^T * A^T, which gives us C = A * B in row-major
    
    status = cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr<float>(), N,
                A.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);
    
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS sgemm failed");
    
    cublasDestroy(handle);
    
    return C;
}
"""

matmul_optimized_cpp_source = "torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for optimized matrix multiplication
matmul_optimized = load_inline(
    name="matmul_optimized_v3",
    cpp_sources=matmul_optimized_cpp_source,
    cuda_sources=matmul_optimized_source,
    functions=["matmul_optimized_cuda"],
    verbose=False,
    with_cuda=True,
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using cuBLAS
    for exact numerical compatibility with PyTorch's torch.matmul
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using cuBLAS.
        
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