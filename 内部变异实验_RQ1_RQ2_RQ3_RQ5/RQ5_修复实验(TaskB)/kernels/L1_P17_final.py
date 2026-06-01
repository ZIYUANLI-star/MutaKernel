import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Use cuBLAS for matrix multiplication to match PyTorch's numerical precision
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// cuBLAS handle management
static cublasHandle_t cublas_handle = nullptr;

void ensure_cublas_handle() {
    if (cublas_handle == nullptr) {
        cublasCreate(&cublas_handle);
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    ensure_cublas_handle();
    
    // A is (M, K), B is (N, K), output C is (M, N) = A * B^T
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();
    
    // Create output tensor
    auto C = torch::empty({M, N}, A.options());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS uses column-major order while PyTorch uses row-major
    // For C = A * B^T where:
    //   A is (M, K) row-major, cuBLAS sees as (K, M) col-major
    //   B is (N, K) row-major, cuBLAS sees as (K, N) col-major
    //   C is (M, N) row-major, cuBLAS sees as (N, M) col-major
    //
    // We need: C_col = B_col^T * A_col
    // Which is: cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, ...)
    
    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,   // transpose B (as seen by cuBLAS)
        CUBLAS_OP_N,   // don't transpose A (as seen by cuBLAS)
        N, M, K,
        &alpha,
        B.data_ptr<float>(), K,  // ldb = K (leading dim of B in memory)
        A.data_ptr<float>(), K,  // lda = K (leading dim of A in memory)
        &beta,
        C.data_ptr<float>(), N   // ldc = N (leading dim of C in memory)
    );
    
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_cuda = load_inline(
    name="matmul_cuda_cublas",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model using cuBLAS for matrix multiplication with proper numerical precision.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using cuBLAS.
        
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_cuda.matmul_cuda(A, B)