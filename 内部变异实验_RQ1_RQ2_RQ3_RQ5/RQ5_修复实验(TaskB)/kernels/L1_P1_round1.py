import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel using cuBLAS for numerically stable matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Global cuBLAS handle for efficiency
static cublasHandle_t cublas_handle = nullptr;
static bool handle_initialized = false;

void ensure_cublas_handle() {
    if (!handle_initialized) {
        cublasCreate(&cublas_handle);
        handle_initialized = true;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    ensure_cublas_handle();
    
    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();
    
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    
    int M = A.size(0);  // rows of A and C
    int K = A.size(1);  // cols of A, rows of B
    int N = B.size(1);  // cols of B and C
    
    auto C = torch::empty({M, N}, A.options());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS uses column-major ordering, PyTorch uses row-major
    // For row-major C = A * B where A is MxK, B is KxN, C is MxN:
    // In cuBLAS column-major view:
    //   - PyTorch's A (row-major MxK) is seen as A^T (column-major KxM)
    //   - PyTorch's B (row-major KxN) is seen as B^T (column-major NxK)
    //   - We want C (row-major MxN) which cuBLAS sees as C^T (column-major NxM)
    // So we compute: C^T = B^T * A^T
    // cublasSgemm computes: C_cublas = alpha * op(A_cublas) * op(B_cublas) + beta * C_cublas
    // Where C_cublas is m x n, op(A_cublas) is m x k, op(B_cublas) is k x n
    // We need: C^T (NxM) = B^T (NxK) * A^T (KxM)
    // So: m=N, n=M, k=K, first_matrix=B, second_matrix=A
    
    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.data_ptr<float>(), N,  // lda = leading dimension of B in column-major = N
        A.data_ptr<float>(), K,  // ldb = leading dimension of A in column-major = K
        &beta,
        C.data_ptr<float>(), N   // ldc = leading dimension of C in column-major = N
    );
    
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS sgemm failed");
    
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
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model with cuBLAS-based CUDA kernel for matrix multiplication.
    Uses cuBLAS for numerically stable results matching PyTorch's matmul.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using cuBLAS.
        
        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).
            
        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.matmul_cuda.matmul_cuda(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N).cuda()
    B = torch.randn(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []