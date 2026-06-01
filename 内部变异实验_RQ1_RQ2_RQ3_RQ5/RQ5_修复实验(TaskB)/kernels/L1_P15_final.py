import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for triangular matrix multiplication
triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triangular_matmul_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        if (col <= row) {
            // Compute full dot product for lower triangular elements
            // This handles cases where input matrices may not be strictly lower triangular
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        } else {
            // Upper triangular elements are zero (tril operation)
            C[row * N + col] = 0.0f;
        }
    }
}

torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, 
                   (N + block_size.y - 1) / block_size.y);
    
    triangular_matmul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
"""

triangular_matmul_cpp_source = """
torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
triangular_matmul = load_inline(
    name="triangular_matmul_v2",
    cpp_sources=triangular_matmul_cpp_source,
    cuda_sources=triangular_matmul_source,
    functions=["triangular_matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of lower triangular matrices
    using a custom CUDA kernel that only computes the lower triangular part.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.triangular_matmul = triangular_matmul
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B
        using custom CUDA kernel.
        
        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).
            
        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N),
            which is also lower triangular.
        """
        return self.triangular_matmul.triangular_matmul_cuda(A, B)