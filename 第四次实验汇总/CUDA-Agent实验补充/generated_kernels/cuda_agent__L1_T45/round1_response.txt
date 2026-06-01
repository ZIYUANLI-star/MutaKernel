import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === lower_tril_mask.cu ===
__global__ void lower_tril_mask_kernel(float* C, int N) {
    int idx = blockIdx.x * 256 + threadIdx.x;

    if (idx < N * N) {
        int row = idx / N;
        int col = idx % N;
        if (col > row) {
            C[idx] = 0.0f;
        }
    }
}

extern "C" void lower_tril_mask_launcher(float* C, int N, cudaStream_t stream) {
    int total = N * N;
    int blocks = (total + 256 - 1) / 256;
    lower_tril_mask_kernel<<<blocks, 256, 0, stream>>>(C, N);
}
"""

_cpp_sources = """
// === lower_tril_mask_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void lower_tril_mask_launcher(float* C, int N, cudaStream_t stream);

torch::Tensor lower_tril_mask(torch::Tensor C) {
    // Input validation
    TORCH_CHECK(C.is_cuda(), "Input C must be a CUDA tensor");
    TORCH_CHECK(C.is_contiguous(), "Input C must be contiguous");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "Input C must be float32");
    TORCH_CHECK(C.dim() == 2, "Input must be 2D");
    TORCH_CHECK(C.size(0) == C.size(1), "Input must be square");

    int N = C.size(0);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    lower_tril_mask_launcher(
        C.data_ptr<float>(),
        N,
        stream
    );

    return C;
}

void register_lower_tril_mask(pybind11::module& m) {
    m.def("lower_tril_mask", &lower_tril_mask,
          "Lower triangular mask",
          py::arg("C"));
}


"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["lower_tril_mask"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of lower triangular matrices A and B
    and returns the lower triangular part of the result.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B
        and returns the lower triangular part of the result.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        # Disable TF32 to ensure numerical precision matches reference
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            C = torch.matmul(A, B)
            # Apply lower triangular mask
            C = cuda_extension.lower_tril_mask(C.contiguous())
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return C