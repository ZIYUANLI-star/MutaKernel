import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === matvec.cu ===
#include <cuda_runtime.h>

__global__ void matvec_kernel(double* output, const float* A, const float* B, int M, int K) {
    // Use double precision for accumulation to avoid numerical instability
    __shared__ double smem[256];

    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (idx < M) {
        double sum = 0.0;
        const float* row = A + (long long)idx * K;

        // Each thread processes elements with stride of blockSize
        for (int k = tid; k < K; k += blockSize) {
            sum += (double)row[k] * (double)B[k];
        }

        smem[tid] = sum;
        __syncthreads();

        // Parallel reduction in shared memory
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] += smem[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[idx] = smem[0];
        }
    }
}

extern "C" void matvec_launcher(double* output, const float* A, const float* B, int M, int K, cudaStream_t stream) {
    matvec_kernel<<<M, 256, 0, stream>>>(output, A, B, M, K);
}

"""

_cpp_sources = """
// === matvec_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void matvec_launcher(double* output, const float* A, const float* B, int M, int K, cudaStream_t stream);

torch::Tensor matvec_forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "Input A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Input B must be contiguous");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");
    TORCH_CHECK(A.dim() == 2, "Input A must be 2D matrix");
    TORCH_CHECK(B.dim() == 2, "Input B must be 2D vector");

    int M = A.size(0);
    int K = A.size(1);

    // Use double precision output tensor for intermediate result
    auto output_double = torch::empty({M, 1}, A.options().dtype(torch::kFloat64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    matvec_launcher(
        output_double.data_ptr<double>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        M,
        K,
        stream
    );

    // Convert back to float32
    return output_double.to(torch::kFloat32);
}

void register_matvec(pybind11::module& m) {
    m.def("matvec_forward", &matvec_forward,
          "Matrix-vector multiplication forward",
          py::arg("A"),
          py::arg("B"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["matvec_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using custom CUDA kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        # Flatten B for the kernel (expects 1D access pattern)
        B_flat = B.view(-1)
        
        # Create a contiguous view of B as 2D for the kernel check
        B_2d = B_flat.view(-1, 1)
        
        return cuda_extension.matvec_forward(A, B_2d)