import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === matvec.cu ===
#include <cuda_runtime.h>

__global__ void matvec_kernel(float* output, const float* A, const float* B, int M, int K) {
    // Use double precision for accumulation to avoid numerical instability
    __shared__ double smem[128];

    int idx = blockIdx.x;
    int tid = threadIdx.x;

    smem[tid] = 0.0;
    __syncthreads();

    if (idx < M) {
        double sum = 0.0;
        const float* row = A + (long long)idx * K;

        // Use float4 for vectorization but accumulate in double
        int vec_size = 4;
        for (int k = tid * vec_size; k < K; k += 128 * vec_size) {
            float4 a_val = reinterpret_cast<const float4*>(row)[k / vec_size];
            float4 b_val = reinterpret_cast<const float4*>(B)[k / vec_size];

            sum += (double)a_val.x * (double)b_val.x;
            sum += (double)a_val.y * (double)b_val.y;
            sum += (double)a_val.z * (double)b_val.z;
            sum += (double)a_val.w * (double)b_val.w;
        }

        smem[tid] = sum;
        __syncthreads();

        // Parallel reduction in shared memory
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] += smem[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[idx] = (float)smem[0];
        }
    }
}

extern "C" void matvec_launcher(float* output, const float* A, const float* B, int M, int K, cudaStream_t stream) {
    matvec_kernel<<<M, 128, 0, stream>>>(output, A, B, M, K);
}

"""

_cpp_sources = """
// === matvec_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void matvec_launcher(float* output, const float* A, const float* B, int M, int K, cudaStream_t stream);

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
    
    // Ensure K is divisible by 4 for float4 vectorization
    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for vectorization");

    auto output = torch::empty({M, 1}, A.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    matvec_launcher(
        output.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        M,
        K,
        stream
    );

    return output;
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
        return cuda_extension.matvec_forward(A, B)