import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_div_gelu.cu ===

#include <cuda_runtime.h>

// Gelu activation function using tanh approximation for better numerical stability
__device__ __forceinline__ float gelu_activation(float x) {
    // Using the exact erf-based GELU formula
    float cdf = 0.5f * (1.0f + erff(x * 0.7071067811865475f));  // 1/sqrt(2)
    return x * cdf;
}

__global__ void fused_div_gelu_kernel(float* output, const float* input, float divisor, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    while (idx < total_elements) {
        float val = input[idx] / divisor;
        output[idx] = gelu_activation(val);
        idx += stride;
    }
}

extern "C" void fused_div_gelu_launcher(float* output, const float* input, float divisor, int total_elements, cudaStream_t stream) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Limit blocks to avoid excessive grid size
    if (blocks > 65535) blocks = 65535;
    fused_div_gelu_kernel<<<blocks, threads, 0, stream>>>(output, input, divisor, total_elements);
}

"""

_cpp_sources = """
// === fused_div_gelu_binding.cpp ===

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_div_gelu_launcher(float* output, const float* input, float divisor, int total_elements, cudaStream_t stream);

torch::Tensor fused_div_gelu_forward(torch::Tensor input, float divisor) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    int total_elements = input.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    fused_div_gelu_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        divisor,
        total_elements,
        stream
    );

    return output;
}

void register_fused_div_gelu(pybind11::module& m) {
    m.def("fused_div_gelu_forward", &fused_div_gelu_forward,
          "Fused division + Gelu forward",
          py::arg("input"),
          py::arg("divisor"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_div_gelu_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        # Disable TF32 for numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Use standard linear layer for matrix multiplication with bias
            matmul_out = self.linear(x)
            
            # Ensure contiguous for CUDA kernel
            if not matmul_out.is_contiguous():
                matmul_out = matmul_out.contiguous()
            
            # Use fused kernel for division + gelu
            output = cuda_extension.fused_div_gelu_forward(matmul_out, float(self.divisor))
            
            return output
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32