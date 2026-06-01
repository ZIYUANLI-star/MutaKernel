import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_bias_relu.cu ===
#include <cuda_runtime.h>

__global__ void fused_bias_relu_kernel(float* output, const float* input, const float* bias, int batch_size, int features) {
    int tid = blockIdx.x * 128 + threadIdx.x;
    int total_elements = batch_size * features;

    while (tid < total_elements) {
        int feature = tid % features;
        float val = input[tid] + bias[feature];
        output[tid] = val > 0 ? val : 0.0f;
        tid += 128 * gridDim.x;
    }
}

extern "C" void fused_bias_relu_launcher(float* output, const float* input, const float* bias, int batch_size, int features, cudaStream_t stream) {
    int blocks = (batch_size * features + 128 - 1) / 128;
    fused_bias_relu_kernel<<<blocks, 128, 0, stream>>>(output, input, bias, batch_size, features);
}

"""

_cpp_sources = """
// === fused_bias_relu_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_bias_relu_launcher(float* output, const float* input, const float* bias, int batch_size, int features, cudaStream_t stream);

torch::Tensor fused_bias_relu_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(input.size(1) == bias.size(0), "Input features must match bias size");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fused_bias_relu_launcher(output.data_ptr<float>(), input.data_ptr<float>(), bias.data_ptr<float>(), input.size(0), input.size(1), stream);

    return output;
}

void register_fused_bias_relu(pybind11::module& m) {
    m.def("fused_bias_relu_forward", &fused_bias_relu_forward,
          "Fused bias addition and ReLU forward",
          py::arg("input"),
          py::arg("bias"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_bias_relu_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that uses fused bias+ReLU kernel with full precision matmul.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        # Use the same parameter name as the original model for state_dict compatibility
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Disable TF32 to ensure full precision for numerical stability
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Compute matrix multiplication using standard Linear layer (full precision)
            x = self.gemm(x)
            
            # Use custom fused bias+ReLU kernel
            x = cuda_extension.fused_bias_relu_forward(x, self.bias)
        finally:
            # Restore original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return x