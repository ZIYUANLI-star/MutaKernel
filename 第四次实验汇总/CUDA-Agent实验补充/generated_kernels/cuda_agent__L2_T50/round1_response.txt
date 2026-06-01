import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_bias_relu_div.cu ===
#include <cuda_runtime.h>

__global__ void fused_bias_relu_div_kernel(float* output, const float* input, const float* bias, 
                                            int total_elements, int out_features, float divisor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < total_elements; i += stride) {
        float val = input[i];
        float b = bias[i % out_features];  // Bias has shape (out_features,)
        val = val + b;             // Add bias
        val = fmaxf(val, 0.0f);    // ReLU
        output[i] = val / divisor; // Division
    }
}

extern "C" void fused_bias_relu_div_launcher(float* output, const float* input, const float* bias, 
                                              int total_elements, int out_features, float divisor,
                                              cudaStream_t stream) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Limit blocks to avoid excessive grid size
    if (blocks > 65535) blocks = 65535;
    fused_bias_relu_div_kernel<<<blocks, threads, 0, stream>>>(output, input, bias, 
                                                                total_elements, out_features, divisor);
}

"""

_cpp_sources = """
// === fused_bias_relu_div_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_bias_relu_div_launcher(float* output, const float* input, const float* bias, 
                                              int total_elements, int out_features, float divisor,
                                              cudaStream_t stream);

torch::Tensor fused_bias_relu_div_forward(torch::Tensor input, torch::Tensor bias, double divisor) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");

    auto output = torch::empty_like(input);

    int total_elements = input.numel();
    int out_features = bias.numel();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    fused_bias_relu_div_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        total_elements,
        out_features,
        static_cast<float>(divisor),
        stream
    );

    return output;
}

void register_fused_bias_relu_div(pybind11::module& m) {
    m.def("fused_bias_relu_div_forward", &fused_bias_relu_div_forward,
          "Fused bias + relu + div forward",
          py::arg("input"),
          py::arg("bias"),
          py::arg("divisor"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_bias_relu_div_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Model that uses fused bias + ReLU + division kernel.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        # Initialize with same state_dict keys as original model
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        # Save and disable TF32 for numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Linear layer without bias (we'll add bias in fused kernel)
            x = torch.matmul(x, self.linear.weight.t())
            
            # Use custom fused kernel for bias + ReLU + division
            x = cuda_extension.fused_bias_relu_div_forward(x, self.linear.bias, self.divisor)
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return x