import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_bias_scale_residual.cu ===

#include <cuda_runtime.h>

__global__ void fused_bias_scale_residual_kernel(
    const float* matmul_out,
    const float* bias,
    float* output,
    float scaling_factor,
    int batch_size,
    int out_features) {
    
    int tid = blockIdx.x * 128 + threadIdx.x;
    
    if (tid < batch_size * out_features) {
        float val = matmul_out[tid];
        float b = bias[tid % out_features];
        float with_bias = val + b;
        // original_x = with_bias (after matmul with bias)
        // x = with_bias * scaling_factor
        // output = x + original_x = with_bias * scaling_factor + with_bias = with_bias * (scaling_factor + 1)
        output[tid] = with_bias * (scaling_factor + 1.0f);
    }
}

extern "C" void fused_bias_scale_residual_launcher(
    const float* matmul_out,
    const float* bias,
    float* output,
    float scaling_factor,
    int batch_size,
    int out_features,
    cudaStream_t stream) {
    
    int blocks = (batch_size * out_features + 128 - 1) / 128;
    fused_bias_scale_residual_kernel<<<blocks, 128, 0, stream>>>(
        matmul_out, bias, output, scaling_factor, batch_size, out_features);
}

"""

_cpp_sources = """
// === fused_bias_scale_residual_binding.cpp ===

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_bias_scale_residual_launcher(
    const float* matmul_out,
    const float* bias,
    float* output,
    float scaling_factor,
    int batch_size,
    int out_features,
    cudaStream_t stream);

torch::Tensor fused_bias_scale_residual_forward(
    torch::Tensor matmul_out,
    torch::Tensor bias,
    float scaling_factor,
    int batch_size,
    int out_features) {
    
    // Input validation
    TORCH_CHECK(matmul_out.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(matmul_out.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(matmul_out.dtype() == torch::kFloat32, "Input must be float32");
    
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    
    auto output = torch::empty_like(matmul_out);
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call CUDA launcher
    fused_bias_scale_residual_launcher(
        matmul_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        out_features,
        stream);
    
    return output;
}

void register_fused_bias_scale_residual(pybind11::module& m) {
    m.def("fused_bias_scale_residual_forward", &fused_bias_scale_residual_forward,
          "Fused bias addition, scaling, and residual forward",
          py::arg("matmul_out"),
          py::arg("bias"),
          py::arg("scaling_factor"),
          py::arg("batch_size"),
          py::arg("out_features"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_bias_scale_residual_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs fused matrix multiplication, scaling, and residual addition.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Initialize parameters with the same names as in the original model
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        """
        Forward pass of the optimized model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size = x.shape[0]
        out_features = self.out_features
        
        # Disable TF32 for numerical precision
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Compute matrix multiplication without bias
            matmul_out = x @ self.matmul.weight.t()
            
            # Use custom fused kernel for bias addition + scaling/residual
            output = cuda_extension.fused_bias_scale_residual_forward(
                matmul_out, self.matmul.bias, self.scaling_factor, batch_size, out_features)
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return output