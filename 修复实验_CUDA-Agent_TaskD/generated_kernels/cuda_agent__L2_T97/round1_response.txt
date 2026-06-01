import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_bias_sigmoid_scaling_residual.cu ===
#include <cuda_runtime.h>

__global__ void fused_bias_sigmoid_scaling_residual_kernel(
    float* output, 
    const float* x, 
    const float* bias,
    float scaling_factor,
    int batch_size,
    int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_size;

    if (tid < total_elements) {
        float val = x[tid] + bias[tid % hidden_size];
        // Use expf for single precision and handle numerical stability
        float neg_val = -val;
        // Clamp to avoid overflow in exp
        neg_val = fminf(fmaxf(neg_val, -88.0f), 88.0f);
        float sig = 1.0f / (1.0f + expf(neg_val));
        output[tid] = val + scaling_factor * sig;
    }
}

extern "C" void fused_bias_sigmoid_scaling_residual_launcher(
    float* output,
    const float* x,
    const float* bias,
    float scaling_factor,
    int batch_size,
    int hidden_size,
    cudaStream_t stream
) {
    int total_elements = batch_size * hidden_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    fused_bias_sigmoid_scaling_residual_kernel<<<blocks, threads, 0, stream>>>(
        output, x, bias, scaling_factor, batch_size, hidden_size
    );
}

"""

_cpp_sources = """
// === fused_bias_sigmoid_scaling_residual_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_bias_sigmoid_scaling_residual_launcher(
    float* output,
    const float* x,
    const float* bias,
    float scaling_factor,
    int batch_size,
    int hidden_size,
    cudaStream_t stream
);

torch::Tensor fused_bias_sigmoid_scaling_residual_forward(
    torch::Tensor x, 
    torch::Tensor bias,
    float scaling_factor
) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    
    int batch_size = x.size(0);
    int hidden_size = x.size(1);
    
    TORCH_CHECK(bias.size(0) == hidden_size, "Bias size must match hidden_size");
    
    auto output = torch::empty_like(x);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    fused_bias_sigmoid_scaling_residual_launcher(
        output.data_ptr<float>(), 
        x.data_ptr<float>(), 
        bias.data_ptr<float>(),
        scaling_factor,
        batch_size,
        hidden_size,
        stream
    );
    
    return output;
}

void register_fused_bias_sigmoid_scaling_residual(pybind11::module& m) {
    m.def("fused_bias_sigmoid_scaling_residual_forward", &fused_bias_sigmoid_scaling_residual_forward,
          "Fused bias addition, sigmoid, scaling, and residual addition");
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_bias_sigmoid_scaling_residual_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd" with fused post-processing.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # Disable TF32 for numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Use the same weight and bias as the original model
            weight = self.gemm.weight
            bias = self.gemm.bias
            
            # Compute matmul without bias
            x = x.matmul(weight.transpose(0, 1))
            
            # Ensure contiguous for CUDA kernel
            x = x.contiguous()
            
            # Use fused kernel for bias + sigmoid + scaling + residual
            x = cuda_extension.fused_bias_sigmoid_scaling_residual_forward(
                x, bias, self.scaling_factor
            )
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return x