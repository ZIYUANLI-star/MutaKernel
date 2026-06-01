import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_bias_postprocess.cu ===

#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bias_postprocess_kernel(float* output, const float* gemm_out, const float* bias, int batch_size, int features) {
    int tid = blockIdx.x * 256 + threadIdx.x;
    int total_elements = batch_size * features;

    if (tid < total_elements) {
        int col = tid % features; // feature index
        
        float x = gemm_out[tid] + bias[col];
        
        // Swish activation: x * sigmoid(x)
        // Use numerically stable sigmoid computation
        float sig;
        if (x >= 0.0f) {
            sig = 1.0f / (1.0f + expf(-x));
        } else {
            float exp_x = expf(x);
            sig = exp_x / (1.0f + exp_x);
        }
        float swish_val = x * sig;
        
        // Divide by 2.0
        float div_val = swish_val * 0.5f;
        
        // Clamp between -1.0 and 1.0
        float clamp1_val = fmaxf(-1.0f, fminf(1.0f, div_val));
        
        // Tanh activation
        float tanh_val = tanhf(clamp1_val);
        
        // Clamp between -1.0 and 1.0
        float clamp2_val = fmaxf(-1.0f, fminf(1.0f, tanh_val));
        
        // Write output
        output[tid] = clamp2_val;
    }
}

extern "C" void fused_bias_postprocess_launcher(
    float* output,
    const float* gemm_out,
    const float* bias,
    int batch_size,
    int features,
    cudaStream_t stream
) {
    int blocks = (batch_size * features + 255) / 256;
    fused_bias_postprocess_kernel<<<blocks, 256, 0, stream>>>(output, gemm_out, bias, batch_size, features);
}

"""

_cpp_sources = """
// === fused_bias_postprocess_binding.cpp ===

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_bias_postprocess_launcher(
    float* output,
    const float* gemm_out,
    const float* bias,
    int batch_size,
    int features,
    cudaStream_t stream
);

torch::Tensor fused_bias_postprocess_forward(torch::Tensor gemm_out, torch::Tensor bias, int batch_size, int features) {
    TORCH_CHECK(gemm_out.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gemm_out.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gemm_out.dtype() == torch::kFloat32, "Input must be float32");

    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");

    auto output = torch::empty_like(gemm_out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fused_bias_postprocess_launcher(
        output.data_ptr<float>(),
        gemm_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        features,
        stream
    );

    return output;
}

void register_fused_bias_postprocess(pybind11::module& m) {
    m.def("fused_bias_postprocess_forward", &fused_bias_postprocess_forward, 
          "Fused bias + postprocess forward",
          py::arg("gemm_out"),
          py::arg("bias"),
          py::arg("batch_size"),
          py::arg("features"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_bias_postprocess_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs gemm with fused bias + postprocess operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Disable TF32 for numerical precision
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Compute gemm without bias
            gemm_out = x.matmul(self.weight.transpose(0, 1))
            
            # Use fused bias + postprocess kernel
            batch_size = x.shape[0]
            out_features = self.out_features
            x = cuda_extension.fused_bias_postprocess_forward(gemm_out, self.bias, batch_size, out_features)
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return x

    def load_state_dict(self, state_dict, strict=True):
        # Remap state dict keys from "gemm.weight" to "weight" and "gemm.bias" to "bias"
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("gemm."):
                new_key = key.replace("gemm.", "")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        super().load_state_dict(new_state_dict, strict=strict)