import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_postprocess.cu ===
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_bias_batchnorm_extrabias_division_swish_kernel(
    float* output,
    const float* gemm_out,
    const float* gemm_bias,
    const float* running_mean,
    const float* running_var,
    const float* bn_weight,
    const float* bn_bias,
    const float* extra_bias,
    float divide_value,
    float eps,
    int batch_size,
    int features
) {
    int tid = blockIdx.x * 128 + threadIdx.x;
    
    if (tid < batch_size * features) {
        int feature_idx = tid % features;
        
        // Get all required values
        float val = gemm_out[tid];
        float gb = gemm_bias[feature_idx];
        float rm = running_mean[feature_idx];
        float rv = running_var[feature_idx];
        float gw = bn_weight[feature_idx];
        float gb_bn = bn_bias[feature_idx];
        float eb = extra_bias[0];
        
        // First add the GEMM bias to get the actual linear output
        val = val + gb;
        
        // Compute inv_std once
        float inv_std = rsqrtf(rv + eps);
        
        // Apply batch normalization: (val - mean) * inv_std * weight + bias
        val = (val - rm) * inv_std * gw + gb_bn;
        
        // Add extra bias
        val = val + eb;
        
        // Divide
        val = val / divide_value;
        
        // Apply swish activation
        float sig = sigmoid(val);
        output[tid] = val * sig;
    }
}

extern "C" void fused_bias_batchnorm_extrabias_division_swish_launcher(
    float* output,
    const float* gemm_out,
    const float* gemm_bias,
    const float* running_mean,
    const float* running_var,
    const float* bn_weight,
    const float* bn_bias,
    const float* extra_bias,
    float divide_value,
    float eps,
    int batch_size,
    int features,
    cudaStream_t stream
) {
    int blocks = (batch_size * features + 127) / 128;
    fused_bias_batchnorm_extrabias_division_swish_kernel<<<blocks, 128, 0, stream>>>(
        output, gemm_out, gemm_bias, running_mean, running_var, bn_weight, bn_bias,
        extra_bias, divide_value, eps, batch_size, features
    );
}

"""

_cpp_sources = """
// === fused_postprocess_binding.cpp ===

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_bias_batchnorm_extrabias_division_swish_launcher(
    float* output,
    const float* gemm_out,
    const float* gemm_bias,
    const float* running_mean,
    const float* running_var,
    const float* bn_weight,
    const float* bn_bias,
    const float* extra_bias,
    float divide_value,
    float eps,
    int batch_size,
    int features,
    cudaStream_t stream
);

torch::Tensor fused_postprocess_forward(
    torch::Tensor gemm_out,
    torch::Tensor gemm_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor extra_bias,
    float divide_value,
    float eps
) {
    TORCH_CHECK(gemm_out.is_cuda() && gemm_out.dtype() == torch::kFloat32 && gemm_out.is_contiguous());
    TORCH_CHECK(gemm_bias.is_cuda() && gemm_bias.dtype() == torch::kFloat32 && gemm_bias.is_contiguous());
    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32 && running_mean.is_contiguous());
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32 && running_var.is_contiguous());
    TORCH_CHECK(bn_weight.is_cuda() && bn_weight.dtype() == torch::kFloat32 && bn_weight.is_contiguous());
    TORCH_CHECK(bn_bias.is_cuda() && bn_bias.dtype() == torch::kFloat32 && bn_bias.is_contiguous());
    TORCH_CHECK(extra_bias.is_cuda() && extra_bias.dtype() == torch::kFloat32 && extra_bias.is_contiguous());
    
    TORCH_CHECK(gemm_out.ndimension() == 2, "gemm_out must be 2D (batch_size, features)");
    TORCH_CHECK(extra_bias.size(0) == 1, "extra_bias must have size 1");
    
    int batch_size = gemm_out.size(0);
    int features = gemm_out.size(1);
    
    TORCH_CHECK(gemm_bias.size(0) == features && running_mean.size(0) == features &&
               running_var.size(0) == features && bn_weight.size(0) == features &&
               bn_bias.size(0) == features);
    
    auto output = torch::empty_like(gemm_out);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    fused_bias_batchnorm_extrabias_division_swish_launcher(
        output.data_ptr<float>(),
        gemm_out.data_ptr<float>(),
        gemm_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        extra_bias.data_ptr<float>(),
        divide_value,
        eps,
        batch_size,
        features,
        stream
    );
    
    return output;
}

void register_fused_postprocess(pybind11::module& m) {
    m.def("fused_postprocess_forward", &fused_postprocess_forward,
          "Fused postprocessing after GEMM: bias + batch norm + extra bias + division + swish",
          py::arg("gemm_out"),
          py::arg("gemm_bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("bn_weight"),
          py::arg("bn_bias"),
          py::arg("extra_bias"),
          py::arg("divide_value"),
          py::arg("eps"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_postprocess_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication with cuBLAS and fuses
    batch normalization, bias addition, division, and Swish activation into a single kernel.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        # Preserve original structure for state_dict compatibility
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # Check if we're in training mode - if so, use the reference implementation
        # because batch norm statistics need to be computed from the current batch
        if self.training:
            # Use standard PyTorch operations in training mode
            x = self.matmul(x)
            x = self.bn(x)
            x = x + self.bias
            x = x / self.divide_value
            x = x * torch.sigmoid(x)
            return x
        
        # In eval mode, use the optimized fused kernel
        # Disable TF32 for numerical precision
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Step 1: Matrix multiplication (without bias, we'll add it in the fused kernel)
            weight = self.matmul.weight
            x = x @ weight.T
            
            # Step 2: Use custom fused kernel for all postprocessing steps
            x = cuda_extension.fused_postprocess_forward(
                x,
                self.matmul.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                self.bias,
                self.divide_value,
                self.bn.eps
            )
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return x