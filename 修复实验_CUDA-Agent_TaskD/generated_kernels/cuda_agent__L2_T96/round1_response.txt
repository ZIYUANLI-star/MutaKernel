import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_all_after_gemm.cu ===
#include <cuda_runtime.h>

__global__ void fused_all_after_gemm_kernel(float* output, const float* input, const float* gemm_bias,
                                            const float* extra_bias, const float* gn_gamma, const float* gn_beta,
                                            int batch_size, int features, int num_groups) {
    __shared__ float sum_val[128];
    __shared__ float sq_sum_val[128];
    
    int idx = threadIdx.x;
    int block = blockIdx.x;
    
    int group_size = features / num_groups;
    
    int batch = block / num_groups;
    int group = block % num_groups;
    
    // Use double precision for accumulation to improve numerical stability
    double sum = 0.0;
    double sq_sum = 0.0;
    
    // Step 1: Compute all operations for this block's group and accumulate sum/sq_sum
    for (int i = idx; i < group_size; i += 128) {
        int feature = group * group_size + i;
        int linear_idx = batch * features + feature;
        
        float x_val = input[linear_idx];  // This is the output of the linear layer (matmul without bias)
        x_val += gemm_bias[feature];      // First bias (from linear layer)
        x_val += extra_bias[feature];     // Second bias
        
        // Hardtanh
        x_val = fmaxf(-1.0f, fminf(1.0f, x_val));
        
        // Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
        float exp_val = expf(x_val);
        float log_val = log1pf(exp_val);
        float tanh_val = tanhf(log_val);
        x_val *= tanh_val;
        
        // Accumulate for mean and variance using double precision
        sum += (double)x_val;
        sq_sum += (double)x_val * (double)x_val;
    }
    
    sum_val[idx] = (float)sum;
    sq_sum_val[idx] = (float)sq_sum;
    __syncthreads();
    
    // Step 2: Reduce sum and sq_sum across threads in the block
    for (int i = 64; i > 0; i >>= 1) {
        if (idx < i) {
            sum_val[idx] += sum_val[idx + i];
            sq_sum_val[idx] += sq_sum_val[idx + i];
        }
        __syncthreads();
    }
    
    float inv_n = 1.0f / (float)group_size;
    float mean = sum_val[0] * inv_n;
    float var = sq_sum_val[0] * inv_n - mean * mean;
    // Ensure variance is non-negative (can become slightly negative due to floating point errors)
    var = fmaxf(var, 0.0f);
    float inv_std = rsqrtf(var + 1e-5f);
    __syncthreads();
    
    // Step 3: Normalize each element in the group
    for (int i = idx; i < group_size; i += 128) {
        int feature = group * group_size + i;
        int linear_idx = batch * features + feature;
        
        float x_val = input[linear_idx];
        x_val += gemm_bias[feature];
        x_val += extra_bias[feature];
        
        // Hardtanh
        x_val = fmaxf(-1.0f, fminf(1.0f, x_val));
        
        // Mish
        float exp_val = expf(x_val);
        float log_val = log1pf(exp_val);
        float tanh_val = tanhf(log_val);
        x_val *= tanh_val;
        
        float normalized_val = gn_gamma[feature] * (x_val - mean) * inv_std + gn_beta[feature];
        output[linear_idx] = normalized_val;
    }
}

extern "C" void fused_all_after_gemm_launcher(float* output, const float* input, const float* gemm_bias,
                                              const float* extra_bias, const float* gn_gamma, const float* gn_beta,
                                              int batch_size, int features, int num_groups) {
    int blocks = batch_size * num_groups;
    fused_all_after_gemm_kernel<<<blocks, 128>>>(output, input, gemm_bias, extra_bias, gn_gamma, gn_beta,
                                                  batch_size, features, num_groups);
}

"""

_cpp_sources = """
// === fused_all_after_gemm_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_all_after_gemm_launcher(float* output, const float* input, const float* gemm_bias,
                                              const float* extra_bias, const float* gn_gamma, const float* gn_beta,
                                              int batch_size, int features, int num_groups);

torch::Tensor fused_all_after_gemm_forward(torch::Tensor input, torch::Tensor gemm_bias, torch::Tensor extra_bias, 
                                          torch::Tensor gn_gamma, torch::Tensor gn_beta, int num_groups) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    TORCH_CHECK(gemm_bias.is_cuda(), "GEMM bias must be a CUDA tensor");
    TORCH_CHECK(gemm_bias.is_contiguous(), "GEMM bias must be contiguous");
    TORCH_CHECK(gemm_bias.dtype() == torch::kFloat32, "GEMM bias must be float32");
    
    TORCH_CHECK(extra_bias.is_cuda(), "Extra bias must be a CUDA tensor");
    TORCH_CHECK(extra_bias.is_contiguous(), "Extra bias must be contiguous");
    TORCH_CHECK(extra_bias.dtype() == torch::kFloat32, "Extra bias must be float32");
    
    TORCH_CHECK(gn_gamma.is_cuda(), "GroupNorm gamma must be a CUDA tensor");
    TORCH_CHECK(gn_gamma.is_contiguous(), "GroupNorm gamma must be contiguous");
    TORCH_CHECK(gn_gamma.dtype() == torch::kFloat32, "GroupNorm gamma must be float32");
    
    TORCH_CHECK(gn_beta.is_cuda(), "GroupNorm beta must be a CUDA tensor");
    TORCH_CHECK(gn_beta.is_contiguous(), "GroupNorm beta must be contiguous");
    TORCH_CHECK(gn_beta.dtype() == torch::kFloat32, "GroupNorm beta must be float32");

    auto output = torch::empty_like(input);
    
    int batch_size = input.size(0);
    int features = input.size(1);

    fused_all_after_gemm_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        gemm_bias.data_ptr<float>(),
        extra_bias.data_ptr<float>(),
        gn_gamma.data_ptr<float>(),
        gn_beta.data_ptr<float>(),
        batch_size,
        features,
        num_groups
    );

    return output;
}

void register_fused_all_after_gemm(pybind11::module& m) {
    m.def("fused_all_after_gemm_forward", &fused_all_after_gemm_forward,
          "Fused all operations after GEMM (bias, hardtanh, mish, groupnorm) forward",
          py::arg("input"),
          py::arg("gemm_bias"),
          py::arg("extra_bias"),
          py::arg("gn_gamma"),
          py::arg("gn_beta"),
          py::arg("num_groups"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_all_after_gemm_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses all operations (Linear bias, additional bias, Hardtanh, Mish, and GroupNorm).
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        # Initialize parameters with names that match the original model
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(out_features))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.num_groups = num_groups

    def load_state_dict(self, state_dict, strict=True):
        # Map original state_dict to our parameters
        new_state_dict = {}
        for key, val in state_dict.items():
            if key == 'gemm.weight':
                new_state_dict['gemm.weight'] = val
            elif key == 'gemm.bias':
                new_state_dict['gemm.bias'] = val
            elif key == 'bias':
                new_state_dict['bias'] = val
            elif key == 'groupnorm.weight':
                new_state_dict['groupnorm.weight'] = val
            elif key == 'groupnorm.bias':
                new_state_dict['groupnorm.bias'] = val
        super().load_state_dict(new_state_dict, strict=strict)

    def forward(self, x):
        # Disable TF32 for numerical precision
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Step 1: Linear layer - just the matmul part (no bias)
            x = torch.matmul(x, self.gemm.weight.T)
            
            # Step 2: Fuse ALL operations - gemm bias, additional bias, hardtanh, mish, and groupnorm
            x = cuda_extension.fused_all_after_gemm_forward(x, self.gemm.bias, self.bias, 
                                                          self.groupnorm.weight, self.groupnorm.bias,
                                                          self.num_groups)
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return x