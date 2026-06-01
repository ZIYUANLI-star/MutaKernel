import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === conv_bias_instance_norm_divide_cuda.cu ===
#include <cuda_runtime.h>

__global__ void conv_bias_instance_norm_divide_kernel(
    float* output, 
    const float* input, 
    const float* bias,
    int batch_size,
    int num_channels,
    int height,
    int width,
    float divide_by
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    
    int block_size = blockDim.x;
    int num_blocks = batch_size * num_channels;
    int elements_per_block = height * width;
    
    if (idx >= num_blocks) return;
    
    int instance = idx / num_channels;
    int channel = idx % num_channels;
    
    // Use double precision for accumulation to avoid numerical instability
    double sum = 0.0;
    double sum_sq = 0.0;
    
    float conv_bias = bias[channel];
    
    // Step 1: Calculate mean and variance for this (instance, channel)
    for (int i = tid; i < elements_per_block; i += block_size) {
        int linear_idx = instance * (num_channels * height * width) + channel * (height * width) + i;
        float val = input[linear_idx] + conv_bias;
        sum += (double)val;
        sum_sq += (double)val * (double)val;
    }
    
    // Block reduce for sum and sum_sq using double precision
    __shared__ double sh_sum[256];
    __shared__ double sh_sum_sq[256];
    
    sh_sum[tid] = sum;
    sh_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_sum[tid] += sh_sum[tid + s];
            sh_sum_sq[tid] += sh_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    // Compute mean and variance using Welford-like stable computation
    double mean = sh_sum[0] / (double)elements_per_block;
    // Use the more numerically stable formula: var = E[X^2] - E[X]^2
    // But compute it carefully to avoid catastrophic cancellation
    double var = (sh_sum_sq[0] / (double)elements_per_block) - (mean * mean);
    
    // Clamp variance to be non-negative (can become slightly negative due to floating point)
    if (var < 0.0) var = 0.0;
    
    float mean_f = (float)mean;
    float inv_std = rsqrtf((float)var + 1e-5f);
    
    __syncthreads();
    
    // Step 2: Normalize and divide
    float gamma = 1.0f;  // No affine in the model
    float beta = 0.0f;
    
    for (int i = tid; i < elements_per_block; i += block_size) {
        int linear_idx = instance * (num_channels * height * width) + channel * (height * width) + i;
        float val = input[linear_idx] + conv_bias;
        
        float normalized_val = gamma * (val - mean_f) * inv_std + beta;
        output[linear_idx] = normalized_val / divide_by;
    }
}

extern "C" void conv_bias_instance_norm_divide_launcher(
    float* output, 
    const float* input, 
    const float* bias, 
    int batch_size,
    int num_channels,
    int height,
    int width,
    float divide_by,
    cudaStream_t stream
) {
    int num_blocks = batch_size * num_channels;
    int block_size = 256;  // Use 256 threads for better occupancy
    
    conv_bias_instance_norm_divide_kernel<<<num_blocks, block_size, 0, stream>>>(
        output, input, bias, batch_size, num_channels, height, width, divide_by
    );
}

"""

_cpp_sources = """
// === conv_bias_instance_norm_divide_cuda_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void conv_bias_instance_norm_divide_launcher(
    float* output, 
    const float* input, 
    const float* bias, 
    int batch_size,
    int num_channels,
    int height,
    int width,
    float divide_by,
    cudaStream_t stream
);

torch::Tensor conv_bias_instance_norm_divide_forward(torch::Tensor input, torch::Tensor bias, float divide_by) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
    
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D tensor");
    
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::empty_like(input);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_bias_instance_norm_divide_launcher(
        output.data_ptr<float>(), 
        input.data_ptr<float>(), 
        bias.data_ptr<float>(),
        batch_size,
        num_channels,
        height,
        width,
        divide_by,
        stream
    );
    
    return output;
}

void register_conv_bias_instance_norm_divide(pybind11::module& m) {
    m.def("conv_bias_instance_norm_divide_forward", &conv_bias_instance_norm_divide_forward, "Fused conv bias + instance norm + divide forward");
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["conv_bias_instance_norm_divide_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model using:
    - PyTorch's convolution (without bias)
    - Custom fused kernel for conv bias + instance norm + division
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)  # Keep for state dict compatibility
        self.divide_by = divide_by

    def forward(self, x):
        # Disable TF32 for numerical stability
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Use PyTorch's convolution without bias
            x = torch.conv2d(x, self.conv.weight, None, self.conv.stride, self.conv.padding, self.conv.dilation)
            
            # Ensure contiguous
            x = x.contiguous()
            
            # Use custom fused kernel for conv bias + instance norm + divide
            x = cuda_extension.conv_bias_instance_norm_divide_forward(x, self.conv.bias, float(self.divide_by))
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return x