import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void fused_ops_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * spatial_size;
    if (idx >= total_elements) return;
    
    int batch_idx = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    
    int base_input_idx = batch_idx * channels * spatial_size + spatial_idx;
    
    // Step 1: LogSumExp with numerical stability - find max
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        float val = input[base_input_idx + c * spatial_size];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    // Step 2: Sum of exponentials
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; c++) {
        float val = input[base_input_idx + c * spatial_size];
        float diff = val - max_val;
        // Clamp to avoid underflow in exp
        if (diff > -88.0f) {
            sum_exp += expf(diff);
        }
    }
    
    // Step 3: Complete LogSumExp
    float logsumexp_result;
    if (sum_exp > 0.0f) {
        logsumexp_result = logf(sum_exp) + max_val;
    } else {
        logsumexp_result = max_val;
    }
    
    // Handle potential inf/nan
    if (isnan(logsumexp_result)) {
        logsumexp_result = max_val;
    }
    
    // Step 4: HardSwish: x * sigmoid(x + 3) / 6
    float x_plus_3 = logsumexp_result + 3.0f;
    float sigmoid_val;
    if (x_plus_3 > 20.0f) {
        sigmoid_val = 1.0f;
    } else if (x_plus_3 < -20.0f) {
        sigmoid_val = 0.0f;
    } else {
        sigmoid_val = 1.0f / (1.0f + expf(-x_plus_3));
    }
    float hardswish_result = logsumexp_result * sigmoid_val / 6.0f;
    
    // Handle nan in hardswish result
    if (isnan(hardswish_result)) {
        hardswish_result = 0.0f;
    }
    
    // Step 5, 6, 7: Subtract each bias, clamp to [-1, 1], and find max
    float max_final = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        float biased = hardswish_result - bias[c];
        float clamped = fminf(fmaxf(biased, -1.0f), 1.0f);
        if (clamped > max_final) {
            max_final = clamped;
        }
    }
    
    output[idx] = max_final;
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    int spatial_size = depth * height * width;
    
    auto output = torch::empty({batch_size, 1, depth, height, width}, 
                               input.options());
    
    int total_elements = batch_size * spatial_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_ops_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias);"

fused_ops = load_inline(
    name="fused_ops_fixed",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for operations after conv transpose.
    Operations: LogSumExp -> HardSwish -> bias subtraction -> clamp -> max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        x = self.fused_ops.fused_ops_cuda(x, bias_flat)
        return x