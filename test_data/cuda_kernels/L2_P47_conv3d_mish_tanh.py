import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Conv3D + Mish + Tanh CUDA kernel
fused_conv3d_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define MISH_THRESHOLD 20.0f

__device__ float mish_activation(float x) {
    if (x > MISH_THRESHOLD) {
        return x;
    } else if (x < -MISH_THRESHOLD) {
        return 0.0f;
    }
    return x * tanhf(log1pf(expf(x)));
}

__device__ float tanh_activation(float x) {
    return tanhf(x);
}

__global__ void fused_conv3d_mish_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int D, const int H, const int W,
    const int kernel_D, const int kernel_H, const int kernel_W,
    const int stride_D, const int stride_H, const int stride_W,
    const int pad_D, const int pad_H, const int pad_W,
    const int out_D, const int out_H, const int out_W
) {
    // Calculate output position
    const int n = blockIdx.x;  // batch index
    const int oc = blockIdx.y; // output channel
    const int od = threadIdx.x; // output depth
    const int oh = threadIdx.y; // output height
    const int ow = threadIdx.z; // output width
    
    if (n >= batch_size || oc >= out_channels || 
        od >= out_D || oh >= out_H || ow >= out_W) {
        return;
    }
    
    float sum = 0.0f;
    
    // Convolution operation
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_D; kd++) {
            for (int kh = 0; kh < kernel_H; kh++) {
                for (int kw = 0; kw < kernel_W; kw++) {
                    int id = od * stride_D + kd - pad_D;
                    int ih = oh * stride_H + kh - pad_H;
                    int iw = ow * stride_W + kw - pad_W;
                    
                    if (id >= 0 && id < D && ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        int input_idx = ((n * in_channels + ic) * D + id) * H * W + ih * W + iw;
                        int weight_idx = ((oc * in_channels + ic) * kernel_D + kd) * kernel_H * kernel_W + 
                                        kh * kernel_W + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    // Apply Mish activation
    sum = mish_activation(sum);
    
    // Apply Tanh activation
    sum = tanh_activation(sum);
    
    // Write output
    int output_idx = ((n * out_channels + oc) * out_D + od) * out_H * out_W + oh * out_W + ow;
    output[output_idx] = sum;
}

torch::Tensor fused_conv3d_mish_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_D, int stride_H, int stride_W,
    int pad_D, int pad_H, int pad_W
) {
    // Get input dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_D = weight.size(2);
    const int kernel_H = weight.size(3);
    const int kernel_W = weight.size(4);
    
    // Calculate output dimensions
    const int out_D = (D + 2 * pad_D - kernel_D) / stride_D + 1;
    const int out_H = (H + 2 * pad_H - kernel_H) / stride_H + 1;
    const int out_W = (W + 2 * pad_W - kernel_W) / stride_W + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_D, out_H, out_W}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Configure kernel launch
    dim3 threads_per_block(out_D, out_H, out_W);
    dim3 num_blocks(batch_size, out_channels, 1);
    
    // Launch kernel
    fused_conv3d_mish_tanh_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        D, H, W,
        kernel_D, kernel_H, kernel_W,
        stride_D, stride_H, stride_W,
        pad_D, pad_H, pad_W,
        out_D, out_H, out_W
    );
    
    return output;
}
"""

fused_conv3d_mish_tanh_cpp_source = """
torch::Tensor fused_conv3d_mish_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_D, int stride_H, int stride_W,
    int pad_D, int pad_H, int pad_W
);
"""

# Compile the inline CUDA code
fused_conv3d_mish_tanh = load_inline(
    name="fused_conv3d_mish_tanh",
    cpp_sources=fused_conv3d_mish_tanh_cpp_source,
    cuda_sources=fused_conv3d_mish_tanh_source,
    functions=["fused_conv3d_mish_tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs fused 3D convolution with Mish and Tanh activations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        
        # Handle stride as int or tuple
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride
        
        # Handle padding as int or tuple
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.padding = padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, 
            kernel_size[0], kernel_size[1], kernel_size[2]
        ))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights using Kaiming initialization for better training
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='linear')
        
        # Load the custom CUDA operator
        self.fused_op = fused_conv3d_mish_tanh

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        return self.fused_op.fused_conv3d_mish_tanh_cuda(
            x,
            self.weight,
            self.bias,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2]
        )