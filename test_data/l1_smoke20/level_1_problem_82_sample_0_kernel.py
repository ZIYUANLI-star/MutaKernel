import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for depthwise convolution with fused ReLU
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

template<typename T>
__global__ void depthwise_conv2d_kernel(
    const T* input,
    const T* weight,
    T* output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const bool use_bias,
    const T* bias) {
    
    const int output_size = output_height * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * channels * output_size) return;
    
    const int batch_idx = idx / (channels * output_size);
    const int channel_idx = (idx / output_size) % channels;
    const int output_idx = idx % output_size;
    
    const int oh = output_idx / output_width;
    const int ow = output_idx % output_width;
    
    const T* input_ptr = input + batch_idx * channels * input_height * input_width + 
                         channel_idx * input_height * input_width;
    const T* weight_ptr = weight + channel_idx * kernel_size * kernel_size;
    
    T acc = T(0);
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int ih = oh * stride + kh - padding;
            const int iw = ow * stride + kw - padding;
            
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const T input_val = input_ptr[ih * input_width + iw];
                const T weight_val = weight_ptr[kh * kernel_size + kw];
                acc += input_val * weight_val;
            }
        }
    }
    
    if (use_bias) {
        acc += bias[channel_idx];
    }
    
    // Fused ReLU activation
    output[idx] = acc > T(0) ? acc : T(0);
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    bool use_bias) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int kernel_size = weight.size(2);
    
    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    const int output_size = output_height * output_width;
    
    auto output = torch::zeros({batch_size, channels, output_height, output_width}, 
                               input.options());
    
    const int total_threads = batch_size * channels * output_size;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            kernel_size,
            output_height,
            output_width,
            stride,
            padding,
            use_bias,
            use_bias ? bias.data_ptr<scalar_t>() : nullptr);
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    bool use_bias);
"""

# Compile the inline CUDA code
depthwise_conv_module = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution with custom CUDA kernel and fused ReLU.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        
        # Initialize weight parameters
        self.weight = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size, kernel_size) * 0.01
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
            
        # Custom CUDA operator
        self.depthwise_conv = depthwise_conv_module
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized depthwise 2D convolution with fused ReLU.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor with fused ReLU activation.
        """
        if self.bias is not None:
            bias_tensor = self.bias
        else:
            # Create dummy tensor for kernel compatibility
            bias_tensor = torch.zeros(self.in_channels, device=x.device, dtype=x.dtype)
            
        return self.depthwise_conv.depthwise_conv2d_cuda(
            x, 
            self.weight, 
            bias_tensor,
            self.stride,
            self.padding,
            self.use_bias
        )