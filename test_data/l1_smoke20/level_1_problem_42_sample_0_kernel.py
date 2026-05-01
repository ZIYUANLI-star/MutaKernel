import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized MaxPool2d
maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename scalar_t>
__global__ void maxpool2d_forward_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t* indices,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w) {
    
    const int n = blockIdx.x;
    const int c = blockIdx.y;
    const int oh = blockIdx.z / output_width;
    const int ow = blockIdx.z % output_width;
    
    if (n >= batch_size || c >= channels || oh >= output_height || ow >= output_width) {
        return;
    }
    
    scalar_t max_val = -FLT_MAX;
    int64_t max_index = -1;
    
    const int h_start = oh * stride_h - pad_h;
    const int w_start = ow * stride_w - pad_w;
    
    for (int kh = 0; kh < kernel_h; kh++) {
        const int h = h_start + kh * dilation_h;
        if (h < 0 || h >= input_height) continue;
        
        for (int kw = 0; kw < kernel_w; kw++) {
            const int w = w_start + kw * dilation_w;
            if (w < 0 || w >= input_width) continue;
            
            const int input_idx = ((n * channels + c) * input_height + h) * input_width + w;
            const scalar_t val = input[input_idx];
            
            if (val > max_val) {
                max_val = val;
                max_index = input_idx;
            }
        }
    }
    
    const int output_idx = ((n * channels + c) * output_height + oh) * output_width + ow;
    output[output_idx] = max_val;
    
    if (indices != nullptr) {
        indices[output_idx] = max_index;
    }
}

torch::Tensor maxpool2d_forward_cuda(
    torch::Tensor input,
    torch::IntArrayRef kernel_size,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    bool return_indices) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int kernel_h = kernel_size[0];
    const int kernel_w = kernel_size.size() > 1 ? kernel_size[1] : kernel_size[0];
    const int stride_h = stride[0];
    const int stride_w = stride.size() > 1 ? stride[1] : stride[0];
    const int pad_h = padding[0];
    const int pad_w = padding.size() > 1 ? padding[1] : padding[0];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation.size() > 1 ? dilation[1] : dilation[0];
    
    const int output_height = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int output_width = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, 
                              input.options());
    
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, channels, output_height, output_width}, 
                              torch::dtype(torch::kLong).device(input.device()));
    }
    
    const int blocks_x = batch_size;
    const int blocks_y = channels;
    const int blocks_z = output_height * output_width;
    
    dim3 grid(blocks_x, blocks_y, blocks_z);
    const int threads = 1;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "maxpool2d_forward_cuda", [&] {
        maxpool2d_forward_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w);
    });
    
    cudaDeviceSynchronize();
    
    if (return_indices) {
        return torch::cat({output.unsqueeze(-1), indices.unsqueeze(-1)}, -1);
    }
    
    return output;
}
"""

maxpool2d_cpp_source = """
torch::Tensor maxpool2d_forward_cuda(
    torch::Tensor input,
    torch::IntArrayRef kernel_size,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    bool return_indices);
"""

# Compile the inline CUDA code
maxpool2d_cuda = load_inline(
    name="maxpool2d_cuda",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["maxpool2d_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D with custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the optimized Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.maxpool_cuda = maxpool2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        if x.is_cuda:
            return self.maxpool_cuda.maxpool2d_forward_cuda(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                False
            )
        else:
            # Fallback to PyTorch implementation for CPU
            return torch.nn.functional.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation
            )