import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized 2D average pooling
avg_pool_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename scalar_t>
__global__ void avg_pool_2d_kernel(
    const scalar_t* input,
    scalar_t* output,
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
    const float inv_kernel_size
) {
    // Calculate output position
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int h = blockIdx.x * blockDim.y + threadIdx.y;
    const int w = threadIdx.x;
    
    if (h >= output_height || w >= output_width) return;
    
    // Calculate input start position
    const int input_h_start = h * stride_h - pad_h;
    const int input_w_start = w * stride_w - pad_w;
    const int input_h_end = min(input_h_start + kernel_h, input_height);
    const int input_w_end = min(input_w_start + kernel_w, input_width);
    const int input_h_start_clamped = max(input_h_start, 0);
    const int input_w_start_clamped = max(input_w_start, 0);
    
    // Compute average pooling
    scalar_t sum = 0.0;
    int count = 0;
    
    for (int kh = input_h_start_clamped; kh < input_h_end; ++kh) {
        for (int kw = input_w_start_clamped; kw < input_w_end; ++kw) {
            const int input_idx = ((n * channels + c) * input_height + kh) * input_width + kw;
            sum += input[input_idx];
            count++;
        }
    }
    
    // Write output
    if (count > 0) {
        const int output_idx = ((n * channels + c) * output_height + h) * output_width + w;
        output[output_idx] = sum / static_cast<scalar_t>(count);
    }
}

torch::Tensor avg_pool_2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    // Validate input
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (batch, channels, height, width)");
    TORCH_CHECK(kernel_size > 0, "Kernel size must be positive");
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    // Calculate output dimensions
    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    // Create output tensor
    auto output = torch::empty({batch_size, channels, output_height, output_width}, 
                              input.options());
    
    // Calculate inverse kernel size for normalization
    const float inv_kernel_size = 1.0f / (kernel_size * kernel_size);
    
    // Configure kernel launch parameters
    const int threads_x = 32;
    const int threads_y = 4;
    dim3 block_size(threads_x, threads_y);
    dim3 grid_size(
        (output_height + threads_y - 1) / threads_y,
        channels,
        batch_size
    );
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "avg_pool_2d_cuda", [&] {
        avg_pool_2d_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            kernel_size,
            stride,
            stride,
            padding,
            padding,
            inv_kernel_size
        );
    });
    
    cudaDeviceSynchronize();
    return output;
}
"""

avg_pool_2d_cpp_source = """
torch::Tensor avg_pool_2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code
avg_pool_2d_custom = load_inline(
    name="avg_pool_2d_custom",
    cpp_sources=avg_pool_2d_cpp_source,
    cuda_sources=avg_pool_2d_source,
    functions=["avg_pool_2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    with_cuda=True,
    extra_include_paths=[]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling with custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the optimized Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_cuda = avg_pool_2d_custom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized 2D Average Pooling to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with optimized Average Pooling applied.
        """
        # Move to CUDA if not already
        if not x.is_cuda:
            x = x.cuda()
            
        return self.avg_pool_cuda.avg_pool_2d_cuda(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding
        )