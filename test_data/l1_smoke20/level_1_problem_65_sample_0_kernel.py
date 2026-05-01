import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

template<typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int output_padding_h,
    const int output_padding_w,
    const int groups) {
    
    const int group_size = out_channels / groups;
    const int group_in_channels = in_channels / groups;
    
    // Each thread block processes one output position across multiple channels
    const int oh = blockIdx.y;
    const int ow = blockIdx.x;
    const int g = blockIdx.z % groups;
    const int b = blockIdx.z / groups;
    
    if (b >= batch_size || oh >= output_height || ow >= output_width) return;
    
    // Each thread processes multiple output channels
    for (int oc_local = threadIdx.x; oc_local < group_size; oc_local += blockDim.x) {
        const int oc = g * group_size + oc_local;
        scalar_t result = 0.0;
        
        // Loop over input channels
        for (int ic = 0; ic < group_in_channels; ic++) {
            // Loop over kernel positions
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Calculate input position for transposed convolution
                    // For transposed convolution: input position = (output_pos + padding - kernel_pos) / stride
                    const int ih = oh + padding_h - kh;
                    const int iw = ow + padding_w - kw;
                    
                    // Check if this input position contributes to the output
                    if (ih >= 0 && iw >= 0 && ih % stride_h == 0 && iw % stride_w == 0) {
                        const int ih_div = ih / stride_h;
                        const int iw_div = iw / stride_w;
                        
                        if (ih_div < input_height && iw_div < input_width) {
                            const int input_idx = ((b * in_channels + g * group_in_channels + ic) * input_height + ih_div) * input_width + iw_div;
                            // Weight shape: [in_channels, out_channels/groups, kernel_h, kernel_w] for transposed conv
                            const int weight_idx = ((ic * group_size + oc_local) * kernel_h + kh) * kernel_w + kw;
                            
                            result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Write result to output
        const int out_idx = ((b * out_channels + oc) * output_height + oh) * output_width + ow;
        output[out_idx] = result;
    }
}

template<typename scalar_t>
__global__ void conv_transpose2d_add_bias_kernel(
    scalar_t* output,
    const scalar_t* bias,
    const int batch_size,
    const int out_channels,
    const int output_height,
    const int output_width) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx < total_elements) {
        const int oc = (idx / (output_height * output_width)) % out_channels;
        output[idx] += bias[oc];
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    
    const auto out_channels = weight.size(1) * groups;  // weight shape: [in_channels, out_channels/groups, k_h, k_w]
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    
    // Output dimensions calculation for transposed convolution
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                              input.options());
    
    // Configure kernel launch parameters
    const dim3 block_size(256);
    const dim3 grid_size(output_width, output_height, batch_size * groups);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            groups);
    }));
    
    cudaDeviceSynchronize();
    
    // Add bias if provided
    if (bias.defined() && bias.numel() > 0) {
        const int total_elements = batch_size * out_channels * output_height * output_width;
        const int block_size_bias = 256;
        const int num_blocks_bias = (total_elements + block_size_bias - 1) / block_size_bias;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv_transpose2d_add_bias_cuda", ([&] {
            conv_transpose2d_add_bias_kernel<scalar_t><<<num_blocks_bias, block_size_bias>>>(
                output.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                batch_size,
                out_channels,
                output_height,
                output_width);
        }));
    }
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups);
"""

# Compile the inline CUDA code
conv_transpose2d_cuda_op = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized transposed 2D convolution with custom CUDA kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Convert parameters to tuples if they are integers
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        
        # Validate parameters
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
        
        # Initialize weight parameters - Correct shape for ConvTranspose2d
        # Weight shape: [in_channels, out_channels/groups, kernel_h, kernel_w]
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            self.kernel_size[0],
            self.kernel_size[1]
        ))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
        # Load custom CUDA operator
        self.conv_transpose2d_cuda = conv_transpose2d_cuda_op
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Ensure tensors are on CUDA
        if not x.is_cuda:
            x = x.cuda()
            self.weight.data = self.weight.data.cuda()
            if self.bias is not None:
                self.bias.data = self.bias.data.cuda()
        
        # Use custom CUDA kernel
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        
        return self.conv_transpose2d_cuda.conv_transpose2d_cuda(
            x,
            self.weight,
            bias_tensor,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.output_padding[0],
            self.output_padding[1],
            self.groups
        )