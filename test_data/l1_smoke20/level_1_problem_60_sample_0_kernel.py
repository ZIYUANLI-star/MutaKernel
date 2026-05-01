import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized 3D convolution
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define TILE_SIZE 16

template<typename scalar_t>
__global__ void conv3d_forward_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int width,
    const int height,
    const int depth,
    const int kernel_w,
    const int kernel_h,
    const int kernel_d,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int out_width,
    const int out_height,
    const int out_depth) {
    
    // Output position indices
    const int n = blockIdx.x;  // batch index
    const int oc = blockIdx.y * TILE_SIZE + threadIdx.y;  // output channel
    const int od = blockIdx.z;  // output depth
    
    if (oc >= out_channels) return;
    
    const int group_size = in_channels / groups;
    const int out_group_size = out_channels / groups;
    const int group_idx = oc / out_group_size;
    const int in_start = group_idx * group_size;
    
    // Loop over output spatial dimensions
    for (int oh = threadIdx.x; oh < out_height; oh += blockDim.x) {
        for (int ow = 0; ow < out_width; ow++) {
            scalar_t result = 0.0;
            
            // Loop over kernel dimensions
            for (int kd = 0; kd < kernel_d; kd++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        // Calculate input position
                        const int iw = ow * stride - padding + kw * dilation;
                        const int ih = oh * stride - padding + kh * dilation;
                        const int id = od * stride - padding + kd * dilation;
                        
                        if (iw >= 0 && iw < width && ih >= 0 && ih < height && id >= 0 && id < depth) {
                            // Loop over input channels in this group
                            for (int ic = 0; ic < group_size; ic++) {
                                const int in_ic = in_start + ic;
                                
                                // Input index
                                const int input_idx = (((n * in_channels + in_ic) * height + ih) * width + iw) * depth + id;
                                
                                // Weight index
                                const int weight_idx = (((oc * group_size + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                                
                                result += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            // Store result
            if (oc < out_channels && oh < out_height && ow < out_width && od < out_depth) {
                const int output_idx = (((n * out_channels + oc) * out_height + oh) * out_width + ow) * out_depth + od;
                output[output_idx] = result;
            }
        }
    }
}

template<typename scalar_t>
__global__ void conv3d_im2col_kernel(
    const scalar_t* input,
    scalar_t* im2col_buff,
    const int batch_size,
    const int in_channels,
    const int width,
    const int height,
    const int depth,
    const int kernel_w,
    const int kernel_h,
    const int kernel_d,
    const int stride,
    const int padding,
    const int dilation,
    const int out_width,
    const int out_height,
    const int out_depth) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_width * out_height * out_depth * kernel_w * kernel_h * kernel_d * in_channels;
    
    if (idx >= total_elements) return;
    
    // Decompose index
    int temp = idx;
    const int b = temp / (out_width * out_height * out_depth * kernel_w * kernel_h * kernel_d * in_channels);
    temp %= out_width * out_height * out_depth * kernel_w * kernel_h * kernel_d * in_channels;
    const int out_x = temp / (out_height * out_depth * kernel_w * kernel_h * kernel_d * in_channels);
    temp %= out_height * out_depth * kernel_w * kernel_h * kernel_d * in_channels;
    const int out_y = temp / (out_depth * kernel_w * kernel_h * kernel_d * in_channels);
    temp %= out_depth * kernel_w * kernel_h * kernel_d * in_channels;
    const int out_z = temp / (kernel_w * kernel_h * kernel_d * in_channels);
    temp %= kernel_w * kernel_h * kernel_d * in_channels;
    const int kx = temp / (kernel_h * kernel_d * in_channels);
    temp %= kernel_h * kernel_d * in_channels;
    const int ky = temp / (kernel_d * in_channels);
    temp %= kernel_d * in_channels;
    const int kz = temp / in_channels;
    const int c = temp % in_channels;
    
    const int in_x = out_x * stride - padding + kx * dilation;
    const int in_y = out_y * stride - padding + ky * dilation;
    const int in_z = out_z * stride - padding + kz * dilation;
    
    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height && in_z >= 0 && in_z < depth) {
        const int input_idx = (((b * in_channels + c) * height + in_y) * width + in_x) * depth + in_z;
        im2col_buff[idx] = input[input_idx];
    } else {
        im2col_buff[idx] = 0.0;
    }
}

torch::Tensor conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int width = input.size(2);
    const int height = input.size(3);
    const int depth = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_w = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_d = weight.size(4);
    
    // Calculate output dimensions
    const int out_width = (width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    const int out_height = (height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_depth = (depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_width, out_height, out_depth}, 
                              input.options());
    
    // Choose kernel based on tensor dtype
    if (input.dtype() == torch::kFloat16) {
        // Use im2col + gemm approach for better performance with half precision
        const int im2col_size = batch_size * out_width * out_height * out_depth * kernel_w * kernel_h * kernel_d * in_channels;
        auto im2col_buff = torch::zeros({im2col_size}, input.options());
        
        // Launch im2col kernel
        const int block_size = 256;
        const int num_blocks = (im2col_size + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv3d_im2col", ([&] {
            conv3d_im2col_kernel<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                im2col_buff.data_ptr<scalar_t>(),
                batch_size,
                in_channels,
                width,
                height,
                depth,
                kernel_w,
                kernel_h,
                kernel_d,
                stride,
                padding,
                dilation,
                out_width,
                out_height,
                out_depth);
        }));
        
        // Reshape and perform batched matrix multiplication
        im2col_buff = im2col_buff.view({batch_size * out_width * out_height * out_depth, 
                                       kernel_w * kernel_h * kernel_d * in_channels});
        weight = weight.view({out_channels, kernel_w * kernel_h * kernel_d * in_channels});
        
        // Perform matrix multiplication
        output = torch::matmul(im2col_buff, weight.t());
        output = output.view({batch_size, out_width, out_height, out_depth, out_channels});
        output = output.permute({0, 4, 1, 2, 3});
        
    } else {
        // Use direct convolution kernel for float32
        dim3 grid_size(batch_size, (out_channels + TILE_SIZE - 1) / TILE_SIZE, out_depth);
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward", ([&] {
            conv3d_forward_kernel<scalar_t><<<grid_size, block_size>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                in_channels,
                out_channels,
                width,
                height,
                depth,
                kernel_w,
                kernel_h,
                kernel_d,
                stride,
                padding,
                dilation,
                groups,
                out_width,
                out_height,
                out_depth);
        }));
    }
    
    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int groups);
"""

# Compile the inline CUDA code
conv3d_cuda = load_inline(
    name="conv3d_cuda",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized 3D convolution with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Validate groups
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
        
        # Initialize weight
        kernel_w, kernel_h, kernel_d = kernel_size
        self.weight = nn.Parameter(torch.randn(
            out_channels, 
            in_channels // groups, 
            kernel_w, kernel_h, kernel_d
        ))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Custom CUDA operator
        self.conv3d_cuda = conv3d_cuda
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform convolution using custom CUDA kernel
        output = self.conv3d_cuda.conv3d_forward_cuda(
            x, 
            self.weight, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        
        return output