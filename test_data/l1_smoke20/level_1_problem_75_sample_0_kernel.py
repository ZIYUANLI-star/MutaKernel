import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel for grouped transposed convolution with dilation
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
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const scalar_t* bias) {
    
    const int group_size = out_channels / groups;
    const int group_in_channels = in_channels / groups;
    
    // Each thread handles one output pixel for one output channel
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int ow = blockIdx.z;
    const int n = blockIdx.w;
    
    if (n >= batch_size || oc >= out_channels || oh >= output_height || ow >= output_width) {
        return;
    }
    
    const int group_idx = oc / group_size;
    const int group_oc = oc % group_size;
    
    scalar_t value = 0;
    
    // Calculate input window for this output position
    const int ih_start = oh - padding_h;
    const int iw_start = ow - padding_w;
    
    // Iterate over kernel positions
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            // Calculate input position considering stride and dilation
            int ih = ih_start + kh * dilation_h;
            int iw = iw_start + kw * dilation_w;
            
            // Check if input position is valid
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                // Check if this position contributes to output (stride alignment)
                if (ih % stride_h == 0 && iw % stride_w == 0) {
                    
                    // Adjust indices for stride
                    ih = ih / stride_h;
                    iw = iw / stride_w;
                    
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        // Iterate over input channels in this group
                        for (int ic = 0; ic < group_in_channels; ic++) {
                            int input_idx = ((n * in_channels + group_idx * group_in_channels + ic) * input_height + ih) * input_width + iw;
                            int weight_idx = ((group_idx * group_size + group_oc) * group_in_channels + ic) * kernel_h * kernel_w + kh * kernel_w + kw;
                            
                            value += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias if present
    if (bias != nullptr) {
        value += bias[oc];
    }
    
    // Write output
    int output_idx = ((n * out_channels + oc) * output_height + oh) * output_width + ow;
    output[output_idx] = value;
}

// Main optimized transposed convolution function
torch::Tensor conv_transpose2d_optimized_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) {
        CHECK_INPUT(bias);
    }
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int padding_h = padding[0];
    const int padding_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    // Calculate output dimensions
    const int output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    const int output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                              input.options());
    
    // Choose optimal block configuration
    dim3 block_size(32, 8, 1);
    dim3 grid_size(
        (out_channels + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        output_width,
        batch_size
    );
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_kernel", ([&] {
        const scalar_t* bias_ptr = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;
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
            dilation_h,
            dilation_w,
            groups,
            bias_ptr
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = """
#include <torch/extension.h>

torch::Tensor conv_transpose2d_optimized_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    int groups);
"""

# Compile the inline CUDA code
try:
    conv_transpose2d_optimized = load_inline(
        name="conv_transpose2d_optimized",
        cpp_sources=conv_transpose2d_cpp_source,
        cuda_sources=conv_transpose2d_source,
        functions=["conv_transpose2d_optimized_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        with_cuda=True
    )
except:
    # Fallback to PyTorch implementation if compilation fails
    conv_transpose2d_optimized = None

class ModelNew(nn.Module):
    """
    Optimized 2D transposed convolution with custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
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
            raise ValueError(f'in_channels must be divisible by groups. Got in_channels={in_channels}, groups={groups}')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels must be divisible by groups. Got out_channels={out_channels}, groups={groups}')
        
        # Initialize weight parameters (note: ConvTranspose2d weight shape is different)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        )
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', torch.tensor([]))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Store whether CUDA kernel is available
        self.use_custom_kernel = conv_transpose2d_optimized is not None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized 2D transposed convolution using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        if self.use_custom_kernel and x.is_cuda:
            # Ensure tensors are on CUDA
            x = x.contiguous()
            weight = self.weight.contiguous()
            
            # Handle bias tensor properly
            if self.bias.numel() > 0:
                bias = self.bias.contiguous().cuda()
            else:
                bias = torch.tensor([], device='cuda', dtype=x.dtype)
            
            return conv_transpose2d_optimized.conv_transpose2d_optimized_cuda(
                x, weight, bias, 
                list(self.stride), 
                list(self.padding), 
                list(self.dilation), 
                self.groups
            )
        else:
            # Fallback to PyTorch implementation
            return F.conv_transpose2d(
                x, self.weight, self.bias if self.bias.numel() > 0 else None,
                stride=self.stride, padding=self.padding, 
                dilation=self.dilation, groups=self.groups
            )