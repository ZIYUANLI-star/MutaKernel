import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + scaling + residual
fused_matmul_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_matmul_residual_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    float scaling_factor,
    int batch_size,
    int in_features,
    int out_features
) {
    // Each thread block handles one output element
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        
        // Compute matmul for this output element
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Apply scaling and residual in one step: output = sum * scaling_factor + sum
        output[row * out_features + col] = sum * (scaling_factor + 1.0f);
    }
}

torch::Tensor fused_matmul_residual_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    float scaling_factor
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Configure thread blocks
    dim3 block_size(16, 16);
    dim3 grid_size(
        (batch_size + block_size.x - 1) / block_size.x,
        (out_features + block_size.y - 1) / block_size.y
    );
    
    fused_matmul_residual_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_matmul_residual_cpp_source = """
torch::Tensor fused_matmul_residual_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    float scaling_factor
);
"""

# Compile the inline CUDA code
fused_matmul_residual = load_inline(
    name="fused_matmul_residual",
    cpp_sources=fused_matmul_residual_cpp_source,
    cuda_sources=fused_matmul_residual_source,
    functions=["fused_matmul_residual_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs fused matrix multiplication, scaling, and residual addition.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_matmul_residual

    def forward(self, x):
        """
        Forward pass using fused CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.fused_op.fused_matmul_residual_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.scaling_factor
        )