import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for linear + subtract + multiply + relu
fused_operation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void fused_linear_sub_mul_relu_kernel(
    const T* __restrict__ input, 
    const T* __restrict__ weight, 
    const T* __restrict__ bias,
    T* __restrict__ output, 
    int batch_size, int in_features, int out_features,
    T subtract_value, T multiply_value) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        T sum = bias[col];
        
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Fused operations: subtract, multiply, and ReLU
        T result = (sum - subtract_value) * multiply_value;
        result = result > static_cast<T>(0) ? result : static_cast<T>(0);
        
        output[row * out_features + col] = result;
    }
}

torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float subtract_value, float multiply_value) {
    
    // Ensure tensors are contiguous
    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();
    auto bias_c = bias.contiguous();
    
    auto batch_size = input_c.size(0);
    auto in_features = input_c.size(1);
    auto out_features = weight_c.size(0);
    
    auto output = torch::empty({batch_size, out_features}, 
                               torch::TensorOptions()
                               .dtype(input_c.dtype())
                               .device(input_c.device()));
    
    // Block and grid dimensions
    dim3 block_size(32, 8);
    dim3 grid_size((out_features + block_size.x - 1) / block_size.x,
                   (batch_size + block_size.y - 1) / block_size.y);
    
    AT_DISPATCH_FLOATING_TYPES(input_c.scalar_type(), "fused_linear_sub_mul_relu_cuda", [&] {
        fused_linear_sub_mul_relu_kernel<scalar_t><<<grid_size, block_size>>>(
            input_c.data_ptr<scalar_t>(),
            weight_c.data_ptr<scalar_t>(),
            bias_c.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_features, out_features,
            static_cast<scalar_t>(subtract_value),
            static_cast<scalar_t>(multiply_value));
    });
    
    return output;
}
"""

fused_operation_cpp_source = """
torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float subtract_value, float multiply_value);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation_v2",
    cpp_sources=fused_operation_cpp_source,
    cuda_sources=fused_operation_source,
    functions=["fused_linear_sub_mul_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for linear + subtract + multiply + ReLU.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_linear_sub_mul_relu_cuda(
            x, self.linear.weight, self.linear.bias, 
            self.subtract_value, self.multiply_value)