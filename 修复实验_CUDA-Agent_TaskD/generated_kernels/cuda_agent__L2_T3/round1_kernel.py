import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_bias_mish.cu ===
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float stable_softplus(float x) {
    // Numerically stable softplus: log(1 + exp(x))
    // For large x, softplus(x) ≈ x
    // For small x, use log1p(exp(x))
    if (x > 20.0f) {
        return x;
    } else if (x < -20.0f) {
        return expf(x);
    } else {
        return log1pf(expf(x));
    }
}

__device__ __forceinline__ float mish_activation(float x) {
    // mish(x) = x * tanh(softplus(x))
    float sp = stable_softplus(x);
    float tanh_val = tanhf(sp);
    return x * tanh_val;
}

__global__ void fused_bias_mish_mish_kernel(float* output, const float* bias, const float* input, int total_elements, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;
    
    if (vec_idx + 3 < total_elements) {
        float4 in = *(float4*)(input + vec_idx);
        
        int col0 = vec_idx % out_features;
        int col1 = (vec_idx + 1) % out_features;
        int col2 = (vec_idx + 2) % out_features;
        int col3 = (vec_idx + 3) % out_features;

        float b0 = bias[col0];
        float b1 = bias[col1];
        float b2 = bias[col2];
        float b3 = bias[col3];

        float4 sum;
        sum.x = in.x + b0;
        sum.y = in.y + b1;
        sum.z = in.z + b2;
        sum.w = in.w + b3;

        // Apply mish twice to each element
        sum.x = mish_activation(sum.x);
        sum.x = mish_activation(sum.x);
        sum.y = mish_activation(sum.y);
        sum.y = mish_activation(sum.y);
        sum.z = mish_activation(sum.z);
        sum.z = mish_activation(sum.z);
        sum.w = mish_activation(sum.w);
        sum.w = mish_activation(sum.w);

        *(float4*)(output + vec_idx) = sum;
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && vec_idx + i < total_elements; i++) {
            int elem_idx = vec_idx + i;
            int col = elem_idx % out_features;
            float val = input[elem_idx] + bias[col];
            val = mish_activation(val);
            val = mish_activation(val);
            output[elem_idx] = val;
        }
    }
}

extern "C" void fused_bias_mish_mish_launcher(float* output, const float* bias, const float* input, int total_elements, int out_features, cudaStream_t stream) {
    int threads = 256;
    int elements_per_block = threads * 4;
    int blocks = (total_elements + elements_per_block - 1) / elements_per_block;
    fused_bias_mish_mish_kernel<<<blocks, threads, 0, stream>>>(output, bias, input, total_elements, out_features);
}

"""

_cpp_sources = """
// === fused_bias_mish_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_bias_mish_mish_launcher(float* output, const float* bias, const float* input, int total_elements, int out_features, cudaStream_t stream);

torch::Tensor fused_bias_mish_mish_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");

    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");

    auto output = torch::empty_like(input);

    int total_elements = input.numel();
    int out_features = bias.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    fused_bias_mish_mish_launcher(output.data_ptr<float>(), bias.data_ptr<float>(), input.data_ptr<float>(), total_elements, out_features, stream);

    return output;
}

void register_fused_bias_mish(pybind11::module& m) {
    m.def("fused_bias_mish_mish_forward", &fused_bias_mish_mish_forward, "Fused bias + mish + mish forward");
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_bias_mish_mish_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Disable TF32 for numerical precision
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Use PyTorch's matrix multiplication (without TF32)
            x = x @ self.linear.weight.T
            
            # Use custom fused kernel for bias + mish + mish
            x = cuda_extension.fused_bias_mish_mish_forward(x.contiguous(), self.linear.bias.contiguous())
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return x