import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_bias_gelu_softmax.cu ===
#include <cuda_runtime.h>

__device__ __forceinline__ float gelu_activation(float x) {
    const float cdf = 0.5f * (1.0f + erff(x * 0.7071067811865475f));
    return x * cdf;
}

__global__ void fused_bias_gelu_softmax_kernel(float* output, const float* matmul_out, const float* bias, int batch_size, int features) {
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (row >= batch_size) return;

    // Step 1: Initialize shared memory
    shared_max[tid] = -INFINITY;
    shared_sum[tid] = 0.0f;
    __syncthreads();

    // Step 2: Compute initial max_val for this thread's portion of the row
    float local_max = -INFINITY;
    for (int col = tid; col < features; col += num_threads) {
        float val = matmul_out[row * features + col] + bias[col];
        float gelu_val = gelu_activation(val);
        local_max = fmaxf(local_max, gelu_val);
    }

    // Step 3: Block-wise reduction for max_val
    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    float row_max = shared_max[0];
    __syncthreads();

    // Step 4: Compute initial sum_exp for this thread's portion of the row using double precision
    double local_sum = 0.0;
    for (int col = tid; col < features; col += num_threads) {
        float val = matmul_out[row * features + col] + bias[col];
        float gelu_val = gelu_activation(val);
        local_sum += (double)expf(gelu_val - row_max);
    }

    // Step 5: Block-wise reduction for sum_exp
    shared_sum[tid] = (float)local_sum;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        __syncthreads();
    }

    float row_sum = shared_sum[0];
    // Add epsilon to prevent division by zero
    float inv_row_sum = 1.0f / fmaxf(row_sum, 1e-12f);

    // Step 6: Compute output for this thread's portion of the row
    for (int col = tid; col < features; col += num_threads) {
        float val = matmul_out[row * features + col] + bias[col];
        float gelu_val = gelu_activation(val);
        float exp_val = expf(gelu_val - row_max);
        output[row * features + col] = exp_val * inv_row_sum;
    }
}

extern "C" void fused_bias_gelu_softmax_launcher(float* output, const float* matmul_out, const float* bias, int batch_size, int features, cudaStream_t stream) {
    int num_threads = 256;
    size_t shared_mem_size = 2 * num_threads * sizeof(float);
    fused_bias_gelu_softmax_kernel<<<batch_size, num_threads, shared_mem_size, stream>>>(output, matmul_out, bias, batch_size, features);
}
"""

_cpp_sources = """
// === fused_bias_gelu_softmax_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_bias_gelu_softmax_launcher(float* output, const float* matmul_out, const float* bias, int batch_size, int features, cudaStream_t stream);

torch::Tensor fused_bias_gelu_softmax_forward(torch::Tensor matmul_out, torch::Tensor bias) {
    // Input validation
    TORCH_CHECK(matmul_out.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(matmul_out.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(matmul_out.dtype() == torch::kFloat32, "Input must be float32");
    
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(bias.ndimension() == 1, "Bias must be 1D");
    TORCH_CHECK(matmul_out.size(1) == bias.size(0), "Bias size mismatch");

    auto output = torch::empty_like(matmul_out);
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call CUDA launcher
    fused_bias_gelu_softmax_launcher(
        output.data_ptr<float>(),
        matmul_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        matmul_out.size(0),
        matmul_out.size(1),
        stream
    );
    
    return output;
}

void register_fused_bias_gelu_softmax(pybind11::module& m) {
    m.def("fused_bias_gelu_softmax_forward", &fused_bias_gelu_softmax_forward, 
          "Fused bias + GELU + Softmax forward",
          py::arg("matmul_out"),
          py::arg("bias"));
}


"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_bias_gelu_softmax_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom fused kernel for bias + GELU + Softmax after Linear layer.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Initialize parameters with the same names as in the original model
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Disable TF32 for numerical precision
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Ensure input is float32 for precision
            x = x.float()
            
            # Step 1: Compute matrix multiplication using PyTorch's optimized implementation
            matmul_out = x @ self.linear.weight.t()
            
            # Ensure contiguous for CUDA kernel
            matmul_out = matmul_out.contiguous()
            bias = self.linear.bias.contiguous()
            
            # Step 2: Apply fused bias + GELU + Softmax using custom CUDA kernel
            output = cuda_extension.fused_bias_gelu_softmax_forward(matmul_out, bias)
            
            return output
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]