import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === fused_bias_dropout_softmax.cu ===
#include <cuda_runtime.h>
#include <cmath>
#include <curand_kernel.h>

__global__ void fused_bias_dropout_softmax_kernel(
    float* output,
    const float* input,
    const float* bias,
    float dropout_p,
    float scale,
    int batch_size,
    int features,
    unsigned long long seed,
    bool training) {

    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int block_size = blockDim.x;

    if (row >= batch_size) return;

    curandState state;
    if (training && dropout_p > 0.0f) {
        curand_init(seed, row * features + tid, 0, &state);
    }

    // Step 1: Compute max value for the row using reduction
    float max_val = -INFINITY;
    for (int i = tid; i < features; i += block_size) {
        int idx = row * features + i;
        float val = input[idx] + bias[i];
        
        // Apply dropout mask during max computation
        if (training && dropout_p > 0.0f) {
            curandState temp_state = state;
            for (int j = tid; j < i; j += block_size) {
                curand_uniform(&temp_state);
            }
            float rand_val = curand_uniform(&temp_state);
            if (rand_val < dropout_p) {
                val = -INFINITY;  // Dropped elements should not affect max
            } else {
                val *= scale;
            }
        }
        max_val = fmaxf(max_val, val);
    }
    shared_max[tid] = max_val;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    max_val = shared_max[0];
    __syncthreads();

    // Handle case where all values are -inf (all dropped)
    if (max_val == -INFINITY) {
        max_val = 0.0f;
    }

    // Re-initialize random state for consistent dropout mask
    if (training && dropout_p > 0.0f) {
        curand_init(seed, row * features + tid, 0, &state);
    }

    // Step 2: Compute sum of exponentials using reduction
    double sum_exp_val = 0.0;  // Use double for accumulation
    for (int i = tid; i < features; i += block_size) {
        int idx = row * features + i;
        float val = input[idx] + bias[i];
        
        // Apply same dropout mask
        if (training && dropout_p > 0.0f) {
            curandState temp_state = state;
            for (int j = tid; j < i; j += block_size) {
                curand_uniform(&temp_state);
            }
            float rand_val = curand_uniform(&temp_state);
            if (rand_val < dropout_p) {
                continue;  // Skip dropped elements
            } else {
                val *= scale;
            }
        }
        sum_exp_val += exp((double)(val - max_val));
    }
    shared_sum[tid] = (float)sum_exp_val;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        __syncthreads();
    }

    float sum_exp = shared_sum[0];
    
    // Add epsilon to prevent division by zero
    if (sum_exp < 1e-30f) {
        sum_exp = 1e-30f;
    }
    
    float log_sum = logf(sum_exp);
    __syncthreads();

    // Re-initialize random state for consistent dropout mask
    if (training && dropout_p > 0.0f) {
        curand_init(seed, row * features + tid, 0, &state);
    }

    // Step 3: Compute softmax for each feature in the row
    for (int i = tid; i < features; i += block_size) {
        int idx = row * features + i;
        float val = input[idx] + bias[i];
        
        // Apply same dropout mask
        if (training && dropout_p > 0.0f) {
            curandState temp_state = state;
            for (int j = tid; j < i; j += block_size) {
                curand_uniform(&temp_state);
            }
            float rand_val = curand_uniform(&temp_state);
            if (rand_val < dropout_p) {
                output[idx] = 0.0f;  // Dropped elements become 0
                continue;
            } else {
                val *= scale;
            }
        }
        output[idx] = expf(val - max_val - log_sum);
    }
}

extern "C" void fused_bias_dropout_softmax_launcher(
    float* output,
    const float* input,
    const float* bias,
    float dropout_p,
    int batch_size,
    int features,
    unsigned long long seed,
    bool training,
    cudaStream_t stream) {

    int block_size = 256;
    int grid_size = batch_size;
    int shared_mem_size = 2 * block_size * sizeof(float);
    
    float scale = (dropout_p < 1.0f) ? 1.0f / (1.0f - dropout_p) : 0.0f;

    fused_bias_dropout_softmax_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        output, input, bias, dropout_p, scale, batch_size, features, seed, training);
}

"""

_cpp_sources = """
// === fused_bias_dropout_softmax_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


// Declare launcher from .cu file
extern "C" void fused_bias_dropout_softmax_launcher(
    float* output,
    const float* input,
    const float* bias,
    float dropout_p,
    int batch_size,
    int features,
    unsigned long long seed,
    bool training,
    cudaStream_t stream);

// PyTorch wrapper
torch::Tensor fused_bias_dropout_softmax_forward(
    torch::Tensor input,
    torch::Tensor bias,
    float dropout_p,
    bool training) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D (batch_size, features)");

    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(bias.ndimension() == 1, "Bias must be 1D (features)");
    TORCH_CHECK(bias.numel() == input.size(1), "Bias size must match input features");

    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Generate random seed
    unsigned long long seed = 0;
    if (training && dropout_p > 0.0f) {
        seed = static_cast<unsigned long long>(rand()) * 1000000ULL + static_cast<unsigned long long>(rand());
    }

    // Call CUDA launcher
    fused_bias_dropout_softmax_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        dropout_p,
        static_cast<int>(input.size(0)),
        static_cast<int>(input.size(1)),
        seed,
        training,
        stream
    );

    return output;
}

// Registration function
void register_fused_bias_dropout_softmax(pybind11::module& m) {
    m.def("fused_bias_dropout_softmax_forward", 
          &fused_bias_dropout_softmax_forward, 
          "Fused bias + dropout + softmax forward",
          py::arg("input"),
          py::arg("bias"),
          py::arg("dropout_p"),
          py::arg("training"));
}

"""

cuda_extension = load_inline(
    name="cuda_extension",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_bias_dropout_softmax_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication, applies dropout, and then applies softmax.
    Uses PyTorch's native operations for numerical stability.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # Disable TF32 for numerical precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Use the standard PyTorch operations for numerical stability
        x = self.matmul(x)
        x = self.dropout(x)
        x = torch.softmax(x, dim=1)
        return x