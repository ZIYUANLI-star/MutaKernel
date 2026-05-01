"""Synthetic CUDA kernel with softmax + layernorm patterns for C-category testing."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

// Softmax kernel with numerical stability
__global__ void softmax_kernel(const float* input, float* output, int N, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* x = input + row * D;
    float* y = output + row * D;

    // Pass 1: find max for numerical stability
    float max_val = -INFINITY;
    for (int j = 0; j < D; j++) {
        if (x[j] > max_val) max_val = x[j];
    }

    // Pass 2: compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (int j = 0; j < D; j++) {
        float shifted = x[j] - max_val;
        float e = expf(shifted);
        y[j] = e;
        sum_exp += e;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum_exp;
    for (int j = 0; j < D; j++) {
        y[j] *= inv_sum;
    }
}

// LayerNorm kernel
__global__ void layernorm_kernel(
    const float* input, const float* gamma, const float* beta,
    float* output, int N, int D
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* x = input + row * D;
    float* y = output + row * D;

    // Mean
    float mean = 0.0f;
    for (int j = 0; j < D; j++) {
        mean += x[j];
    }
    mean /= (float)D;

    // Variance
    float var = 0.0f;
    for (int j = 0; j < D; j++) {
        float diff = x[j] - mean;
        var += diff * diff;
    }
    var /= (float)D;

    // Normalize: (x - mean) * rsqrt(var + eps) * gamma + beta
    float eps = 1e-5f;
    float inv_std = rsqrtf(var + eps);
    for (int j = 0; j < D; j++) {
        y[j] = (x[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

// Attention scale computation
__device__ float attention_scale(int head_dim) {
    return 1.0f / sqrtf(static_cast<float>(head_dim));
}

// Mixed precision helper
__device__ float safe_divide(float a, float b) {
    return a / (b + 1e-8f);
}

// Batch norm with FP32 accumulation
__global__ void batchnorm_kernel(
    const float* input, const float* running_mean,
    const float* running_var, const float* weight, const float* bias,
    float* output, int N, int C, int HW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (idx >= total) return;

    int c = (idx / HW) % C;
    float x = input[idx];
    float mean = running_mean[c];
    float var = running_var[c];
    float eps = 1e-5f;

    float normalized = (x - mean) * rsqrtf(var + eps);
    output[idx] = normalized * weight[c] + bias[c];
}

// Min/max reduction with initializers
__global__ void minmax_reduce(const float* input, float* out_min, float* out_max, int N) {
    int tid = threadIdx.x;
    __shared__ float s_min[256];
    __shared__ float s_max[256];

    s_min[tid] = FLT_MAX;
    s_max[tid] = -FLT_MAX;

    int i = blockIdx.x * blockDim.x + tid;
    if (i < N) {
        s_min[tid] = input[i];
        s_max[tid] = input[i];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_min[blockIdx.x] = s_min[0];
        out_max[blockIdx.x] = s_max[0];
    }
}

// Half precision kernel with casts
__global__ void half_precision_kernel(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float val = in[i];
    float result = static_cast<float>(expf(val - 2.0f));
    out[i] = result;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N = input.size(0);
    int D = input.size(1);
    softmax_kernel<<<(N + 255) / 256, 256>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, D);
    return output;
}
"""

cpp_source = "torch::Tensor softmax_cuda(torch::Tensor input);"

module = load_inline(
    name="test_c_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = module

    def forward(self, x):
        return self.op.softmax_cuda(x)
