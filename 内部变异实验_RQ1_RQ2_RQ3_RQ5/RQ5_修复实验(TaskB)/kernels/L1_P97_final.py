import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fully fused cosine similarity loss with proper numerical stability
cosine_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template<int THREADS_PER_BLOCK, int ELEMENTS_PER_THREAD>
__global__ void cosine_loss_fused_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int batch_size,
    int feature_size
) {
    __shared__ double shared_dot[THREADS_PER_BLOCK];
    __shared__ double shared_pred_norm[THREADS_PER_BLOCK];
    __shared__ double shared_target_norm[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* pred_ptr = predictions + batch_idx * feature_size;
    const float* target_ptr = targets + batch_idx * feature_size;
    
    // Use double precision for accumulation to handle near-overflow cases
    double dot_product = 0.0;
    double norm_pred = 0.0;
    double norm_target = 0.0;
    
    // Vectorized processing for better memory bandwidth utilization
    for (int base = 0; base < feature_size; base += THREADS_PER_BLOCK * ELEMENTS_PER_THREAD) {
        int idx = base + tid * ELEMENTS_PER_THREAD;
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int element_idx = idx + i;
            if (element_idx < feature_size) {
                double pred_val = (double)pred_ptr[element_idx];
                double target_val = (double)target_ptr[element_idx];
                dot_product += pred_val * target_val;
                norm_pred += pred_val * pred_val;
                norm_target += target_val * target_val;
            }
        }
    }
    
    // Store thread results in shared memory
    shared_dot[tid] = dot_product;
    shared_pred_norm[tid] = norm_pred;
    shared_target_norm[tid] = norm_target;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
            shared_pred_norm[tid] += shared_pred_norm[tid + stride];
            shared_target_norm[tid] += shared_target_norm[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread computes final loss for this batch
    if (tid == 0) {
        const double epsilon = 1e-8;
        double dot = shared_dot[0];
        double norm_pred_sq = shared_pred_norm[0];
        double norm_target_sq = shared_target_norm[0];
        
        // Compute norms
        double norm_pred_val = sqrt(norm_pred_sq);
        double norm_target_val = sqrt(norm_target_sq);
        
        // Match PyTorch's max(norm, eps) behavior for cosine_similarity
        norm_pred_val = fmax(norm_pred_val, epsilon);
        norm_target_val = fmax(norm_target_val, epsilon);
        
        double norm_product = norm_pred_val * norm_target_val;
        
        double cosine_sim;
        // Handle overflow/underflow cases
        if (!isfinite(norm_product) || !isfinite(dot)) {
            // For overflow/nan, cosine similarity should be 0
            cosine_sim = 0.0;
        } else {
            cosine_sim = dot / norm_product;
            // Clamp to valid range for numerical stability
            cosine_sim = fmin(fmax(cosine_sim, -1.0), 1.0);
        }
        
        float loss = (float)(1.0 - cosine_sim);
        
        // Atomic add to accumulate mean
        atomicAdd(output, loss / batch_size);
    }
}

torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto feature_size = predictions.size(1);
    
    // Allocate output tensor (single value)
    auto options = torch::TensorOptions()
        .dtype(predictions.dtype())
        .device(predictions.device());
    auto output = torch::zeros({1}, options);
    
    // Choose optimal configuration based on feature size
    const int THREADS = 256;
    const int ELEMENTS_PER_THREAD = 4;
    
    dim3 grid_size(batch_size, 1, 1);
    dim3 block_size(THREADS, 1, 1);
    
    // Launch fused kernel - one block per batch element
    cosine_loss_fused_kernel<THREADS, ELEMENTS_PER_THREAD><<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        feature_size
    );
    
    return output.squeeze();
}
"""

cosine_loss_cpp_source = "torch::Tensor cosine_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code
cosine_loss = load_inline(
    name="cosine_loss_v2",
    cpp_sources=cosine_loss_cpp_source,
    cuda_sources=cosine_loss_source,
    functions=["cosine_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fully fused custom CUDA kernel for cosine similarity loss.
    Features double precision accumulation and proper numerical stability handling.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_loss = cosine_loss

    def forward(self, predictions, targets):
        # Use custom CUDA kernel for fully fused cosine similarity loss computation
        return self.cosine_loss.cosine_loss_cuda(predictions, targets)