import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for optimized triplet margin loss with fused operations
triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int ELEMENTS_PER_THREAD = 8;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void triplet_margin_loss_fused_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    float margin,
    int batch_size,
    int feature_dim
) {
    __shared__ float shared_pos[32];
    __shared__ float shared_neg[32];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (batch_idx >= batch_size) return;
    
    const float* anchor_ptr = anchor + batch_idx * feature_dim;
    const float* positive_ptr = positive + batch_idx * feature_dim;
    const float* negative_ptr = negative + batch_idx * feature_dim;
    
    // Vectorized distance computation with fused sqrt and margin
    float pos_dist = 0.0f;
    float neg_dist = 0.0f;
    
    // Process multiple elements per thread for better memory coalescing
    for (int i = tid * ELEMENTS_PER_THREAD; i < feature_dim; i += blockDim.x * ELEMENTS_PER_THREAD) {
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            int idx = i + j;
            if (idx < feature_dim) {
                float diff_pos = anchor_ptr[idx] - positive_ptr[idx];
                float diff_neg = anchor_ptr[idx] - negative_ptr[idx];
                pos_dist += diff_pos * diff_pos;
                neg_dist += diff_neg * diff_neg;
            }
        }
    }
    
    // Warp-level reduction
    pos_dist = warp_reduce_sum(pos_dist);
    neg_dist = warp_reduce_sum(neg_dist);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        shared_pos[warp_id] = pos_dist;
        shared_neg[warp_id] = neg_dist;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        float final_pos = lane_id < (blockDim.x / 32) ? shared_pos[lane_id] : 0.0f;
        float final_neg = lane_id < (blockDim.x / 32) ? shared_neg[lane_id] : 0.0f;
        
        final_pos = warp_reduce_sum(final_pos);
        final_neg = warp_reduce_sum(final_neg);
        
        // Thread 0 computes final loss with sqrt and margin
        if (lane_id == 0) {
            float pos_sqrt = sqrtf(final_pos);
            float neg_sqrt = sqrtf(final_neg);
            float loss = fmaxf(0.0f, pos_sqrt - neg_sqrt + margin);
            output[batch_idx] = loss;
        }
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
) {
    // Input validation
    TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");
    TORCH_CHECK(anchor.dtype() == torch::kFloat32, "anchor must be float32");
    TORCH_CHECK(positive.dtype() == torch::kFloat32, "positive must be float32");
    TORCH_CHECK(negative.dtype() == torch::kFloat32, "negative must be float32");
    
    auto batch_size = anchor.size(0);
    auto feature_dim = anchor.numel() / batch_size;
    
    TORCH_CHECK(positive.sizes() == anchor.sizes(),
                "positive must have same shape as anchor");
    TORCH_CHECK(negative.sizes() == anchor.sizes(),
                "negative must have same shape as anchor");
    
    // Allocate output buffer
    auto output = torch::empty({batch_size}, anchor.options());
    
    // Launch fused kernel: computes distances, sqrt, and margin in one pass
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    triplet_margin_loss_fused_kernel<<<grid, block>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feature_dim
    );
    
    // Return mean of all losses
    return output.mean();
}
"""

triplet_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin
);
"""

# Compile the inline CUDA code
triplet_loss_module = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-ffast-math"],
    extra_ldflags=[""],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss with fully fused custom CUDA kernel.
    
    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss_cuda = triplet_loss_module.triplet_margin_loss_cuda

    def forward(self, anchor, positive, negative):
        # Ensure tensors are on CUDA and have correct dtype
        anchor = anchor.contiguous().cuda().float()
        positive = positive.contiguous().cuda().float()
        negative = negative.contiguous().cuda().float()
        
        return self.triplet_loss_cuda(anchor, positive, negative, self.margin)