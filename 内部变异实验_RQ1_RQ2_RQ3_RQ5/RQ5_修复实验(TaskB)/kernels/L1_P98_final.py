import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define optimized CUDA kernel for fused KL divergence
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template<typename scalar_t>
__global__ void fused_kl_div_kernel(
    const scalar_t* __restrict__ predictions,
    const scalar_t* __restrict__ targets,
    scalar_t* __restrict__ output,
    int batch_size,
    int seq_len
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const scalar_t* pred_batch = predictions + batch_idx * seq_len;
    const scalar_t* target_batch = targets + batch_idx * seq_len;
    
    // Thread block and warp setup
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    float thread_sum = 0.0f;
    
    // Process elements with stride equal to total threads in block
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    for (int idx = tid; idx < seq_len; idx += total_threads) {
        float p = static_cast<float>(pred_batch[idx]);
        float t = static_cast<float>(target_batch[idx]);
        
        // KL divergence matching PyTorch's kl_div:
        // PyTorch kl_div(log_input, target) = target * (log(target) - log_input)
        // Since we get predictions (not log), we compute:
        // target * (log(target) - log(predictions))
        // For target <= 0, contribution is 0 (by convention 0*log(0) = 0)
        if (t > 0.0f) {
            float log_p = (p > 0.0f) ? logf(p) : -1e10f;  // Handle p=0 case
            float log_t = logf(t);
            thread_sum += t * (log_t - log_p);
        }
    }
    
    // Warp reduction
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    
    // First thread in warp writes to shared memory
    __shared__ float block_sums[32];
    if (warp.thread_rank() == 0) {
        block_sums[warp.meta_group_rank()] = warp_sum;
    }
    block.sync();
    
    // First warp reduces all warp sums
    if (warp.meta_group_rank() == 0) {
        float block_sum = 0.0f;
        int num_warps = (total_threads + 31) / 32;
        if (warp.thread_rank() < num_warps) {
            block_sum = block_sums[warp.thread_rank()];
        }
        
        float final_sum = cg::reduce(warp, block_sum, cg::plus<float>());
        
        if (warp.thread_rank() == 0) {
            // Store per-batch result
            output[batch_idx] = static_cast<scalar_t>(final_sum);
        }
    }
}

torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure inputs are on GPU and contiguous
    predictions = predictions.contiguous();
    targets = targets.contiguous();
    
    auto batch_size = predictions.size(0);
    auto seq_len = predictions.size(1);
    
    // Create output tensor for per-batch results
    auto output = torch::zeros({batch_size}, predictions.options());
    
    // Optimized kernel launch configuration
    int threads = 256;
    int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "fused_kl_div_kernel", ([&] {
        fused_kl_div_kernel<scalar_t><<<blocks, threads>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len
        );
    }));
    
    // Return batchmean reduction (sum over all elements / batch_size)
    return output.mean();
}
"""

kl_div_cpp_source = """
torch::Tensor fused_kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code
fused_kl_div = load_inline(
    name="fused_kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["fused_kl_div_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using fused CUDA kernels.
    Matches PyTorch's torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_kl_div = fused_kl_div
        
    def forward(self, predictions, targets):
        # Input validation and preprocessing
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        # Only clamp predictions to avoid log(0), but DO NOT normalize
        # This matches PyTorch's behavior which takes raw inputs
        predictions = torch.clamp(predictions, min=1e-12)
        
        # Use fused CUDA kernel
        return self.fused_kl_div.fused_kl_div_cuda(predictions, targets)