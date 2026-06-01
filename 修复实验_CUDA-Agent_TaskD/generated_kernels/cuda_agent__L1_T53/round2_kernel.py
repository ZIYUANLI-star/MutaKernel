import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
#include <cuda_runtime.h>
#include <cstdio>

__device__ __forceinline__ double warp_reduce_sum(double val) {
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

__device__ __forceinline__ double block_reduce_sum(double val, double* smem, int block_size) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // First reduce within warp
    val = warp_reduce_sum(val);
    
    // Write warp sums to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    // Now reduce the warp sums
    int num_warps = block_size / 32;
    if (tid < 32) {
        val = (tid < num_warps) ? smem[tid] : 0.0;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

__global__ void fused_hinge_loss_sum_kernel(double* output, const float* predictions, const float* targets, long long total_elements, int input_dim) {
    const int block_size = 256;

    __shared__ double smem[32];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_blocks = gridDim.x;

    double sum = 0.0;

    const long long threads = (long long)block_size * num_blocks;

    long long base_idx = (long long)bid * block_size + tid;

    for (long long i = base_idx; i < total_elements; i += threads) {
        int batch_idx = (int)(i / input_dim);
        double t = (double)targets[batch_idx];
        double p = (double)predictions[i];

        double val = 1.0 - p * t;
        if (val > 0.0) {
            sum += val;
        }
    }

    sum = block_reduce_sum(sum, smem, block_size);

    if (tid == 0) {
        output[bid] = sum;
    }
}

__global__ void sum_reduction_kernel(double* output, const double* block_sums, int num_blocks) {
    const int block_size = 256;
    __shared__ double smem[32];

    const int tid = threadIdx.x;

    double sum = 0.0;

    for (int i = tid; i < num_blocks; i += block_size) {
        sum += block_sums[i];
    }

    sum = block_reduce_sum(sum, smem, block_size);

    if (tid == 0) {
        *output = sum;
    }
}

extern "C" void fused_hinge_loss_sum_launcher(double* output, const float* predictions, const float* targets, long long total_elements, int input_dim, int num_blocks, double* d_block_sums, cudaStream_t stream) {

    fused_hinge_loss_sum_kernel<<<num_blocks, 256, 0, stream>>>(d_block_sums, predictions, targets, total_elements, input_dim);

    sum_reduction_kernel<<<1, 256, 0, stream>>>(output, d_block_sums, num_blocks);
}

"""

_cpp_sources = """
#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>

extern "C" void fused_hinge_loss_sum_launcher(double* output, const float* predictions, const float* targets, long long total_elements, int input_dim, int num_blocks, double* d_block_sums, cudaStream_t stream);

torch::Tensor fused_hinge_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "Predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "Targets must be a CUDA tensor");
    
    auto preds_contig = predictions.contiguous();
    auto targs_contig = targets.contiguous();
    
    auto preds_f32 = preds_contig.to(torch::kFloat32);
    auto targs_f32 = targs_contig.to(torch::kFloat32);

    auto output = torch::empty(1, torch::TensorOptions().dtype(torch::kFloat64).device(predictions.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long total_elements = predictions.numel();
    int batch_size = targets.numel();
    int input_dim = total_elements / batch_size;
    
    int num_blocks = std::min(1024, (int)((total_elements + 255) / 256));
    if (num_blocks < 1) num_blocks = 1;

    auto block_sums = torch::empty(num_blocks, torch::TensorOptions().dtype(torch::kFloat64).device(predictions.device()));

    fused_hinge_loss_sum_launcher(
        output.data_ptr<double>(),
        preds_f32.data_ptr<float>(),
        targs_f32.data_ptr<float>(),
        total_elements,
        input_dim,
        num_blocks,
        block_sums.data_ptr<double>(),
        stream
    );

    return output;
}

"""

cuda_extension = load_inline(
    name="cuda_extension_hinge",
    cpp_sources=[_cpp_sources],
    cuda_sources=[_cuda_sources],
    functions=["fused_hinge_loss_forward"],
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized Hinge Loss model using fused CUDA kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Disable TF32 for numerical precision
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            sum_val = cuda_extension.fused_hinge_loss_forward(predictions, targets)
            
            total_elements = predictions.numel()
            mean_val = sum_val / float(total_elements)
            
            return mean_val.float().squeeze()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32