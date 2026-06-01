import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_sources = """
// === hinge_loss.cu ===
#include <cuda_runtime.h>

__device__ __forceinline__ double warp_reduce_sum(double val) {
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

__device__ __forceinline__ double block_reduce_sum(double val, volatile double* smem, int block_size) {
    int tid = threadIdx.x;
    
    // First reduce within warp
    val = warp_reduce_sum(val);
    
    // Write warp sums to shared memory
    if ((tid % 32) == 0) {
        smem[tid / 32] = val;
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
    const int block_size = 1024;

    __shared__ double smem[32];  // 32 warp sums per block

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_blocks = gridDim.x;

    double sum = 0.0;

    const long long threads = (long long)block_size * num_blocks;

    // Calculate starting index for this thread
    long long base_idx = (long long)bid * block_size + tid;

    for (long long i = base_idx; i < total_elements; i += threads) {
        // Calculate batch index
        int batch_idx = (int)(i / input_dim);
        float t = targets[batch_idx];
        float p = predictions[i];

        // Calculate hinge loss value
        double val = 1.0 - (double)p * (double)t;
        if (val > 0.0) {
            sum += val;
        }
    }

    // Reduce within block
    sum = block_reduce_sum(sum, smem, block_size);

    if (tid == 0) {
        output[bid] = sum;
    }
}

__global__ void sum_reduction_kernel(double* output, const double* block_sums, int num_blocks) {
    const int block_size = 1024;
    __shared__ double smem[32];

    const int tid = threadIdx.x;

    double sum = 0.0;

    // Read block sums
    for (int i = tid; i < num_blocks; i += block_size) {
        sum += block_sums[i];
    }

    // Reduce within block
    sum = block_reduce_sum(sum, smem, block_size);

    if (tid == 0) {
        *output = sum;
    }
}

extern "C" void fused_hinge_loss_sum_launcher(double* output, const float* predictions, const float* targets, long long total_elements, int input_dim, int num_blocks, cudaStream_t stream) {

    // Allocate temporary storage for block sums
    double* d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(double));

    // Step 1: Compute sum of each block
    fused_hinge_loss_sum_kernel<<<num_blocks, 1024, 32 * sizeof(double), stream>>>(d_block_sums, predictions, targets, total_elements, input_dim);

    // Step 2: Reduce block sums to total sum
    sum_reduction_kernel<<<1, 1024, 32 * sizeof(double), stream>>>(output, d_block_sums, num_blocks);

    // Free temporary storage
    cudaFree(d_block_sums);
}

"""

_cpp_sources = """
// === hinge_loss_binding.cpp ===
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


extern "C" void fused_hinge_loss_sum_launcher(double* output, const float* predictions, const float* targets, long long total_elements, int input_dim, int num_blocks, cudaStream_t stream);

torch::Tensor fused_hinge_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    // Input validation
    TORCH_CHECK(predictions.is_cuda(), "Predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "Targets must be a CUDA tensor");
    TORCH_CHECK(predictions.is_contiguous(), "Predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "Targets must be contiguous");
    
    // Convert to float32 if needed
    auto preds_f32 = predictions.to(torch::kFloat32);
    auto targs_f32 = targets.to(torch::kFloat32);

    // Output in double precision for accurate accumulation
    auto output = torch::empty(1, torch::TensorOptions().dtype(torch::kFloat64).device(predictions.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long total_elements = predictions.numel();
    int batch_size = targets.numel();
    int input_dim = total_elements / batch_size;
    
    // Calculate number of blocks based on total elements
    int num_blocks = min(2048, (int)((total_elements + 1023) / 1024));
    if (num_blocks < 1) num_blocks = 1;

    fused_hinge_loss_sum_launcher(
        output.data_ptr<double>(),
        preds_f32.data_ptr<float>(),
        targs_f32.data_ptr<float>(),
        total_elements,
        input_dim,
        num_blocks,
        stream
    );

    return output;
}

void register_hinge_loss(pybind11::module& m) {
    m.def("fused_hinge_loss_forward", &fused_hinge_loss_forward,
          "Fused hinge loss sum calculation",
          py::arg("predictions"),
          py::arg("targets"));
}



"""

cuda_extension = load_inline(
    name="cuda_extension",
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
        # Expand targets to match predictions shape if needed
        if predictions.dim() > 1 and targets.dim() == 1:
            # targets is (batch_size,), predictions is (batch_size, input_dim)
            pass  # The kernel handles this case
        
        # Use the fused kernel to compute sum of clamped values
        sum_val = cuda_extension.fused_hinge_loss_forward(predictions, targets)
        
        # Compute mean
        total_elements = predictions.numel()
        mean_val = sum_val / total_elements
        
        return mean_val.float().squeeze()