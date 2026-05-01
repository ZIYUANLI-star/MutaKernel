import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for parallel prefix sum
scan_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void inclusive_scan_kernel(T* data, int n) {
    extern __shared__ T temp[];
    
    int tid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    int ai = tid * 2;
    int bi = tid * 2 + 1;
    
    if (ai < n) temp[ai] = data[ai];
    if (bi < n) temp[bi] = data[bi];
    
    __syncthreads();
    
    // Up-sweep phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    
    // Down-sweep phase
    if (tid == 0) {
        temp[n - 1] = 0;
    }
    
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    // Write results back to global memory
    if (ai < n) data[ai] = temp[ai];
    if (bi < n) data[bi] = temp[bi];
}

template<typename T>
__global__ void row_scan_kernel(const T* input, T* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    extern __shared__ T shared[];
    
    // Load row into shared memory
    for (int i = tid; i < cols; i += blockDim.x) {
        shared[i] = input[row * cols + i];
    }
    __syncthreads();
    
    // Perform sequential scan in shared memory
    if (tid == 0) {
        for (int i = 1; i < cols; i++) {
            shared[i] += shared[i - 1];
        }
    }
    __syncthreads();
    
    // Store results
    for (int i = tid; i < cols; i += blockDim.x) {
        output[row * cols + i] = shared[i];
    }
}

torch::Tensor parallel_scan_cuda(torch::Tensor x, int dim) {
    // Get tensor dimensions
    auto sizes = x.sizes();
    int n = sizes[dim];
    int total_elements = x.numel();
    int outer_elements = total_elements / n;
    
    // Reshape to 2D: (outer_elements, n) for easier processing
    auto x_reshaped = x.reshape({outer_elements, n}).contiguous();
    auto out = torch::zeros_like(x_reshaped);
    
    // Get data pointers
    auto x_ptr = x_reshaped.data_ptr<float>();
    auto out_ptr = out.data_ptr<float>();
    
    // Configure kernel launch
    int threads = 256;
    int shared_mem_size = n * sizeof(float);
    
    // Launch one block per row
    row_scan_kernel<float><<<outer_elements, threads, shared_mem_size>>>(
        x_ptr, out_ptr, outer_elements, n
    );
    
    // Reshape back to original shape
    return out.reshape(sizes);
}
"""

scan_cpp_source = """
torch::Tensor parallel_scan_cuda(torch::Tensor x, int dim);
"""

# Compile the inline CUDA code
scan_module = load_inline(
    name="parallel_scan",
    cpp_sources=scan_cpp_source,
    cuda_sources=scan_source,
    functions=["parallel_scan_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs cumulative sum using custom CUDA kernels.
    
    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized Scan model.
        
        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.scan_module = scan_module

    def forward(self, x):
        """
        Forward pass using optimized parallel scan.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Tensor after applying cumulative sum along specified dimension.
        """
        return self.scan_module.parallel_scan_cuda(x, self.dim)