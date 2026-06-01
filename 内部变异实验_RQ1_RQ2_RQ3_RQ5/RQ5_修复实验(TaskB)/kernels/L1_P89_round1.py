import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative sum
scan_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel that computes cumsum for each "row" (slice along the cumsum dimension)
// Uses a running sum approach to match PyTorch's behavior
__global__ void cumsum_rows_kernel(const float* __restrict__ input, 
                                    float* __restrict__ output, 
                                    int num_rows, 
                                    int row_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    
    int offset = row * row_size;
    
    // Compute cumulative sum using running sum (matches PyTorch's approach)
    float running_sum = 0.0f;
    for (int i = 0; i < row_size; i++) {
        running_sum += input[offset + i];
        output[offset + i] = running_sum;
    }
}

// General kernel for arbitrary dimension cumsum
// Handles the case where cumsum is along an arbitrary dimension
__global__ void cumsum_general_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int outer_size,
                                       int dim_size,
                                       int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_slices = outer_size * inner_size;
    if (idx >= total_slices) return;
    
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
    // Compute cumsum along the middle dimension
    float running_sum = 0.0f;
    for (int i = 0; i < dim_size; i++) {
        int pos = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
        running_sum += input[pos];
        output[pos] = running_sum;
    }
}

torch::Tensor parallel_scan_cuda(torch::Tensor x, int dim) {
    // Ensure input is contiguous and on CUDA
    auto x_contig = x.contiguous();
    
    // Handle negative dimension
    if (dim < 0) {
        dim = x_contig.dim() + dim;
    }
    
    // Get tensor dimensions
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    
    // Calculate outer_size (product of dimensions before dim)
    // dim_size (size of the dimension we're cumsum'ing)
    // inner_size (product of dimensions after dim)
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t dim_size = sizes[dim];
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }
    
    // Create output tensor
    auto out = torch::empty_like(x_contig);
    
    // Get data pointers
    auto x_ptr = x_contig.data_ptr<float>();
    auto out_ptr = out.data_ptr<float>();
    
    // Choose kernel based on inner_size
    if (inner_size == 1) {
        // Optimized path: cumsum is along the last dimension
        // Each thread handles one row
        int num_rows = outer_size;
        int row_size = dim_size;
        
        int threads = 256;
        int blocks = (num_rows + threads - 1) / threads;
        
        cumsum_rows_kernel<<<blocks, threads>>>(x_ptr, out_ptr, num_rows, row_size);
    } else {
        // General path: cumsum along arbitrary dimension
        int64_t total_slices = outer_size * inner_size;
        
        int threads = 256;
        int blocks = (total_slices + threads - 1) / threads;
        
        cumsum_general_kernel<<<blocks, threads>>>(
            x_ptr, out_ptr, 
            (int)outer_size, (int)dim_size, (int)inner_size
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Fall back to PyTorch's cumsum on error
        return torch::cumsum(x, dim);
    }
    
    return out;
}
"""

scan_cpp_source = """
torch::Tensor parallel_scan_cuda(torch::Tensor x, int dim);
"""

# Compile the inline CUDA code
scan_module = load_inline(
    name="parallel_scan_v2",
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
        # Use custom CUDA kernel for float32 CUDA tensors
        if x.is_cuda and x.dtype == torch.float32:
            return self.scan_module.parallel_scan_cuda(x, self.dim)
        else:
            # Fall back to PyTorch for other dtypes or CPU tensors
            return torch.cumsum(x, dim=self.dim)