import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmin with optimized memory access
argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template<typename scalar_t>
__global__ void argmin_kernel_2d(const scalar_t* input, int64_t* output,
                                 int64_t rows, int64_t cols, int64_t dim) {
    // Each thread block handles one row (if dim=1) or one column (if dim=0)
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dim == 1) {
        // Reduce along columns (dim=1)
        if (idx < rows) {
            scalar_t min_val = std::numeric_limits<scalar_t>::max();
            int64_t min_idx = 0;
            
            // Unrolled loop for better memory access
            for (int64_t col = 0; col < cols; col++) {
                scalar_t val = input[idx * cols + col];
                if (val < min_val) {
                    min_val = val;
                    min_idx = col;
                }
            }
            output[idx] = min_idx;
        }
    } else if (dim == 0) {
        // Reduce along rows (dim=0)
        if (idx < cols) {
            scalar_t min_val = std::numeric_limits<scalar_t>::max();
            int64_t min_idx = 0;
            
            for (int64_t row = 0; row < rows; row++) {
                scalar_t val = input[row * cols + idx];
                if (val < min_val) {
                    min_val = val;
                    min_idx = row;
                }
            }
            output[idx] = min_idx;
        }
    }
}

template<typename scalar_t>
__global__ void argmin_kernel_3d(const scalar_t* input, int64_t* output,
                                 int64_t batch, int64_t rows, int64_t cols, int64_t dim) {
    // Optimized for 3D tensors with batch dimension
    int64_t batch_idx = blockIdx.x;
    int64_t spatial_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch) return;
    
    if (dim == 2) {
        // Reduce along last dimension (cols)
        if (spatial_idx < rows) {
            scalar_t min_val = std::numeric_limits<scalar_t>::max();
            int64_t min_idx = 0;
            
            int64_t base_offset = batch_idx * rows * cols + spatial_idx * cols;
            for (int64_t col = 0; col < cols; col++) {
                scalar_t val = input[base_offset + col];
                if (val < min_val) {
                    min_val = val;
                    min_idx = col;
                }
            }
            output[batch_idx * rows + spatial_idx] = min_idx;
        }
    } else if (dim == 1) {
        // Reduce along middle dimension (rows)
        if (spatial_idx < cols) {
            scalar_t min_val = std::numeric_limits<scalar_t>::max();
            int64_t min_idx = 0;
            
            int64_t base_offset = batch_idx * rows * cols + spatial_idx;
            for (int64_t row = 0; row < rows; row++) {
                scalar_t val = input[base_offset + row * cols];
                if (val < min_val) {
                    min_val = val;
                    min_idx = row;
                }
            }
            output[batch_idx * cols + spatial_idx] = min_idx;
        }
    } else if (dim == 0) {
        // Reduce along batch dimension
        if (spatial_idx < rows * cols) {
            scalar_t min_val = std::numeric_limits<scalar_t>::max();
            int64_t min_idx = 0;
            
            int64_t row = spatial_idx / cols;
            int64_t col = spatial_idx % cols;
            int64_t base_offset = row * cols + col;
            
            for (int64_t b = 0; b < batch; b++) {
                scalar_t val = input[b * rows * cols + base_offset];
                if (val < min_val) {
                    min_val = val;
                    min_idx = b;
                }
            }
            output[spatial_idx] = min_idx;
        }
    }
}

torch::Tensor argmin_cuda(torch::Tensor input, int64_t dim) {
    auto sizes = input.sizes();
    auto ndim = input.dim();
    
    // Calculate output shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_shape.push_back(sizes[i]);
        }
    }
    
    auto output = torch::empty(output_shape, torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    
    // Optimize kernel selection based on tensor dimensions
    if (ndim == 2) {
        int64_t rows = sizes[0];
        int64_t cols = sizes[1];
        
        // Use 256 threads per block for better occupancy
        int block_size = 256;
        int grid_size = 0;
        
        if (dim == 1) {
            grid_size = (rows + block_size - 1) / block_size;
        } else if (dim == 0) {
            grid_size = (cols + block_size - 1) / block_size;
        }
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmin_2d", ([&] {
            argmin_kernel_2d<scalar_t><<<grid_size, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<int64_t>(),
                rows, cols, dim
            );
        }));
        
    } else if (ndim == 3) {
        int64_t batch = sizes[0];
        int64_t rows = sizes[1];
        int64_t cols = sizes[2];
        
        // Optimize block and grid dimensions for 3D case
        int block_size = 256;
        
        if (dim == 2) {
            // Reduce along last dimension
            dim3 grid(batch, (rows + block_size - 1) / block_size);
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmin_3d_dim2", ([&] {
                argmin_kernel_3d<scalar_t><<<grid, block_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<int64_t>(),
                    batch, rows, cols, dim
                );
            }));
        } else if (dim == 1) {
            // Reduce along middle dimension
            dim3 grid(batch, (cols + block_size - 1) / block_size);
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmin_3d_dim1", ([&] {
                argmin_kernel_3d<scalar_t><<<grid, block_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<int64_t>(),
                    batch, rows, cols, dim
                );
            }));
        } else if (dim == 0) {
            // Reduce along batch dimension
            int grid_size = (rows * cols + block_size - 1) / block_size;
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmin_3d_dim0", ([&] {
                argmin_kernel_3d<scalar_t><<<grid_size, block_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<int64_t>(),
                    batch, rows, cols, dim
                );
            }));
        }
    } else {
        // Fallback for other dimensions - use PyTorch's implementation
        output = torch::argmin(input, dim);
    }
    
    return output;
}
"""

argmin_cpp_source = "torch::Tensor argmin_cuda(torch::Tensor input, int64_t dim);"

# Compile the inline CUDA code for argmin
argmin_module = load_inline(
    name="argmin",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension
    using a custom CUDA kernel with optimized memory access patterns.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin_cuda = argmin_module.argmin_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension
        using custom CUDA kernel optimized for 2D and 3D tensors.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        if x.is_cuda and x.dim() in [2, 3]:
            return self.argmin_cuda(x, self.dim)
        else:
            # Fallback to PyTorch implementation for CPU tensors or unsupported dimensions
            return torch.argmin(x, dim=self.dim)