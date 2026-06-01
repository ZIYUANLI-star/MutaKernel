import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for optimized masked cumulative sum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* __restrict__ input,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each block handles one outer position
    const int64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;
    
    const int64_t base_offset = outer_idx * dim_size * inner_size;
    
    // Process inner positions in parallel
    for (int64_t inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        // Use double precision for accumulation to improve numerical stability
        // This helps match PyTorch's behavior for large magnitude values
        double cum_sum = 0.0;
        
        // Sequential scan along the cumulative dimension
        for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            const int64_t offset = base_offset + dim_idx * inner_size + inner_idx;
            // Multiply by mask value (0 or 1) to match PyTorch's x * mask behavior
            double val = static_cast<double>(input[offset]) * static_cast<double>(mask[offset] ? 1 : 0);
            cum_sum += val;
            output[offset] = static_cast<scalar_t>(cum_sum);
        }
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor input, torch::Tensor mask, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) {
        dim = input.dim() + dim;
    }
    
    // Validate inputs
    TORCH_CHECK(input.sizes() == mask.sizes(), "Input and mask must have same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "Mask must be boolean tensor");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Dimension out of range");
    
    // Ensure tensors are contiguous for correct memory access
    auto input_contig = input.contiguous();
    auto mask_contig = mask.contiguous();
    
    auto output = torch::empty_like(input_contig);
    
    // Handle empty tensor case
    if (input_contig.numel() == 0) {
        return output;
    }
    
    // Get tensor dimensions
    auto sizes = input_contig.sizes();
    
    // Calculate dimensions for kernel launch
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t dim_size = sizes[dim];
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input_contig.dim(); i++) {
        inner_size *= sizes[i];
    }
    
    // Handle edge case where outer_size is 0
    if (outer_size == 0 || dim_size == 0) {
        return output;
    }
    
    // Launch kernel with optimized configuration
    const int block_size = 256;
    dim3 grid_size(outer_size);
    int threads = (inner_size > 0) ? std::min(block_size, (int)inner_size) : 1;
    dim3 block_size_dim(threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_contig.scalar_type(),
        "masked_cumsum_kernel",
        [&] {
            masked_cumsum_kernel<scalar_t><<<grid_size, block_size_dim>>>(
                input_contig.data_ptr<scalar_t>(),
                mask_contig.data_ptr<bool>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }
    );
    
    return output;
}
"""

masked_cumsum_cpp_source = """
torch::Tensor masked_cumsum_cuda(torch::Tensor input, torch::Tensor mask, int64_t dim);
"""

# Compile the inline CUDA code
masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using custom CUDA kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """
    
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum
    
    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.
            
        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)