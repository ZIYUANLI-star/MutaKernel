import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumulative sum.
    Fixed to match reference implementation exactly.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Disable TF32 for numerical precision
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Match the reference implementation exactly:
            # 1. Select the slice at position 0 along self.dim
            # 2. Create zeros with same shape, unsqueeze to add back the dimension
            # 3. Concatenate zeros at the beginning along self.dim
            # 4. Slice off the last element along self.dim
            # 5. Compute cumsum
            
            # Create zeros matching the shape of a slice along self.dim
            zeros_shape = list(x.shape)
            zeros_shape[self.dim] = 1
            zeros = torch.zeros(zeros_shape, dtype=x.dtype, device=x.device)
            
            # Concatenate zeros at the beginning
            x_with_zeros = torch.cat((zeros, x), dim=self.dim)
            
            # Slice to remove the last element along self.dim (to match original shape)
            # This creates the "exclusive" part - we prepend 0 and remove the last element
            slices = [slice(None)] * x_with_zeros.dim()
            slices[self.dim] = slice(None, -1)
            exclusive_input = x_with_zeros[tuple(slices)]
            
            # Compute cumulative sum along the specified dimension
            result = torch.cumsum(exclusive_input, dim=self.dim)
            
            return result
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32