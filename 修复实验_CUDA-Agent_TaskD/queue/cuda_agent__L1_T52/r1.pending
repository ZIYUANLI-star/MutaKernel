import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized model that performs masked cumulative sum using PyTorch operations.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        # Disable TF32 for better numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # For better numerical stability, especially with float16/bfloat16,
            # perform the computation in float32 and cast back
            input_dtype = x.dtype
            
            if input_dtype in (torch.float16, torch.bfloat16):
                # Upcast to float32 for accumulation
                x_float = x.float()
                mask_float = mask.float()
                masked = x_float * mask_float
                result = torch.cumsum(masked, dim=self.dim)
                return result.to(input_dtype)
            else:
                # For float32 and float64, use standard computation
                # Ensure mask is properly cast to the same dtype as x
                masked = x * mask.to(x.dtype)
                return torch.cumsum(masked, dim=self.dim)
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32