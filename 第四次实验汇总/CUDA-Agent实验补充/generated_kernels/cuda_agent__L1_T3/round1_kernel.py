import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized model for cumulative sum that maintains numerical stability.
    """

    def __init__(self, dim):
        """
        Initialize the optimized Scan model.
        
        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass computing cumulative sum along the specified dimension.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            
        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum.
        """
        # Ensure contiguous memory layout for efficient access
        x_contig = x.contiguous()
        
        # Perform cumulative sum in the input's dtype to maintain precision
        # For float32 inputs, this preserves full precision
        # For float16/bfloat16 inputs, this respects the user's precision choice
        result = torch.cumsum(x_contig, dim=self.dim)
        
        return result