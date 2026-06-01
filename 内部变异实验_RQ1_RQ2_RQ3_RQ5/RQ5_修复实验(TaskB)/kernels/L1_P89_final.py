import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs cumulative sum using PyTorch's native cumsum.
    
    The native torch.cumsum is highly optimized and provides consistent numerical
    results across different input magnitudes and distributions.
    
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

    def forward(self, x):
        """
        Forward pass using PyTorch's native cumsum for optimal numerical stability.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Tensor after applying cumulative sum along specified dimension.
        """
        return torch.cumsum(x, dim=self.dim)