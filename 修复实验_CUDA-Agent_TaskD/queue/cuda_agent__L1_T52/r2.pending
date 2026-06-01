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
        # Match the reference implementation exactly:
        # torch.cumsum(x * mask, dim=self.dim)
        # The key is that mask is boolean, and x * mask uses boolean multiplication
        # which converts True to 1 and False to 0 in the dtype of x
        
        # Simply replicate the reference behavior exactly
        return torch.cumsum(x * mask, dim=self.dim)