import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a reverse cumulative sum operation.
    Uses PyTorch's native operations to ensure numerical correctness.
    
    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use PyTorch's native flip and cumsum for exact numerical match with reference
        # This is equivalent to: torch.cumsum(x.flip(dim), dim=dim).flip(dim)
        # which computes reverse cumulative sum
        return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)