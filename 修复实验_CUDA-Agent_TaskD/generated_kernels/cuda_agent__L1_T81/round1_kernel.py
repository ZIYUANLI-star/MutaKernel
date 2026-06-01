import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Perform sum reduction in the original dtype (float32) to maintain precision
        # Using float32 accumulation avoids precision loss from float16 conversion
        # which was causing failures with large magnitudes and near-overflow values
        return torch.sum(x, dim=self.dim, keepdim=True)