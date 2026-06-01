import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation with numerical stability.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Softplus activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        # Use PyTorch's built-in softplus which handles numerical stability correctly
        # softplus(x) = log(1 + exp(x))
        # For large x: softplus(x) ≈ x (to avoid overflow in exp)
        # For small x: softplus(x) ≈ exp(x) (to avoid log(1) precision issues)
        return torch.nn.functional.softplus(x)