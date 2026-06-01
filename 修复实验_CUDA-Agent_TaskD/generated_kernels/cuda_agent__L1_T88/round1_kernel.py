import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction using PyTorch's optimized operations.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        # Use PyTorch's built-in mean which handles numerical stability properly
        # Convert to float64 for accumulation to handle near-overflow values
        original_dtype = x.dtype
        
        # For numerical stability with extreme values, use float64 accumulation
        if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            # Accumulate in float64 for better precision with extreme values
            result = torch.mean(x.to(torch.float64), dim=self.dim)
            # Convert back to original dtype if it was float32, otherwise keep float64
            if original_dtype == torch.float32:
                result = result.to(torch.float32)
            elif original_dtype in (torch.float16, torch.bfloat16):
                result = result.to(original_dtype)
        else:
            result = torch.mean(x, dim=self.dim)
        
        return result