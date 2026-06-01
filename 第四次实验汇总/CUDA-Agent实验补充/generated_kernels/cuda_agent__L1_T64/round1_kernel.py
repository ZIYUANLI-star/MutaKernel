import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized transposed 1D convolution that matches the reference implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use the actual nn.ConvTranspose1d to ensure correct behavior and state_dict compatibility
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Disable TF32 for numerical stability
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            output = self.conv1d_transpose(x)
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return output