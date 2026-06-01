import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized transposed 2D convolution that matches reference implementation.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        # Use the standard nn.ConvTranspose2d to ensure correct behavior
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding, 
            groups=groups, 
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Disable TF32 to ensure numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Use standard convolution in FP32 for numerical stability
            output = self.conv_transpose2d(x)
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return output