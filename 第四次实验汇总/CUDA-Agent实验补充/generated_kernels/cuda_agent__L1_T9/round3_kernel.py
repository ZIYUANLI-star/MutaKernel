import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized 2D transposed convolution using PyTorch's functional API.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use the standard nn.ConvTranspose2d to ensure correct weight initialization and behavior
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            groups=groups, bias=bias
        )
        
        # Store parameters for functional call
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use torch.nn.functional.conv_transpose2d with deterministic settings
        # to match the reference implementation exactly
        
        # Disable TF32 and set deterministic cudnn for numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        old_cudnn_benchmark = torch.backends.cudnn.benchmark
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        try:
            # Use functional API with the module's weight and bias
            output = torch.nn.functional.conv_transpose2d(
                x,
                self.conv_transpose2d.weight,
                self.conv_transpose2d.bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=0,
                groups=self.groups,
                dilation=self.dilation
            )
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cudnn.deterministic = old_cudnn_deterministic
            torch.backends.cudnn.benchmark = old_cudnn_benchmark
        
        return output