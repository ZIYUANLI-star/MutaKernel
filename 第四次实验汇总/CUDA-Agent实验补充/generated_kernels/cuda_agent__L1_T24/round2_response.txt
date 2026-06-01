import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized 1D convolution using PyTorch's native Conv1d with deterministic settings.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        
        # Create the same conv1d as in the original model
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Save original settings
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        old_cudnn_benchmark = torch.backends.cudnn.benchmark
        old_cudnn_enabled = torch.backends.cudnn.enabled
        
        try:
            # Disable TF32 and enable deterministic mode for numerical precision
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Disable cuDNN entirely to use deterministic algorithms
            torch.backends.cudnn.enabled = False
            
            # Use F.conv1d with the module's weight and bias for consistent behavior
            output = F.conv1d(
                x, 
                self.conv1d.weight, 
                self.conv1d.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        finally:
            # Restore previous settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cudnn.deterministic = old_cudnn_deterministic
            torch.backends.cudnn.benchmark = old_cudnn_benchmark
            torch.backends.cudnn.enabled = old_cudnn_enabled
        
        return output