import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    Uses PyTorch's cuDNN implementation with specific optimizations.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize parameters - preserve original structure for state_dict compatibility
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size, kernel_size), 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding,
            groups=groups, 
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using PyTorch's cuDNN implementation.
        Disables TF32 to ensure numerical precision matches reference.
        """
        # Save current TF32 settings
        old_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
        old_cuda_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        
        try:
            # Disable TF32 for numerical precision
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            
            # Use the nn.ConvTranspose3d module directly
            return self.conv_transpose3d(x)
        finally:
            # Restore TF32 settings
            torch.backends.cudnn.allow_tf32 = old_cudnn_allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = old_cuda_allow_tf32