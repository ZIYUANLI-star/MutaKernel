import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super().__init__()
        # Use the actual nn.ConvTranspose2d to preserve correct behavior and state_dict structure
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original settings
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_cudnn_enabled = torch.backends.cudnn.enabled
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        
        # Disable TF32 and cuDNN non-determinism for numerical precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        
        try:
            # Convert to float64 for maximum precision with extreme values
            original_dtype = x.dtype
            
            # Use double precision for computation to handle extreme values
            x_double = x.double()
            weight_double = self.conv_transpose2d.weight.double()
            
            if self.conv_transpose2d.bias is not None:
                bias_double = self.conv_transpose2d.bias.double()
            else:
                bias_double = None
            
            # Perform the transposed convolution in double precision
            output = F.conv_transpose2d(
                x_double,
                weight_double,
                bias_double,
                stride=self.conv_transpose2d.stride,
                padding=self.conv_transpose2d.padding,
                output_padding=self.conv_transpose2d.output_padding,
                groups=self.conv_transpose2d.groups,
                dilation=self.conv_transpose2d.dilation
            )
            
            # Convert back to original dtype
            output = output.to(original_dtype)
            
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cudnn.deterministic = old_cudnn_deterministic
        
        return output