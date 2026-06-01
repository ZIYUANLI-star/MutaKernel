import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super().__init__()
        # Use the standard nn.ConvTranspose3d to ensure correct weight initialization and behavior
        self.conv_transpose3d = nn.ConvTranspose3d(
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
        # Disable TF32 to ensure numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Perform the convolution in full precision (float32)
            # Ensure input and weights are in float32 for numerical stability
            input_dtype = x.dtype
            
            if x.dtype != torch.float32:
                x = x.float()
            
            output = self.conv_transpose3d(x)
            
            # Convert back to original dtype if needed
            if input_dtype != torch.float32:
                output = output.to(input_dtype)
                
            return output
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32