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
        # Save original settings
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_cudnn_enabled = torch.backends.cudnn.enabled
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        
        # Disable TF32 and use deterministic algorithms for numerical precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        
        try:
            # Store original dtype
            input_dtype = x.dtype
            
            # Convert to float64 for maximum numerical precision
            # This is especially important for large magnitude and near-overflow cases
            if x.dtype != torch.float64:
                x_fp64 = x.double()
            else:
                x_fp64 = x
            
            # Also convert weights to float64
            weight_dtype = self.conv_transpose3d.weight.dtype
            original_weight = self.conv_transpose3d.weight.data.clone()
            self.conv_transpose3d.weight.data = self.conv_transpose3d.weight.data.double()
            
            original_bias = None
            if self.conv_transpose3d.bias is not None:
                original_bias = self.conv_transpose3d.bias.data.clone()
                self.conv_transpose3d.bias.data = self.conv_transpose3d.bias.data.double()
            
            # Perform the convolution in float64
            output = self.conv_transpose3d(x_fp64)
            
            # Restore original weight dtype
            self.conv_transpose3d.weight.data = original_weight
            if original_bias is not None:
                self.conv_transpose3d.bias.data = original_bias
            
            # Convert back to original dtype
            if input_dtype != torch.float64:
                output = output.to(input_dtype)
                
            return output
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cudnn.deterministic = old_cudnn_deterministic