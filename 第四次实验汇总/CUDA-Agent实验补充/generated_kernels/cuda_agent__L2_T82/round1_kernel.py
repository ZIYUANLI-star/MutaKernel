import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model with numerical stability fixes
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        # Initialize parameters - preserve original structure for state_dict compatibility
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        # Disable TF32 for numerical stability
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Ensure input is float32 for numerical stability
            x = x.float()
            
            # Use standard convolution in float32
            x = self.conv(x)
            
            # LeakyReLU with negative_slope=0.2
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
            
            # Add sum_tensor (ensure it's float32)
            x = x + self.sum_tensor.float()
            
            # Clamp
            x = torch.clamp(x, min=-1.0, max=1.0)
            
            # GELU activation using PyTorch's implementation for consistency
            x = torch.nn.functional.gelu(x)
            
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
        
        return x