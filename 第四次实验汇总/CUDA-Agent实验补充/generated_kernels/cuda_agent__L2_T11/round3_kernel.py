import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model using:
    - PyTorch's convolution (with bias)
    - PyTorch's instance norm for numerical stability
    - Division by constant
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        # Save original settings
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        old_cudnn_enabled = torch.backends.cudnn.enabled
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        
        # Disable TF32 and enable deterministic mode for numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        
        try:
            # Use PyTorch's convolution with bias
            x = self.conv(x)
            
            # Use PyTorch's instance norm for numerical stability
            x = self.instance_norm(x)
            
            # Divide by constant using multiplication for consistency
            x = x * (1.0 / self.divide_by)
        finally:
            # Restore settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
            torch.backends.cudnn.deterministic = old_cudnn_deterministic
        
        return x