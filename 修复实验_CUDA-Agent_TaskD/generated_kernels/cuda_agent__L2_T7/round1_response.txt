import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Disable TF32 for numerical stability
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Use native convolution through nn.Conv2d
            x = self.conv(x)
            
            # Apply GELU using PyTorch's native implementation for numerical stability
            x = torch.nn.functional.gelu(x)
            
            # Global average pooling with explicit float32 accumulation for stability
            if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
                # Cast to float32 for stable reduction
                x_float = x.float()
                x = x_float.mean(dim=(2, 3)).to(x.dtype)
            else:
                x = x.mean(dim=(2, 3))
            
            return x
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32