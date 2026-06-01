import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    Optimized version that maintains numerical stability.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Initialize parameters - preserve original structure for state_dict compatibility
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # Disable TF32 for numerical stability
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Step 1: 3D convolution
            x = self.conv(x)
            
            # Step 2: Multiply with multiplier
            x = x * self.multiplier
            
            # Step 3: Instance normalization using the built-in module for consistency
            x = self.instance_norm(x)
            
            # Step 4: Clamp and multiply with multiplier again
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
            x = x * self.multiplier
            
            # Step 5: Max over dim=1
            x = torch.max(x, dim=1)[0]
            
            return x
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32