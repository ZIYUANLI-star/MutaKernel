import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    Fixed for numerical stability.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        # Disable TF32 for numerical stability
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Step 1: Transposed convolution
            x = self.conv_transpose(x)
            
            # Step 2: ReLU
            x = self.relu(x)
            
            # Step 3: Group normalization
            x = self.group_norm(x)
            
            return x
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32