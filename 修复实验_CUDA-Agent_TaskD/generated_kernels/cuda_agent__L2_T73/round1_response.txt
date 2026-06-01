import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that matches reference implementation behavior
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, x):
        # Disable TF32 for numerical stability
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        
        try:
            # Ensure input is float32 for numerical stability
            x = x.float()
            
            # Step 1: Transposed convolution
            x = self.conv_transpose(x)
            
            # Step 2: Scale
            x = x * self.scale_factor
            
            # Step 3: Batch normalization
            x = self.batch_norm(x)
            
            # Step 4: Global average pooling
            x = self.global_avg_pool(x)
            
            return x
        finally:
            # Restore TF32 settings
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32