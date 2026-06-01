import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication, scaling, and batch normalization.
    This version properly handles training vs eval mode for batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        # Disable TF32 to ensure numerical precision
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Standard GEMM operation
            x = self.gemm(x)
            
            # Apply scaling
            x = x * self.scale
            
            # Apply batch normalization (handles training/eval mode correctly)
            x = self.bn(x)
            
            return x
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn