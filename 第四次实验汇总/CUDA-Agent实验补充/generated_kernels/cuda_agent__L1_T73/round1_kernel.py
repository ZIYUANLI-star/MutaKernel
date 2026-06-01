import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication with numerical stability.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        """
        Performs the matrix multiplication with full precision to match reference.
        """
        # Save current TF32 settings
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        try:
            # Disable TF32 to ensure numerical precision matches reference
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            result = torch.matmul(A, B)
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return result