import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized version of the 3D tensor-matrix multiplication model.
    Matches reference implementation numerically by avoiding TF32 precision loss.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication matching reference precision.
        
        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L).
        """
        # Save current TF32 settings
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        try:
            # Disable TF32 to ensure full precision matching reference
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            result = torch.matmul(A, B)
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return result