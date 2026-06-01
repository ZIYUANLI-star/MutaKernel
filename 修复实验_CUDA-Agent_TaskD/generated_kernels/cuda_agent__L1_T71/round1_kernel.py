import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized model for matrix multiplication of two symmetric matrices.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        """
        Performs optimized matrix multiplication of two symmetric matrices.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        # Save original TF32 settings
        orig_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        try:
            # Disable TF32 for numerical precision
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            # Perform matrix multiplication
            C = torch.matmul(A, B)
        finally:
            # Restore original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul_tf32
            torch.backends.cudnn.allow_tf32 = orig_cudnn_tf32
        
        return C