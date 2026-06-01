import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication with full precision
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B with full precision.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        # Save the current TF32 settings
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        try:
            # Disable TensorFloat32 for full precision matrix multiplication
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            result = torch.matmul(A, B)
        finally:
            # Restore the original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return result