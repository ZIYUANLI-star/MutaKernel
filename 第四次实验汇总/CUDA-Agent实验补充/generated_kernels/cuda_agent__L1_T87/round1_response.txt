import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Performs 4D tensor-matrix multiplication using optimized matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        # Save original TF32 settings
        orig_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        # Disable TF32 for numerical precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Reshape A to 2D and perform matrix multiplication
            batch_size = A.shape[0]
            i = A.shape[1]
            j = A.shape[2]
            l = A.shape[3]
            k = B.shape[1]
            
            A_reshaped = A.reshape(-1, l)
            result = A_reshaped @ B
            output = result.reshape(batch_size, i, j, k)
            
            return output
        finally:
            # Restore original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul_tf32
            torch.backends.cudnn.allow_tf32 = orig_cudnn_tf32