import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Disable TF32 for numerical stability
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Convert to float32 for numerical stability
            original_dtype = x.dtype
            x = x.float()
            
            # GEMM operation
            x = self.gemm(x)
            
            # Max operation along max_dim with keepdim=True
            x = torch.max(x, dim=self.max_dim, keepdim=True).values
            
            # Subtract mean along dim=1
            x = x - x.mean(dim=1, keepdim=True)
            
            # GELU activation
            x = torch.nn.functional.gelu(x)
            
            # Convert back to original dtype if needed
            if original_dtype != torch.float32:
                x = x.to(original_dtype)
            
            return x
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32