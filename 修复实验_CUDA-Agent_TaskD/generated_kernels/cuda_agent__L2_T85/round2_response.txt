import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication, applies dropout, and then applies softmax.
    Uses PyTorch's native operations for numerical stability.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x):
        # Disable TF32 for numerical precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Matrix multiplication
        x = self.matmul(x)
        
        # Apply dropout manually to ensure consistent behavior
        if self.training and self.dropout_p > 0.0:
            # Generate dropout mask
            mask = torch.bernoulli(torch.full_like(x, 1.0 - self.dropout_p))
            # Apply mask and scale
            scale = 1.0 / (1.0 - self.dropout_p)
            x = x * mask * scale
        
        # Apply softmax with numerical stability
        # Subtract max for numerical stability (PyTorch's softmax does this internally)
        x = torch.softmax(x, dim=1)
        
        return x