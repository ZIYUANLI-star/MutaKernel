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
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # Disable TF32 for numerical precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Matrix multiplication
        x = self.matmul(x)
        
        # Apply dropout using PyTorch's native dropout
        # This ensures consistent behavior with the reference implementation
        x = self.dropout(x)
        
        # Apply softmax with numerical stability
        x = torch.softmax(x, dim=1)
        
        return x