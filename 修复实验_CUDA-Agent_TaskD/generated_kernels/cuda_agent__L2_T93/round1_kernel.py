import torch
import torch.nn as nn

# Disable TF32 to ensure numerical precision matches reference
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        # Initialize with the same state dict keys as original
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        # Calculate linear layer using the standard nn.Linear
        linear_out = self.linear(x)
        
        # Fused operation: min with constant and subtract constant
        output = torch.min(linear_out, self.constant) - self.constant
        
        return output