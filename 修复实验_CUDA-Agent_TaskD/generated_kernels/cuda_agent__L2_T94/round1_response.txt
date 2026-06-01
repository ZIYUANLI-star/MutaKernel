import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that matches reference implementation numerically
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        # Ensure input is float32 for numerical stability
        x = x.float()
        
        # Use the module's forward for proper behavior
        x = self.conv_transpose(x)
        
        # Mish activation: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
        # Use PyTorch's built-in mish for numerical stability
        x = torch.nn.functional.mish(x)
        
        # Add value
        x = x + self.add_value
        
        # Hardtanh activation
        x = torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0)
        
        # Scale
        x = x * self.scale
        
        return x