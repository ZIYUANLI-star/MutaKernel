import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        # Disable TF32 for numerical precision
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        
        try:
            # Run convolution
            x = self.conv(x)
            
            # Apply activations exactly as in reference
            x = torch.relu(x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
            x = torch.nn.functional.gelu(x)
            x = torch.sigmoid(x)
            x = x + self.bias
        finally:
            # Restore TF32 settings
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = old_cuda_tf32
        
        return x