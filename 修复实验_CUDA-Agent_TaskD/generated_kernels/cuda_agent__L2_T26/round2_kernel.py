import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        # Save original settings
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_cudnn_enabled = torch.backends.cudnn.enabled
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        
        # Disable TF32 and enable deterministic mode for numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        
        try:
            # Convert to float64 for higher precision computation
            original_dtype = x.dtype
            x = x.to(torch.float64)
            
            # Convert model parameters to float64 temporarily
            conv_weight = self.conv.weight.to(torch.float64)
            conv_bias = self.conv.bias.to(torch.float64) if self.conv.bias is not None else None
            bias = self.bias.to(torch.float64)
            scale = self.scale.to(torch.float64)
            gn_weight = self.group_norm.weight.to(torch.float64) if self.group_norm.weight is not None else None
            gn_bias = self.group_norm.bias.to(torch.float64) if self.group_norm.bias is not None else None
            
            # Step 1: Convolution in float64
            x = torch.nn.functional.conv2d(x, conv_weight, conv_bias, 
                                           self.conv.stride, self.conv.padding, 
                                           self.conv.dilation, self.conv.groups)
            
            # Step 2: Add bias
            x = x + bias
            
            # Step 3: Scale
            x = x * scale
            
            # Step 4: Sigmoid with clamping to avoid extreme values
            x = torch.clamp(x, min=-50.0, max=50.0)
            x = torch.sigmoid(x)
            
            # Step 5: Group normalization in float64
            # Manual implementation for better numerical stability
            N, C, H, W = x.shape
            num_groups = self.group_norm.num_groups
            eps = self.group_norm.eps
            
            # Reshape for group norm: (N, num_groups, C//num_groups, H, W)
            x = x.view(N, num_groups, C // num_groups, H, W)
            
            # Compute mean and variance with higher precision
            mean = x.mean(dim=(2, 3, 4), keepdim=True)
            var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
            
            # Normalize
            x = (x - mean) / torch.sqrt(var + eps)
            
            # Reshape back
            x = x.view(N, C, H, W)
            
            # Apply affine transformation if present
            if gn_weight is not None:
                x = x * gn_weight.view(1, C, 1, 1)
            if gn_bias is not None:
                x = x + gn_bias.view(1, C, 1, 1)
            
            # Convert back to original dtype
            x = x.to(original_dtype)
            
            return x
        finally:
            # Restore settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cudnn.deterministic = old_cudnn_deterministic