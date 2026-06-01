import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x):
        # Save original dtype
        orig_dtype = x.dtype
        
        # Disable TF32 for numerical stability
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Convert to float64 for maximum numerical precision
            x = x.to(torch.float64)
            
            # Store original conv weights dtype and convert to float64
            conv_weight_dtype = self.conv_transpose.weight.dtype
            conv_bias_dtype = self.conv_transpose.bias.dtype if self.conv_transpose.bias is not None else None
            
            orig_weight = self.conv_transpose.weight.data
            orig_bias = self.conv_transpose.bias.data if self.conv_transpose.bias is not None else None
            
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(torch.float64)
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.data = self.conv_transpose.bias.data.to(torch.float64)
            
            # ConvTranspose3d
            x = self.conv_transpose(x)
            
            # Restore conv weights
            self.conv_transpose.weight.data = orig_weight
            if orig_bias is not None:
                self.conv_transpose.bias.data = orig_bias
            
            # MaxPool3d
            x = self.max_pool(x)
            
            # Numerically stable softmax across channels (dim=1)
            # Subtract max for numerical stability
            x_max = x.max(dim=1, keepdim=True)[0]
            x_shifted = x - x_max
            exp_x = torch.exp(x_shifted)
            sum_exp = exp_x.sum(dim=1, keepdim=True)
            x = exp_x / (sum_exp + 1e-15)  # Add small epsilon to prevent division by zero
            
            # Subtract across channels
            subtract_f64 = self.subtract.to(torch.float64)
            x = x - subtract_f64.view(1, -1, 1, 1, 1)
            
            # Swish activation: sigmoid(x) * x
            # Use numerically stable sigmoid
            sigmoid_x = torch.sigmoid(x)
            x = sigmoid_x * x
            
            # Max pooling across channels
            x = torch.max(x, dim=1)[0]
            
            # Convert back to original dtype
            x = x.to(orig_dtype)
            
            return x
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32