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
            # Use float32 to match reference implementation exactly
            # The reference uses default float32, so we should too
            x = x.float()
            
            # ConvTranspose3d - ensure weights are float32
            with torch.no_grad():
                orig_weight = self.conv_transpose.weight.data.clone()
                orig_bias = self.conv_transpose.bias.data.clone() if self.conv_transpose.bias is not None else None
            
            # Temporarily convert weights to float32 if needed
            weight_was_converted = False
            if self.conv_transpose.weight.dtype != torch.float32:
                weight_was_converted = True
                self.conv_transpose.weight.data = self.conv_transpose.weight.data.float()
                if self.conv_transpose.bias is not None:
                    self.conv_transpose.bias.data = self.conv_transpose.bias.data.float()
            
            x = self.conv_transpose(x)
            
            # Restore weights if converted
            if weight_was_converted:
                self.conv_transpose.weight.data = orig_weight
                if orig_bias is not None:
                    self.conv_transpose.bias.data = orig_bias
            
            # MaxPool3d
            x = self.max_pool(x)
            
            # Use PyTorch's built-in softmax which is numerically stable
            # This matches the reference implementation exactly
            x = torch.softmax(x, dim=1)
            
            # Subtract across channels - ensure subtract param is float32
            subtract_f32 = self.subtract.float()
            x = x - subtract_f32.view(1, -1, 1, 1, 1)
            
            # Swish activation: sigmoid(x) * x
            # Use PyTorch's built-in sigmoid which handles edge cases
            x = torch.sigmoid(x) * x
            
            # Max pooling across channels
            x = torch.max(x, dim=1)[0]
            
            # Convert back to original dtype
            x = x.to(orig_dtype)
            
            return x
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32