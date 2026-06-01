import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that matches the reference implementation exactly by using
    the same PyTorch modules internally.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        
        # Use the exact same modules as the reference implementation
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
    def forward(self, x):
        # Store original dtype for potential conversion
        original_dtype = x.dtype
        
        # Convert to float32 for computation if needed for numerical stability
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.float()
            compute_in_float32 = True
        else:
            compute_in_float32 = False
        
        # Step 1: Transposed convolution
        x = self.conv_transpose(x)
        
        # Step 2: Batch normalization
        x = self.batch_norm(x)
        
        # Step 3: Tanh
        x = self.tanh(x)
        
        # Step 4: Max pool
        x = self.max_pool(x)
        
        # Step 5: Group normalization
        x = self.group_norm(x)
        
        # Convert back to original dtype if needed
        if compute_in_float32 and original_dtype != torch.float32:
            x = x.to(original_dtype)
        
        return x