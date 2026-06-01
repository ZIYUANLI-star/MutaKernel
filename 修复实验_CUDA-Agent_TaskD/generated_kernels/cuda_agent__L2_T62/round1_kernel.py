import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that matches the reference implementation exactly.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        
        # Initialize parameters to preserve state_dict compatibility
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.bn_running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.bn_running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))
        
        # Dummy parameter to capture num_batches_tracked (will ignore its value)
        self.num_batches_tracked = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
        
        # Track training state for batch norm
        self.bn_momentum = 0.1
        self.bn_eps = 1e-5
        self.gn_eps = 1e-5
        
    def load_state_dict(self, state_dict, strict=True):
        # Map old keys to new keys
        new_state_dict = {}
        for old_key, value in state_dict.items():
            if old_key == "conv_transpose.weight":
                new_key = "weight"
            elif old_key == "conv_transpose.bias":
                new_key = "bias"
            elif old_key == "batch_norm.weight":
                new_key = "bn_weight"
            elif old_key == "batch_norm.bias":
                new_key = "bn_bias"
            elif old_key == "batch_norm.running_mean":
                new_key = "bn_running_mean"
            elif old_key == "batch_norm.running_var":
                new_key = "bn_running_var"
            elif old_key == "batch_norm.num_batches_tracked":
                new_key = "num_batches_tracked"
            elif old_key == "group_norm.weight":
                new_key = "gn_weight"
            elif old_key == "group_norm.bias":
                new_key = "gn_bias"
            else:
                continue  # Ignore other keys if any
            new_state_dict[new_key] = value
        
        # Load the mapped state_dict
        super(ModelNew, self).load_state_dict(new_state_dict, strict=False)
        
    def forward(self, x):
        # Disable TF32 for numerical stability
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Step 1: Transposed convolution
            conv_out = torch.conv_transpose2d(
                x, self.weight, self.bias, stride=self.stride, padding=self.padding, output_padding=0
            )
            
            # Step 2: Batch normalization
            if self.training:
                # Compute batch statistics
                batch_mean = conv_out.mean(dim=(0, 2, 3))
                batch_var = conv_out.var(dim=(0, 2, 3), unbiased=False)
                
                # Update running statistics
                with torch.no_grad():
                    self.bn_running_mean.data = (1 - self.bn_momentum) * self.bn_running_mean.data + self.bn_momentum * batch_mean
                    self.bn_running_var.data = (1 - self.bn_momentum) * self.bn_running_var.data + self.bn_momentum * batch_var
                
                # Normalize using batch statistics
                mean = batch_mean[None, :, None, None]
                var = batch_var[None, :, None, None]
            else:
                # Use running statistics
                mean = self.bn_running_mean[None, :, None, None]
                var = self.bn_running_var[None, :, None, None]
            
            gamma = self.bn_weight[None, :, None, None]
            beta = self.bn_bias[None, :, None, None]
            conv_out = gamma * (conv_out - mean) / torch.sqrt(var + self.bn_eps) + beta
            
            # Step 3: Tanh
            conv_out = torch.tanh(conv_out)
            
            # Step 4: Max pool
            pooled = torch.max_pool2d(conv_out, kernel_size=2, stride=2)
            
            # Step 5: Group normalization
            batch_size, channels, height, width = pooled.shape
            channels_per_group = channels // self.num_groups
            
            # Reshape for group norm: (N, G, C//G, H, W)
            pooled = pooled.reshape(batch_size, self.num_groups, channels_per_group, height, width)
            
            # Calculate mean and variance per group
            mean = pooled.mean(dim=(2, 3, 4), keepdim=True)
            var = pooled.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
            
            # Normalize
            pooled = (pooled - mean) / torch.sqrt(var + self.gn_eps)
            
            # Reshape back to (N, C, H, W)
            pooled = pooled.reshape(batch_size, channels, height, width)
            
            # Apply learnable parameters
            gamma = self.gn_weight[None, :, None, None]
            beta = self.gn_bias[None, :, None, None]
            out = gamma * pooled + beta
            
            return out
            
        finally:
            # Restore original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn