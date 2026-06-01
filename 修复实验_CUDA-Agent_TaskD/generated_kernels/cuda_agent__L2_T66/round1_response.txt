import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Use the same parameter names as original for state_dict compatibility
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
    def forward(self, x):
        # Save original TF32 settings
        orig_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        # Disable TF32 for numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Ensure we're working in float32 for numerical stability
            input_dtype = x.dtype
            if x.dtype != torch.float32:
                x = x.float()
            
            # Step 1: GEMM
            x = self.gemm(x)
            
            # Step 2: Group Normalization
            x = self.group_norm(x)
            
            # Step 3: Minimum value across dim=1
            x = torch.min(x, dim=1, keepdim=True)[0]
            
            # Step 4: Bias addition
            x = x + self.bias
            
            # Convert back to original dtype if needed
            if input_dtype != torch.float32:
                x = x.to(input_dtype)
            
            return x
        finally:
            # Restore original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul_tf32
            torch.backends.cudnn.allow_tf32 = orig_cudnn_tf32