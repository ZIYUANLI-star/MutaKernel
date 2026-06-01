import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super().__init__()
        # Initialize parameters with the same names as the original model
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # Branch3x3 has two Conv2d layers
        self.branch3x3_0 = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_1 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        
        # Branch5x5 has two Conv2d layers
        self.branch5x5_0 = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        
        # BranchPool has a Conv2d layer at index 1
        self.branch_pool_1 = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

    def load_state_dict(self, state_dict, strict=True):
        # Remap state dict keys from original model to this model
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == 'branch3x3.0.weight':
                new_state_dict['branch3x3_0.weight'] = v
            elif k == 'branch3x3.0.bias':
                new_state_dict['branch3x3_0.bias'] = v
            elif k == 'branch3x3.1.weight':
                new_state_dict['branch3x3_1.weight'] = v
            elif k == 'branch3x3.1.bias':
                new_state_dict['branch3x3_1.bias'] = v
            elif k == 'branch5x5.0.weight':
                new_state_dict['branch5x5_0.weight'] = v
            elif k == 'branch5x5.0.bias':
                new_state_dict['branch5x5_0.bias'] = v
            elif k == 'branch5x5.1.weight':
                new_state_dict['branch5x5_1.weight'] = v
            elif k == 'branch5x5.1.bias':
                new_state_dict['branch5x5_1.bias'] = v
            elif k == 'branch_pool.1.weight':
                new_state_dict['branch_pool_1.weight'] = v
            elif k == 'branch_pool.1.bias':
                new_state_dict['branch_pool_1.bias'] = v
            else:
                new_state_dict[k] = v
        return super().load_state_dict(new_state_dict, strict)

    def state_dict(self, *args, **kwargs):
        # Remap state dict keys from this model to match original model
        state_dict = super().state_dict(*args, **kwargs)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == 'branch3x3_0.weight':
                new_state_dict['branch3x3.0.weight'] = v
            elif k == 'branch3x3_0.bias':
                new_state_dict['branch3x3.0.bias'] = v
            elif k == 'branch3x3_1.weight':
                new_state_dict['branch3x3.1.weight'] = v
            elif k == 'branch3x3_1.bias':
                new_state_dict['branch3x3.1.bias'] = v
            elif k == 'branch5x5_0.weight':
                new_state_dict['branch5x5.0.weight'] = v
            elif k == 'branch5x5_0.bias':
                new_state_dict['branch5x5.0.bias'] = v
            elif k == 'branch5x5_1.weight':
                new_state_dict['branch5x5.1.weight'] = v
            elif k == 'branch5x5_1.bias':
                new_state_dict['branch5x5.1.bias'] = v
            elif k == 'branch_pool_1.weight':
                new_state_dict['branch_pool.1.weight'] = v
            elif k == 'branch_pool_1.bias':
                new_state_dict['branch_pool.1.bias'] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def forward(self, x):
        # Disable TF32 for numerical stability
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Set deterministic algorithms for reproducibility
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True
        
        try:
            # Branch 1x1
            branch1x1 = self.branch1x1(x)

            # Branch 3x3
            reduce3x3 = self.branch3x3_0(x)
            branch3x3 = self.branch3x3_1(reduce3x3)

            # Branch 5x5
            reduce5x5 = self.branch5x5_0(x)
            branch5x5 = self.branch5x5_1(reduce5x5)

            # Branch max pooling + conv
            branch_pool = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
            branch_pool = self.branch_pool_1(branch_pool)

            # Concatenate all branches
            outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
            result = torch.cat(outputs, 1)
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.backends.cudnn.deterministic = old_cudnn_deterministic
        
        return result