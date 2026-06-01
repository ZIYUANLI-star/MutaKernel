import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        
        # Initialize parameters - we'll rename them later to match original
        self.fc1_weight = nn.Parameter(torch.randn(layer_sizes[0], input_size))
        self.fc1_bias = nn.Parameter(torch.zeros(layer_sizes[0]))
        self.fc2_weight = nn.Parameter(torch.randn(layer_sizes[1], layer_sizes[0]))
        self.fc2_bias = nn.Parameter(torch.zeros(layer_sizes[1]))
        self.fc3_weight = nn.Parameter(torch.randn(output_size, layer_sizes[1]))
        self.fc3_bias = nn.Parameter(torch.zeros(output_size))

    def load_state_dict(self, state_dict, strict=True):
        # Rename keys to match
        new_state_dict = {}
        new_state_dict['fc1_weight'] = state_dict['network.0.weight']
        new_state_dict['fc1_bias'] = state_dict['network.0.bias']
        new_state_dict['fc2_weight'] = state_dict['network.2.weight']
        new_state_dict['fc2_bias'] = state_dict['network.2.bias']
        new_state_dict['fc3_weight'] = state_dict['network.4.weight']
        new_state_dict['fc3_bias'] = state_dict['network.4.bias']
        super().load_state_dict(new_state_dict, strict=strict)

    def forward(self, x):
        # Disable TF32 for numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Layer 1: Linear + ReLU
            x = torch.nn.functional.linear(x, self.fc1_weight, self.fc1_bias)
            x = torch.nn.functional.relu(x)
            
            # Layer 2: Linear + ReLU
            x = torch.nn.functional.linear(x, self.fc2_weight, self.fc2_bias)
            x = torch.nn.functional.relu(x)
            
            # Layer 3: Linear (output)
            x = torch.nn.functional.linear(x, self.fc3_weight, self.fc3_bias)
        finally:
            # Restore original settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return x