import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        
        # Create a Sequential to match the original state_dict structure
        layers = []
        
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            # Add Linear layer
            linear = nn.Linear(current_input_size, hidden_size, bias=True)
            layers.append(linear)
            # Add ReLU
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_input_size, output_size, bias=True))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Use the standard Sequential forward pass to ensure numerical stability
        # This avoids FP16 overflow issues with extreme values
        return self.network(x)