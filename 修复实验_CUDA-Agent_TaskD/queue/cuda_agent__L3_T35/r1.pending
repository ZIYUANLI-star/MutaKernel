import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        # Initialize parameters with the same names as in the original model
        self.network = nn.ModuleList()
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            linear_layer = nn.Linear(current_input_size, hidden_size)
            self.network.append(linear_layer)
            # Add ReLU layer to match original structure
            self.network.append(nn.ReLU())
            current_input_size = hidden_size
        
        self.network.append(nn.Linear(current_input_size, output_size))

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Save original TF32 settings
        orig_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        
        # Disable TF32 for numerical precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Simply iterate through all layers in order
            for layer in self.network:
                x = layer(x)
        finally:
            # Restore original TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul_tf32
            torch.backends.cudnn.allow_tf32 = orig_cudnn_tf32
        
        return x