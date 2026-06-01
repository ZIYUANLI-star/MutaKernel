import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the optimized Vanilla RNN model.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Preserve original structure for state_dict compatibility
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with numerical stability improvements.
        
        :param x: Input tensor of shape (seq_len, batch_size, input_size)
        :param h0: Initial hidden state tensor of shape (batch_size, hidden_size)
        :return: Output tensor of shape (seq_len, batch_size, output_size)
        """
        # Disable TF32 for numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            seq_len, batch_size, input_size = x.size()
            
            # Keep everything in float32 for numerical stability
            # This avoids float16 overflow issues with near_overflow test cases
            x_fp32 = x.float()
            hidden = h0.float().to(x.device)
            
            # Get weights in float32
            w_i2h = self.i2h.weight.float()
            b_i2h = self.i2h.bias.float()
            w_h2o = self.h2o.weight.float()
            b_h2o = self.h2o.bias.float()
            
            # Split weights into input and hidden parts
            w_ih = w_i2h[:, :input_size]
            w_hh = w_i2h[:, input_size:]
            
            outputs = []
            
            for t in range(seq_len):
                # Linear layers using matrix multiplication in float32
                ih = x_fp32[t] @ w_ih.T
                hh = hidden @ w_hh.T
                hidden = torch.tanh(ih + hh + b_i2h)
                output = hidden @ w_h2o.T + b_h2o
                outputs.append(output)
            
            # Stack outputs
            output = torch.stack(outputs, dim=0)
            
            return output
            
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32