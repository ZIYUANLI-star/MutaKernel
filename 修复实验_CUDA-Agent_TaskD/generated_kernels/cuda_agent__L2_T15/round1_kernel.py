import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that uses column sum optimization.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        # We'll compute weight_column_sum on demand and cache it
        self.register_buffer('_weight_column_sum', None)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # Disable TF32 for numerical precision
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # During training, weight can change, so we need to recompute
            # Also recompute if dtype or device changes
            if self.training or self._weight_column_sum is None:
                # Compute sum of columns of weight (sum over hidden_size dimension)
                # weight is (hidden_size, input_size), sum over dim=0 gives (input_size,)
                weight_column_sum = self.weight.sum(dim=0)
            else:
                # Check if cached sum matches current weight dtype/device
                if (self._weight_column_sum.dtype != self.weight.dtype or 
                    self._weight_column_sum.device != self.weight.device):
                    weight_column_sum = self.weight.sum(dim=0)
                    self._weight_column_sum = weight_column_sum.detach()
                else:
                    weight_column_sum = self._weight_column_sum
            
            # Cache for inference mode
            if not self.training and self._weight_column_sum is None:
                self._weight_column_sum = weight_column_sum.detach()
            
            # Compute dot product of x with the column sum
            # x is (batch_size, input_size), weight_column_sum is (input_size,)
            # Result is (batch_size,)
            result = torch.mv(x, weight_column_sum) if x.dim() == 2 else x.matmul(weight_column_sum)
            
            # Apply division and scaling: (x / 2) * scaling_factor
            result = result * (self.scaling_factor / 2.0)
            
            # Reshape to (batch_size, 1)
            result = result.unsqueeze(-1)
            
            return result
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn