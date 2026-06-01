import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        # Use proper nn.BatchNorm2d and nn.Conv2d modules
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Build the transition sequential to match expected interface
        self.transition = nn.Sequential(
            self.bn,
            nn.ReLU(inplace=True),
            self.conv,
            self.avgpool
        )

    def forward(self, x):
        # Disable TF32 for numerical stability
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Use the standard transition sequential
            return self.transition(x)
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32