import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation optimized for PyTorch compilation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.hidden_dim = in_channels * expand_ratio
        self.stride = stride
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        
        # Initialize parameters - preserve original structure for state_dict compatibility
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=self.hidden_dim, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x