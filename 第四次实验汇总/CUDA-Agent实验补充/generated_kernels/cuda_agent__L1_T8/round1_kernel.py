import torch
import torch.nn as nn


class ModelNew(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        # Initialize parameters to match the original model
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        # Use the standard conv2d operation without mixed precision
        # to ensure numerical stability across all test conditions
        output = self.conv1(x)
        return output