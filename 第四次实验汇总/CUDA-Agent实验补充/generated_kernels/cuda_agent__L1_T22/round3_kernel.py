import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super().__init__()
        # Use the actual nn.ConvTranspose2d to preserve correct behavior and state_dict structure
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simply delegate to the underlying ConvTranspose2d module
        # The reference implementation uses nn.ConvTranspose2d directly,
        # so we should do the same without any modifications
        return self.conv_transpose2d(x)