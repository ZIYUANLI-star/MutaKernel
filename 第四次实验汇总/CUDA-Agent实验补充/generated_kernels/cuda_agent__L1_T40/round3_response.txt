import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super().__init__()
        # Use the standard nn.ConvTranspose3d to ensure correct weight initialization and behavior
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding, 
            groups=groups, 
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simply delegate to the standard PyTorch implementation
        # The reference implementation uses nn.ConvTranspose3d directly,
        # so we should do the same without any modifications
        return self.conv_transpose3d(x)