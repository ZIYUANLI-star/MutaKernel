import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)