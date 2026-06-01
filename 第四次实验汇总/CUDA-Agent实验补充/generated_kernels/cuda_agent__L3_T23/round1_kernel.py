import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.num_classes = num_classes
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # Create features sequential
        self.features = nn.Sequential()
        
        # First layer: Conv2d
        self.features.append(nn.Conv2d(3, 32, 3, 2, 1, bias=False))
        # Second layer: BatchNorm2d
        self.features.append(nn.BatchNorm2d(32))
        # Third layer: ReLU6
        self.features.append(nn.ReLU6(inplace=True))
        
        # Inverted residual blocks
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        input_channel = 32
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                
                # Create sequential for each inverted residual block
                block = nn.Sequential()
                hidden_dim = int(input_channel * t)
                
                if t != 1:
                    # Pointwise convolution
                    block.append(nn.Conv2d(input_channel, hidden_dim, 1, 1, 0, bias=False))
                    block.append(nn.BatchNorm2d(hidden_dim))
                    block.append(nn.ReLU6(inplace=True))
                
                # Depthwise convolution
                block.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
                block.append(nn.BatchNorm2d(hidden_dim))
                block.append(nn.ReLU6(inplace=True))
                
                # Pointwise linear convolution
                block.append(nn.Conv2d(hidden_dim, output_channel, 1, 1, 0, bias=False))
                block.append(nn.BatchNorm2d(output_channel))
                
                self.features.append(block)
                input_channel = output_channel
        
        # Last convolution
        self.features.append(nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False))
        # BatchNorm
        self.features.append(nn.BatchNorm2d(1280))
        # ReLU6
        self.features.append(nn.ReLU6(inplace=True))
        # AdaptiveAvgPool2d
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(1280, num_classes),
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Disable TF32 for numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return x