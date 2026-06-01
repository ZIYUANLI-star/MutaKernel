import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.alpha = alpha
        
        # Store all weights as a dictionary to make it easier to map
        self.weights = nn.Module()
        
        # Initial conv
        self.weights.conv1_weight = nn.Parameter(torch.empty(int(32 * alpha), input_channels, 3, 3))
        
        # Batch norm for initial conv
        self.weights.conv1_bn_weight = nn.Parameter(torch.empty(int(32 * alpha)))
        self.weights.conv1_bn_bias = nn.Parameter(torch.empty(int(32 * alpha)))
        self.register_buffer('conv1_bn_running_mean', torch.zeros(int(32 * alpha)))
        self.register_buffer('conv1_bn_running_var', torch.ones(int(32 * alpha)))
        
        # Depth-wise and point-wise convolutions for each block
        self._initialize_dw_pw_weights()
        
        # Linear layer
        self.weights.fc_weight = nn.Parameter(torch.empty(num_classes, int(1024 * alpha)))
        self.weights.fc_bias = nn.Parameter(torch.empty(num_classes))
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_dw_pw_weights(self):
        # Create weights for all conv_dw blocks (13 blocks)
        depths = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024]
        depths = [int(d * self.alpha) for d in depths]
        
        for i in range(1, 14):  # 13 conv_dw blocks
            in_channels = depths[i-1]
            out_channels = depths[i] if i < 13 else int(1024 * self.alpha)
            
            # Depth-wise
            setattr(self.weights, f'dw_conv{i}_weight', nn.Parameter(torch.empty(in_channels, 1, 3, 3)))
            setattr(self.weights, f'dw_conv{i}_bn_weight', nn.Parameter(torch.empty(in_channels)))
            setattr(self.weights, f'dw_conv{i}_bn_bias', nn.Parameter(torch.empty(in_channels)))
            self.register_buffer(f'dw_conv{i}_bn_running_mean', torch.zeros(in_channels))
            self.register_buffer(f'dw_conv{i}_bn_running_var', torch.ones(in_channels))
            
            # Point-wise
            setattr(self.weights, f'pw_conv{i}_weight', nn.Parameter(torch.empty(out_channels, in_channels, 1, 1)))
            setattr(self.weights, f'pw_conv{i}_bn_weight', nn.Parameter(torch.empty(out_channels)))
            setattr(self.weights, f'pw_conv{i}_bn_bias', nn.Parameter(torch.empty(out_channels)))
            self.register_buffer(f'pw_conv{i}_bn_running_mean', torch.zeros(out_channels))
            self.register_buffer(f'pw_conv{i}_bn_running_var', torch.ones(out_channels))

    def _initialize_weights(self):
        # Initialize all weights
        for name, param in self.weights.named_parameters():
            if 'weight' in name and param.dim() == 4 and param.size(2) == 3 and param.size(3) == 3:
                # Depth-wise or regular 3x3 conv
                nn.init.xavier_uniform_(param)
            elif 'weight' in name and param.dim() == 4 and param.size(2) == 1 and param.size(3) == 1:
                # Point-wise conv
                nn.init.xavier_uniform_(param)
            elif 'weight' in name and param.dim() == 2:
                # Linear weight
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def load_state_dict(self, state_dict, strict=True):
        # Custom state dict mapping
        new_state_dict = {}
        
        # Initial conv
        new_state_dict['weights.conv1_weight'] = state_dict['model.0.0.weight']
        
        # Batch norm for initial conv
        new_state_dict['weights.conv1_bn_weight'] = state_dict['model.0.1.weight']
        new_state_dict['weights.conv1_bn_bias'] = state_dict['model.0.1.bias']
        new_state_dict['conv1_bn_running_mean'] = state_dict['model.0.1.running_mean']
        new_state_dict['conv1_bn_running_var'] = state_dict['model.0.1.running_var']
        
        # Conv_dw blocks - these are layers 1 to 13 in model.model
        for i in range(1, 14):
            block_idx = i
            # Depth-wise
            new_state_dict[f'weights.dw_conv{i}_weight'] = state_dict[f'model.{block_idx}.0.weight']
            new_state_dict[f'weights.dw_conv{i}_bn_weight'] = state_dict[f'model.{block_idx}.1.weight']
            new_state_dict[f'weights.dw_conv{i}_bn_bias'] = state_dict[f'model.{block_idx}.1.bias']
            new_state_dict[f'dw_conv{i}_bn_running_mean'] = state_dict[f'model.{block_idx}.1.running_mean']
            new_state_dict[f'dw_conv{i}_bn_running_var'] = state_dict[f'model.{block_idx}.1.running_var']
            
            # Point-wise
            new_state_dict[f'weights.pw_conv{i}_weight'] = state_dict[f'model.{block_idx}.3.weight']
            new_state_dict[f'weights.pw_conv{i}_bn_weight'] = state_dict[f'model.{block_idx}.4.weight']
            new_state_dict[f'weights.pw_conv{i}_bn_bias'] = state_dict[f'model.{block_idx}.4.bias']
            new_state_dict[f'pw_conv{i}_bn_running_mean'] = state_dict[f'model.{block_idx}.4.running_mean']
            new_state_dict[f'pw_conv{i}_bn_running_var'] = state_dict[f'model.{block_idx}.4.running_var']
        
        # Linear layer
        new_state_dict['weights.fc_weight'] = state_dict['fc.weight']
        new_state_dict['weights.fc_bias'] = state_dict['fc.bias']
        
        super().load_state_dict(new_state_dict, strict=strict)

    def _apply_bn_relu(self, x, bn_weight, bn_bias, running_mean, running_var):
        """Apply batch normalization and ReLU with proper numerical stability."""
        # Use F.batch_norm for proper handling
        x = F.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, 
                        training=self.training, momentum=0.1, eps=1e-05)
        x = F.relu(x, inplace=False)
        return x

    def forward(self, x):
        # Disable TF32 for numerical precision
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        try:
            # Initial conv
            x = F.conv2d(x, self.weights.conv1_weight, None, stride=2, padding=1)
            
            # Batch norm + relu
            x = self._apply_bn_relu(x, self.weights.conv1_bn_weight, self.weights.conv1_bn_bias,
                                   self.conv1_bn_running_mean, self.conv1_bn_running_var)
            
            # Conv_dw blocks
            conv_dw_strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2]
            
            for i in range(1, 14):
                stride = conv_dw_strides[i-1]
                
                # Depth-wise conv
                dw_weight = getattr(self.weights, f'dw_conv{i}_weight')
                x = F.conv2d(x, dw_weight, None, stride=stride, padding=1, groups=dw_weight.size(0))
                
                # Batch norm + relu for depth-wise
                dw_bn_weight = getattr(self.weights, f'dw_conv{i}_bn_weight')
                dw_bn_bias = getattr(self.weights, f'dw_conv{i}_bn_bias')
                dw_running_mean = getattr(self, f'dw_conv{i}_bn_running_mean')
                dw_running_var = getattr(self, f'dw_conv{i}_bn_running_var')
                x = self._apply_bn_relu(x, dw_bn_weight, dw_bn_bias, dw_running_mean, dw_running_var)
                
                # Point-wise conv
                pw_weight = getattr(self.weights, f'pw_conv{i}_weight')
                x = F.conv2d(x, pw_weight, None, stride=1, padding=0)
                
                # Batch norm + relu for point-wise
                pw_bn_weight = getattr(self.weights, f'pw_conv{i}_bn_weight')
                pw_bn_bias = getattr(self.weights, f'pw_conv{i}_bn_bias')
                pw_running_mean = getattr(self, f'pw_conv{i}_bn_running_mean')
                pw_running_var = getattr(self, f'pw_conv{i}_bn_running_var')
                x = self._apply_bn_relu(x, pw_bn_weight, pw_bn_bias, pw_running_mean, pw_running_var)
            
            # AvgPool2d (7x7)
            x = F.avg_pool2d(x, kernel_size=7)
            
            # Linear layer
            x = x.view(x.size(0), -1)
            x = F.linear(x, self.weights.fc_weight, self.weights.fc_bias)
            
        finally:
            # Restore TF32 settings
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        
        return x