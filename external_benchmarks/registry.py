"""External CUDA kernel registry for differential testing.

Each entry defines:
  - id: unique identifier
  - repo: source repository
  - kernel_name: human-readable name
  - reference_file: path to PyTorch reference (Model + get_inputs)
  - kernel_source: Python source string wrapping the external kernel as ModelNew
  - requires: pip packages that must be installed
"""

EXTERNAL_KERNELS = [
    # ---- NVIDIA Apex (CUDA C++) ----
    {
        "id": "apex__fused_layer_norm",
        "repo": "NVIDIA/apex",
        "kernel_name": "FusedLayerNorm",
        "reference_file": "external_benchmarks/apex/fused_layer_norm_ref.py",
        "requires": ["apex"],
        "kernel_source": '''
import torch
import torch.nn as nn
from apex.normalization import FusedLayerNorm

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = FusedLayerNorm(normalized_shape)

    def forward(self, x):
        return self.ln(x)
''',
    },
    {
        "id": "apex__fused_rms_norm",
        "repo": "NVIDIA/apex",
        "kernel_name": "FusedRMSNorm",
        "reference_file": "external_benchmarks/apex/fused_rms_norm_ref.py",
        "requires": ["apex"],
        "kernel_source": '''
import torch
import torch.nn as nn
from apex.normalization import FusedRMSNorm

class ModelNew(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.rms = FusedRMSNorm(hidden_size, eps=eps)

    def forward(self, x):
        return self.rms(x)
''',
    },
    # FusedSoftmax removed: apex.transformer module not available in this
    # Apex build (requires Megatron-specific compilation flags).
    {
        "id": "apex__fused_dense",
        "repo": "NVIDIA/apex",
        "kernel_name": "FusedDense",
        "reference_file": "external_benchmarks/apex/fused_dense_ref.py",
        "requires": ["apex"],
        "kernel_source": '''
import torch
import torch.nn as nn
from apex.fused_dense import FusedDense

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.dense = FusedDense(in_features, out_features)

    def forward(self, x):
        return self.dense(x)
''',
    },
    {
        "id": "apex__fused_dense_gelu_dense",
        "repo": "NVIDIA/apex",
        "kernel_name": "FusedDenseGeluDense",
        "reference_file": "external_benchmarks/apex/fused_dense_gelu_dense_ref.py",
        "requires": ["apex"],
        "kernel_source": '''
import torch
import torch.nn as nn
from apex.fused_dense import FusedDenseGeluDense

class ModelNew(nn.Module):
    def __init__(self, in_features, intermediate, out_features):
        super().__init__()
        self.mlp = FusedDenseGeluDense(in_features, intermediate, out_features)

    def forward(self, x):
        return self.mlp(x)
''',
    },
    {
        "id": "apex__mlp",
        "repo": "NVIDIA/apex",
        "kernel_name": "MLP",
        "reference_file": "external_benchmarks/apex/mlp_ref.py",
        "requires": ["apex"],
        "kernel_source": '''
import torch
import torch.nn as nn
from apex.mlp import MLP

class ModelNew(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.mlp = MLP([in_features, hidden_features, out_features])

    def forward(self, x):
        return self.mlp(x)
''',
    },
    # ---- FlashAttention (CUDA C++) ----
    {
        "id": "flash_attn__flash_attention_2",
        "repo": "Dao-AILab/flash-attention",
        "kernel_name": "FlashAttention2",
        "reference_file": "external_benchmarks/flash_attention/flash_attn_ref.py",
        "requires": ["flash_attn"],
        "kernel_source": '''
import torch
import torch.nn as nn
from flash_attn import flash_attn_func

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return flash_attn_func(q, k, v)
''',
    },
]
