#!/usr/bin/env python3
"""Quick environment check for external differential testing."""
import sys
print(f"Python: {sys.version}")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

for pkg in ["apex", "flash_attn", "triton"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{pkg}: {ver}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")
