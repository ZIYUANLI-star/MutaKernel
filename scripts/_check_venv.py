import sys
print("python:", sys.executable)
try:
    import apex
    print("apex OK:", apex.__path__)
except Exception as e:
    print("apex FAIL:", e)
try:
    import flash_attn
    print("flash_attn OK:", flash_attn.__version__)
except Exception as e:
    print("flash_attn FAIL:", e)
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
