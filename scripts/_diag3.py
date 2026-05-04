import apex.contrib
import os, pkgutil

print("apex.contrib submodules:")
for importer, modname, ispkg in pkgutil.walk_packages(
    apex.contrib.__path__, prefix="apex.contrib."
):
    print(f"  {modname} {'(pkg)' if ispkg else ''}")

# Check FusedDenseGeluDense with 2D input and weight sync
import torch
from apex.fused_dense import FusedDenseGeluDense

IN_F, INTER, OUT_F = 1024, 4096, 1024

ext = FusedDenseGeluDense(IN_F, INTER, OUT_F).cuda().eval()
print(f"\nFusedDenseGeluDense state_dict:")
for k, v in ext.state_dict().items():
    print(f"  {k}: {v.shape}")

# Build matching reference
ref_l1 = torch.nn.Linear(IN_F, INTER).cuda()
ref_l2 = torch.nn.Linear(INTER, OUT_F).cuda()
print(f"\nReference Linear state_dict:")
for k, v in ref_l1.state_dict().items():
    print(f"  linear1.{k}: {v.shape}")
for k, v in ref_l2.state_dict().items():
    print(f"  linear2.{k}: {v.shape}")

# Copy by shape order
ref_params = list(ref_l1.state_dict().values()) + list(ref_l2.state_dict().values())
ext_sd = ext.state_dict()
ext_keys = list(ext_sd.keys())
for i, key in enumerate(ext_keys):
    if i < len(ref_params) and ext_sd[key].shape == ref_params[i].shape:
        ext_sd[key] = ref_params[i].clone()
        print(f"  Copied ref param {i} -> ext {key}")
ext.load_state_dict(ext_sd)

# Test 2D input
x = torch.randn(32, IN_F, device='cuda')
ref_out = ref_l2(torch.nn.functional.gelu(ref_l1(x)))
ext_out = ext(x)
diff = (ref_out - ext_out).abs()
print(f"\nAfter weight sync, 2D input:")
print(f"  max_diff={diff.max().item():.6e}")
print(f"  allclose(atol=1e-3): {torch.allclose(ref_out, ext_out, atol=1e-3)}")

# Test 3D input
x3d = torch.randn(32, 128, IN_F, device='cuda')
try:
    ext_out3d = ext(x3d)
    ref_out3d = ref_l2(torch.nn.functional.gelu(ref_l1(x3d)))
    print(f"\n3D input:")
    print(f"  ref shape={ref_out3d.shape}, ext shape={ext_out3d.shape}")
except Exception as e:
    print(f"\n3D input error: {e}")
