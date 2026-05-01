import sys, importlib

# Check what softmax-related modules exist in apex
import apex
print("apex path:", apex.__path__)

for submod in ["apex.contrib", "apex.contrib.csrc", "apex.fused_softmax",
               "apex.normalization", "apex.fused_dense",
               "apex.transformer", "apex.transformer.functional"]:
    try:
        importlib.import_module(submod)
        print(f"  {submod}: OK")
    except ImportError as e:
        print(f"  {submod}: MISSING ({e})")

# Check apex top-level contents
print("\napex dir:", [x for x in dir(apex) if not x.startswith('_')])

# Check if fused_softmax C++ extension is available
try:
    import torch
    print("\ntorch.ops:", hasattr(torch.ops, 'fused_softmax'))
    # Check scaled_masked_softmax_cuda
    from apex._C import scaled_masked_softmax as sms
    print("scaled_masked_softmax C ext: OK")
except Exception as e:
    print(f"scaled_masked_softmax C ext: FAIL ({e})")

# Check FusedDense 2D vs 3D
try:
    import torch
    from apex.fused_dense import FusedDense
    d = FusedDense(64, 128).cuda().eval()
    x2d = torch.randn(8, 64, device='cuda')
    x3d = torch.randn(8, 16, 64, device='cuda')
    out2d = d(x2d)
    print(f"\nFusedDense 2D: input={x2d.shape} -> output={out2d.shape}")
    try:
        out3d = d(x3d)
        print(f"FusedDense 3D: input={x3d.shape} -> output={out3d.shape}")
    except Exception as e:
        print(f"FusedDense 3D: FAIL ({e})")

    # Check state_dict key compatibility
    ref = torch.nn.Linear(64, 128).cuda()
    print(f"\nFusedDense keys: {list(d.state_dict().keys())}")
    print(f"nn.Linear keys:  {list(ref.state_dict().keys())}")
    # Can we load?
    d.load_state_dict(ref.state_dict())
    print("FusedDense load_state_dict from Linear: OK")
except Exception as e:
    print(f"FusedDense test: FAIL ({e})")
