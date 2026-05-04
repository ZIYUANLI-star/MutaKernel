#!/usr/bin/env python3
"""Smoke-test a single kernel from the AI CUDA Engineer dataset."""
from __future__ import annotations
import json, sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
kid = sys.argv[1] if len(sys.argv) > 1 else "sakana__L1_T25"

reg_path = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"
with open(reg_path, encoding="utf-8") as f:
    data = json.load(f)

entry = None
for e in data:
    if e["id"] == kid:
        entry = e
        break

if not entry:
    print(f"ERROR: kernel {kid} not found in registry")
    sys.exit(1)

ref_file = entry["reference_file"]
if not os.path.isabs(ref_file):
    ref_file = str(PROJECT_ROOT / ref_file)

kernel_code = entry["kernel_source"]
print(f"=== Testing: {kid} ===")
print(f"  ref: {ref_file}")
print(f"  kernel code: {len(kernel_code)} chars")
print(f"  has load_inline: {'load_inline' in kernel_code}")

# Step 1: Load reference
print("\n[Step 1] Loading reference module...")
from src.bridge.eval_bridge import _load_module_from_path
try:
    ref_mod = _load_module_from_path(ref_file, "smoke_ref")
    print("  OK: reference loaded")
except Exception as ex:
    print(f"  FAIL: {ex}")
    sys.exit(1)

# Step 2: Test reference
import torch
get_inputs = ref_mod.get_inputs
get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
init_args = get_init_inputs()
print(f"  init_args: {init_args}")

inputs = get_inputs()
print(f"  inputs: {[x.shape if hasattr(x, 'shape') else type(x) for x in inputs]}")

ref_model = ref_mod.Model(*init_args) if init_args else ref_mod.Model()
ref_model = ref_model.cuda().eval()
with torch.no_grad():
    ref_out = ref_model(*[x.cuda() for x in inputs])
print(f"  ref_out shape: {ref_out.shape if hasattr(ref_out, 'shape') else type(ref_out)}")

# Step 3: Load kernel (triggers CUDA JIT compilation)
print("\n[Step 2] Loading kernel module (CUDA JIT compilation)...")
print("  This may take 30-120 seconds for CUDA JIT compilation...")
sys.stdout.flush()
t0 = time.time()
from src.mutengine.mutant_runner import _load_module_from_source, CompilationError
import tempfile
tmp = tempfile.mkdtemp(prefix="smoke_")
try:
    kern_mod = _load_module_from_source(kernel_code, f"smoke_kern_{kid}", tmp)
    elapsed = time.time() - t0
    print(f"  OK: kernel compiled in {elapsed:.1f}s")
except CompilationError as ex:
    elapsed = time.time() - t0
    full_err = str(ex)
    print(f"  COMPILE FAIL ({elapsed:.1f}s):")
    print(f"  --- Full Error ({len(full_err)} chars total) ---")
    # Print first 3000 chars to see the actual error
    print(full_err[:3000])
    print(f"  --- End Error ---")
    sys.exit(1)
except Exception as ex:
    elapsed = time.time() - t0
    full_err = str(ex)
    print(f"  FAIL ({elapsed:.1f}s):")
    print(full_err[-2000:])
    sys.exit(1)

# Step 4: Instantiate ModelNew
print("\n[Step 3] Instantiating ModelNew...")
kern_cls = getattr(kern_mod, "ModelNew", None) or getattr(kern_mod, "Model")
kern_model = kern_cls(*init_args) if init_args else kern_cls()
kern_model = kern_model.cuda().eval()
print("  OK: ModelNew instantiated")

# Step 5: Forward pass
print("\n[Step 4] Running forward pass...")
with torch.no_grad():
    kern_out = kern_model(*[x.cuda() for x in inputs])
print(f"  kern_out shape: {kern_out.shape if hasattr(kern_out, 'shape') else type(kern_out)}")

# Step 6: Compare
print("\n[Step 5] Comparing outputs...")
match = False
if isinstance(ref_out, torch.Tensor) and isinstance(kern_out, torch.Tensor):
    match = torch.allclose(ref_out, kern_out, atol=1e-5, rtol=1e-5)
    max_diff = (ref_out.float() - kern_out.float()).abs().max().item()
    mean_diff = (ref_out.float() - kern_out.float()).abs().mean().item()
    print(f"  allclose: {match}")
    print(f"  max_diff: {max_diff:.6e}")
    print(f"  mean_diff: {mean_diff:.6e}")
else:
    print(f"  ref type: {type(ref_out)}, kern type: {type(kern_out)}")

print(f"\n=== RESULT: {'PASS' if match else 'FAIL (outputs differ)'} ===")
