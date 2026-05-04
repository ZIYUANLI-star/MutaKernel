#!/usr/bin/env python3
"""Test baseline loading for specific kernels to diagnose compilation failures."""
from __future__ import annotations
import json, sys, os, traceback, importlib.util, tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_kernel(reg_path, kid):
    with open(reg_path, encoding="utf-8") as f:
        registry = json.load(f)

    entry = None
    for e in registry:
        if e["id"] == kid:
            entry = e
            break
    if not entry:
        print(f"  {kid}: NOT FOUND")
        return

    ref_file = entry["reference_file"]
    if not os.path.isabs(ref_file):
        ref_file = str(PROJECT_ROOT / ref_file)

    kernel_source = entry["kernel_source"]
    print(f"=== Testing {kid} ===")

    # Step 1: Load reference module
    print(f"  Step 1: Loading reference module...")
    try:
        ref_mod = load_module(ref_file, "ref_test")
        get_inputs = ref_mod.get_inputs
        get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
        init_args = get_init_inputs()
        print(f"    OK. init_args={init_args}")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return

    # Step 2: Instantiate Model (reference)
    print(f"  Step 2: Instantiate Model...")
    try:
        ref_model = ref_mod.Model(*init_args).cuda()
        print(f"    OK: {ref_model.__class__.__name__}")
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
        return

    # Step 3: Write kernel_source to temp file and load it
    print(f"  Step 3: Loading kernel module (this triggers CUDA compilation)...")
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmp.write(kernel_source)
    tmp.close()
    try:
        kernel_mod = load_module(tmp.name, "kernel_test")
        print(f"    OK: module loaded")
    except Exception as e:
        print(f"    FAILED during load/compile: {type(e).__name__}: {e}")
        traceback.print_exc()
        return
    finally:
        os.unlink(tmp.name)

    # Step 4: Instantiate ModelNew
    print(f"  Step 4: Instantiate ModelNew...")
    try:
        new_model = kernel_mod.ModelNew(*init_args).cuda()
        print(f"    OK: {new_model.__class__.__name__}")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # Step 5: Run forward
    print(f"  Step 5: Forward pass...")
    try:
        inputs = get_inputs()
        inputs = [x.cuda() if hasattr(x, "cuda") else x for x in inputs]
        ref_out = ref_model(*inputs)
        new_out = new_model(*inputs)
        print(f"    ref_out shape: {ref_out.shape}")
        print(f"    new_out shape: {new_out.shape}")
        diff = (ref_out - new_out).abs().max().item()
        print(f"    max_diff: {diff}")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

    print()


if __name__ == "__main__":
    reg = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"
    for kid in ["sakana__L1_T34", "sakana__L1_T15"]:
        test_kernel(reg, kid)
