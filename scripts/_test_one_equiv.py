#!/usr/bin/env python3
"""Minimal test: check one mutant for equivalence."""
import hashlib
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)

from src.mutengine.mutant_runner import _load_module_from_source, CompilationError
from src.bridge.eval_bridge import _load_module_from_path

DEVICE = "cuda"
KB_ROOT = "/home/kbuser/projects/KernelBench-0"

details_dir = os.path.join(PROJECT_ROOT, "第一次实验汇总", "full_block12_results", "details")
best_kernels = json.load(open(os.path.join(PROJECT_ROOT, "best_kernels.json")))

first_file = sorted(os.listdir(details_dir))[0]
print(f"Loading: {first_file}", flush=True)
data = json.load(open(os.path.join(details_dir, first_file), encoding="utf-8"))

kernel_name = data["kernel"]["problem_name"]
bk = best_kernels[kernel_name]
kernel_path = bk["kernel_path"]
problem_id = bk["problem_id"]
level = bk["level"]

problem_dir = os.path.join(KB_ROOT, "KernelBench", "level1" if level == "L1" else "level2")
problem_file = None
for f in os.listdir(problem_dir):
    if f.startswith(f"{problem_id}_") and f.endswith(".py"):
        problem_file = os.path.join(problem_dir, f)
        break

print(f"Kernel: {kernel_name}, Problem: {problem_file}", flush=True)

survived = [m for m in data["mutants"] if m["status"] == "survived"]
if not survived:
    print("No survived mutants")
    sys.exit(0)

m = survived[0]
print(f"Mutant: {m['id']}", flush=True)

kernel_code = open(kernel_path, encoding="utf-8").read()
mutated_code = m["mutated_code"]

tmp_dir = tempfile.mkdtemp(prefix="eqtest_")

print("Loading problem module...", flush=True)
ref_mod = _load_module_from_path(problem_file, "ref_test")
get_inputs = ref_mod.get_inputs
get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
init_args = get_init_inputs()
print(f"  init_args: {init_args}", flush=True)

print("Compiling original kernel...", flush=True)
t0 = time.time()
orig_hash = hashlib.md5(kernel_code.encode()).hexdigest()[:10]
orig_mod = _load_module_from_source(kernel_code, f"eqo_{orig_hash}", tmp_dir)
print(f"  Original compiled in {time.time()-t0:.1f}s", flush=True)

orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
orig_model = (orig_cls(*init_args) if isinstance(init_args, (list, tuple))
              else orig_cls())
orig_model = orig_model.to(DEVICE).eval()
print(f"  Original model on {DEVICE}", flush=True)

safe_id = m["id"].replace("-", "_").replace(".", "_")
print("Compiling mutant kernel...", flush=True)
t0 = time.time()
mut_mod = _load_module_from_source(mutated_code, f"eqm_{safe_id}", tmp_dir)
print(f"  Mutant compiled in {time.time()-t0:.1f}s", flush=True)

mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
mut_model = (mut_cls(*init_args) if isinstance(init_args, (list, tuple))
             else mut_cls())
mut_model = mut_model.to(DEVICE).eval()
print(f"  Mutant model on {DEVICE}", flush=True)

print(f"  GPU mem: {torch.cuda.memory_allocated()/1024**2:.0f} MB", flush=True)

print("\nRunning 10 random iterations...", flush=True)
for i in range(10):
    torch.manual_seed(10000 + i)
    inputs = get_inputs()
    moved = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x for x in inputs]
    with torch.no_grad():
        orig_out = orig_model(*moved)
        mut_out = mut_model(*moved)
    same = torch.equal(orig_out, mut_out) if isinstance(orig_out, torch.Tensor) else (orig_out == mut_out)
    print(f"  iter {i}: same={same}", flush=True)
    if not same:
        print("  -> NOT EQUIVALENT!")
        break
    del orig_out, mut_out, moved, inputs

print("\nDone!", flush=True)
