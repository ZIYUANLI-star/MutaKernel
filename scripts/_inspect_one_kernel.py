#!/usr/bin/env python3
"""Print kernel CUDA forward signature for a given kernel id."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"

with open(REG, encoding="utf-8") as f:
    reg_list = json.load(f)
registry = {e["id"]: e for e in reg_list}

kid = sys.argv[1] if len(sys.argv) > 1 else "sakana__L1_T94"
e = registry[kid]
ref_path = ROOT / e["reference_file"]
cuda = e.get("kernel_source", "")

print(f"=== {kid} ({e.get('kernel_name','')}) ===")
print(f"reference_file: {ref_path}")
print()

# Extract Python module_fn body from problem file
text = ref_path.read_text(encoding="utf-8")
print("--- problem.py (relevant excerpt) ---")
m = re.search(r'def module_fn\([^)]*\)[^:]*:\n(?:    .*\n)+', text)
if m:
    print(m.group(0))

# Extract get_inputs
m = re.search(r'def get_inputs\(\)[^:]*:\n(?:    .*\n)+', text)
if m:
    print(m.group(0))

print()
print("--- CUDA pybind & forward signature ---")
# Find the C++ forward declaration
fwd = re.findall(r'(?:torch::Tensor|at::Tensor)\s+forward\s*\([^)]*\)', cuda, re.DOTALL)
for f in fwd[:3]:
    print(f"  {f.strip()[:300]}")

# Find PYBIND11_MODULE bindings
pyb = re.findall(r'PYBIND11_MODULE\s*\(([^{]+)\{([^}]+)\}', cuda, re.DOTALL)
for header, body in pyb[:1]:
    print()
    print(f"  PYBIND11 header: {header.strip()}")
    print(f"  PYBIND11 body:")
    for line in body.strip().split("\n"):
        line = line.strip()
        if line.startswith("m.def"):
            print(f"    {line[:200]}")

# Find return type/style of CUDA forward (sum vs mean)
print()
print("--- Reduction hints in CUDA code ---")
for keyword in ['/ static_cast', '/ float(', '/ N', '* (1.0f /', 'AvgPool', '__shfl_down', 'atomicAdd', 'reduction']:
    if keyword in cuda:
        # Find a snippet
        idx = cuda.find(keyword)
        snip = cuda[max(0, idx-50):idx+100].replace('\n', ' ')
        print(f"  '{keyword}' found: ...{snip}...")
