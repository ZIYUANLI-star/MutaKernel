#!/usr/bin/env python3
"""Check what format AI CUDA Engineer kernel codes are in."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
reg_path = ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"
with open(reg_path) as f:
    reg = json.load(f)

has_modelnew = 0
has_load_inline = 0
has_include = 0
has_pybind = 0
total = len(reg)

for e in reg:
    ks = e["kernel_source"]
    if "class ModelNew" in ks:
        has_modelnew += 1
    if "load_inline" in ks:
        has_load_inline += 1
    if "#include" in ks:
        has_include += 1
    if "PYBIND11" in ks or "pybind" in ks.lower():
        has_pybind += 1

print(f"Total: {total}")
print(f"Has class ModelNew: {has_modelnew}")
print(f"Has load_inline: {has_load_inline}")
print(f"Has #include: {has_include}")
print(f"Has pybind: {has_pybind}")

# Show 3 examples
for i in [0, 100, 200]:
    if i < total:
        e = reg[i]
        ks = e["kernel_source"]
        print(f"\n--- Entry {i}: {e['id']} ---")
        print(f"First 500 chars: {ks[:500]}")
        print(f"Last 200 chars: ...{ks[-200:]}")
