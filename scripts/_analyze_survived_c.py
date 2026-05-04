#!/usr/bin/env python3
"""提取所有存活的 C 类变异体的详细信息。"""
import json
from pathlib import Path

d = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/full_block12_results/details")

for f in sorted(d.glob("*.json")):
    data = json.load(open(f))
    kname = f.stem
    for m in data.get("mutants", []):
        if m.get("operator_category") != "C":
            continue
        if m.get("status") != "survived":
            continue
        print(f"{'='*70}")
        print(f"Kernel: {kname}")
        print(f"Operator: {m['operator_name']}")
        print(f"Line: {m['site']['line_start']}")
        print(f"Status: {m['status']}")
        print(f"Description: {m.get('description', 'N/A')}")
        print(f"Original site code: {m['site'].get('original_code', 'N/A')}")
        print(f"Error msg: {m.get('error_message', '')}")
        print()
