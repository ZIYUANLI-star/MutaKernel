#!/usr/bin/env python3
"""For each failing loss kernel, look at actual return-shape construction."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "external_benchmarks/ai_cuda_engineer/registry.json"

with open(REG, encoding="utf-8") as f:
    reg = json.load(f)
by_id = {e["id"]: e for e in reg}

loss_ids = [
    "sakana__L1_T94", "sakana__L1_T96", "sakana__L1_T97",
    "sakana__L1_T98", "sakana__L1_T100",
    "sakana__L2_T42", "sakana__L2_T45", "sakana__L2_T65", "sakana__L2_T95",
    "sakana__L1_T34",
]

for kid in loss_ids:
    e = by_id[kid]
    cuda = e["kernel_source"]
    name = e.get("kernel_name", "")
    print(f"=== {kid} ({name}) ===")

    # Find return statements / output tensor allocation
    # auto output = torch::xxx({shape});
    out_alloc = re.findall(r'(?:auto\s+(\w+)\s*=\s*|torch::Tensor\s+(\w+)\s*=\s*)?'
                           r'torch::(?:zeros|empty|ones|empty_like|zeros_like)\s*\(([^,)]+)',
                           cuda)
    for grp in out_alloc[:5]:
        var = grp[0] or grp[1] or "_"
        shape_arg = grp[2].strip()
        print(f"  alloc:  {var:<14} shape_arg = {shape_arg[:100]}")

    # Look at last `return` lines
    returns = re.findall(r'return\s+(\w+)\s*;', cuda)
    for r in returns[-3:]:
        print(f"  return: {r}")

    # Final tensor reshape/view ops
    reshapes = re.findall(r'\.(view|reshape|squeeze|unsqueeze)\s*\(([^)]*)\)', cuda)
    for op, arg in reshapes[:5]:
        print(f"  reshape: .{op}({arg[:60]})")
    print()
