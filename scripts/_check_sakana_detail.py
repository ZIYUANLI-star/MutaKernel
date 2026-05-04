#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
reg = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"

with open(reg, encoding="utf-8") as f:
    data = json.load(f)

for kid in ["sakana__L1_T34", "sakana__L1_T15"]:
    for e in data:
        if e["id"] == kid:
            ks = e["kernel_source"]
            ref = e["reference_file"]
            p = Path(ref) if Path(ref).is_absolute() else PROJECT_ROOT / ref
            with open(p, encoding="utf-8") as f2:
                ref_code = f2.read()

            has_model_new = "class ModelNew" in ks
            has_load_inline = "load_inline" in ks

            print(f"{kid}:")
            print(f"  kernel length: {len(ks)} chars")
            print(f"  has ModelNew: {has_model_new}")
            print(f"  has load_inline: {has_load_inline}")
            print(f"  ref has proper Model: {'class Model' in ref_code}")
            print(f"  ref has get_inputs: {'get_inputs' in ref_code}")
            print(f"  ref has get_init_inputs: {'get_init_inputs' in ref_code}")
            print(f"  ref is identity passthrough: {'identity passthrough' in ref_code}")

            for line in ref_code.split("\n"):
                s = line.strip()
                if any(kw in s for kw in ["batch_size", "dim", "randn", "features", "N =", "M ="]):
                    print(f"  ref: {s}")

            # Check init params match
            for line in ks.split("\n"):
                if "class ModelNew" in line:
                    print(f"  kernel: {line.strip()}")
                if "def __init__" in line and "ModelNew" not in line:
                    print(f"  kernel init: {line.strip()}")

            print()
            break
