#!/usr/bin/env python3
"""Inspect the schema and a sample row of the Sakana AI CUDA Engineer parquet."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "external_benchmarks/ai_cuda_engineer/parquet/level_1.parquet"

import re

def scan_all(level_files):
    fn_name_counts = {}
    has_module_fn_kw = 0
    fn_signatures = {}
    pybind_signatures = {}
    total_correct = 0

    for lf in level_files:
        df = pd.read_parquet(lf)
        correct = df[df["Correct"] == True]
        total_correct += len(correct)

        for _, row in correct.iterrows():
            functional = str(row.get("PyTorch_Code_Functional", ""))
            cuda = str(row.get("CUDA_Code", ""))
            op = str(row.get("Op_Name", "?"))
            level = row.get("Level_ID", "?")
            task = row.get("Task_ID", "?")

            # Find the top-level function (the "module_fn" candidate)
            # Pattern: a free function that takes x as first arg
            funcs = re.findall(r'^def\s+(\w+)\s*\(', functional, re.MULTILINE)
            top_fns = [f for f in funcs if f not in ("get_inputs", "get_init_inputs", "forward", "__init__")]
            if top_fns:
                main_fn = top_fns[0]
                fn_name_counts[main_fn] = fn_name_counts.get(main_fn, 0) + 1

            if "fn=module_fn" in functional or "fn = module_fn" in functional:
                has_module_fn_kw += 1

            # Capture the def line for the main function
            m = re.search(r'^def\s+(\w+)\s*\(([^)]*?)\)\s*(?:->\s*[^:]+)?:', functional, re.MULTILINE | re.DOTALL)
            if m and m.group(1) in top_fns:
                sig = re.sub(r'\s+', ' ', m.group(0))[:300]
                fn_signatures[f"L{level}_T{task}_{op}"] = sig

            # Capture pybind signature - 'm.def("forward", &<fn_name>...)'
            pb = re.search(r'm\.def\(\s*"(\w+)"\s*,\s*&(\w+)', cuda)
            if pb:
                # Find that fn signature
                fn_def_m = re.search(rf'\b(?:torch::Tensor|at::Tensor|void)\s+{pb.group(2)}\s*\(([^)]*)\)', cuda)
                if fn_def_m:
                    args = re.sub(r'\s+', ' ', fn_def_m.group(1))[:250]
                    pybind_signatures[f"L{level}_T{task}_{op}"] = (pb.group(1), args)

    print(f"Total correct entries: {total_correct}")
    print(f"\nTop-level fn name distribution (top 10):")
    for n, c in sorted(fn_name_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n}: {c}")
    print(f"\nEntries with `fn=module_fn` keyword default: {has_module_fn_kw}/{total_correct}")

    print("\nSample PyTorch_Code_Functional signatures (first 8):")
    for k, sig in list(fn_signatures.items())[:8]:
        print(f"  [{k}]")
        print(f"    {sig}")

    print("\nSample CUDA pybind signatures (first 8):")
    for k, (fn, args) in list(pybind_signatures.items())[:8]:
        print(f"  [{k}] m.def(\"{fn}\") -> {args}")


level_files = [
    ROOT / "external_benchmarks/ai_cuda_engineer/parquet/level_1.parquet",
    ROOT / "external_benchmarks/ai_cuda_engineer/parquet/level_2.parquet",
    ROOT / "external_benchmarks/ai_cuda_engineer/parquet/level_3.parquet",
]
scan_all(level_files)

