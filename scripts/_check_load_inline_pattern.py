#!/usr/bin/env python3
"""Verify whether CUDA_Code already contains PYBIND11_MODULE.

If yes, load_inline must be called with functions=None (or not specified)
and the embedded PYBIND11_MODULE handles the binding.

If we pass `functions=[...]` while code also has PYBIND11_MODULE, we get
a duplicate registration linker error.
"""
from __future__ import annotations
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with_pybind = 0
without_pybind = 0
total = 0
for level in [1, 2, 3]:
    df = pd.read_parquet(ROOT / f"external_benchmarks/ai_cuda_engineer/parquet/level_{level}.parquet")
    correct = df[df["Correct"] == True]
    for _, row in correct.iterrows():
        total += 1
        code = str(row["CUDA_Code"])
        if "PYBIND11_MODULE" in code:
            with_pybind += 1
        else:
            without_pybind += 1

print(f"Total correct: {total}")
print(f"With PYBIND11_MODULE: {with_pybind}")
print(f"Without PYBIND11_MODULE: {without_pybind}")
