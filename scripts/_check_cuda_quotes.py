#!/usr/bin/env python3
"""Sanity check for embedding CUDA_Code into Python raw r\"\"\"...\"\"\" string."""
from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

triple = 0
ends_with_quote = 0
total = 0
for level in [1, 2, 3]:
    df = pd.read_parquet(ROOT / f"external_benchmarks/ai_cuda_engineer/parquet/level_{level}.parquet")
    correct = df[df["Correct"] == True]
    total += len(correct)
    for _, row in correct.iterrows():
        code = str(row["CUDA_Code"])
        if '"""' in code:
            triple += 1
        if code.rstrip().endswith('"'):
            ends_with_quote += 1

print(f"Total correct entries: {total}")
print(f"Contains '\"\"\"' inside CUDA: {triple}")
print(f"Ends with '\"': {ends_with_quote}")
