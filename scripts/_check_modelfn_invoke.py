#!/usr/bin/env python3
"""Inspect how Model.forward invokes module_fn across the dataset."""
from __future__ import annotations
import pandas as pd
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

invoke_patterns = Counter()
forward_sigs = Counter()
samples = {}

for level in [1, 2, 3]:
    df = pd.read_parquet(ROOT / f"external_benchmarks/ai_cuda_engineer/parquet/level_{level}.parquet")
    correct = df[df["Correct"] == True]
    for _, row in correct.iterrows():
        code = str(row["PyTorch_Code_Functional"])
        # Capture the forward method body
        m = re.search(
            r'def\s+forward\s*\(\s*self([^)]*?)\)[^:]*:(.*?)(?=^\s*(?:def|class)\s|\Z)',
            code, re.MULTILINE | re.DOTALL)
        if not m:
            continue
        sig = m.group(1).strip().rstrip(",")
        forward_sigs[sig[:120]] += 1
        body = m.group(2)
        for inv in re.findall(r'(?:fn|module_fn)\s*\(([^)]*)\)', body):
            simple = re.sub(r'\s+', ' ', inv).strip()
            invoke_patterns[simple[:120]] += 1
            if simple not in samples:
                samples[simple] = (level, row["Task_ID"], row["Op_Name"])

print(f"Forward signatures (top 15):")
for sig, c in forward_sigs.most_common(15):
    print(f"  [{c:5d}]  ({sig})")

print(f"\nmodule_fn(...) invocation patterns (top 20):")
for inv, c in invoke_patterns.most_common(20):
    s = samples[inv]
    print(f"  [{c:5d}]  fn({inv})   e.g. L{s[0]}_T{s[1]}_{s[2]}")
