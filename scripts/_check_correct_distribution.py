#!/usr/bin/env python3
"""Confirm the Correct=True/False distribution in the Sakana parquet
and the composition of our current registry."""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

print("=" * 72)
print("Step 1: Sakana parquet Correct distribution")
print("=" * 72)

total_rows = 0
total_true = 0
total_false = 0
unique_tasks_true = set()
unique_tasks_false = set()

for level in [1, 2, 3]:
    df = pd.read_parquet(ROOT / f"external_benchmarks/ai_cuda_engineer/parquet/level_{level}.parquet")
    n = len(df)
    n_true = int((df["Correct"] == True).sum())
    n_false = int((df["Correct"] == False).sum())
    n_other = n - n_true - n_false
    total_rows += n
    total_true += n_true
    total_false += n_false
    print(f"\n  level_{level}: {n} rows")
    print(f"    Correct=True : {n_true:5d}  ({n_true/n*100:.1f}%)")
    print(f"    Correct=False: {n_false:5d}  ({n_false/n*100:.1f}%)")
    if n_other:
        print(f"    Other        : {n_other:5d}")
    unique_t = set()
    unique_f = set()
    for _, row in df.iterrows():
        key = (row["Level_ID"], row["Task_ID"])
        if row["Correct"] == True:
            unique_t.add(key)
            unique_tasks_true.add(key)
        else:
            unique_f.add(key)
            unique_tasks_false.add(key)
    print(f"    Distinct (Level,Task) with at least one Correct=True : {len(unique_t)}")
    print(f"    Distinct (Level,Task) with at least one Correct=False: {len(unique_f)}")

print(f"\n  TOTAL: {total_rows} rows")
print(f"    Correct=True : {total_true:5d} rows  ({total_true/total_rows*100:.1f}%)")
print(f"    Correct=False: {total_false:5d} rows  ({total_false/total_rows*100:.1f}%)")
print(f"    Distinct tasks with at least one Correct=True : {len(unique_tasks_true)}")
print(f"    Distinct tasks ONLY having Correct=False (none correct): "
      f"{len(unique_tasks_false - unique_tasks_true)}")

print()
print("=" * 72)
print("Step 2: Current registry composition")
print("=" * 72)

reg_path = ROOT / "external_benchmarks/ai_cuda_engineer/registry.json"
with open(reg_path, encoding="utf-8") as f:
    registry = json.load(f)

print(f"\n  Total entries in registry.json: {len(registry)}")
print(f"  Filtering rule (in prepare_external_datasets.py):")
print(f"    1. Correct == True (filtered at parquet load)")
print(f"    2. Best speedup per (Level_ID, Task_ID)  [dedup]")
print(f"    3. Has both CUDA_Code and PyTorch_Code_Functional")

# Distribution by level
by_level = {1: 0, 2: 0, 3: 0}
for e in registry:
    lid = e.get("level_id", 0)
    by_level[lid] = by_level.get(lid, 0) + 1
print(f"\n  Registry distribution by level:")
for lid, cnt in sorted(by_level.items()):
    print(f"    Level {lid}: {cnt} kernels")

print()
print("=" * 72)
print("Conclusion:")
print("=" * 72)
print(f"""
  - Sakana DOES have Correct=True/False labels (binary).
  - Of {total_rows} total entries: {total_true} ({total_true/total_rows*100:.1f}%) are Correct=True,
    and {total_false} ({total_false/total_rows*100:.1f}%) are Correct=False.
  - Many tasks have multiple attempts; we already keep ONLY the best
    Correct=True attempt per (Level_ID, Task_ID), giving {len(registry)} kernels in
    the registry.
  - This is the right starting set: all entries are claimed correct
    by Sakana. Any baseline failure we observe is a "Sakana validator
    miss" or hardware/wrapper issue.
""")
