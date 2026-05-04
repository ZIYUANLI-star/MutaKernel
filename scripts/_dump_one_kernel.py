#!/usr/bin/env python3
"""Dump the wrapped kernel_source for a given kid from registry."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
reg = ROOT / "external_benchmarks/ai_cuda_engineer/registry.json"

kid = sys.argv[1] if len(sys.argv) > 1 else "sakana__L1_T34"

with open(reg, encoding="utf-8") as f:
    data = json.load(f)
for e in data:
    if e["id"] == kid:
        ks = e["kernel_source"]
        # Print first ~30 lines and last 50 lines
        lines = ks.split("\n")
        print(f"=== kid={kid}  total lines={len(lines)} ===")
        print("--- HEAD (first 25 lines) ---")
        for i, ln in enumerate(lines[:25], 1):
            print(f"{i:4d}| {ln}")
        print("...")
        print(f"--- TAIL (last 50 lines) ---")
        n = len(lines)
        for i, ln in enumerate(lines[max(0, n - 50):], n - min(50, n) + 1):
            print(f"{i:4d}| {ln}")
        break
else:
    print(f"NOT FOUND: {kid}")
