#!/usr/bin/env python3
"""Quick diagnostic script for FusedSoftmax results."""
import json, sys

path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

print("=== Baseline (first 3) ===")
for r in d["baseline"]["results"][:3]:
    print(json.dumps(r, indent=2))

print("\n=== First stress policy ===")
for k, v in list(d["stress"].items())[:1]:
    print(f"{k}:", json.dumps(v, indent=2))
