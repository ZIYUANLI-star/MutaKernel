#!/usr/bin/env python3
"""Syntax-check every wrapped kernel_source and the problem_*.py reference.

This catches any Python-level issue (NameError, IndentError, etc.) before
we try to compile CUDA. It does NOT execute load_inline.
"""
from __future__ import annotations
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "external_benchmarks/ai_cuda_engineer/registry.json"

with open(REG, encoding="utf-8") as f:
    data = json.load(f)

ok = 0
ks_fail = []
ref_fail = []

for entry in data:
    kid = entry["id"]
    try:
        ast.parse(entry["kernel_source"])
    except SyntaxError as e:
        ks_fail.append((kid, str(e)))
        continue

    ref = ROOT / entry["reference_file"]
    if not ref.exists():
        ref_fail.append((kid, "FILE NOT FOUND"))
        continue
    try:
        ref_src = ref.read_text(encoding="utf-8")
        ast.parse(ref_src)
    except SyntaxError as e:
        ref_fail.append((kid, str(e)))
        continue
    ok += 1

print(f"Total entries: {len(data)}")
print(f"OK (both kernel_source and reference parse): {ok}")
print(f"Kernel-source syntax errors: {len(ks_fail)}")
for kid, err in ks_fail[:5]:
    print(f"  - {kid}: {err}")
print(f"Reference syntax errors: {len(ref_fail)}")
for kid, err in ref_fail[:5]:
    print(f"  - {kid}: {err}")

# Also verify ModelNew is defined in every kernel_source
import re
no_modelnew = []
no_model_in_ref = []
for entry in data:
    if not re.search(r'^class\s+ModelNew\b', entry["kernel_source"], re.MULTILINE):
        no_modelnew.append(entry["id"])
    ref = (ROOT / entry["reference_file"]).read_text(encoding="utf-8")
    if not re.search(r'^class\s+Model\b', ref, re.MULTILINE):
        no_model_in_ref.append(entry["id"])

print(f"\nMissing ModelNew in kernel_source: {len(no_modelnew)}")
for kid in no_modelnew[:5]:
    print(f"  - {kid}")
print(f"Missing Model in reference: {len(no_model_in_ref)}")
for kid in no_model_in_ref[:5]:
    print(f"  - {kid}")

if ks_fail or ref_fail or no_modelnew or no_model_in_ref:
    sys.exit(1)
