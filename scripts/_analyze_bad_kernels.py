#!/usr/bin/env python3
"""Static analysis of BAD kernels in the live baseline run.

Reads the current checkpoint, finds all BAD/ERR kernels, extracts:
  - reference_file's module_fn signature & body
  - kernel_source's CUDA forward() signature
  - Reduction semantics for losses
  - Any obvious mismatches

Categorizes each into:
  A. SAKANA_VALIDATOR_MISS   (true numerical bug in their kernel; ours is correct)
  B. WRAPPER_BUG             (our wrapper introduces an error)
  C. HARDWARE_LIMIT          (OOM / unsupported on small GPU)
  D. UNKNOWN                 (need runtime debug)
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "第三次实验汇总" / "results" / "ai_cuda_engineer" / "baseline_checkpoint.json"
REG = ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"

with open(CKPT, encoding="utf-8") as f:
    ckpt = json.load(f)
with open(REG, encoding="utf-8") as f:
    reg_list = json.load(f)
registry = {e["id"]: e for e in reg_list}

bad = [r for r in ckpt.values() if r.get("verdict") in ("BAD", "ERR")]
print(f"Total BAD/ERR so far: {len(bad)}")
print()

# Extract pybind forward signature from CUDA code
def extract_pybind_args(cuda_code: str) -> str | None:
    # Look for forward(...) - typical patterns
    matches = re.findall(r'(?:torch::Tensor|at::Tensor)\s+forward\s*\(([^)]*)\)', cuda_code)
    if matches:
        return matches[0].strip()
    return None


def extract_module_fn_args(ref_path: Path) -> str | None:
    if not ref_path.exists():
        return None
    text = ref_path.read_text(encoding="utf-8")
    # Note: our wrapper rewrote module_fn body, so we look for it in the
    # raw functional_source on the registry instead.
    m = re.search(r'def\s+module_fn\s*\(([^)]*)\)', text)
    return m.group(1).strip() if m else None


def extract_module_fn_body(ref_path: Path) -> str | None:
    if not ref_path.exists():
        return None
    text = ref_path.read_text(encoding="utf-8")
    # Get the function body - lines after def module_fn until next def or class
    m = re.search(r'def\s+module_fn\s*\([^)]*\)[^:]*:\n((?:    .*\n)+)', text)
    return m.group(1) if m else None


# Categorize
print("=" * 88)
print(f"{'Kernel':<22} {'Verdict':<5} {'Time':>6}  {'p/f/e':<7}  Module fn signature")
print("-" * 88)

for r in sorted(bad, key=lambda x: x.get("level_id", 0) * 1000 + int(x["id"].split("T")[-1])):
    kid = r["id"]
    entry = registry.get(kid, {})
    ref_path = ROOT / entry.get("reference_file", "")
    cuda_code = entry.get("kernel_source", "")

    mod_args = extract_module_fn_args(ref_path) or "?"
    mod_body = extract_module_fn_body(ref_path) or ""

    # Truncate
    mod_args_short = mod_args[:60] + "..." if len(mod_args) > 60 else mod_args
    pfe = f"{r.get('baseline_pass',0)}/{r.get('baseline_fail',0)}/{r.get('baseline_err',0)}"

    print(f"{kid:<22} {r['verdict']:<5} {r.get('elapsed_s',0):>5.1f}s  {pfe:<7}  {mod_args_short}")
    if mod_body:
        body_first_line = mod_body.strip().split("\n")[0].strip()[:80]
        print(f"{'':<22} {'':<5} {'':<6}  {'':<7}  body: {body_first_line}")
    if r.get("reason"):
        print(f"{'':<22} {'':<5} {'':<6}  reason: {r['reason'][:80]}")
print("=" * 88)
print()

# Category-by-category analysis
print()
print("=" * 88)
print("CATEGORY ANALYSIS")
print("=" * 88)

losses = []
matmuls = []
norms = []
composites = []
others = []
hardware = []

for r in bad:
    name = (r.get("name", "") or "").lower()
    if r["verdict"] == "ERR":
        hardware.append(r)
    elif "loss" in name:
        losses.append(r)
    elif "matmul" in name or "gemm" in name:
        matmuls.append(r)
    elif "norm" in name and r.get("level_id", 0) == 1:
        norms.append(r)
    elif r.get("level_id", 0) >= 2:
        composites.append(r)
    else:
        others.append(r)

if losses:
    print(f"\n--- Losses ({len(losses)}) ---")
    for r in losses:
        print(f"  {r['id']:<22}  {r.get('name','')}")

if matmuls:
    print(f"\n--- Matmuls ({len(matmuls)}) ---")
    for r in matmuls:
        print(f"  {r['id']:<22}  {r.get('name','')}")

if norms:
    print(f"\n--- L1 Norms ({len(norms)}) ---")
    for r in norms:
        print(f"  {r['id']:<22}  {r.get('name','')}")

if composites:
    print(f"\n--- L2/L3 Composites ({len(composites)}) ---")
    for r in composites:
        print(f"  {r['id']:<22}  {r.get('name','')}")

if hardware:
    print(f"\n--- Hardware/ERR ({len(hardware)}) ---")
    for r in hardware:
        print(f"  {r['id']:<22}  {r.get('name','')}")
        if r.get('reason'):
            print(f"      reason: {r['reason'][:120]}")

if others:
    print(f"\n--- Others ({len(others)}) ---")
    for r in others:
        print(f"  {r['id']:<22}  {r.get('name','')}")
