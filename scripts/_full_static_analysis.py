#!/usr/bin/env python3
"""Deeper static analysis: for each BAD/ERR kernel, print:
- module_fn signature & body
- CUDA forward signature
- Any reduction/atomicAdd/output-shape issues
- Matched signature length (should equal)
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
bad.sort(key=lambda x: (x.get("level_id", 0), x.get("name", "")))

for r in bad:
    kid = r["id"]
    entry = registry.get(kid, {})
    ref_path = ROOT / entry.get("reference_file", "")
    cuda = entry.get("kernel_source", "")

    print("=" * 88)
    print(f"  {kid}   |   {entry.get('kernel_name','')}   |   {r['verdict']}  p/f/e={r.get('baseline_pass',0)}/{r.get('baseline_fail',0)}/{r.get('baseline_err',0)}")
    print("=" * 88)

    # 1. module_fn signature
    if ref_path.exists():
        text = ref_path.read_text(encoding="utf-8")
        m = re.search(r'def module_fn\(([^)]*)\)', text, re.DOTALL)
        py_args = m.group(1).strip() if m else "?"
        py_args_clean = " ".join(py_args.split())
        print(f"  module_fn args: {py_args_clean[:200]}")

        # 2. one-liner of body
        m = re.search(r'def module_fn\([^)]*\)[^:]*:\s*\n(?:    """(?:[^"]|"[^"])*"""\s*\n)?\s*(.+?)(?=\n(?:def |class |\Z))',
                      text, re.DOTALL)
        if m:
            # Find return statement
            body_lines = [l.strip() for l in m.group(1).split("\n") if l.strip() and not l.strip().startswith("#")]
            for line in body_lines[:3]:
                print(f"  body: {line[:160]}")

    # 3. CUDA forward signature (params)
    fwd = re.findall(r'(?:torch::Tensor|at::Tensor)\s+forward\s*\(([^)]*)\)', cuda, re.DOTALL)
    if fwd:
        cu_args = " ".join(fwd[0].split())
        print(f"  CUDA forward args: {cu_args[:200]}")
        n_py = py_args_clean.count(",") + 1 if py_args_clean and py_args_clean != "?" else 0
        n_cu = cu_args.count(",") + 1 if cu_args else 0
        match = "MATCH" if n_py == n_cu else f"MISMATCH (py={n_py}, cu={n_cu})"
        print(f"  arg-count check: {match}")

    # 4. Reduction hints
    reductions = []
    if "atomicAdd" in cuda:
        reductions.append("atomicAdd")
    if "__shfl_down" in cuda or "warp_reduce" in cuda.lower():
        reductions.append("warp-reduce")
    if "__syncthreads" in cuda:
        reductions.append("block-sync")
    if "/ static_cast" in cuda or "/ float(" in cuda:
        reductions.append("manual-divide")
    if reductions:
        print(f"  reduction style: {','.join(reductions)}")

    # 5. Output shape signature: does it return scalar via empty({1}) or {} ?
    out_shape_hints = []
    if "torch::zeros({1}" in cuda or "torch::empty({1}" in cuda:
        out_shape_hints.append("scalar-as-[1]")
    if "torch::zeros({})" in cuda or "torch::empty({})" in cuda:
        out_shape_hints.append("scalar-as-[]")
    if out_shape_hints:
        print(f"  output shape: {','.join(out_shape_hints)}")

    # 6. Special: count how many distinct kernel names (loss, fused etc.)
    kernel_calls = re.findall(r'(\w+_kernel)\s*<<<', cuda)
    kernel_calls = list(set(kernel_calls))
    if kernel_calls:
        print(f"  CUDA kernels invoked: {kernel_calls[:6]}")

    print()
