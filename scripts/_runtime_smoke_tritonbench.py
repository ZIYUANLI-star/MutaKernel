#!/usr/bin/env python3
"""Runtime smoke test for TritonBench-G prep fixes (CPU-only, no GPU).

Verifies that each problem_*.py file:
  1. Can be parsed as Python (ast.parse).
  2. Defines class Model with a forward signature whose arity matches
     ``len(get_inputs())`` -- the exact bug class flagged in the analysis.
  3. The kernel_source string also has a ModelNew with matching arity.
  4. ``kernel_source`` does not contain a recursive ``return forward(...)``
     pattern that would NameError at runtime.

We do NOT actually execute the .py files because triton kernel imports
require a GPU. AST analysis catches the bugs that caused the original
49 skips (NameError + TypeError) without needing a CUDA device.
"""
from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

reg_path = PROJECT_ROOT / "external_benchmarks" / "tritonbench_g" / "registry.json"
data = json.load(open(reg_path, encoding="utf-8"))


def _forward_arity(src: str, class_name: str) -> int:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return -2
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return len([a for a in item.args.args if a.arg != "self"])
    return -1


def _get_inputs_count(src: str) -> int:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return -2
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
                    return len(stmt.value.elts)
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
                    return 0
    return -1


def _modelnew_calls_undefined_forward(kernel_src: str) -> bool:
    """Check kernel_source's ModelNew.forward doesn't recursively call bare ``forward()``."""
    try:
        tree = ast.parse(kernel_src)
    except SyntaxError:
        return False
    module_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    module_names.add(t.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    for stmt in ast.walk(item):
                        if (isinstance(stmt, ast.Call) and
                                isinstance(stmt.func, ast.Name)):
                            fn = stmt.func.id
                            if fn == "forward" and "forward" not in module_names:
                                return True
    return False


def _module_names(src: str) -> set[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    names = {"torch", "nn", "F", "triton", "tl"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, ast.Tuple):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _undef_in_class_forward(src: str, class_name: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    mod_names = _module_names(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    params = {a.arg for a in item.args.args}
                    used = {n.id for n in ast.walk(item) if isinstance(n, ast.Name)}
                    return sorted(used - params - mod_names - {"self"})
    return []


bugs: list[tuple[str, str]] = []
ok = 0
mode_status: Counter[str] = Counter()

for entry in data:
    kid = entry["id"]
    mode = entry.get("ref_mode", "?")

    ref_path_rel = entry["reference_file"]
    ref_path = (PROJECT_ROOT / ref_path_rel
                if not Path(ref_path_rel).is_absolute()
                else Path(ref_path_rel))
    if not ref_path.exists():
        bugs.append((kid, f"problem missing"))
        continue

    problem_src = ref_path.read_text(encoding="utf-8")
    kernel_src = entry.get("kernel_source", "")

    p_arity = _forward_arity(problem_src, "Model")
    p_inputs = _get_inputs_count(problem_src)
    if p_arity != p_inputs:
        bugs.append((kid, f"problem arity mismatch: forward({p_arity}) vs get_inputs={p_inputs}"))
        continue

    k_arity = _forward_arity(kernel_src, "ModelNew")
    if k_arity != p_inputs:
        bugs.append((kid, f"kernel ModelNew arity mismatch: forward({k_arity}) vs get_inputs={p_inputs}"))
        continue

    p_undef = _undef_in_class_forward(problem_src, "Model")
    if p_undef:
        bugs.append((kid, f"Model.forward references undefined names: {p_undef}"))
        continue

    k_undef = _undef_in_class_forward(kernel_src, "ModelNew")
    if k_undef:
        bugs.append((kid, f"ModelNew.forward references undefined names: {k_undef}"))
        continue

    if _modelnew_calls_undefined_forward(kernel_src):
        bugs.append((kid, "ModelNew calls undefined forward()"))
        continue

    mode_status[mode] += 1
    ok += 1


print(f"Total kernels: {len(data)}")
print(f"OK: {ok}")
print(f"Bugs: {len(bugs)}")
print()
print("OK by ref_mode:")
for m, c in mode_status.most_common():
    print(f"  {m}: {c}")
print()
if bugs:
    print(f"Sample bugs (first 20):")
    for kid, reason in bugs[:20]:
        print(f"  {kid}: {reason}")
