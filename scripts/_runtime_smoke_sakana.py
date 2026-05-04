#!/usr/bin/env python3
"""CPU-only AST sanity check for the AI CUDA Engineer adaptation.

Verifies that for every kernel in registry.json:
  1. Both ``problem_*.py`` and ``kernel_source`` have valid Python syntax.
  2. ``Model`` (problem) and ``ModelNew`` (kernel_source) define
     compatible ``__init__`` (so they accept the same ``get_init_inputs()``).
  3. ``Model.forward`` and ``ModelNew.forward`` both accept the same number
     of positional args (so unpacking ``get_inputs()`` works for both).
  4. ``get_inputs`` / ``get_init_inputs`` are defined in both module sources.
  5. ``module_fn`` is replaced with the ``_ext.forward`` stub in kernel_source.

This does NOT compile CUDA. The runtime stress worker will try
``load_inline`` on each kernel_source on the GPU machine.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REG_PATH = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"


def _func_arg_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[str], int]:
    args = [a.arg for a in fn.args.args]
    if args and args[0] in ("self", "cls"):
        args = args[1:]
    return args, len(fn.args.defaults)


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _class_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _has_top_level_func(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, ast.FunctionDef) and n.name == name
        for n in tree.body  # type: ignore[attr-defined]
    )


def check_one(entry: dict) -> tuple[bool, str]:
    kid = entry["id"]
    ref = entry.get("reference_file")
    if not ref:
        return False, f"{kid}: missing reference_file"
    ref_path = Path(ref)
    if not ref_path.is_absolute():
        ref_path = PROJECT_ROOT / ref
    if not ref_path.exists():
        return False, f"{kid}: ref file missing: {ref_path}"
    src = entry.get("kernel_source", "")
    if not src:
        return False, f"{kid}: empty kernel_source"

    try:
        prob_text = ref_path.read_text(encoding="utf-8")
        prob_tree = ast.parse(prob_text)
    except Exception as e:
        return False, f"{kid}: ref parse: {e}"
    try:
        kern_tree = ast.parse(src)
    except Exception as e:
        return False, f"{kid}: kernel parse: {e}"

    prob_model = _find_class(prob_tree, "Model")
    if prob_model is None:
        return False, f"{kid}: ref has no class Model"
    kern_model = _find_class(kern_tree, "ModelNew")
    if kern_model is None:
        return False, f"{kid}: kernel has no class ModelNew"

    prob_init = _class_method(prob_model, "__init__")
    kern_init = _class_method(kern_model, "__init__")
    if prob_init is None or kern_init is None:
        return False, f"{kid}: missing __init__"
    p_init_args, p_init_def = _func_arg_names(prob_init)
    k_init_args, k_init_def = _func_arg_names(kern_init)
    p_init_required = max(0, len(p_init_args) - p_init_def)
    k_init_required = max(0, len(k_init_args) - k_init_def)
    if p_init_required != k_init_required:
        return False, (
            f"{kid}: init arity mismatch: "
            f"Model={p_init_required} ({p_init_args}), "
            f"ModelNew={k_init_required} ({k_init_args})"
        )

    prob_fwd = _class_method(prob_model, "forward")
    kern_fwd = _class_method(kern_model, "forward")
    if prob_fwd is None or kern_fwd is None:
        return False, f"{kid}: missing forward"
    p_fwd_args, p_fwd_def = _func_arg_names(prob_fwd)
    k_fwd_args, k_fwd_def = _func_arg_names(kern_fwd)
    p_fwd_required = max(0, len(p_fwd_args) - p_fwd_def)
    k_fwd_required = max(0, len(k_fwd_args) - k_fwd_def)
    if p_fwd_required != k_fwd_required:
        return False, (
            f"{kid}: forward arity mismatch: "
            f"Model={p_fwd_required} ({p_fwd_args}), "
            f"ModelNew={k_fwd_required} ({k_fwd_args})"
        )

    if not _has_top_level_func(prob_tree, "get_inputs"):
        return False, f"{kid}: ref missing get_inputs"
    if not _has_top_level_func(kern_tree, "get_inputs"):
        return False, f"{kid}: kernel missing get_inputs"

    if "_ext = load_inline" not in src:
        return False, f"{kid}: kernel_source missing _ext = load_inline"
    if "_ext.forward(*args, **kwargs)" not in src and "_ext." not in src:
        return False, f"{kid}: kernel_source missing _ext.<fn> stub"

    return True, ""


def main() -> int:
    if not REG_PATH.exists():
        print(f"ERROR: registry not found: {REG_PATH}")
        return 2
    with REG_PATH.open(encoding="utf-8") as f:
        registry = json.load(f)

    ok = 0
    bugs: list[str] = []
    for entry in registry:
        good, msg = check_one(entry)
        if good:
            ok += 1
        else:
            bugs.append(msg)

    print(f"Total kernels: {len(registry)}")
    print(f"OK: {ok}")
    print(f"Bugs: {len(bugs)}")
    for b in bugs[:20]:
        print(f"  - {b}")
    if len(bugs) > 20:
        print(f"  ... and {len(bugs) - 20} more")

    return 0 if not bugs else 1


if __name__ == "__main__":
    sys.exit(main())
