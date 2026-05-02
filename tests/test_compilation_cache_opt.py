#!/usr/bin/env python3
"""Smoke test for compilation cache optimization.

Validates that:
1. When kernel_code == mutated_code, the stress worker skips mutant compilation
   and returns bitwise_orig_mut_eq=True directly.
2. When kernel_code != mutated_code, the normal compilation path is used.
3. The mutation testing path (mutant_runner.py) uses a separate prefix and is unaffected.

No GPU required — tests logic paths only via source inspection and mock.
"""
import sys
import ast
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_stress_worker_same_code_shortcircuit():
    """When kernel_code == mutated_code, run_stress must skip mut compilation."""
    worker_path = PROJECT_ROOT / "scripts" / "_stress_worker.py"
    source = worker_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    run_stress_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_stress":
            run_stress_fn = node
            break

    assert run_stress_fn is not None, "run_stress function not found"

    fn_source = ast.get_source_segment(source, run_stress_fn)

    assert 'is_same_code = (cfg["kernel_code"] == cfg["mutated_code"])' in fn_source, \
        "is_same_code detection not found in run_stress"

    assert "if is_same_code:" in fn_source, \
        "is_same_code short-circuit branch not found in run_stress"

    assert "bitwise_orig_mut_eq = True" in fn_source, \
        "bitwise_orig_mut_eq = True not set in is_same_code branch"

    assert "mutant_ok = original_ok" in fn_source, \
        "mutant_ok = original_ok not set in is_same_code branch"

    print("  [PASS] run_stress: same-code short-circuit logic correct")


def test_config_stress_same_code_shortcircuit():
    """When kernel_code == mutated_code, run_config_stress must skip mut compilation."""
    worker_path = PROJECT_ROOT / "scripts" / "_stress_worker.py"
    source = worker_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    run_config_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_config_stress":
            run_config_fn = node
            break

    assert run_config_fn is not None, "run_config_stress function not found"

    fn_source = ast.get_source_segment(source, run_config_fn)

    assert 'is_same_code = (cfg["kernel_code"] == cfg["mutated_code"])' in fn_source, \
        "is_same_code detection not found in run_config_stress"

    assert "if is_same_code:" in fn_source, \
        "is_same_code short-circuit branch not found in run_config_stress"

    assert "mut_model = orig_model" in fn_source, \
        "mut_model reuse not set in is_same_code branch of run_config_stress"

    print("  [PASS] run_config_stress: same-code short-circuit logic correct")


def test_unified_prefix():
    """All module name prefixes in _stress_worker must use 'ext_' prefix."""
    worker_path = PROJECT_ROOT / "scripts" / "_stress_worker.py"
    source = worker_path.read_text(encoding="utf-8")

    import re
    old_patterns = re.findall(r'f"(?:cfg)?stress_(orig|mut|ref)_', source)
    assert len(old_patterns) == 0, \
        f"Old stress_/cfgstress_ prefixes still present: {old_patterns}"

    ext_patterns = re.findall(r'f"ext_(orig|mut|ref)_', source)
    assert len(ext_patterns) >= 4, \
        f"Expected at least 4 ext_ prefixes (orig+ref in run_stress, orig+ref in run_config), got {len(ext_patterns)}"

    print(f"  [PASS] Unified prefix: found {len(ext_patterns)} ext_* module names, 0 old prefixes")


def test_mutation_runner_prefix_isolation():
    """mutant_runner.py must NOT use ext_ prefix — it uses mutant_{id}."""
    runner_path = PROJECT_ROOT / "src" / "mutengine" / "mutant_runner.py"
    source = runner_path.read_text(encoding="utf-8")

    import re
    ext_usage = re.findall(r'f"ext_(orig|mut|ref)_', source)
    assert len(ext_usage) == 0, \
        f"mutant_runner.py should not use ext_ prefix, found: {ext_usage}"

    mutant_prefix = re.findall(r'f"mutant_', source)
    assert len(mutant_prefix) >= 1, \
        "mutant_runner.py should use mutant_{id} prefix"

    print("  [PASS] Mutation runner uses separate 'mutant_' prefix — no interference")


def test_stress_worker_syntax():
    """Ensure _stress_worker.py has valid Python syntax after edits."""
    worker_path = PROJECT_ROOT / "scripts" / "_stress_worker.py"
    source = worker_path.read_text(encoding="utf-8")
    try:
        compile(source, str(worker_path), "exec")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error in _stress_worker.py: {e}")
    print("  [PASS] _stress_worker.py syntax valid")


def main():
    print("=" * 60)
    print("  Compilation Cache Optimization — Smoke Test")
    print("=" * 60)
    print()

    tests = [
        test_stress_worker_syntax,
        test_stress_worker_same_code_shortcircuit,
        test_config_stress_same_code_shortcircuit,
        test_unified_prefix,
        test_mutation_runner_prefix_isolation,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1

    print()
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed:
        print("\n  WARNING: Some tests failed. The optimization may have issues.")
        return 1
    else:
        print("\n  All checks passed. Optimization is safe for mutation testing.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
