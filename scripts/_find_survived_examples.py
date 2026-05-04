#!/usr/bin/env python3
"""从 JSON 结果中找出 A/B/C 各一个存活变异体的详细信息。"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "full_block12_results" / "details"

examples = {"A": None, "B": None, "C": None}

for jf in sorted(RESULTS_DIR.glob("*.json")):
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
    kernel_name = data["kernel"]["problem_name"]
    for m in data.get("mutants", []):
        cat = m["operator_category"]
        if cat in examples and examples[cat] is None and m["status"] == "survived":
            examples[cat] = {
                "kernel_file": jf.stem,
                "kernel_name": kernel_name,
                "mutant_id": m["id"],
                "operator_name": m["operator_name"],
                "description": m["description"],
                "site": m["site"],
            }
    if all(v is not None for v in examples.values()):
        break

for cat in ["A", "B", "C"]:
    ex = examples[cat]
    if ex is None:
        print(f"\n=== Category {cat}: NO SURVIVED MUTANT FOUND ===")
        continue
    print(f"\n{'='*70}")
    print(f"Category {cat} survived example")
    print(f"{'='*70}")
    print(f"  Kernel: {ex['kernel_file']} ({ex['kernel_name']})")
    print(f"  Mutant ID: {ex['mutant_id']}")
    print(f"  Operator: {ex['operator_name']}")
    print(f"  Description: {ex['description']}")
    print(f"  Site line: {ex['site']['line_start']}")
    print(f"  Original code: {ex['site']['original_code']}")
    print(f"  Node type: {ex['site'].get('node_type', 'N/A')}")
