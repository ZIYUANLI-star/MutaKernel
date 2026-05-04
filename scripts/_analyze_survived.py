#!/usr/bin/env python3
"""Deep analysis of survived mutants: what mutation, why survived."""
import json
from pathlib import Path
from collections import Counter

D = Path(__file__).resolve().parent.parent / "第二次实验汇总" / "stress_enhance_results" / "details"

survived = []
for f in sorted(D.glob("*.json")):
    data = json.loads(f.read_text())
    if data.get("any_killed"):
        continue
    survived.append(data)

print(f"{'='*70}")
print(f"  Survived Mutants Deep Analysis ({len(survived)})")
print(f"{'='*70}")

# Categorize by mutation pattern
pattern_groups = Counter()

for i, data in enumerate(survived):
    mid = data["mutant_id"]
    op = data.get("operator_name", "?")
    ed = data.get("equiv_detail", {})

    # Extract mutation diff
    l0 = ed.get("layer0", {})
    diffs = l0.get("cuda_diff_lines", [])
    diff_str = ""
    for d in diffs:
        orig = d.get("original", "")
        mut = d.get("mutated", "")
        diff_str = f"{orig}  →  {mut}"

    # LLM analysis
    llm = data.get("llm_iterative_analysis", {})
    rounds = llm.get("rounds", [])
    r1 = rounds[0] if rounds else {}
    reason_cat = r1.get("reason_category", "?")
    proof = r1.get("proof_sketch", "")
    killable = r1.get("killable", "?")

    # How many dimensions tested, how many ref_fail
    mt = data.get("main_track", {})
    ref_fail_count = 0
    total_policy_runs = 0
    for dim_name, dim_data in mt.items():
        for pr in dim_data.get("policy_results", []) + dim_data.get("results", []):
            total_policy_runs += 1
            if not pr.get("ref_ok", True) or pr.get("ref_fail"):
                ref_fail_count += 1

    print(f"\n{'─'*70}")
    print(f"[{i+1}] {mid}")
    print(f"  Operator: {op}")
    print(f"  Mutation: {diff_str}")
    print(f"  LLM reason: {reason_cat}")
    print(f"  LLM proof:  {proof[:200]}")
    print(f"  Policy runs: {total_policy_runs}, ref_fail: {ref_fail_count}")

    pattern_groups[reason_cat] += 1

print(f"\n{'='*70}")
print(f"  Summary by reason_category:")
for cat, cnt in pattern_groups.most_common():
    print(f"    {cat}: {cnt}")
