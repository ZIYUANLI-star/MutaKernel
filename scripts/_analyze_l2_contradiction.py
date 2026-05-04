#!/usr/bin/env python3
"""Analyze the contradiction: Layer 2 rejected equivalence, but LLM says equivalent.
What did Layer 2 actually find?"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "第二次实验汇总" / "stress_enhance_results" / "details"

for f in sorted(D.glob("*.json")):
    data = json.loads(f.read_text())
    if data.get("any_killed"):
        continue

    mid = data["mutant_id"]
    ed = data.get("equiv_detail", {})
    l2 = ed.get("layer2", {})

    print(f"\n{'='*70}")
    print(f"{mid}")
    print(f"  Layer 2 is_equivalent: {l2.get('is_equivalent')}")
    print(f"  Layer 2 verdict:       {l2.get('verdict')}")
    print(f"  Layer 2 equiv_runs:    {l2.get('equiv_runs')}")
    print(f"  Layer 2 divergent_run: {l2.get('divergent_run')}")

    # Show divergence detail
    div = l2.get("divergence", {})
    if div:
        print(f"  Divergence detail:")
        print(f"    policy:    {div.get('policy')}")
        print(f"    seed:      {div.get('seed')}")
        print(f"    max_diff:  {div.get('max_diff')}")
        print(f"    mean_diff: {div.get('mean_diff')}")
        print(f"    orig_range: {div.get('orig_range')}")
        print(f"    mut_range:  {div.get('mut_range')}")
    else:
        print(f"  No divergence detail recorded")

    # Show all L2 policy results if available
    l2_policies = l2.get("policies", [])
    if l2_policies:
        passed = [p for p in l2_policies if p.get("verdict") == "passed" or p.get("bitwise_eq")]
        failed = [p for p in l2_policies if p.get("verdict") != "passed" and not p.get("bitwise_eq", True)]
        print(f"  L2 policies: {len(l2_policies)} total, {len(passed)} passed, {len(failed)} diverged")
        for fp in failed[:5]:
            print(f"    DIVERGED: {fp}")

    # Mutation diff for context
    l0 = ed.get("layer0", {})
    diffs = l0.get("cuda_diff_lines", [])
    for d in diffs:
        print(f"  Mutation: {d.get('original','')}  →  {d.get('mutated','')}")

    # What the enhanced testing found
    # Check if tier1_replay succeeded
    mt = data.get("main_track", {})
    replay = mt.get("tier1_replay", {})
    if replay.get("executed"):
        print(f"  Tier1 replay result: killed={replay.get('killed')}, detail={replay.get('detail',{})}")
    else:
        det = replay.get("detail", {})
        print(f"  Tier1 replay: not executed, reason={det.get('reason','?')}")

    # LLM reason
    llm = data.get("llm_iterative_analysis", {})
    rounds = llm.get("rounds", [])
    if rounds:
        r1 = rounds[0]
        print(f"  LLM reason_category: {r1.get('reason_category')}")
        print(f"  LLM proof: {r1.get('proof_sketch','')[:150]}")
