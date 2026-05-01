"""Analyze policy-level results from stress testing."""
import json
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
STRESS = PROJECT / "stress_enhance_results" / "details"

stats = Counter()
orig_fail_policies = Counter()
ref_fail_policies = Counter()
killed_policies = Counter()

for jf in sorted(STRESS.glob("*.json")):
    d = json.loads(jf.read_text())
    for pr in d.get("policy_results", []):
        pol = pr["policy"]
        if not pr["ref_ok"]:
            stats["REF_FAIL"] += 1
            ref_fail_policies[pol] += 1
        elif not pr["original_ok"]:
            stats["ORIG_ALSO_FAILS"] += 1
            orig_fail_policies[pol] += 1
        elif not pr["mutant_ok"]:
            stats["KILLED"] += 1
            killed_policies[pol] += 1
        else:
            stats["both_OK"] += 1

print("=== Overall policy-level results ===")
total = sum(stats.values())
for k, v in stats.most_common():
    print(f"  {k}: {v} ({v/total*100:.1f}%)")
print(f"  TOTAL test executions: {total}")

print("\n=== REF_FAIL by policy (ref impl crashes on stress input) ===")
for k, v in ref_fail_policies.most_common():
    print(f"  {k}: {v}")

print("\n=== ORIG_ALSO_FAILS by policy (original kernel also fails) ===")
for k, v in orig_fail_policies.most_common():
    print(f"  {k}: {v}")

print("\n=== KILLED by policy ===")
for k, v in killed_policies.most_common():
    print(f"  {k}: {v}")

# Per-mutant summary
print("\n=== Per-mutant summary ===")
for jf in sorted(STRESS.glob("*.json")):
    d = json.loads(jf.read_text())
    mid = d["mutant_id"]
    op = d["operator_name"]
    cat = d["operator_category"]
    killed = d.get("killed", False)
    
    policy_stats = Counter()
    for pr in d.get("policy_results", []):
        if not pr["ref_ok"]:
            policy_stats["ref_fail"] += 1
        elif not pr["original_ok"]:
            policy_stats["orig_fail"] += 1
        elif not pr["mutant_ok"]:
            policy_stats["killed"] += 1
        else:
            policy_stats["both_ok"] += 1
    
    total_pr = sum(policy_stats.values())
    status = "KILLED" if killed else "survived"
    detail = ", ".join(f"{k}={v}" for k, v in policy_stats.most_common())
    print(f"  [{status:8s}] {mid:40s} | {op:18s} ({cat}) | {detail}")
