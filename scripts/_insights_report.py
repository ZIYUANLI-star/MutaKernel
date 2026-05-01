"""Extract actionable insights: test rules from kills + robustness findings."""
import json
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent
DETAILS_DIR = PROJECT / "llm_analysis_results" / "details"

killed = []
survived = []

for jf in sorted(DETAILS_DIR.glob("*.json")):
    d = json.loads(jf.read_text())
    if d.get("killed_by_llm"):
        killed.append(d)
    else:
        survived.append(d)

# ===== Part 1: Test Construction Rules from Killed Mutants =====
print("=" * 80)
print("  PART 1: TEST CONSTRUCTION RULES (from killed mutants)")
print("=" * 80)

for d in killed:
    print(f"\n{'─'*70}")
    print(f"  {d['mutant_id']}  (op={d['operator_name']}, kernel={d['kernel_name']})")

    rule = d.get("test_construction_rule")
    if rule:
        print(f"  Rule Name: {rule.get('rule_name', 'N/A')}")
        print(f"  Description: {rule.get('rule_description', 'N/A')}")
        print(f"  Applicable Ops: {rule.get('applicable_operators', [])}")
        policy = rule.get("policy_code", "")
        if policy:
            print(f"  Policy Code:")
            for line in policy.strip().split("\n"):
                print(f"    {line}")
    else:
        print(f"  [No test construction rule extracted]")

    # Also show the killing input pattern
    for r in d.get("rounds", []):
        if r.get("killed"):
            print(f"\n  Killing Input (Round {r['round']}):")
            code = r.get("suggested_code", "")
            if code:
                for line in code.strip().split("\n"):
                    print(f"    {line}")
            break

# ===== Part 2: Robustness Issues in Code Under Test =====
print(f"\n\n{'=' * 80}")
print("  PART 2: CODE-UNDER-TEST WEAKNESSES (from survived mutants)")
print("=" * 80)

# Group by kernel
by_kernel = defaultdict(list)
for d in survived:
    by_kernel[d["kernel_name"]].append(d)

for kernel, mutants in sorted(by_kernel.items()):
    print(f"\n{'━'*70}")
    print(f"  Kernel: {kernel}")
    print(f"  Survived mutants: {len(mutants)}")

    # Collect unique robustness themes
    for d in mutants:
        rob = d.get("robustness_suggestion", "")
        if not rob:
            continue
        print(f"\n  ┌─ {d['mutant_id']} (op={d['operator_name']})")
        print(f"  │  Survival: {d.get('survival_reason', '')[:150]}...")
        print(f"  │  Robustness Issue:")
        for line in rob.split(". "):
            line = line.strip()
            if line:
                print(f"  │    • {line}.")
        print(f"  └─")

# ===== Part 3: orig_also_fails cases (test code bugs) =====
print(f"\n\n{'=' * 80}")
print("  PART 3: ORIGINAL KERNEL ALSO FAILS (potential test code bugs)")
print("=" * 80)

orig_fail_count = 0
for d in survived:
    for r in d.get("rounds", []):
        detail = r.get("detail", {})
        if detail.get("original_ok") is False and detail.get("ref_ok") is True:
            orig_fail_count += 1
            diff = detail.get("diff_summary", "")
            print(f"\n  {d['mutant_id']} Round {r['round']}:")
            print(f"    ref_ok=True, original_ok=False, mutant_ok={detail.get('mutant_ok')}")
            if diff:
                print(f"    Diff: {diff[:150]}")
            print(f"    Survival reason: {d.get('survival_reason', '')[:200]}")

if orig_fail_count == 0:
    print("\n  (No cases found)")

# ===== Part 4: Summary Statistics =====
print(f"\n\n{'=' * 80}")
print("  SUMMARY")
print("=" * 80)
rules_count = sum(1 for d in killed if d.get("test_construction_rule"))
rob_count = sum(1 for d in survived if d.get("robustness_suggestion"))
print(f"  Killed mutants: {len(killed)}")
print(f"  Test rules extracted: {rules_count}")
print(f"  Survived mutants: {len(survived)}")
print(f"  With robustness advice: {rob_count}")
print(f"  Original-also-fails: {orig_fail_count}")
