"""Could the LLM have killed the 50 mutants that stress test killed?
Analyze the killing mechanism to determine if LLM reasoning alone could find them."""
import json
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parent.parent
STRESS_DIR = PROJECT / "stress_enhance_results" / "details"

# Analyze the 50 mutants killed by stress test
killed_analysis = []

for jf in sorted(STRESS_DIR.glob("*.json")):
    d = json.loads(jf.read_text())
    if not d.get("killed"):
        continue
    
    mid = d["mutant_id"]
    op = d["operator_name"]
    policy = d.get("killing_policy", "unknown")
    
    # Check: was the kill due to a specific numerical value, or structural?
    # Look at the policy results to understand what happened
    policy_results = d.get("policy_results", [])
    
    # Count how many policies could kill this mutant
    killing_policies = set()
    both_ok_policies = set()
    orig_fail_policies = set()
    
    for pr in policy_results:
        p = pr["policy"]
        ref = pr.get("ref_ok", False)
        orig = pr.get("original_ok", False)
        mut = pr.get("mutant_ok", True)
        
        if ref and orig and not mut:
            killing_policies.add(p)
        elif ref and orig and mut:
            both_ok_policies.add(p)
        elif not orig:
            orig_fail_policies.add(p)
    
    killed_analysis.append({
        "id": mid, "op": op,
        "killing_policy": policy,
        "n_killing_policies": len(killing_policies),
        "killing_policies": killing_policies,
        "n_both_ok": len(both_ok_policies),
        "n_orig_fail": len(orig_fail_policies),
    })

# Categorize
print("=" * 80)
print("  ANALYSIS: Could LLM replace stress test?")
print("=" * 80)

# Category 1: Only 1 policy kills it (needle in haystack)
needle = [x for x in killed_analysis if x["n_killing_policies"] == 1]
# Category 2: 2-3 policies kill it
few = [x for x in killed_analysis if 2 <= x["n_killing_policies"] <= 3]
# Category 3: Many policies kill it (easy to find)
many = [x for x in killed_analysis if x["n_killing_policies"] > 3]

print(f"\n--- Difficulty of finding the killing input ---")
print(f"  Only 1 policy kills (hard to find):    {len(needle)} / {len(killed_analysis)}")
print(f"  2-3 policies kill (moderate):           {len(few)} / {len(killed_analysis)}")
print(f"  4+ policies kill (easy to find):        {len(many)} / {len(killed_analysis)}")

print(f"\n--- 'Needle' cases (only 1 policy kills) ---")
for x in needle:
    print(f"  {x['id']:45s}  op={x['op']:20s}  killed_by={list(x['killing_policies'])[0]}")

print(f"\n--- Killing policy distribution ---")
all_killing = Counter()
for x in killed_analysis:
    for p in x["killing_policies"]:
        all_killing[p] += 1
for p, c in all_killing.most_common():
    print(f"  {p:30s}: kills {c} mutants")

# Key question: how many of these require specific NUMERICAL values?
print(f"\n--- Nature of killing inputs ---")
value_policies = {"near_zero", "large_magnitude", "near_overflow", "denormals", 
                  "mixed_extremes", "all_negative", "all_positive", "alternating_sign",
                  "head_heavy", "tail_heavy"}
structure_policies = {"structured_ramp", "boundary_last_element", "sparse", "uniform_constant"}

value_kills = sum(1 for x in killed_analysis 
                  if x["killing_policies"] & value_policies and not x["killing_policies"] & structure_policies)
structure_kills = sum(1 for x in killed_analysis 
                      if x["killing_policies"] & structure_policies and not x["killing_policies"] & value_policies)
both_kills = sum(1 for x in killed_analysis 
                 if x["killing_policies"] & value_policies and x["killing_policies"] & structure_policies)

print(f"  Killed only by value policies (near_zero, large_mag, etc.): {value_kills}")
print(f"  Killed only by structure policies (ramp, boundary, sparse): {structure_kills}")
print(f"  Killed by both types:                                       {both_kills}")

# Estimate: could LLM reason about value-sensitive kills?
print(f"\n--- Would LLM likely reason about these? ---")
for x in killed_analysis[:10]:
    kp = list(x["killing_policies"])
    needs_reasoning = any(p in structure_policies for p in kp)
    needs_numerical = any(p in value_policies for p in kp)
    difficulty = "LLM_CAN_REASON" if needs_reasoning else "NEEDS_NUMERICAL_SEARCH"
    print(f"  {x['id']:40s}  killed_by={kp}  -> {difficulty}")
