"""Compare: What did stress test kill vs what did LLM kill?
Analyze the complementarity of the two approaches."""
import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT = Path(__file__).resolve().parent.parent
STRESS_DIR = PROJECT / "stress_enhance_results" / "details"
LLM_DIR = PROJECT / "llm_analysis_results" / "details"

# 1. What did Phase 1 stress test kill?
stress_killed = []
stress_survived = []
stress_killed_by_layer = Counter()
stress_killed_by_policy = Counter()
stress_killed_ops = Counter()

for jf in sorted(STRESS_DIR.glob("*.json")):
    d = json.loads(jf.read_text())
    mid = d.get("mutant_id", jf.stem)
    op = d.get("operator_name", "")
    killed = False
    
    if d.get("killed"):
        killed = True
        stress_killed_by_layer["layer1_value"] += 1
        kp = d.get("killing_policy", "unknown")
        stress_killed_by_policy[kp] += 1
    elif d.get("dtype_killed"):
        killed = True
        stress_killed_by_layer["layer2_dtype"] += 1
    elif d.get("repeated_killed"):
        killed = True
        stress_killed_by_layer["layer3_repeated"] += 1
    elif d.get("training_killed"):
        killed = True
        stress_killed_by_layer["layer4_training"] += 1
    
    if killed:
        stress_killed.append({"id": mid, "op": op})
        stress_killed_ops[op] += 1
    else:
        stress_survived.append({"id": mid, "op": op})

# 2. What did LLM kill (from the survived set)?
llm_killed = []
llm_survived = []
llm_killed_ops = Counter()
llm_kill_mechanisms = []

for jf in sorted(LLM_DIR.glob("*.json")):
    d = json.loads(jf.read_text())
    mid = d["mutant_id"]
    op = d["operator_name"]
    if d.get("killed_by_llm"):
        llm_killed.append(d)
        llm_killed_ops[op] += 1
        # Analyze HOW the LLM killed it
        for r in d.get("rounds", []):
            if r.get("killed"):
                code = r.get("suggested_code", "")
                # Check if the kill used shape changes or value changes
                uses_shape = any(kw in code.lower() for kw in ["size", "shape", "65536", "512", "1000", "1025", "33"])
                uses_extreme_val = any(kw in code.lower() for kw in ["1e6", "1e-6", "inf", "nan", "1e10"])
                mechanism = []
                if uses_shape:
                    mechanism.append("shape_change")
                if uses_extreme_val:
                    mechanism.append("extreme_values")
                if not mechanism:
                    mechanism.append("targeted_values")
                llm_kill_mechanisms.append({
                    "id": mid, "op": op, 
                    "mechanism": "+".join(mechanism),
                    "round": r["round"],
                })
                break
    else:
        llm_survived.append(d)

# 3. Report
print("=" * 80)
print("  STRESS TEST vs LLM: COMPLEMENTARITY ANALYSIS")
print("=" * 80)

print(f"\n--- Phase 1: Stress Test Results ---")
print(f"  Total mutants tested: {len(stress_killed) + len(stress_survived)}")
print(f"  Killed by stress test: {len(stress_killed)}")
print(f"  Survived stress test: {len(stress_survived)}")

print(f"\n  Killed by layer:")
for layer, cnt in stress_killed_by_layer.most_common():
    print(f"    {layer}: {cnt}")

print(f"\n  Killed by operator:")
for op, cnt in stress_killed_ops.most_common():
    print(f"    {op}: {cnt}")

print(f"\n  Top killing policies:")
for pol, cnt in stress_killed_by_policy.most_common(10):
    print(f"    {pol}: {cnt}")

print(f"\n--- Phase 2: LLM Results (so far) ---")
print(f"  Total analyzed by LLM: {len(llm_killed) + len(llm_survived)}")
print(f"  Killed by LLM: {len(llm_killed)}")
print(f"  Survived LLM: {len(llm_survived)}")

print(f"\n  LLM killed by operator:")
for op, cnt in llm_killed_ops.most_common():
    print(f"    {op}: {cnt}")

print(f"\n  LLM kill mechanisms:")
for km in llm_kill_mechanisms:
    print(f"    {km['id']} (op={km['op']}): {km['mechanism']} (round {km['round']})")

# 4. Key comparison: Could stress test have killed LLM-killed mutants?
print(f"\n--- Could stress test have killed LLM's kills? ---")
for d in llm_killed:
    mid = d["mutant_id"]
    # Find the stress test result for this mutant
    sf = STRESS_DIR / f"{mid}.json"
    if sf.exists():
        sd = json.loads(sf.read_text())
        # Check if any policy came close
        all_both_ok = all(
            pr.get("ref_ok") and pr.get("original_ok") and pr.get("mutant_ok")
            for pr in sd.get("policy_results", [])
            if pr.get("ref_ok") and pr.get("original_ok")
        )
        orig_fails = sd.get("original_failures", [])
        print(f"\n  {mid} (op={d['operator_name']}):")
        print(f"    Stress test: all_both_ok={all_both_ok}")
        print(f"    Original failures: {orig_fails}")
        # Show what the LLM used to kill it
        for r in d.get("rounds", []):
            if r.get("killed"):
                code = r.get("suggested_code", "")
                # Extract the key difference
                for line in code.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("import") and not line.startswith("def"):
                        if any(kw in line for kw in ["size", "shape", "512", "1000", "65536", "1025", "33", "ones", "zeros", "randn"]):
                            print(f"    LLM key input: {line}")
                break

# 5. Overlap analysis: what ops does each method kill?
print(f"\n--- Operator Coverage Comparison ---")
all_ops = set(list(stress_killed_ops.keys()) + list(llm_killed_ops.keys()))
print(f"  {'Operator':30s}  {'Stress':>8s}  {'LLM':>8s}  {'Total':>8s}")
for op in sorted(all_ops):
    s = stress_killed_ops.get(op, 0)
    l = llm_killed_ops.get(op, 0)
    print(f"  {op:30s}  {s:8d}  {l:8d}  {s+l:8d}")
