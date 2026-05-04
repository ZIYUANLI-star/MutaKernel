"""Show detailed stress test results for a specific mutant."""
import json, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
f = PROJECT / "stress_enhance_results" / "details" / "L1_P1__const_perturb__1.json"
d = json.loads(f.read_text())

print("=== Policy Results for L1_P1__const_perturb__1 (N=2048->2047) ===\n")
for pr in d.get("policy_results", []):
    p = pr["policy"]
    ref = pr["ref_ok"]
    orig = pr["original_ok"]
    mut = pr["mutant_ok"]
    err = pr.get("error", "")
    if ref and orig and not mut:
        status = "KILLED"
    elif not orig and not mut:
        status = "ORIG_ALSO_FAIL"
    elif not ref:
        status = "REF_FAIL"
    elif ref and orig and mut:
        status = "both_ok"
    else:
        status = f"other(ref={ref},orig={orig},mut={mut})"
    line = f"  {p:28s}  ref={str(ref):5s}  orig={str(orig):5s}  mut={str(mut):5s}  -> {status}"
    if err:
        line += f"  [{err}]"
    print(line)

print(f"\nOriginal failures: {d.get('original_failures', [])}")
print(f"dtype_killed: {d.get('dtype_killed')}")
print(f"repeated_killed: {d.get('repeated_killed')}")
print(f"training_killed: {d.get('training_killed')}")
