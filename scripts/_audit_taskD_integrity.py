"""Pre-flight integrity check for Task D (5th experiment).
- For each kernel in checkpoint.json, check the LAST round test_result.
- Count: ref_fail (GPU_PREFLIGHT_FAIL), normal_pass (all dims 0 discrepancies),
         normal_fail (some discrepancies), other.
- Important: also look at status (FIXED / NOT_FIXED / TIMEOUT).
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

CKPT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json")

data = json.load(open(CKPT))
print(f"Total kernels in checkpoint: {len(data)}")

status_counter = Counter()
for kid, entry in data.items():
    status_counter[entry.get("status")] += 1
print("\nStatus distribution:")
for s, c in status_counter.most_common():
    print(f"  {s}: {c}")

# For each FIXED kernel, examine the test_result of fixed_round
preflight_fail_when_fixed = []
real_pass = []
no_test_result = []
abnormal_fixed = []

DIMS = ("value_stress", "dtype_stress", "boundary_stress", "perf_stress", "repeated_stress",
        "tier1_replay", "training_stress", "config_stress")

for kid, entry in data.items():
    if entry.get("status") != "FIXED":
        continue
    fr = entry.get("fixed_round")
    rounds = entry.get("rounds", {})
    r = rounds.get(str(fr)) or rounds.get(fr)
    if not r:
        no_test_result.append(kid)
        continue
    tr = r.get("test_result", {})
    if not tr:
        no_test_result.append(kid)
        continue
    # Examine each dim's test_cases for status patterns
    total_cases = 0
    ref_fail_cases = 0
    pass_cases = 0
    fail_cases = 0
    other_cases = 0
    total_discrepancies = 0
    for dim in DIMS:
        d = tr.get(dim) or {}
        cases = d.get("test_cases", []) or []
        total_discrepancies += (d.get("discrepancies") or 0)
        for c in cases:
            total_cases += 1
            st = c.get("status")
            if st == "ref_fail":
                ref_fail_cases += 1
            elif st in ("pass", "ok"):
                pass_cases += 1
            elif st in ("fail", "discrepancy"):
                fail_cases += 1
            else:
                other_cases += 1
    if total_cases == 0:
        no_test_result.append(kid)
    elif ref_fail_cases == total_cases:
        preflight_fail_when_fixed.append((kid, total_cases))
    elif pass_cases == total_cases:
        real_pass.append((kid, total_cases))
    else:
        abnormal_fixed.append((kid, dict(total=total_cases, ref_fail=ref_fail_cases,
                                          pass_=pass_cases, fail=fail_cases, other=other_cases,
                                          discrepancies=total_discrepancies)))

print(f"\n=== FIXED kernel breakdown by underlying test_result ===")
print(f"Real pass (all cases status=pass): {len(real_pass)}")
print(f"All cases ref_fail (GPU_PREFLIGHT_FAIL or similar): {len(preflight_fail_when_fixed)}")
print(f"No usable test_result: {len(no_test_result)}")
print(f"Mixed/abnormal: {len(abnormal_fixed)}")

if preflight_fail_when_fixed:
    print(f"\nKernels marked FIXED but all test cases are ref_fail:")
    for kid, tc in preflight_fail_when_fixed[:20]:
        print(f"  {kid}  ({tc} cases)")
    if len(preflight_fail_when_fixed) > 20:
        print(f"  ... and {len(preflight_fail_when_fixed)-20} more")

if abnormal_fixed:
    print(f"\nFirst 10 abnormal FIXED cases:")
    for kid, d in abnormal_fixed[:10]:
        print(f"  {kid}  {d}")

# Also look at specific error messages
print(f"\n=== Sample ref_fail errors among FIXED kernels ===")
error_counter = Counter()
for kid in [k for k,_ in preflight_fail_when_fixed[:10]]:
    entry = data[kid]
    fr = entry.get("fixed_round")
    r = entry.get("rounds", {}).get(str(fr)) or entry.get("rounds", {}).get(fr)
    tr = r.get("test_result", {})
    for dim in DIMS:
        for c in (tr.get(dim) or {}).get("test_cases", []):
            if c.get("status") == "ref_fail":
                error_counter[c.get("error", "")] += 1
for err, n in error_counter.most_common(5):
    print(f"  ({n}x) {err}")
