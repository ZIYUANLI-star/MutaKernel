"""Detailed audit per FIXED kernel: how many real PASS, how many ref_fail per dimension.
- If a kernel has >50% ref_fail across all dims → high suspicion of being a pseudo-fix.
- If only dtype_stress has ref_fail (typical: fp16/bf16 reference fails) → acceptable.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

CKPT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json")
data = json.load(open(CKPT))

DIMS = ("value_stress", "dtype_stress", "boundary_stress", "perf_stress", "repeated_stress",
        "tier1_replay", "training_stress", "config_stress")

print(f"{'kernel_id':<28}{'total':>6}{'pass':>6}{'reffail':>9}{'fail':>6}{'disc':>6} ref_fail_dims")
print("-" * 110)

high_susp = []   # >=50% ref_fail
medium_susp = [] # 25-50% ref_fail
low_susp = []    # 1-25% ref_fail
clean = []       # 0% ref_fail

for kid, entry in data.items():
    if entry.get("status") != "FIXED":
        continue
    fr = entry.get("fixed_round")
    rounds = entry.get("rounds", {})
    r = rounds.get(str(fr)) or rounds.get(fr)
    tr = (r or {}).get("test_result", {})
    
    total = pass_ = reff = fail = disc = 0
    ref_fail_dims = []
    for dim in DIMS:
        d = tr.get(dim) or {}
        cases = d.get("test_cases", []) or []
        n_ref = sum(1 for c in cases if c.get("status") == "ref_fail")
        if n_ref:
            ref_fail_dims.append(f"{dim}({n_ref}/{len(cases)})")
        disc += d.get("discrepancies") or 0
        for c in cases:
            total += 1
            st = c.get("status")
            if st == "ref_fail": reff += 1
            elif st in ("pass", "ok"): pass_ += 1
            elif st in ("fail", "discrepancy"): fail += 1
    
    ratio = reff / max(total, 1)
    bucket = "high" if ratio >= 0.5 else "med" if ratio >= 0.25 else "low" if ratio > 0 else "clean"
    
    if bucket == "high":
        high_susp.append((kid, total, pass_, reff, fail, disc, ref_fail_dims))
    elif bucket == "med":
        medium_susp.append((kid, total, pass_, reff, fail, disc, ref_fail_dims))
    elif bucket == "low":
        low_susp.append((kid, total, pass_, reff, fail, disc, ref_fail_dims))
    else:
        clean.append((kid, total, pass_, reff, fail, disc))

print("HIGH suspicion (>= 50% ref_fail):")
for k,t,p,r,f,d,dims in high_susp:
    print(f"  {k:<28}{t:>6}{p:>6}{r:>9}{f:>6}{d:>6}  {dims}")
print(f"\nMEDIUM suspicion (25-50% ref_fail):")
for k,t,p,r,f,d,dims in medium_susp:
    print(f"  {k:<28}{t:>6}{p:>6}{r:>9}{f:>6}{d:>6}  {dims}")
print(f"\nLOW suspicion (<25% ref_fail) — first 10:")
for k,t,p,r,f,d,dims in low_susp[:10]:
    print(f"  {k:<28}{t:>6}{p:>6}{r:>9}{f:>6}{d:>6}  {dims}")
print(f"  ... and {max(0, len(low_susp)-10)} more")

print(f"\n=== Summary ===")
print(f"FIXED kernels:           {len(high_susp)+len(medium_susp)+len(low_susp)+len(clean)}")
print(f"  HIGH suspicion (>=50%): {len(high_susp)}")
print(f"  MEDIUM (25-50%):        {len(medium_susp)}")
print(f"  LOW (<25%):             {len(low_susp)}")
print(f"  CLEAN (0% ref_fail):    {len(clean)}")
