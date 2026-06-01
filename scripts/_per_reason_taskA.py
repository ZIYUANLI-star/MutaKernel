"""Per-reason_category (DeepSeek-R1's) cross-check with Opus 4.5 killable."""
import json
from pathlib import Path
from collections import defaultdict

PHASE2 = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/stress_enhance_results/details")
TASKA = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details")

# Build {mid -> DeepSeek-R1 reason_category}
ds_cat = {}
for p in sorted(PHASE2.glob("*.json")):
    d = json.load(open(p))
    if not (d.get("tier") in (1, 2) and not d.get("any_killed", False)):
        continue
    rounds = d.get("llm_iterative_analysis", {}).get("rounds", [])
    cat = rounds[0].get("reason_category") if rounds else None
    ds_cat[d.get("mutant_id")] = cat

# For each reason_category, count Opus all_False vs any_True
per_cat = defaultdict(lambda: {"total": 0, "all_False": 0, "any_True": 0})
for mid, cat in ds_cat.items():
    p = TASKA / f"{mid}.json"
    if not p.exists():
        continue
    d = json.load(open(p))
    seq = [r.get("killable") for r in d.get("rounds", [])]
    per_cat[cat]["total"] += 1
    if all(k is False for k in seq) and seq:
        per_cat[cat]["all_False"] += 1
    if any(k is True for k in seq):
        per_cat[cat]["any_True"] += 1

print(f"{'reason_category':<28}{'total':>8}{'all_False':>12}{'any_True':>11}")
print("-" * 60)
for cat in ("predicate_unreachable", "value_insensitive", "path_not_triggered",
            "infection_no_propagation", "requires_config_change"):
    s = per_cat[cat]
    pct = (s["all_False"] / s["total"] * 100) if s["total"] else 0
    print(f"{cat:<28}{s['total']:>8}{s['all_False']:>9} ({pct:.1f}%){s['any_True']:>11}")
