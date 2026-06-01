"""Precise reason_category breakdown for Tier 1+2 残留 (123 mutants)
- DeepSeek-R1 (from Phase 2 details, reason_category in llm_iterative_analysis.rounds[0])
- Opus 4.5 (from Task A details, rounds[0].reason_category)
"""
import json
from pathlib import Path
from collections import Counter

PHASE2 = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/stress_enhance_results/details")
TASKA = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details")

# Build the 123 Tier 1+2 残留 set from Phase 2 details
target_ids = []
for p in sorted(PHASE2.glob("*.json")):
    d = json.load(open(p))
    if d.get("tier") in (1, 2) and not d.get("any_killed", False):
        target_ids.append(d.get("mutant_id"))

print(f"Tier 1+2 残留 mutant 数量 = {len(target_ids)}")

# DeepSeek-R1 reason_category in Phase 2
ds_counter = Counter()
for mid in target_ids:
    p = PHASE2 / f"{mid}.json"
    if not p.exists(): continue
    d = json.load(open(p))
    rounds = d.get("llm_iterative_analysis", {}).get("rounds", [])
    if rounds:
        cat = rounds[0].get("reason_category") or "(none)"
        ds_counter[cat] += 1
    else:
        ds_counter["(no_rounds)"] += 1

# Opus 4.5 reason_category in Task A (round 1)
op_counter_r1 = Counter()
# Also: per-mutant "any round killable=True"
killable_true_once = 0
killable_false_all = 0

for mid in target_ids:
    p = TASKA / f"{mid}.json"
    if not p.exists(): continue
    d = json.load(open(p))
    rounds = d.get("rounds", [])
    if rounds:
        cat = rounds[0].get("reason_category") or "(none)"
        op_counter_r1[cat] += 1
    # killable summary
    seq = [r.get("killable") for r in rounds]
    if any(k is True for k in seq):
        killable_true_once += 1
    elif all(k is False for k in seq) and seq:
        killable_false_all += 1

print(f"\n=== DeepSeek-R1 (Phase 2 第1轮) reason_category for 123 Tier 1+2 残留 ===")
total = sum(ds_counter.values())
for cat, n in ds_counter.most_common():
    print(f"  {cat:30} {n:>3}  ({n/total*100:.1f}%)")

print(f"\n=== Opus 4.5 (Task A 第1轮) reason_category for 123 Tier 1+2 残留 ===")
total_op = sum(op_counter_r1.values())
for cat, n in op_counter_r1.most_common():
    print(f"  {cat:30} {n:>3}  ({n/total_op*100:.1f}%)")

print(f"\n=== Opus 4.5 killable 判定 for 123 Tier 1+2 残留 ===")
print(f"  任一轮 killable=True: {killable_true_once}")
print(f"  五轮全 killable=False: {killable_false_all}")
print(f"  合计: {killable_true_once + killable_false_all}")
