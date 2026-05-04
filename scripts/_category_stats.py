#!/usr/bin/env python3
"""统计 A~D 各类变异体的 killed/survived/stillborn/equivalent 分布。"""
import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "full_block12_results" / "details"

cat_stats = defaultdict(lambda: defaultdict(int))
op_stats = defaultdict(lambda: {"cat": "", "killed": 0, "survived": 0, "stillborn": 0, "equivalent": 0, "total": 0})

for jf in sorted(RESULTS_DIR.glob("*.json")):
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
    for m in data.get("mutants", []):
        cat = m["operator_category"]
        status = m["status"]
        cat_stats[cat][status] += 1
        cat_stats[cat]["total"] += 1

        op = m["operator_name"]
        op_stats[op]["cat"] = cat
        op_stats[op][status] += 1
        op_stats[op]["total"] += 1

print("=" * 80)
print("按类别 (A/B/C/D) 统计")
print("=" * 80)
print(f"{'Cat':<5} {'Total':>6} {'Killed':>8} {'Survived':>9} {'Stillborn':>10} {'Equiv':>7} {'Adj.Score':>10}")
print("-" * 80)

grand = defaultdict(int)
for cat in sorted(cat_stats.keys()):
    s = cat_stats[cat]
    denom = s["total"] - s["stillborn"] - s["equivalent"]
    score = s["killed"] / denom * 100 if denom > 0 else 0
    print(f"{cat:<5} {s['total']:>6} {s['killed']:>8} {s['survived']:>9} {s['stillborn']:>10} {s['equivalent']:>7} {score:>9.1f}%")
    for k in ["total", "killed", "survived", "stillborn", "equivalent"]:
        grand[k] += s[k]

denom = grand["total"] - grand["stillborn"] - grand["equivalent"]
score = grand["killed"] / denom * 100 if denom > 0 else 0
print("-" * 80)
print(f"{'ALL':<5} {grand['total']:>6} {grand['killed']:>8} {grand['survived']:>9} {grand['stillborn']:>10} {grand['equivalent']:>7} {score:>9.1f}%")

print()
print("=" * 80)
print("按算子统计")
print("=" * 80)
print(f"{'Operator':<25} {'Cat':<4} {'Total':>6} {'Killed':>7} {'Surv':>6} {'Still':>6} {'Equiv':>6} {'Score':>8}")
print("-" * 80)

for cat in ["A", "B", "C", "D"]:
    ops_in_cat = [(op, s) for op, s in op_stats.items() if s["cat"] == cat]
    ops_in_cat.sort(key=lambda x: x[0])
    for op, s in ops_in_cat:
        denom = s["total"] - s["stillborn"] - s["equivalent"]
        score = s["killed"] / denom * 100 if denom > 0 else 0
        print(f"{op:<25} {cat:<4} {s['total']:>6} {s['killed']:>7} {s['survived']:>6} {s['stillborn']:>6} {s['equivalent']:>6} {score:>7.1f}%")
    if ops_in_cat:
        print()
