"""Generate progress report from stress_enhance_results."""
import json
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
STRESS = PROJECT / "stress_enhance_results" / "details"

killed_by = Counter()
killed_details = []
survived_ops = Counter()
survived_cats = Counter()
killed_ops = Counter()
killed_cats = Counter()
total = 0

for jf in sorted(STRESS.glob("*.json")):
    d = json.loads(jf.read_text())
    total += 1
    op = d["operator_name"]
    cat = d["operator_category"]
    if d.get("killed"):
        mode = d.get("killing_mode", "unknown")
        policy = d.get("killing_policy") or d.get("killing_dtype", "")
        killed_by[f"{mode}/{policy}"] += 1
        killed_ops[op] += 1
        killed_cats[cat] += 1
        killed_details.append(f"  {d['mutant_id']:45s} | {op:18s} ({cat}) | {mode}/{policy}")
    else:
        survived_ops[op] += 1
        survived_cats[cat] += 1

killed_total = sum(killed_by.values())
survived_total = sum(survived_ops.values())

print(f"=== PROGRESS REPORT ===")
print(f"Total completed: {total} / 322")
print(f"Killed: {killed_total} ({killed_total/total*100:.1f}%)")
print(f"Survived: {survived_total}")
print()
print(f"=== Kill breakdown by method ===")
for k, v in killed_by.most_common():
    print(f"  {k}: {v}")
print()
print(f"=== Kill breakdown by operator ===")
for k, v in killed_ops.most_common():
    print(f"  {k}: {v}")
print()
print(f"=== Kill breakdown by category ===")
for k, v in killed_cats.most_common():
    print(f"  {k}: {v}")
print()
print(f"=== Survived by operator ===")
for k, v in survived_ops.most_common():
    print(f"  {k}: {v}")
print()
print(f"=== Survived by category ===")
for k, v in survived_cats.most_common():
    print(f"  {k}: {v}")
print()
print(f"=== Killed mutant details ===")
for d in killed_details:
    print(d)
