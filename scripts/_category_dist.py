"""Check category distribution of first 50 survived mutants."""
import json
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BLOCK12 = PROJECT / "full_block12_results" / "details"

survived = []
for jf in sorted(BLOCK12.glob("*.json")):
    data = json.loads(jf.read_text())
    for m in data.get("mutants", []):
        if m["status"] == "survived":
            survived.append(m)

total = len(survived)
first50 = survived[:50]
rest = survived[50:]

print(f"Total survived: {total}")
print(f"\n=== First 50 (currently testing) ===")
cat_count = Counter(m["operator_category"] for m in first50)
op_count = Counter(m["operator_name"] for m in first50)
for cat in sorted(cat_count):
    print(f"  Category {cat}: {cat_count[cat]}")
print(f"  Operators: {dict(op_count)}")

print(f"\n=== Remaining {len(rest)} (not yet tested) ===")
cat_count2 = Counter(m["operator_category"] for m in rest)
op_count2 = Counter(m["operator_name"] for m in rest)
for cat in sorted(cat_count2):
    print(f"  Category {cat}: {cat_count2[cat]}")
print(f"  Operators: {dict(op_count2)}")

print(f"\n=== C-class detail (ALL) ===")
for i, m in enumerate(survived):
    if m["operator_category"] == "C":
        marker = "IN FIRST 50" if i < 50 else f"#{i+1}"
        print(f"  [{marker}] {m['id']:45s} | {m['operator_name']:20s}")
