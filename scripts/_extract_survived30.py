#!/usr/bin/env python3
"""Extract mutation site info for survived mutants from the 30-mutant stress test."""
import json
from pathlib import Path

ROOT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
RESULTS = ROOT / "pilot_stress_results" / "stress30_results.json"
DETAILS = ROOT / "full_block12_results" / "details"

results = json.loads(RESULTS.read_text())
survived = [r for r in results if not any([r["l1_killed"], r.get("l2_killed"), r.get("l3_killed"), r.get("l4_killed")])]

for r in survived:
    mid = r["mutant_id"]
    kn = r["kernel"]
    detail_path = DETAILS / f"{kn}.json"
    if not detail_path.exists():
        print(f"\n=== {mid} === FILE NOT FOUND")
        continue
    data = json.loads(detail_path.read_text())
    for m in data.get("mutants", []):
        if m["id"] == mid:
            s = m["site"]
            print(f"\n=== {mid} ===")
            print(f"  operator: {m['operator_name']}")
            print(f"  category: {m['operator_category']}")
            print(f"  attribution: {r['attribution']}")
            print(f"  original_failures: {r.get('original_failures', [])}")
            print(f"  line: {s['line_start']}")
            print(f"  original_code: {s.get('original_code', '')[:200]}")
            mc = m.get("mutated_code", "")
            if mc:
                lines = mc.split("\n")
                target_line = s["line_start"] - 1
                start = max(0, target_line - 2)
                end = min(len(lines), target_line + 3)
                print(f"  mutated region (L{start+1}-{end}):")
                for i in range(start, end):
                    marker = ">>>" if i == target_line else "   "
                    print(f"    {marker} {i+1:3d}| {lines[i]}")
            break
