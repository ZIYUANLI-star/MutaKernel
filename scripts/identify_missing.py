"""Identify the 35 mutants that were filtered out by should_challenge_tier3."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLOCK12_DIR = PROJECT_ROOT / "第二次实验汇总" / "full_block12_results"
STRESS_DIR = PROJECT_ROOT / "第二次实验汇总" / "stress_enhance_results"

ENHANCEABLE_STATUSES = {"survived", "candidate_equivalent"}

completed = set()
completed_file = STRESS_DIR / "completed.json"
if completed_file.exists():
    completed = set(json.loads(completed_file.read_text()))

print(f"Already completed: {len(completed)} mutants")

all_enhanceable = []
details_dir = BLOCK12_DIR / "details"
for jf in sorted(details_dir.glob("*.json")):
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        continue
    kernel_name = data["kernel"]["problem_name"]
    for m in data.get("mutants", []):
        if m.get("status", "") in ENHANCEABLE_STATUSES:
            all_enhanceable.append({
                "kernel": kernel_name,
                "id": m["id"],
                "status": m["status"],
                "operator_name": m["operator_name"],
                "equiv_detail": m.get("equiv_detail", {}),
            })

print(f"Total enhanceable (Phase 1 survived+cand_eq): {len(all_enhanceable)}")

missing = [m for m in all_enhanceable if m["id"] not in completed]
print(f"Missing (not in completed.json): {len(missing)}")

from collections import Counter
by_op = Counter(m["operator_name"] for m in missing)
by_status = Counter(m["status"] for m in missing)

print(f"\nBy operator: {dict(by_op)}")
print(f"By status: {dict(by_status)}")
print(f"\nMissing mutant IDs:")
for m in missing:
    ed = m["equiv_detail"]
    l3 = ed.get("layer3", {})
    conf = l3.get("confidence", "N/A")
    print(f"  {m['id']:40s}  op={m['operator_name']:20s}  status={m['status']:25s}  L3_conf={conf}")
