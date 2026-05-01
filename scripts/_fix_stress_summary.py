"""Rebuild stress_summary.json from detail files + find missing mutants."""
import json
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
STRESS_DIR = PROJECT / "stress_enhance_results"
DETAILS_DIR = STRESS_DIR / "details"
COMPLETED_FILE = STRESS_DIR / "completed.json"

completed = json.loads(COMPLETED_FILE.read_text(encoding="utf-8"))
print(f"completed.json: {len(completed)} mutant IDs")

detail_files = sorted(DETAILS_DIR.glob("*.json"))
detail_ids = {f.stem for f in detail_files}
print(f"details/ folder: {len(detail_files)} files")

# Find missing
completed_set = set(completed)
missing_from_details = completed_set - detail_ids
extra_in_details = detail_ids - completed_set
print(f"\nIn completed but NO detail file: {len(missing_from_details)}")
for m in sorted(missing_from_details):
    print(f"  - {m}")
print(f"In details but NOT in completed: {len(extra_in_details)}")
for m in sorted(extra_in_details):
    print(f"  - {m}")

# Rebuild summary from detail files
killed_count = 0
survived_count = 0
per_layer_kills = defaultdict(int)
per_policy_kills = defaultdict(int)
per_mode_kills = defaultdict(int)
per_mutant = []

for df in detail_files:
    try:
        sd = json.loads(df.read_text(encoding="utf-8"))
    except Exception:
        continue
    is_killed = sd.get("killed", False)
    if is_killed:
        killed_count += 1
        layer = sd.get("killing_layer", "unknown")
        policy = sd.get("killing_policy", "unknown")
        mode = sd.get("killing_mode", "unknown")
        per_layer_kills[layer] += 1
        if policy:
            per_policy_kills[policy] += 1
        if mode:
            per_mode_kills[mode] += 1
    else:
        survived_count += 1

    per_mutant.append({
        "mutant_id": sd["mutant_id"],
        "kernel_name": sd["kernel_name"],
        "killed": is_killed,
        "killing_layer": sd.get("killing_layer"),
        "killing_policy": sd.get("killing_policy"),
        "killing_seed": sd.get("killing_seed"),
        "killing_mode": sd.get("killing_mode"),
    })

total = killed_count + survived_count
summary = {
    "total_tested": total,
    "killed_count": killed_count,
    "survived_count": survived_count,
    "kill_rate": round(killed_count / total, 4) if total > 0 else 0.0,
    "per_layer_kills": dict(per_layer_kills),
    "per_policy_kills": dict(per_policy_kills),
    "per_mode_kills": dict(per_mode_kills),
    "per_mutant": per_mutant,
}

out_path = STRESS_DIR / "stress_summary.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nRebuilt stress_summary.json:")
print(f"  total_tested: {total}")
print(f"  killed: {killed_count}")
print(f"  survived: {survived_count}")
print(f"  kill_rate: {summary['kill_rate']}")
print(f"  per_layer_kills: {dict(per_layer_kills)}")
print(f"  per_policy_kills: {dict(per_policy_kills)}")
print(f"  per_mode_kills: {dict(per_mode_kills)}")
