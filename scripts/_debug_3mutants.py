"""Debug: trace why --max-mutants 3 only runs 2."""
import json
from pathlib import Path

details_dir = Path(r"d:\doctor_learning\Academic_Project\paper_1\MutaKernel\第二次实验汇总\full_block12_results\details")

ENHANCEABLE_STATUSES = {"survived", "candidate_equivalent"}

items = []
for jf in sorted(details_dir.glob("*.json")):
    data = json.loads(jf.read_text(encoding="utf-8"))
    kname = data["kernel"]["problem_name"]
    for m in data.get("mutants", []):
        if m.get("status", "") in ENHANCEABLE_STATUSES:
            items.append((kname, data["kernel"], m))

print(f"Total enhanceable: {len(items)}")
print(f"\nFirst 5 in load order:")
for i, (kn, km, m) in enumerate(items[:5]):
    print(f"  [{i}] {m['id']} status={m['status']}")

limited = items[:3]
print(f"\n--- After --max-mutants 3, got {len(limited)} items ---")

tier_groups = {1: [], 2: [], 3: []}
for kn, km, m in limited:
    status = m.get("status", "survived")
    ed = m.get("equiv_detail", {})
    if status == "candidate_equivalent":
        tier = 3
    elif ed.get("layer2", {}).get("is_equivalent") is False:
        tier = 1
    elif ed.get("layer3", {}).get("verdict") == "possibly_killable":
        tier = 2
    else:
        tier = 2
    tier_groups[tier].append((kn, km, m))
    print(f"  {m['id']} -> Tier {tier} (status={status}, L2_eq={ed.get('layer2',{}).get('is_equivalent')}, L3_verdict={ed.get('layer3',{}).get('verdict')})")

print(f"\nTier counts: T1={len(tier_groups[1])}, T2={len(tier_groups[2])}, T3={len(tier_groups[3])}")

# Tier 3 filter
def should_challenge_tier3(m):
    ed = m.get("equiv_detail", {})
    l3 = ed.get("layer3", {})
    confidence = l3.get("confidence", 1.0)
    if confidence < 0.98:
        return True
    op_name = m.get("operator_name", "")
    if op_name in ("sync_remove", "launch_config_mutate", "mask_boundary",
                    "index_replace", "relop_replace", "const_perturb"):
        return True
    return False

tier3_filtered = [item for item in tier_groups[3] if should_challenge_tier3(item[2])]
print(f"\nTier 3 after filter: {len(tier3_filtered)}")
for kn, km, m in tier_groups[3]:
    passed = should_challenge_tier3(m)
    l3 = m.get("equiv_detail", {}).get("layer3", {})
    print(f"  {m['id']} confidence={l3.get('confidence')} op={m['operator_name']} -> passed_filter={passed}")

execution_order = (
    [(1, item) for item in tier_groups[1]]
    + [(2, item) for item in tier_groups[2]]
    + [(3, item) for item in tier3_filtered]
)
print(f"\nFinal execution_order: {len(execution_order)} mutants")
for tier, (kn, km, m) in execution_order:
    print(f"  Tier {tier}: {m['id']}")
