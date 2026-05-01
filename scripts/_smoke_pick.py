"""Pick 3 enhanceable mutants from different tiers for smoke test."""
import json
from pathlib import Path
from collections import defaultdict

details_dir = Path(r"d:\doctor_learning\Academic_Project\paper_1\MutaKernel\第二次实验汇总\full_block12_results\details")

tier_picks = defaultdict(list)

for jf in sorted(details_dir.glob("*.json")):
    data = json.loads(jf.read_text(encoding="utf-8"))
    kname = data["kernel"]["problem_name"]
    for m in data.get("mutants", []):
        status = m.get("status", "")
        if status not in ("survived", "candidate_equivalent"):
            continue
        ed = m.get("equiv_detail", {})
        if status == "candidate_equivalent":
            tier = 3
        elif ed.get("layer2", {}).get("is_equivalent") is False:
            tier = 1
        elif ed.get("layer3", {}).get("verdict") == "possibly_killable":
            tier = 2
        else:
            tier = 2
        mid = m["id"]
        op = m["operator_name"]
        line = m["site"]["line_start"]
        tier_picks[tier].append(f"  Tier {tier}: {mid} ({op} @ L{line}) [{kname}] status={status}")

for t in [1, 2, 3]:
    print(f"\nTier {t}: {len(tier_picks[t])} mutants")
    for s in tier_picks[t][:5]:
        print(s)
