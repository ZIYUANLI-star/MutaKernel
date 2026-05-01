import json, sys
from pathlib import Path

f = Path(sys.argv[1])
data = json.loads(f.read_text(encoding="utf-8"))

survived = [d for d in data
            if not d.get("l1_killed") and not d.get("l2_killed")
            and not d.get("l3_killed") and not d.get("l4_killed")]

print(f"Total: {len(data)},  Survived (not killed by L1-L4): {len(survived)}")
for d in survived:
    print(f"  {d['mutant_id']:45s}  op={d['operator']:25s}  attr={d.get('attribution','')}")
