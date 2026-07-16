"""Extract all KGB survivors (survived + strict/candidate equivalent) with code,
from the existing kgb_ext details, into a single work file for LLM-EMD."""
import json, os, collections

BASE = r"外部Benchmark差分测试_RQ4/MutaKernel-KGB/MutaKernel-KGB/MutaKernel/runs/kgb_ext"
DET = os.path.join(BASE, "details")
OUT = os.path.join(BASE, "llm_emd", "survivors_input.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

status_ct = collections.Counter()
survivors = []
missing_code = 0
for fn in sorted(os.listdir(DET)):
    if not fn.endswith(".json"):
        continue
    with open(os.path.join(DET, fn), encoding="utf-8") as f:
        d = json.load(f)
    kname = d["kernel"]["problem_name"]
    pid = d["kernel"]["problem_id"]
    for m in d.get("mutants", []):
        st = m["status"]
        status_ct[st] += 1
        if st in ("survived", "strict_equivalent", "candidate_equivalent"):
            oc = m.get("original_code", "")
            mc = m.get("mutated_code", "")
            if not oc or not mc:
                missing_code += 1
            survivors.append({
                "kernel_file": fn,
                "kernel_name": kname,
                "problem_id": pid,
                "id": m["id"],
                "operator_name": m["operator_name"],
                "operator_category": m["operator_category"],
                "description": m.get("description", ""),
                "site_line": m["site"]["line_start"],
                "deterministic_emd_status": st,
                "deterministic_emd_detail": m.get("equiv_detail", {}),
                "original_code": oc,
                "mutated_code": mc,
            })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(survivors, f, ensure_ascii=False, indent=2)

print("status totals:", dict(status_ct))
print("survivors (survived+strict+candidate):", len(survivors))
print("  survived(true_escape):", sum(1 for s in survivors if s["deterministic_emd_status"] == "survived"))
print("  strict_equivalent   :", sum(1 for s in survivors if s["deterministic_emd_status"] == "strict_equivalent"))
print("  candidate_equivalent:", sum(1 for s in survivors if s["deterministic_emd_status"] == "candidate_equivalent"))
print("missing_code:", missing_code)
print("written:", OUT)
