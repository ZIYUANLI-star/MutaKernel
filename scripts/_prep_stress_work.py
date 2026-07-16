"""Prepare assets for the differential stress-enhancement phase:
  1. gather one refmod per problem id (from the 6 shard dirs) into stress/refmods/
  2. build stress_work.json: every non-strict survivor with final_emd_status +
     original_code + mutated_code (input to the stress runner).
"""
import json, os, re, shutil, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "外部Benchmark差分测试_RQ4", "MutaKernel-KGB",
                    "MutaKernel-KGB", "MutaKernel", "runs")
KGB = os.path.join(RUNS, "kgb_ext")
LLMEMD = os.path.join(RUNS, "kgb_ext_llmemd")
DET = os.path.join(LLMEMD, "details")
OUTDIR = os.path.join(LLMEMD, "stress")
REFOUT = os.path.join(OUTDIR, "refmods")
os.makedirs(REFOUT, exist_ok=True)

# --- 1. gather refmods (problem_id -> path), prefer any shard ---
ref_by_pid = {}
for s in range(6):
    rd = os.path.join(RUNS, f"kgb_ext_s{s}", "_refmods")
    if not os.path.isdir(rd):
        continue
    for fn in os.listdir(rd):
        m = re.match(r"L0_P(\d+)_(.+)\.py$", fn)
        if not m:
            continue
        pid = int(m.group(1))
        if pid not in ref_by_pid:
            ref_by_pid[pid] = os.path.join(rd, fn)

# --- 2. build work items from llmemd details ---
work = []
status_ct = collections.Counter()
missing_ref = set()
for fn in sorted(os.listdir(DET)):
    if not fn.endswith(".json"):
        continue
    d = json.load(open(os.path.join(DET, fn), encoding="utf-8"))
    pid = d["kernel"]["problem_id"]
    kname = d["kernel"]["problem_name"]
    op = kname.split("__", 1)[0]
    # copy refmod for this pid
    if pid in ref_by_pid:
        dst = os.path.join(REFOUT, f"L0_P{pid}.py")
        if not os.path.exists(dst):
            shutil.copyfile(ref_by_pid[pid], dst)
    else:
        missing_ref.add(pid)
    for m in d["mutants"]:
        fs = m.get("final_emd_status", m["status"])
        status_ct[fs] += 1
        if fs in ("killed", "stillborn", "strict_equivalent"):
            continue  # killed/stillborn done; strict = textually identical kernel
        work.append({
            "problem_id": pid,
            "kernel_file": fn,
            "kernel_name": kname,
            "operator": op,
            "id": m["id"],
            "operator_name": m["operator_name"],
            "operator_category": m["operator_category"],
            "final_emd_status": fs,   # 'survived'(true_escape) or 'candidate_equivalent'
            "llm_equivalent": (m.get("equiv_detail", {}).get("layer3", {}) or {}).get("llm_equivalent"),
            "original_code": m["original_code"],
            "mutated_code": m["mutated_code"],
        })

json.dump(work, open(os.path.join(OUTDIR, "stress_work.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
print("final-status totals:", dict(status_ct))
print("refmods gathered:", len(os.listdir(REFOUT)), "| problems missing ref:", sorted(missing_ref))
print("stress work items (non-strict survivors):", len(work))
print("  true_escape(survived):", sum(1 for w in work if w["final_emd_status"] == "survived"))
print("  candidate_equivalent :", sum(1 for w in work if w["final_emd_status"] == "candidate_equivalent"))
print("written ->", os.path.join(OUTDIR, "stress_work.json"))
