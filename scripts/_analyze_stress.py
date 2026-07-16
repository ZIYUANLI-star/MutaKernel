import json, collections, os

RES = "外部Benchmark差分测试_RQ4/MutaKernel-KGB/MutaKernel-KGB/MutaKernel/runs/kgb_ext_llmemd/stress/results.jsonl"
rows = [json.loads(l) for l in open(RES, encoding="utf-8") if l.strip()]
print("total rows:", len(rows))
ids = set(r["mutant_id"] for r in rows)
print("unique ids:", len(ids))

by_status = collections.Counter(r["final_emd_status"] for r in rows)
print("by final_emd_status:", dict(by_status))

killed = [r for r in rows if r.get("any_killed")]
notk = [r for r in rows if not r.get("any_killed")]
print("killed:", len(killed), "not_killed:", len(notk))

crash = [r for r in rows if r.get("_crash_kill")]
print("crash/timeout kills:", len(crash))
for r in crash:
    print("  ", r["mutant_id"], r["first_kill_mode"], r["kernel_name"])

# kills split by emd status
print("\n-- kills by emd status --")
for st in by_status:
    k = sum(1 for r in killed if r["final_emd_status"] == st)
    print(f"  {st}: killed {k}/{by_status[st]}")

# by dimension (first kill mode)
print("\n-- first_kill_mode --")
print(dict(collections.Counter(r.get("first_kill_mode") for r in killed)))

# dimension kill counts (any dim that killed)
dimc = collections.Counter()
for r in killed:
    for d in r.get("killed_dimensions", []):
        dimc[d] += 1
print("\n-- killed_dimensions (count of mutants killed by each dim) --")
print(dict(dimc))

# killing policy breakdown
polc = collections.Counter()
for r in killed:
    for trk in ("main_track", "config_track"):
        for dim, info in (r.get(trk) or {}).items():
            if isinstance(info, dict) and info.get("killed") and info.get("killing_policy"):
                polc[(dim, info["killing_policy"])] += 1
print("\n-- killing_policy (dim, policy) --")
for k, v in polc.most_common():
    print(f"  {k}: {v}")

# by operator
print("\n-- by operator (killed/total) --")
opt = collections.Counter(r["operator_name"] for r in rows)
opk = collections.Counter(r["operator_name"] for r in killed)
for op in sorted(opt):
    print(f"  {op}: {opk[op]}/{opt[op]}")
