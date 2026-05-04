#!/usr/bin/env python3
"""Comprehensive report of stress_enhance experiment results so far."""
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "第二次实验汇总" / "stress_enhance_results" / "details"

if not D.exists():
    print("No details dir"); exit()

files = sorted(D.glob("*.json"))
print(f"{'='*70}")
print(f"  增强测试阶段实验结果汇报 ({len(files)} mutants)")
print(f"{'='*70}")

killed_total = 0
survived_total = 0
tier_counts = Counter()
tier_kills = Counter()
op_counts = Counter()
op_kills = Counter()
kernel_counts = Counter()
kernel_kills = Counter()
dim_kills = Counter()
policy_kills = Counter()
multi_dim = 0
cat_counts = Counter()
cat_kills = Counter()

llm_executed = 0
llm_killed = 0
llm_reason_cats = Counter()
llm_killable_count = 0
llm_unkillable_count = 0
llm_rounds_dist = Counter()
ref_nan_fallback_count = 0
ref_nan_fallback_kills = 0

survived_list = []
original_failure_policies = Counter()

for f in files:
    data = json.loads(f.read_text())
    mid = data["mutant_id"]
    op = data.get("operator_name", "?")
    cat = data.get("operator_category", "?")
    kernel = data.get("kernel_name", "?")
    tier = data.get("tier", 0)
    ak = data.get("any_killed", False)

    tier_counts[tier] += 1
    op_counts[op] += 1
    kernel_counts[kernel] += 1
    cat_counts[cat] += 1

    if ak:
        killed_total += 1
        tier_kills[tier] += 1
        op_kills[op] += 1
        kernel_kills[kernel] += 1
        cat_kills[cat] += 1
    else:
        survived_total += 1

    ks = data.get("kill_summary", {})
    main_killed = ks.get("main_track_killed_by", [])
    config_killed = ks.get("config_track_killed_by", [])
    all_killed_dims = main_killed + config_killed
    for d_name in all_killed_dims:
        dim_kills[d_name] += 1
    if len(all_killed_dims) > 1:
        multi_dim += 1

    # Per-policy kills
    mt = data.get("main_track", {})
    for dim_name, dim_data_item in mt.items():
        if dim_data_item.get("killed"):
            kp = dim_data_item.get("killing_policy", "")
            if kp:
                policy_kills[kp] += 1

    # Original failures
    for of_policy in data.get("original_failures", []):
        original_failure_policies[of_policy] += 1

    # ref_nan_fallback analysis
    for dim_name in ["value_stress", "training_stress"]:
        dd = data.get("main_track", {}).get(dim_name, {})
        for pr in dd.get("policy_results", []) + dd.get("results", []):
            if pr.get("ref_nan_fallback"):
                ref_nan_fallback_count += 1

    # LLM analysis
    llm = data.get("llm_iterative_analysis", {})
    if llm.get("executed"):
        llm_executed += 1
        n_rounds = len(llm.get("rounds", []))
        llm_rounds_dist[n_rounds] += 1

        if llm.get("killed"):
            llm_killed += 1
            dim_kills["llm_iterative_analysis"] += 1

        for r in llm.get("rounds", []):
            rc = r.get("reason_category", "")
            if rc:
                llm_reason_cats[rc] += 1
            if r.get("killable"):
                llm_killable_count += 1
            else:
                llm_unkillable_count += 1
            er = r.get("execution_result") or {}
            if er.get("ref_nan_fallback"):
                ref_nan_fallback_count += 1
                if er.get("killed"):
                    ref_nan_fallback_kills += 1

        if not llm.get("killed"):
            last_round = llm["rounds"][-1] if llm.get("rounds") else {}
            survived_list.append({
                "id": mid,
                "op": op,
                "cat": cat,
                "kernel": kernel,
                "llm_killable": last_round.get("killable", "?"),
                "reason_cat": last_round.get("reason_category", "?"),
                "n_rounds": n_rounds,
            })

# === Print Report ===
print(f"\n--- 1. 总体结果 ---")
print(f"  总测试:    {killed_total + survived_total}")
print(f"  杀死:      {killed_total}")
print(f"  存活:      {survived_total}")
print(f"  杀死率:    {killed_total/max(1,killed_total+survived_total)*100:.1f}%")
print(f"  多维度杀死: {multi_dim}")

print(f"\n--- 2. Per-Tier ---")
for t in sorted(tier_counts.keys()):
    total = tier_counts[t]
    k = tier_kills[t]
    s = total - k
    print(f"  Tier {t}: tested={total}, killed={k}, survived={s}, "
          f"kill_rate={k/max(1,total)*100:.1f}%")

print(f"\n--- 3. Per-Operator (按杀死率排序) ---")
for op in sorted(op_counts.keys(), key=lambda x: op_kills[x]/max(1,op_counts[x]), reverse=True):
    total = op_counts[op]
    k = op_kills[op]
    print(f"  {op:25s}: tested={total:3d}, killed={k:3d}, survived={total-k:3d}, "
          f"kill_rate={k/max(1,total)*100:.1f}%")

print(f"\n--- 4. Per-Category ---")
for c in sorted(cat_counts.keys()):
    total = cat_counts[c]
    k = cat_kills[c]
    print(f"  Category {c}: tested={total}, killed={k}, survived={total-k}, "
          f"kill_rate={k/max(1,total)*100:.1f}%")

print(f"\n--- 5. Per-Dimension 杀死数 ---")
for d_name, cnt in sorted(dim_kills.items(), key=lambda x: -x[1]):
    print(f"  {d_name:25s}: {cnt}")

print(f"\n--- 6. Per-Policy 杀死数 (Top 15) ---")
for p, cnt in policy_kills.most_common(15):
    print(f"  {p:30s}: {cnt}")

print(f"\n--- 7. Original Failures (kernel 自身也失败的 policy, Top 10) ---")
for p, cnt in original_failure_policies.most_common(10):
    print(f"  {p:30s}: {cnt} mutants")

print(f"\n--- 8. Per-Kernel (仅显示有 survived 的) ---")
for k in sorted(kernel_counts.keys()):
    total = kernel_counts[k]
    kills = kernel_kills[k]
    if total - kills > 0:
        print(f"  {k:12s}: tested={total:2d}, killed={kills:2d}, survived={total-kills:2d}")

print(f"\n--- 9. LLM 迭代分析 ---")
print(f"  触发次数:      {llm_executed}")
print(f"  杀死次数:      {llm_killed}")
print(f"  轮次分布:      {dict(llm_rounds_dist)}")
print(f"  判定可杀轮数:  {llm_killable_count}")
print(f"  判定不可杀轮数: {llm_unkillable_count}")
print(f"\n  reason_category 分布:")
for rc, cnt in llm_reason_cats.most_common():
    print(f"    {rc:35s}: {cnt}")

print(f"\n--- 10. ref_nan_fallback 机制 ---")
print(f"  触发次数 (policy runs): {ref_nan_fallback_count}")
print(f"  由 fallback 杀死:       {ref_nan_fallback_kills}")

print(f"\n--- 11. Survived Mutants 详情 ({len(survived_list)}) ---")
for s in survived_list:
    killable_str = "killable" if s["llm_killable"] else "unkillable"
    print(f"  {s['id']:45s} ({s['op']:20s} [{s['cat']}]) "
          f"R{s['n_rounds']} {killable_str:10s} {s['reason_cat']}")
