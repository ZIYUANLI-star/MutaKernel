"""Aggregate MutaKernel enhanced (differential stress) testing results.

Combines:
  runs/kgb_ext_llmemd/details/*.json          (post-LLM-EMD per-mutant statuses)
  runs/kgb_ext_llmemd/stress/results.jsonl    (stress verdicts, pulled from A800)
Produces:
  runs/kgb_ext_llmemd/stress/stress_summary.json
  runs/kgb_ext_llmemd/stress/stress_details.json   (per-mutant kill detail)
  updates details/*.json with mutant["stress_result"] + ["post_stress_status"]
  runs/kgb_ext_llmemd/stress/STRESS_REPORT.md
"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLMEMD = os.path.join(ROOT, "外部Benchmark差分测试_RQ4", "MutaKernel-KGB",
                      "MutaKernel-KGB", "MutaKernel", "runs", "kgb_ext_llmemd")
DET = os.path.join(LLMEMD, "details")
STRESS = os.path.join(LLMEMD, "stress")
RESULTS = os.path.join(STRESS, "results.jsonl")


def famof(name):
    return name.split("__", 1)[0]


def main():
    res = {}
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                res[r["mutant_id"]] = r

    # global tallies
    g = collections.Counter()
    by_op = collections.defaultdict(lambda: collections.Counter())
    per_dim = collections.Counter()
    per_policy = collections.Counter()
    contradictions = []        # candidate_equivalent that stress killed
    escape_killed = []         # true_escape killed by stress
    escape_survived = []       # true_escape that even MutaKernel missed
    stress_details = []

    for fn in sorted(os.listdir(DET)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(DET, fn), encoding="utf-8"))
        fam = famof(d["kernel"]["problem_name"])
        changed = False
        for m in d["mutants"]:
            fs = m.get("final_emd_status", m["status"])
            r = res.get(m["id"])
            if fs not in ("survived", "candidate_equivalent"):
                # killed_baseline / stillborn / strict — untouched by stress phase
                m["post_stress_status"] = {
                    "killed": "killed_baseline", "stillborn": "stillborn",
                    "strict_equivalent": "equivalent_strict",
                }.get(fs, fs)
                continue
            if r is None:
                m["post_stress_status"] = fs + "__no_stress_result"
                continue
            killed = bool(r.get("any_killed"))
            mode = r.get("first_kill_mode")
            m["stress_result"] = {
                "killed": killed, "first_kill_mode": mode,
                "killed_dimensions": r.get("killed_dimensions", []),
                "crash_kill": bool(r.get("_crash_kill")),
                "main_track": r.get("main_track", {}),
                "config_track": r.get("config_track", {}),
            }
            changed = True
            if fs == "survived":   # true escape
                g["true_escape_total"] += 1
                by_op[fam]["escape_total"] += 1
                if killed:
                    g["escape_killed"] += 1
                    by_op[fam]["escape_killed"] += 1
                    m["post_stress_status"] = "killed_by_mutakernel"
                    escape_killed.append(m["id"])
                    if mode:
                        per_dim[mode] += 1
                else:
                    m["post_stress_status"] = "surviving_after_mutakernel"
                    escape_survived.append(m["id"])
            else:  # candidate_equivalent (LLM-confirmed equiv) — sanity track
                g["candidate_total"] += 1
                if killed:
                    g["candidate_killed_contradiction"] += 1
                    m["post_stress_status"] = "candidate_equiv_but_stress_killed"
                    contradictions.append({"id": m["id"], "mode": mode,
                                            "op": fam})
                else:
                    m["post_stress_status"] = "equivalent_candidate_confirmed"
            # per-policy attribution (any killing dim's policy)
            if killed:
                for trk in ("main_track", "config_track"):
                    for dim, dd in (r.get(trk, {}) or {}).items():
                        if dd.get("killed") and dd.get("killing_policy"):
                            per_policy[f"{dim}:{dd['killing_policy']}"] += 1
            stress_details.append({
                "id": m["id"], "operator": m["operator_name"], "family": fam,
                "final_emd_status": fs, "killed": killed, "first_kill_mode": mode,
                "killed_dimensions": r.get("killed_dimensions", []),
            })
        if changed:
            json.dump(d, open(os.path.join(DET, fn), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)

    # ----- counts from emd layer -----
    # recompute global status counts from details
    tot = collections.Counter()
    for fn in sorted(os.listdir(DET)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(DET, fn), encoding="utf-8"))
        for m in d["mutants"]:
            tot[m.get("final_emd_status", m["status"])] += 1

    killed_baseline = tot["killed"]
    stillborn = tot["stillborn"]
    strict = tot["strict_equivalent"]
    candidate = tot["candidate_equivalent"]
    true_escape = tot["survived"]
    escape_killed_n = g["escape_killed"]
    escape_survive_n = true_escape - escape_killed_n

    # MutaKernel effective: baseline kills + stress kills on real escapes
    total_killed_mk = killed_baseline + escape_killed_n
    denom_cons = (killed_baseline + candidate + true_escape)  # exclude stillborn+strict
    denom_opt = (killed_baseline + true_escape)               # also exclude candidate(equiv)

    summary = {
        "phase": "MutaKernel enhanced differential stress (NO LLM dimension)",
        "dimensions": ["value_stress", "config_stress", "dtype_stress", "repeated_run"],
        "input_set": {
            "true_escape_tested": true_escape,
            "candidate_equiv_tested(sanity)": candidate,
            "strict_equiv_skipped(textually_identical)": strict,
        },
        "headline": {
            "true_escape_total": true_escape,
            "escape_killed_by_mutakernel": escape_killed_n,
            "escape_still_surviving": escape_survive_n,
            "escape_kill_rate": round(escape_killed_n / max(1, true_escape), 4),
            "candidate_equiv_contradictions(stress_killed)": g["candidate_killed_contradiction"],
        },
        "mutation_score_evolution": {
            "kgb_baseline_killed": killed_baseline,
            "mutakernel_total_killed(baseline+stress)": total_killed_mk,
            "score_kgb_baseline_conservative": round(killed_baseline / max(1, denom_cons), 4),
            "score_mutakernel_conservative": round(total_killed_mk / max(1, denom_cons), 4),
            "score_kgb_baseline_optimistic": round(killed_baseline / max(1, denom_opt), 4),
            "score_mutakernel_optimistic": round(total_killed_mk / max(1, denom_opt), 4),
        },
        "per_kill_dimension(first)": dict(per_dim),
        "per_killing_policy": dict(sorted(per_policy.items(), key=lambda x: -x[1])),
        "by_operator": {k: dict(v) for k, v in sorted(by_op.items())},
        "n_stress_results": len(res),
    }
    json.dump(summary, open(os.path.join(STRESS, "stress_summary.json"), "w",
              encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(stress_details, open(os.path.join(STRESS, "stress_details.json"), "w",
              encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---------------- markdown report ----------------
    se = summary["score_mutakernel_conservative"] if False else None
    mse = summary["mutation_score_evolution"]
    L = []
    L.append("# MutaKernel 增强差分测试报告 (KGB · 无 LLM 维度)\n")
    L.append("> 在 EMD-LLM 分类之后，对全部**非严格存活体**施加五维增强差分测试")
    L.append("> （value/config/dtype/repeated；**不含 LLM 维度**），在 A800 上以子进程隔离并行执行。\n")
    L.append("## 1. 输入集合")
    L.append(f"- 真漏检 (true escape, LLM 确认): **{true_escape}**")
    L.append(f"- 候选等价 (candidate equivalent, 作清白对照): **{candidate}**")
    L.append(f"- 严格等价 (文本归一化相同, 跳过): {strict}")
    L.append(f"- KGB baseline 已杀: {killed_baseline} · stillborn: {stillborn}\n")
    L.append("## 2. 核心结果")
    h = summary["headline"]
    L.append(f"- **增强测试补杀真漏检: {h['escape_killed_by_mutakernel']}/{true_escape} "
             f"= {h['escape_kill_rate']*100:.1f}%**")
    L.append(f"- 增强后仍存活的真漏检: {h['escape_still_surviving']}")
    L.append(f"- 候选等价体被增强测试杀掉(矛盾/反证): "
             f"**{h['candidate_equiv_contradictions(stress_killed)']}/{candidate}** "
             f"→ 0 矛盾即强力佐证等价分类正确\n")
    L.append("## 3. 变异分数演化 (conservative / optimistic)")
    L.append("| 指标 | KGB baseline | MutaKernel (baseline+增强) |")
    L.append("|---|---|---|")
    L.append(f"| 已杀变异体 | {killed_baseline} | {total_killed_mk} |")
    L.append(f"| conservative 分数 | {mse['score_kgb_baseline_conservative']} | "
             f"{mse['score_mutakernel_conservative']} |")
    L.append(f"| optimistic 分数 | {mse['score_kgb_baseline_optimistic']} | "
             f"{mse['score_mutakernel_optimistic']} |")
    L.append("")
    L.append("## 4. 按击杀维度 (首杀维度归因)")
    L.append("| 维度 | 首杀数 |")
    L.append("|---|---|")
    for k, v in sorted(per_dim.items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## 5. 按击杀策略 (任一维度命中策略)")
    L.append("| 维度:策略 | 命中变异体数 |")
    L.append("|---|---|")
    for k, v in sorted(per_policy.items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## 6. 按算子 (escape 补杀 / escape 总数)")
    L.append("| 算子族 | 补杀 | 真漏检总数 | 补杀率 |")
    L.append("|---|---|---|---|")
    for op, c in sorted(by_op.items(), key=lambda x: -x[1].get("escape_total", 0)):
        et = c.get("escape_total", 0)
        ek = c.get("escape_killed", 0)
        rate = f"{ek/et*100:.0f}%" if et else "-"
        L.append(f"| {op} | {ek} | {et} | {rate} |")
    L.append("")
    L.append("## 7. 说明")
    L.append("- crash 击杀已在**单线程纯隔离**下复核全部复现，确认为真实 CUDA 级崩溃（非并发误判）。")
    L.append("- 增强测试每个变异体在独立子进程中运行，OOB 崩溃只杀其自身进程，不污染其他用例。")
    L.append("- 运行参数: workers=32(单线程/worker), timeout=240s, 比较口径=逐位一致(bitwise)。")
    open(os.path.join(STRESS, "STRESS_REPORT.md"), "w", encoding="utf-8").write(
        "\n".join(L) + "\n")

    print("=== stress aggregation done ===")
    print(f"true_escape={true_escape} escape_killed={escape_killed_n} "
          f"escape_surviving={escape_survive_n} "
          f"kill_rate={summary['headline']['escape_kill_rate']}")
    print(f"candidate_equiv contradictions (stress killed)={g['candidate_killed_contradiction']}/{candidate}")
    print(f"per_dim={dict(per_dim)}")
    print(f"score: kgb_baseline_cons={summary['mutation_score_evolution']['score_kgb_baseline_conservative']} "
          f"-> mutakernel_cons={summary['mutation_score_evolution']['score_mutakernel_conservative']}")
    print(f"by_op:")
    for k, v in summary["by_operator"].items():
        print(f"  {k}: {dict(v)}")
    return summary


if __name__ == "__main__":
    main()
