"""Aggregate Task A results and compare with Phase II + Task C on the same
input set (Phase I 后未杀死的 534 个 mutant)."""
import json, glob, os
from collections import Counter

TASK_A_DIR = "/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details"
TASK_C_DIR = "/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/details"
PHASE2_DIR = "/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/stress_enhance_results/details"


def scan_task_ac(detail_dir, label):
    files = sorted(glob.glob(os.path.join(detail_dir, "*.json")))
    killed = 0
    killed_by_round = Counter()
    unkillable = 0
    not_killed_other = 0
    rounds_total = 0
    in_tok = out_tok = 0

    by_phase1 = {"survived": [0,0], "candidate_equivalent": [0,0],
                 "unknown": [0,0]}
    killed_mids = set()
    unkillable_mids = set()
    notkilled_mids = set()

    for f in files:
        with open(f) as fp: d = json.load(fp)
        mid = d.get("mutant_id") or os.path.basename(f).replace(".json","")
        is_killed = bool(d.get("killed"))
        kround = d.get("killing_round", 0)
        p1 = d.get("phase1_status", "unknown") or "unknown"
        if p1 not in by_phase1: p1 = "unknown"
        by_phase1[p1][0] += 1
        if is_killed:
            killed += 1
            killed_by_round[kround] += 1
            killed_mids.add(mid)
            by_phase1[p1][1] += 1
        else:
            rounds = d.get("rounds", [])
            if rounds and any(r.get("killable") is False for r in rounds):
                unkillable += 1
                unkillable_mids.add(mid)
            else:
                not_killed_other += 1
                notkilled_mids.add(mid)
        rounds_total += len(d.get("rounds", []))
        for r in d.get("rounds", []):
            u = r.get("usage", {}) or r.get("tokens", {})
            in_tok  += u.get("prompt_tokens", 0) or u.get("input", 0)
            out_tok += u.get("completion_tokens", 0) or u.get("output", 0)

    return {
        "label": label,
        "total": len(files),
        "killed": killed,
        "killed_by_round": dict(killed_by_round.most_common()),
        "unkillable": unkillable,
        "not_killed_other": not_killed_other,
        "by_phase1": by_phase1,
        "killed_mids": killed_mids,
        "unkillable_mids": unkillable_mids,
        "notkilled_mids": notkilled_mids,
        "rounds_total": rounds_total,
        "tokens_in": in_tok,
        "tokens_out": out_tok,
    }


def scan_phase2():
    """Phase II: per-mutant any_killed status from stress_enhance_results."""
    files = sorted(glob.glob(os.path.join(PHASE2_DIR, "*.json")))
    killed = 0; not_killed = 0
    killed_mids = set(); notkilled_mids = set()
    p2llm_killed = 0
    p2llm_executed = 0
    for f in files:
        with open(f) as fp: d = json.load(fp)
        mid = d.get("mutant_id") or os.path.basename(f).replace(".json","")
        if d.get("any_killed"):
            killed += 1
            killed_mids.add(mid)
        else:
            not_killed += 1
            notkilled_mids.add(mid)
        # LLM iterative analysis stats
        lli = d.get("llm_iterative_analysis", {})
        if lli.get("executed"):
            p2llm_executed += 1
            if lli.get("killed"):
                p2llm_killed += 1
    return {
        "label": "Phase II",
        "total": len(files),
        "killed": killed,
        "not_killed": not_killed,
        "killed_mids": killed_mids,
        "notkilled_mids": notkilled_mids,
        "internal_llm_executed": p2llm_executed,
        "internal_llm_killed": p2llm_killed,
    }


print("=" * 80)
print("Phase II / Task A / Task C 三方在同一 534 输入上的对照")
print("=" * 80)

p2 = scan_phase2()
print(f"\n[Phase II]  total={p2['total']}  killed={p2['killed']}  not_killed={p2['not_killed']}")
print(f"  其中内置 LLM 迭代分析：executed={p2['internal_llm_executed']}, "
      f"killed={p2['internal_llm_killed']}")

ta = scan_task_ac(TASK_A_DIR, "Task A")
print(f"\n[Task A]    total={ta['total']}  killed={ta['killed']}  "
      f"LLM_unkillable={ta['unkillable']}  not_killed_other={ta['not_killed_other']}")
print(f"  killed by round: {ta['killed_by_round']}")
print(f"  by phase1 status:")
for k, (t, kl) in ta['by_phase1'].items():
    if t > 0:
        print(f"    {k:25s}: {kl}/{t}  ({kl/t*100:.1f}%)")
print(f"  tokens: in={ta['tokens_in']:,}  out={ta['tokens_out']:,}  "
      f"total={ta['tokens_in']+ta['tokens_out']:,}")

tc = scan_task_ac(TASK_C_DIR, "Task C")
print(f"\n[Task C]    total={tc['total']}  killed={tc['killed']}  "
      f"LLM_unkillable={tc['unkillable']}  not_killed_other={tc['not_killed_other']}")
print(f"  killed by round: {tc['killed_by_round']}")
print(f"  by phase1 status:")
for k, (t, kl) in tc['by_phase1'].items():
    if t > 0:
        print(f"    {k:25s}: {kl}/{t}  ({kl/t*100:.1f}%)")
print(f"  tokens: in={tc['tokens_in']:,}  out={tc['tokens_out']:,}  "
      f"total={tc['tokens_in']+tc['tokens_out']:,}")

# === 级联视角 ===
print("\n" + "=" * 80)
print("级联视角：Phase I 后 534 → Phase II → Task A → 最终")
print("=" * 80)
print(f"\nPhase I 后未杀死: 534")
print(f"  ↓ Phase II 增强测试（5 维 stress + DeepSeek 3 轮）")
print(f"  Phase II killed:  {p2['killed']}  ({p2['killed']/534*100:.1f}%)")
print(f"  Phase II 未杀:    {p2['not_killed']}  ({p2['not_killed']/534*100:.1f}%) → 进入 Task A")
print(f"  ↓ Task A 用 Opus 4.5 重跑 Phase II 的 LLM 模块（5 轮）")
print(f"  Task A killed:    {ta['killed']} / {ta['total']}  ({ta['killed']/ta['total']*100:.1f}%)")
print(f"  Task A unkillable: {ta['unkillable']} / {ta['total']}  ({ta['unkillable']/ta['total']*100:.1f}%)")
print(f"  Task A not_killed: {ta['not_killed_other']} / {ta['total']}  ({ta['not_killed_other']/ta['total']*100:.1f}%)")
print()
print(f"=== Phase II + Task A 合并后 ===")
final_killed = p2['killed'] + ta['killed']
final_notkilled = 534 - final_killed
print(f"  共 killed:        {p2['killed']} (Phase II) + {ta['killed']} (Task A) = {final_killed}  ({final_killed/534*100:.1f}%)")
print(f"  最终未杀:         534 - {final_killed} = {final_notkilled}  ({final_notkilled/534*100:.1f}%)")

# === Task C 对照 ===
print()
print(f"=== Task C (旁路对照：跳过 Phase II 直接 Opus) ===")
print(f"  same input set: 534")
print(f"  Task C killed:     {tc['killed']}  ({tc['killed']/tc['total']*100:.1f}%)")
print(f"  Task C unkillable: {tc['unkillable']}  ({tc['unkillable']/tc['total']*100:.1f}%)")
print(f"  Task C not_killed: {tc['not_killed_other']}  ({tc['not_killed_other']/tc['total']*100:.1f}%)")

# === 交集分析 ===
print("\n" + "=" * 80)
print("交集分析（kill 重叠）")
print("=" * 80)
p2_k = p2['killed_mids']
ta_k = ta['killed_mids']
tc_k = tc['killed_mids']
print(f"  Phase II killed:        {len(p2_k)}")
print(f"  Task A   killed:        {len(ta_k)}")
print(f"  Task C   killed:        {len(tc_k)}")
print()
print(f"  Phase II ∩ Task A:      {len(p2_k & ta_k)}  (Task A 重复杀的)")
print(f"  Phase II ∩ Task C:      {len(p2_k & tc_k)}  (Task C 与 Phase II 重叠)")
print(f"  Task A   ∩ Task C:      {len(ta_k & tc_k)}  (Opus 两种使用模式都能杀)")
print()
print(f"  ★ Only Phase II:          {len(p2_k - ta_k - tc_k)}  (只有 Phase II 杀得到)")
print(f"  ★ Only Task A:            {len(ta_k - p2_k - tc_k)}  (Phase II + Task C 都没杀，仅 Task A 杀到)")
print(f"  ★ Only Task C:            {len(tc_k - p2_k - ta_k)}  (绕过 Phase II 直接 LLM 杀到)")
print(f"  ★ All three:              {len(p2_k & ta_k & tc_k)}  (三种方法都杀)")
print(f"  ★ Phase II + Task A:      {len((p2_k | ta_k) - tc_k)}  (Phase II ∪ Task A) - Task C")
print(f"  ★ Union of all:           {len(p2_k | ta_k | tc_k)}  (任一方法杀到的总和)")
