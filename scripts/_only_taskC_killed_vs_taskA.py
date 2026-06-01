"""Find mutants killed by Task C but NOT by Phase II, and check Task A status.

Outputs a side-by-side comparison of LLM reasoning between Task A and Task C
for each such mutant.
"""
import json, glob, os
from pathlib import Path

ROOT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总")
PHASE2_DIR = ROOT / "stress_enhance_results" / "details"
TASK_A_DIR = ROOT / "第二次实验汇总_补充" / "task_a_phase2_rerun" / "details"
TASK_C_DIR = ROOT / "第二次实验汇总_补充" / "task_c_phase1_direct" / "details"

# ---- collect Phase II any_killed per mutant ----
p2_killed = {}
p2_first_dim = {}
for f in PHASE2_DIR.glob("*.json"):
    d = json.loads(f.read_text(encoding="utf-8"))
    mid = d.get("mutant_id") or f.stem
    p2_killed[mid] = bool(d.get("any_killed"))
    if d.get("any_killed"):
        # Try to identify which dimension first killed
        mt = d.get("main_track", {})
        for dim in ("value_stress", "dtype_stress", "training_stress",
                     "config_stress", "repeated_run"):
            if mt.get(dim, {}).get("killed"):
                p2_first_dim[mid] = dim
                break
        if mid not in p2_first_dim:
            lli = d.get("llm_iterative_analysis", {})
            if lli.get("killed"):
                p2_first_dim[mid] = "llm_iterative_analysis"

# ---- collect Task A summaries ----
task_a = {}
for f in TASK_A_DIR.glob("*.json"):
    d = json.loads(f.read_text(encoding="utf-8"))
    mid = d.get("mutant_id") or f.stem
    rounds = d.get("rounds", [])
    task_a[mid] = {
        "killed": bool(d.get("killed")),
        "killing_round": d.get("killing_round", 0),
        "rounds_count": len(rounds),
        "killable_decisions": [r.get("killable") for r in rounds],
        "reason_categories": [r.get("reason_category") for r in rounds],
        "first_proof": (rounds[0].get("proof_sketch", "")[:300] if rounds else ""),
        "first_survival": (rounds[0].get("survival_reason", "")[:300] if rounds else ""),
        "first_kill_strategy": (rounds[0].get("kill_strategy", "")[:300] if rounds else ""),
    }

# ---- collect Task C summaries (and find killed by Task C) ----
task_c_killed_mids = []
task_c = {}
for f in TASK_C_DIR.glob("*.json"):
    d = json.loads(f.read_text(encoding="utf-8"))
    mid = d.get("mutant_id") or f.stem
    rounds = d.get("rounds", [])
    rec = {
        "killed": bool(d.get("killed")),
        "killing_round": d.get("killing_round", 0),
        "rounds_count": len(rounds),
        "phase1_status": d.get("phase1_status"),
        "operator_name": d.get("operator_name"),
        "kernel_name": d.get("kernel_name"),
    }
    if rec["killed"] and rec["killing_round"] > 0:
        kr = rounds[rec["killing_round"] - 1]
        rec["kill_strategy"] = kr.get("kill_strategy", "")
        rec["proof_sketch"] = kr.get("proof_sketch", "")
        rec["suggested_code"] = kr.get("suggested_code", "")
        rec["exec_result"] = kr.get("execution_result", {})
        rec["all_rounds_killable"] = [r.get("killable") for r in rounds]
        task_c_killed_mids.append(mid)
    task_c[mid] = rec

# ---- intersection analysis ----
only_taskC = []
for mid in task_c_killed_mids:
    if not p2_killed.get(mid, False):
        only_taskC.append(mid)

print(f"Phase II killed total:                  {sum(p2_killed.values())}")
print(f"Task C killed total (current snapshot): {len(task_c_killed_mids)}")
print(f"Task A killed total:                    {sum(1 for v in task_a.values() if v['killed'])}")
print(f"")
print(f">>> ★ Task C killed but Phase II NOT killed: {len(only_taskC)} mutants ★")
print(f">>> (这些就是 Task C 的'独立发现')")
print()

# ---- detailed side-by-side for each ----
for i, mid in enumerate(only_taskC, 1):
    tc = task_c[mid]
    ta = task_a.get(mid, None)
    print(f"\n{'='*78}")
    print(f"[{i}] {mid}  (Phase1={tc['phase1_status']})")
    print(f"{'='*78}")
    print(f"  kernel: {tc.get('kernel_name')}    operator: {tc.get('operator_name')}")
    print(f"  Phase II any_killed: {p2_killed.get(mid)}  → Phase II 未杀")
    print(f"  Task C: killed at round {tc['killing_round']} / {tc['rounds_count']}")
    print(f"          all_rounds killable verdicts: {tc.get('all_rounds_killable')}")
    print(f"          ★ Task C killing strategy:")
    ks = tc.get('kill_strategy', '')[:400]
    print(f"            {ks}")
    print(f"          ★ Task C proof_sketch:")
    ps = tc.get('proof_sketch', '')[:300]
    print(f"            {ps}")

    if ta is None:
        print(f"  Task A: NOT FOUND (该 mutant 不在 Task A 目标集中)")
        # 检查为什么不在 Task A 目标集
        print(f"          → 原因：Task A 只覆盖 Phase II any_killed=False 的 mutant")
        print(f"            既然 Phase II not killed，应该在 Task A 中。检查 Task A 是否漏跑。")
    else:
        print(f"  Task A: killed = {ta['killed']} | rounds = {ta['rounds_count']}/5")
        print(f"          per-round killable: {ta['killable_decisions']}")
        print(f"          reason_categories:  {ta['reason_categories']}")
        print(f"          ★ Task A round-1 reason: {ta['reason_categories'][0] if ta['reason_categories'] else '?'}")
        print(f"          ★ Task A round-1 proof_sketch:")
        print(f"            {ta['first_proof']}")
        print(f"          ★ Task A round-1 survival_reason:")
        print(f"            {ta['first_survival']}")
        if ta.get('first_kill_strategy'):
            print(f"          ★ Task A round-1 kill_strategy:")
            print(f"            {ta['first_kill_strategy']}")

# ---- aggregate Task A reason categories for these specific mutants ----
print(f"\n\n{'='*78}")
print("聚合：在这些'Task C 杀掉但 Phase II 没杀'的 mutant 上，Task A 给出的判定")
print(f"{'='*78}")
from collections import Counter
reason_round1 = Counter()
killable_round1 = Counter()
for mid in only_taskC:
    ta = task_a.get(mid)
    if ta and ta['reason_categories']:
        reason_round1[ta['reason_categories'][0]] += 1
    if ta and ta['killable_decisions']:
        killable_round1[ta['killable_decisions'][0]] += 1
print(f"\nTask A round-1 'killable' 决策分布:")
for k, v in killable_round1.most_common():
    print(f"  killable={k}: {v}")
print(f"\nTask A round-1 reason_category 分布:")
for k, v in reason_round1.most_common():
    print(f"  {k}: {v}")
