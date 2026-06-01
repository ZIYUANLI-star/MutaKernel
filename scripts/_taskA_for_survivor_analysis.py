"""Cross-check 123 non-equivalent survivors (Tier 1+2 残留) with Task A verdicts.

Output mapping: each mutant_id → Task A 5轮 killable 序列 + reason_categories + any kill suggestion.
"""
import json
from pathlib import Path

TASKA = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details")
TASKC = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/details")

# B class 16 (from 未杀死变异体逐项分析.md §5.1.2)
B_CLASS = [
    "L1_P19__relop_replace__1",
    "L1_P29__const_perturb__6",
    "L1_P33__launch_config_mutate__1",
    "L1_P33__scale_modify__1",
    "L1_P41__init_modify__2",
    "L1_P49__init_modify__6",
    "L1_P49__launch_config_mutate__1",
    "L1_P51__mask_boundary__2",
    "L1_P51__relop_replace__5",
    "L1_P52__arith_replace__11",
    "L1_P53__arith_replace__16",
    "L1_P53__arith_replace__17",
    "L1_P53__mask_boundary__5",
    "L1_P96__arith_replace__12",
    "L1_P96__launch_config_mutate__0",
    "L1_P100__sync_remove__0",
]

def get_taskA(mid: str) -> dict:
    p = TASKA / f"{mid}.json"
    if not p.exists():
        return {"found": False}
    d = json.load(open(p))
    rounds = d.get("rounds", [])
    killable_seq = [r.get("killable") for r in rounds]
    reasons = [r.get("reason_category") for r in rounds if r.get("reason_category")]
    any_True = any(k is True for k in killable_seq)
    all_False = all(k is False for k in killable_seq) and len(killable_seq) > 0
    return {
        "found": True,
        "tier": d.get("tier"),
        "killable_seq": killable_seq,
        "any_True": any_True,
        "all_False": all_False,
        "reasons": reasons,
        "killed": d.get("killed", False),
    }

def get_taskC(mid: str) -> dict:
    p = TASKC / f"{mid}.json"
    if not p.exists():
        return {"found": False}
    d = json.load(open(p))
    return {"found": True, "killed": d.get("killed", False),
            "killing_round": d.get("killing_round", 0)}

print("=" * 100)
print("Task A 对 §5.1 B 类 16 个 mutant 的二次审查")
print("=" * 100)
print(f"{'#':>3} {'mutant_id':<40} {'tier':>4} {'killable 5轮':<25} {'verdict':<12} {'C 杀?'}")
print("-" * 100)
b_class_taskA = []
for i, m in enumerate(B_CLASS, 1):
    a = get_taskA(m)
    c = get_taskC(m)
    if not a["found"]:
        print(f"{i:>3} {m:<40}  TASK A NOT FOUND")
        continue
    seq_str = "/".join(['T' if k is True else 'F' if k is False else '?' for k in a["killable_seq"]])
    verdict = "all_False" if a["all_False"] else ("any_True" if a["any_True"] else "mixed")
    c_str = ("YES r" + str(c.get("killing_round", "?"))) if c.get("killed") else ("no" if c.get("found") else "no_data")
    print(f"{i:>3} {m:<40} {a['tier']:>4} {seq_str:<25} {verdict:<12} {c_str}")
    b_class_taskA.append({"mutant_id": m, "tier": a["tier"], "verdict": verdict, "task_c_killed": c.get("killed", False)})

print()
n_all_false = sum(1 for r in b_class_taskA if r["verdict"] == "all_False")
n_any_true = sum(1 for r in b_class_taskA if r["verdict"] == "any_True")
print(f"B 类 16 个：Opus 4.5 五轮全 killable=False = {n_all_false}; 任一轮 killable=True = {n_any_true}")

# 全 123 个的统计已经从其他脚本拿到，重复 print 一下
print()
print("=" * 100)
print("Task A 总览 (跨全部 365 个 Phase II 后存活 mutant)")
print("=" * 100)
counts = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}  # [all_False, any_True, killed]
killed_by_taskC_phase2_survived = []
for p in sorted(TASKA.glob("*.json")):
    d = json.load(open(p))
    tier = d.get("tier", 0)
    if tier not in (1, 2, 3): continue
    seq = [r.get("killable") for r in d.get("rounds", [])]
    if all(k is False for k in seq) and seq:
        counts[tier][0] += 1
    if any(k is True for k in seq):
        counts[tier][1] += 1
    if d.get("killed"):
        counts[tier][2] += 1
    # cross-check with task C kill
    c_path = TASKC / p.name
    if c_path.exists():
        c = json.load(open(c_path))
        if c.get("killed") and not d.get("killed"):
            killed_by_taskC_phase2_survived.append(d.get("mutant_id"))

for t in (1, 2, 3):
    al, at, kl = counts[t]
    print(f"Tier {t}: all_False={al}  any_True={at}  killed_in_taskA={kl}")
print()
print(f"Phase 2 后存活但 Task C 杀掉的 mutant：{len(killed_by_taskC_phase2_survived)}")
for m in killed_by_taskC_phase2_survived:
    print(f"  {m}")
