"""Aggregate raw statistics for Task A, B, C from per-mutant/per-kernel JSON."""
import json, glob
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充")


def aggregate_task_a():
    print("=" * 78)
    print("TASK A — Phase II unkilled 重跑 (Opus 4.5, 5 rounds)")
    print("=" * 78)
    det_dir = ROOT / "task_a_phase2_rerun" / "details"
    files = sorted(det_dir.glob("*.json"))
    print(f"详情文件总数: {len(files)}")

    killable_counter = Counter()
    reason_counter = Counter()
    killed_counter = Counter()
    rounds_used = Counter()
    operator_counter = Counter()
    by_operator_killed = defaultdict(lambda: {"total": 0, "killed": 0, "killable": 0})
    elapsed_total = 0.0
    rounds_with_killable_true = 0
    rounds_with_killable_false = 0

    killed_mutants = []

    for f in files:
        d = json.load(open(f))
        op = d.get("operator_name", "?")
        operator_counter[op] += 1
        if d.get("killed"):
            killed_counter["killed_true"] += 1
            killed_mutants.append({"id": d["mutant_id"], "kr": d.get("killing_round", 0)})
        else:
            killed_counter["killed_false"] += 1
        rounds_used[d.get("killing_round", 0)] += 1
        elapsed_total += float(d.get("elapsed_sec", 0))
        by_operator_killed[op]["total"] += 1
        if d.get("killed"):
            by_operator_killed[op]["killed"] += 1
        for r in d.get("rounds", []):
            kb = r.get("killable")
            if kb is True:
                rounds_with_killable_true += 1
                by_operator_killed[op]["killable"] += 1
                break  # only count once per mutant
            elif kb is False:
                rounds_with_killable_false += 1
            cat = r.get("reason_category") or "(none)"
            reason_counter[cat] += 1

    print(f"\nkilled=true: {killed_counter['killed_true']}")
    print(f"killed=false: {killed_counter['killed_false']}")
    print(f"\n累计 LLM rounds 调用：{sum(rounds_used.values())}")
    print(f"任一轮 killable=True 的 mutant 数: {rounds_with_killable_true}")
    print(f"任一轮 killable=False 的 mutant 数 (累计): {rounds_with_killable_false}")
    print(f"\n--- killing_round 分布 (0 = 未杀) ---")
    for k in sorted(rounds_used):
        print(f"  round {k}: {rounds_used[k]}")
    print(f"\n--- 算子分布 ---")
    for op, c in operator_counter.most_common():
        bk = by_operator_killed[op]
        print(f"  {op:25} {c:3} candidates, killed={bk['killed']:2}, killable_claimed_once={bk['killable']:2}")
    print(f"\n--- reason_category top 12 ---")
    for cat, c in reason_counter.most_common(12):
        print(f"  {cat:40} {c}")
    print(f"\n累计 elapsed: {elapsed_total:.0f} sec = {elapsed_total/3600:.2f} h")
    print(f"\n被杀的 mutant 名单: {killed_mutants}")


def aggregate_task_c():
    print()
    print("=" * 78)
    print("TASK C — Phase I unkilled 直接用 Opus 4.5 (5 rounds, no Phase II info)")
    print("=" * 78)
    det_dir = ROOT / "task_c_phase1_direct" / "details"
    files = sorted(det_dir.glob("*.json"))
    print(f"详情文件总数: {len(files)}")

    killed = []
    survived = 0
    operator_counter = Counter()
    rounds_used = Counter()
    by_operator_killed = defaultdict(lambda: {"total": 0, "killed": 0})
    elapsed_total = 0.0

    for f in files:
        d = json.load(open(f))
        op = d.get("operator_name", "?")
        operator_counter[op] += 1
        by_operator_killed[op]["total"] += 1
        if d.get("killed"):
            killed.append({"id": d["mutant_id"], "kr": d.get("killing_round", 0),
                          "operator": op})
            by_operator_killed[op]["killed"] += 1
        else:
            survived += 1
        rounds_used[d.get("killing_round", 0)] += 1
        elapsed_total += float(d.get("elapsed_sec", 0))

    print(f"\nkilled: {len(killed)} / {len(files)} = {len(killed)/len(files)*100:.2f}%")
    print(f"survived: {survived}")
    print(f"\n--- killing_round 分布 (0 = 未杀) ---")
    for k in sorted(rounds_used):
        print(f"  round {k}: {rounds_used[k]}")
    print(f"\n--- 算子杀率 ---")
    for op, c in operator_counter.most_common():
        bk = by_operator_killed[op]
        rate = bk["killed"] / bk["total"] * 100 if bk["total"] else 0
        print(f"  {op:25} {c:3} candidates, killed={bk['killed']:2} ({rate:.1f}%)")
    print(f"\n累计 elapsed: {elapsed_total:.0f} sec = {elapsed_total/3600:.2f} h")
    print(f"\n--- 被杀 mutant 名单 (id, killing_round, operator) ---")
    for m in sorted(killed, key=lambda x: x["id"]):
        print(f"  {m['id']:50} r{m['kr']}  {m['operator']}")


def aggregate_task_b():
    print()
    print("=" * 78)
    print("TASK B — buggy kernel 重生成 (Opus 4.5, 3 rounds + R0 + double V)")
    print("=" * 78)
    det_dir = ROOT / "task_b_regenerate" / "details"
    files = sorted(det_dir.glob("*.json"))
    print(f"详情文件总数: {len(files)}")

    fixed = 0; failed = 0
    by_round = Counter()
    r0_pseudo = []
    r0_partial = []
    r0_clean = []
    fixed_list = []
    failed_list = []

    for f in files:
        d = json.load(open(f))
        name = d.get("kernel_name", f.stem)
        status = d.get("final_status", "")
        fr = d.get("final_round", 0)
        r0 = d.get("round0_stats", {})
        n_total = r0.get("n_total", 0)
        n_conf = r0.get("n_confirmed_buggy", 0)
        n_unexp = r0.get("n_unexpected_pass", 0)

        if status.startswith("fixed"):
            fixed += 1
            fixed_list.append({"name": name, "final_round": fr,
                              "n_conf": n_conf, "n_unexp": n_unexp,
                              "n_total": n_total})
            by_round[fr] += 1
        else:
            failed += 1
            failed_list.append({"name": name, "status": status})

        if n_conf == 0 and n_unexp > 0:
            r0_pseudo.append((name, n_total, n_unexp))
        elif n_unexp >= n_conf and n_conf > 0:
            r0_partial.append((name, n_total, n_conf, n_unexp))
        else:
            r0_clean.append((name, n_total, n_conf, n_unexp))

    print(f"\nfixed: {fixed} / failed: {failed}")
    print(f"--- 修复轮数分布 ---")
    for r in sorted(by_round):
        print(f"  round {r}: {by_round[r]}")

    print(f"\n--- 修复 kernel 名单 ---")
    for d in fixed_list:
        print(f"  {d['name']:8} R{d['final_round']}  R0: 确认buggy={d['n_conf']:>3}/{d['n_total']:>3}  unexpected_pass={d['n_unexp']}")
    print(f"\n--- 未修复 kernel ---")
    for d in failed_list:
        print(f"  {d['name']:8} status={d['status']}")

    print(f"\n--- R0 全确认 buggy (clean) ---")
    for n, t, c, u in r0_clean:
        print(f"  {n:8} {c}/{t} confirmed buggy, {u} unexpected pass")
    print(f"\n--- R0 部分 unexpected_pass (partial pseudo) ---")
    for n, t, c, u in r0_partial:
        print(f"  {n:8} total={t} confirmed_buggy={c} unexpected_pass={u}")
    print(f"\n--- R0 全 unexpected pass (PSEUDO_FIX) ---")
    for n, t, u in r0_pseudo:
        print(f"  {n:8} total={t} unexpected_pass={u}  ← LLM 修复了一个并不 buggy 的 kernel")


if __name__ == "__main__":
    aggregate_task_a()
    aggregate_task_c()
    aggregate_task_b()
