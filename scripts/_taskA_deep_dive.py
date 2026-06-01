import json, glob, os
from collections import Counter

d_dir = "/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details"
files = sorted(glob.glob(os.path.join(d_dir, "*.json")))

# Aggregate stats
killable_counter = Counter()
reason_counter = Counter()
rounds_count = Counter()
early_stop_round1 = 0
ran_full_5 = 0
exec_failed = 0
suggested_input_failed_to_kill = 0
mids_per_kernel = Counter()
deepseek_baseline_killed_in_p2 = 0

# Pick a sample to show prompt structure
sample_round1 = None
sample_round5 = None

for f in files:
    d = json.load(open(f))
    rs = d.get("rounds", [])
    rounds_count[len(rs)] += 1
    if len(rs) == 1:
        early_stop_round1 += 1
    if len(rs) == 5:
        ran_full_5 += 1
    for r in rs:
        killable_counter[r.get("killable")] += 1
        reason_counter[r.get("reason_category")] += 1
        er = r.get("execution_result") or {}
        if er.get("executed") and not er.get("killed", False):
            suggested_input_failed_to_kill += 1
        if er.get("executed") is False:
            exec_failed += 1
    ds = d.get("deepseek_baseline", {})
    if ds.get("killed"):
        deepseek_baseline_killed_in_p2 += 1
    mids_per_kernel[d.get("kernel_name")] += 1
    if sample_round1 is None and len(rs) >= 1:
        sample_round1 = (d["mutant_id"], rs[0])
    if sample_round5 is None and len(rs) >= 5:
        sample_round5 = (d["mutant_id"], rs)

print(f"\n=== rounds count distribution ===")
for k, v in sorted(rounds_count.items()):
    print(f"  rounds={k}: {v}")

print(f"\n  早停（仅 1 轮 Opus 判 unkillable 后停）: {early_stop_round1}/{len(files)}")
print(f"  跑满 5 轮:                              {ran_full_5}/{len(files)}")

print(f"\n=== 每轮 killable 判定 ===")
for k, v in killable_counter.most_common():
    print(f"  killable={k}: {v}")

print(f"\n=== 每轮 reason_category ===")
for k, v in reason_counter.most_common(10):
    print(f"  {k}: {v}")

print(f"\n=== 执行结果统计 ===")
print(f"  提供输入但未杀死 (executed=True, killed=False): {suggested_input_failed_to_kill}")
print(f"  代码无法执行 (executed=False):                  {exec_failed}")

print(f"\n=== DeepSeek baseline 对比（Phase II 内置 LLM 的结果） ===")
print(f"  Phase II DeepSeek 杀掉的 mutant 进入 Task A: {deepseek_baseline_killed_in_p2}")

print(f"\n=== Task A 单个 mutant 样本（首个 round 1） ===")
mid, r = sample_round1
print(f"mutant: {mid}")
for k, v in r.items():
    if isinstance(v, dict):
        s = json.dumps(v, ensure_ascii=False)
    else:
        s = str(v)
    if len(s) > 250: s = s[:250] + "..."
    print(f"  {k}: {s}")

print(f"\n=== Task A 跑满 5 轮的一个样本 ===")
if sample_round5:
    mid, rs = sample_round5
    print(f"mutant: {mid}")
    for r in rs:
        print(f"  --- round {r.get('round')} ---")
        print(f"    killable: {r.get('killable')}")
        print(f"    reason_category: {r.get('reason_category')}")
        er = r.get("execution_result", {})
        print(f"    execution_result.executed: {er.get('executed')}")
        print(f"    execution_result.killed:   {er.get('killed')}")
        print(f"    execution_result.reason:   {(er.get('reason') or '')[:200]}")
else:
    print("  (none ran full 5)")
