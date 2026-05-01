"""分析当前等价检测实验的详细进度。"""
import json, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel\第二次实验汇总"

results = json.load(open(f"{ROOT}\\equiv_recheck_results.json", encoding="utf-8"))

# 按 result 统计
counts = defaultdict(int)
# 按算子统计
op_counts = defaultdict(lambda: defaultdict(int))
# 按 kernel 统计
kernel_counts = defaultdict(lambda: defaultdict(int))

for r in results:
    res = r["result"]
    op = r.get("operator", "unknown")
    cat = r.get("category", "unknown")
    mid = r["mutant_id"]
    kernel = "_".join(mid.split("__")[0].split("_")[:2])  # e.g. L1_P1

    counts[res] += 1
    op_counts[op][res] += 1
    kernel_counts[kernel][res] += 1

total = len(results)
print(f"{'='*70}")
print(f"  增强等价检测实验进度报告")
print(f"{'='*70}")
print(f"  总检测: {total} / 322")
print(f"  剩余: {322 - total}")
print()

print(f"  ┌─────────────┬───────┬───────┐")
print(f"  │ 结果        │ 数量  │ 占比  │")
print(f"  ├─────────────┼───────┼───────┤")
for res in ["equivalent", "survived", "timeout", "error"]:
    c = counts[res]
    pct = c / total * 100 if total > 0 else 0
    print(f"  │ {res:<11} │ {c:>5} │ {pct:>5.1f}% │")
print(f"  └─────────────┴───────┴───────┘")

print(f"\n  按变异算子分类:")
print(f"  {'算子':<25} {'等价':>5} {'存活':>5} {'超时':>5} {'总计':>5} {'等价率':>7}")
print(f"  {'─'*25} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*7}")
for op in sorted(op_counts.keys()):
    d = op_counts[op]
    eq = d.get("equivalent", 0)
    sv = d.get("survived", 0)
    to = d.get("timeout", 0)
    er = d.get("error", 0)
    t = eq + sv + to + er
    eq_rate = eq / t * 100 if t > 0 else 0
    print(f"  {op:<25} {eq:>5} {sv:>5} {to:>5} {t:>5} {eq_rate:>6.1f}%")

print(f"\n  按 Kernel 分类 (仅显示有存活的):")
print(f"  {'Kernel':<15} {'等价':>5} {'存活':>5} {'超时':>5} {'总计':>5}")
print(f"  {'─'*15} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")
for k in sorted(kernel_counts.keys()):
    d = kernel_counts[k]
    sv = d.get("survived", 0)
    if sv > 0:
        eq = d.get("equivalent", 0)
        to = d.get("timeout", 0)
        er = d.get("error", 0)
        t = eq + sv + to + er
        print(f"  {k:<15} {eq:>5} {sv:>5} {to:>5} {t:>5}")

# survived 变异体详细列表
print(f"\n  真正存活的 {counts['survived']} 个变异体:")
for r in results:
    if r["result"] == "survived":
        div = ""
        if r.get("diverged_at") is not None:
            div = f" (diverged @iter={r['diverged_at']}, policy={r.get('diverged_policy','')})"
        print(f"    {r['mutant_id']}: {r['operator']}{div}")

print(f"\n{'='*70}")
print(f"  意义: 第一次实验中 322 个'存活'变异体中，增强检测已识别出")
print(f"  {counts['equivalent']} 个为等价变异体 ({counts['equivalent']/total*100:.1f}%)，")
print(f"  仅 {counts['survived']} 个为真正存活 ({counts['survived']/total*100:.1f}%)")
print(f"{'='*70}")
