"""Cross-check: Phase II 后存活的 365 个 mutant，按 tier 分类 + Task A 处理结果。"""
import json, glob
from pathlib import Path
from collections import Counter

DET_DIR = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details")

tier_counter = Counter()
killable_by_tier = {1: {"True_once": 0, "False_all": 0},
                    2: {"True_once": 0, "False_all": 0},
                    3: {"True_once": 0, "False_all": 0}}
reason_by_tier = {1: Counter(), 2: Counter(), 3: Counter()}
killed_by_tier = {1: 0, 2: 0, 3: 0}

unknown_tier = []

for f in sorted(DET_DIR.glob("*.json")):
    d = json.load(open(f))
    tier = d.get("tier", None)
    if tier not in (1, 2, 3):
        unknown_tier.append((d.get("mutant_id"), tier))
        continue
    tier_counter[tier] += 1
    if d.get("killed"): killed_by_tier[tier] += 1
    any_killable_true = False
    for r in d.get("rounds", []):
        kb = r.get("killable")
        if kb is True:
            any_killable_true = True
        cat = r.get("reason_category") or "(none)"
        reason_by_tier[tier][cat] += 1
    if any_killable_true:
        killable_by_tier[tier]["True_once"] += 1
    else:
        killable_by_tier[tier]["False_all"] += 1

print(f"Total Task A 详情 = {sum(tier_counter.values())}")
print(f"未知 tier 条目 = {len(unknown_tier)}")
print(f"\n=== Task A 365 个 mutant 按 tier 分布 ===")
for t in (1, 2, 3):
    print(f"  Tier {t}: {tier_counter[t]}  (killed={killed_by_tier[t]})")
print(f"  总计 = {sum(tier_counter.values())}")

print(f"\n=== Task A killable 判定 by tier ===")
for t in (1, 2, 3):
    kb = killable_by_tier[t]
    total = kb["True_once"] + kb["False_all"]
    print(f"  Tier {t}: 5轮中任一killable=True {kb['True_once']:3} ({kb['True_once']/total*100:.1f}%) | "
          f"5轮全部killable=False {kb['False_all']:3} ({kb['False_all']/total*100:.1f}%)")

print(f"\n=== Task A reason_category by tier (top 5) ===")
for t in (1, 2, 3):
    print(f"  Tier {t}:")
    for cat, n in reason_by_tier[t].most_common(5):
        print(f"    {cat:30} {n}")

# 对照: Phase II 后 Tier 分布应该是：
# Tier 1 残留 = 151 - 128 = 23
# Tier 2 残留 = 119 - 19 = 100
# Tier 3 残留 = 264 - 22 = 242
# 合计 = 23 + 100 + 242 = 365 ✓
print(f"\n=== 对照: Phase II 后存活 tier 分布（理论值） ===")
print(f"  Tier 1 残留: 151 - 128 = 23")
print(f"  Tier 2 残留: 119 - 19  = 100")
print(f"  Tier 3 残留: 264 - 22  = 242")
print(f"  合计 = 365")
