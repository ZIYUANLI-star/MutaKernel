#!/usr/bin/env python3
"""统计已完成 kernel 中 C/D 类变异体的详细情况。"""
import json
from pathlib import Path
from collections import defaultdict

d = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/full_block12_results/details")
if not d.exists():
    print("No results yet")
    exit()

# Per-operator stats
op_stats = defaultdict(lambda: {"K": 0, "S": 0, "SB": 0, "EQ": 0, "cat": ""})
# Per-kernel C/D details
kernel_cd = []

for f in sorted(d.glob("*.json")):
    data = json.load(open(f))
    kname = f.stem
    c_mutants = []
    d_mutants = []
    for m in data.get("mutants", []):
        cat = m.get("operator_category", "")
        op = m.get("operator_name", "")
        status = m.get("status", "")
        if cat not in ("C", "D"):
            continue
        entry = {"op": op, "status": status, "line": m.get("site", {}).get("line_start", "?")}
        if cat == "C":
            c_mutants.append(entry)
        else:
            d_mutants.append(entry)

        s = op_stats[op]
        s["cat"] = cat
        if status == "killed":
            s["K"] += 1
        elif status == "survived":
            s["S"] += 1
        elif status == "stillborn":
            s["SB"] += 1
        elif status == "equivalent":
            s["EQ"] += 1

    if c_mutants or d_mutants:
        kernel_cd.append((kname, c_mutants, d_mutants))

# Print per-operator summary
print("=" * 65)
print("  C/D 类变异算子汇总统计")
print("=" * 65)
print(f"  {'Cat':<4} {'Operator':<22} {'Kill':>5} {'Surv':>5} {'SB':>5} {'EQ':>5} {'Total':>6} {'Score':>8}")
print(f"  {'-'*60}")

total = {"K": 0, "S": 0, "SB": 0, "EQ": 0}
for op in sorted(op_stats, key=lambda x: (op_stats[x]["cat"], x)):
    s = op_stats[op]
    t = s["K"] + s["S"] + s["SB"] + s["EQ"]
    denom = s["K"] + s["S"]
    score = s["K"] / denom if denom > 0 else 0
    print(f"  {s['cat']:<4} {op:<22} {s['K']:>5} {s['S']:>5} {s['SB']:>5} {s['EQ']:>5} {t:>6} {score:>7.1%}")
    for k in ("K", "S", "SB", "EQ"):
        total[k] += s[k]

t_all = sum(total.values())
t_denom = total["K"] + total["S"]
t_score = total["K"] / t_denom if t_denom > 0 else 0
print(f"  {'-'*60}")
print(f"  {'':4} {'TOTAL':<22} {total['K']:>5} {total['S']:>5} {total['SB']:>5} {total['EQ']:>5} {t_all:>6} {t_score:>7.1%}")

# Print per-kernel detail
print(f"\n{'=' * 65}")
print("  含有 C/D 变异体的 Kernel 明细")
print("=" * 65)
for kname, c_list, d_list in kernel_cd:
    print(f"\n  {kname}:")
    for label, mlist in [("C", c_list), ("D", d_list)]:
        if not mlist:
            continue
        for m in mlist:
            tag = m["status"].upper()
            print(f"    [{label}] {m['op']:22s} L{m['line']:<4}  {tag}")

print(f"\n  共 {len(kernel_cd)} 个 kernel 含有 C/D 变异体 (已完成 {len(list(d.glob('*.json')))} 个)")
