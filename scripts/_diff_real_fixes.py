"""Compare buggy vs fixed kernels for REAL_CUDA_FIX cases to understand the fix logic."""
import json, difflib, re
from pathlib import Path

PROJECT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
BEST = json.loads((PROJECT / "best_kernels.json").read_text())
TASKB = PROJECT / "第二次实验汇总" / "第二次实验汇总_补充" / "task_b_regenerate"

REAL_FIXES = ["L1_P14", "L1_P22", "L1_P39", "L1_P97"]

print("=" * 78)
print("REAL_CUDA_FIX 详细 diff 分析")
print("=" * 78)

for name in REAL_FIXES:
    detail = json.load(open(TASKB / "details" / f"{name}.json"))
    final_r = detail["final_round"]
    buggy = Path(BEST[name]["kernel_path"]).read_text(encoding="utf-8").splitlines()
    fixed = (TASKB / "kernels" / f"{name}_round{final_r}.py").read_text(encoding="utf-8").splitlines()

    # 标识算子
    problem = detail.get("problem_id"); lvl = detail.get("level")
    n_inputs = detail.get("round0_stats", {}).get("n_total", "?")
    n_buggy = detail.get("round0_stats", {}).get("n_confirmed_buggy", "?")

    print(f"\n{'='*78}\n{name}  (round {final_r}) — {lvl}/P{problem}")
    print(f"  R0: {n_buggy}/{n_inputs} 输入确认 buggy")
    print(f"  buggy_lines={len(buggy)}  fixed_lines={len(fixed)}")
    print(f"{'='*78}")

    diff = list(difflib.unified_diff(buggy, fixed,
                                      fromfile=f"{name}_buggy",
                                      tofile=f"{name}_fixed_r{final_r}",
                                      n=1, lineterm=""))
    out = "\n".join(diff)
    # Limit
    if len(out) > 6500:
        out = out[:4500] + f"\n... ({len(out)-4500} 字符省略) ...\n" + out[-1500:]
    print(out)
