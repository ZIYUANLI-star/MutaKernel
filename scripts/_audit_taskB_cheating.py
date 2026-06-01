"""Audit Task B fixed kernels for cheating patterns.

Detect:
1. Fallback to pure PyTorch (forward() bypasses load_inline kernels entirely)
2. Conditional patch (if-branch detecting special inputs, redirecting to ref)
3. Removed CUDA/Triton kernels (no longer compiles custom kernel)
4. Massive bloat (huge new code added unrelated to kernel)
"""
import json, re, os, difflib
from pathlib import Path

TASKB = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_b_regenerate")
DET_DIR = TASKB / "details"
KERN_DIR = TASKB / "kernels"
BEST = json.loads(Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/best_kernels.json").read_text())


# ---- heuristics ----
KERNEL_INDICATORS = [
    "load_inline", "cuda_source", "cpp_source",
    "@triton.jit", "tl.load", "tl.store",
    "__global__", "__device__",
]

PYTORCH_FALLBACK_IN_FORWARD = [
    # high-level pytorch ops that replace custom CUDA kernels
    "torch.nn.functional.", "F.conv", "F.linear", "F.matmul",
    "torch.matmul", "torch.bmm", "torch.einsum",
    "@torch.compile",
    "nn.Conv1d(", "nn.Conv2d(", "nn.Conv3d(", "nn.Linear(",
    "nn.BatchNorm", "nn.LayerNorm",
    "nn.ReLU(", "nn.GELU(", "nn.Softmax(",
]

CHEATING_PATTERNS = [
    # special-case branching
    (r"if\s+\w+\.\s*(isnan|isinf|abs)\s*\(", "special-value branch"),
    (r"(torch\.allclose|torch\.eq)\s*\(", "tolerance check"),
    (r"if\s+.*(==|!=).*ref\b", "ref equality check"),
    (r"clamp_?\s*\(.*[-+]?[0-9]+e[+-]?\d", "magic numerical clamp"),
    (r"masked_fill_?\s*\(.*nan", "nan mask"),
    # Pytorch fallback patterns
    (r"return\s+self\.[a-zA-Z_]+\([^)]*\)\s*$", "delegate to self.attribute"),
]

def count_kernel_indicators(code: str) -> int:
    return sum(1 for ind in KERNEL_INDICATORS if ind in code)

def count_pytorch_fallback(code: str) -> int:
    return sum(1 for p in PYTORCH_FALLBACK_IN_FORWARD if p in code)

def detect_cheating(code: str):
    hits = []
    for pat, name in CHEATING_PATTERNS:
        for m in re.finditer(pat, code):
            hits.append((name, m.group(0)[:80]))
    return hits

def extract_forward(code: str) -> str:
    """Pull out the body of the forward() method of ModelNew."""
    m = re.search(r"class\s+ModelNew\b.*?def\s+forward\s*\([^)]*\)\s*:(.*?)(?=\n\S|\Z)",
                  code, re.DOTALL)
    return m.group(1) if m else ""

def kernel_to_forward_summary(code: str) -> dict:
    fwd = extract_forward(code)
    return {
        "total_lines": code.count("\n") + 1,
        "kernel_indicators": count_kernel_indicators(code),
        "pytorch_fallback_in_full": count_pytorch_fallback(code),
        "pytorch_fallback_in_forward": count_pytorch_fallback(fwd),
        "forward_lines": fwd.count("\n") + 1,
        "cheating_hits": detect_cheating(code),
    }


# ---- per-kernel pairwise audit ----
print(f"{'Kernel':<10} {'buggy_lines':>11} {'fixed_lines':>11} {'b_kern':>6} {'f_kern':>6} "
      f"{'b_fb':>5} {'f_fb':>5} {'cheat':>5} {'verdict':<28}")
print("-" * 130)

results = []
truly_fixed_list = []
for f in sorted(DET_DIR.glob("*.json")):
    d = json.load(open(f))
    name = d.get("kernel_name", f.stem)
    final_status = d.get("final_status", "")
    if not final_status.startswith("fixed"):
        continue
    final_round = d.get("final_round", 0)
    if final_round == 0:
        continue
    # buggy code path
    if name not in BEST:
        continue
    buggy_code = Path(BEST[name]["kernel_path"]).read_text(encoding="utf-8")
    fixed_p = KERN_DIR / f"{name}_round{final_round}.py"
    if not fixed_p.exists():
        continue
    fixed_code = fixed_p.read_text(encoding="utf-8")

    b = kernel_to_forward_summary(buggy_code)
    f_sum = kernel_to_forward_summary(fixed_code)

    # Verdict heuristic
    verdict = "OK_REAL_FIX"
    if f_sum["kernel_indicators"] < b["kernel_indicators"] and b["kernel_indicators"] > 0:
        verdict = "⚠ KERNEL_REMOVED"
    elif f_sum["pytorch_fallback_in_forward"] > b["pytorch_fallback_in_forward"]:
        verdict = "⚠ ADDED_PYTORCH_IN_FWD"
    elif f_sum["cheating_hits"] and not b["cheating_hits"]:
        verdict = f"⚠ NEW_PATTERN ({len(f_sum['cheating_hits'])})"
    elif f_sum["kernel_indicators"] == 0 and b["kernel_indicators"] > 0:
        verdict = "⚠ NO_KERNEL_LEFT"

    print(f"{name:<10} {b['total_lines']:>11} {f_sum['total_lines']:>11} "
          f"{b['kernel_indicators']:>6} {f_sum['kernel_indicators']:>6} "
          f"{b['pytorch_fallback_in_forward']:>5} {f_sum['pytorch_fallback_in_forward']:>5} "
          f"{len(f_sum['cheating_hits']):>5} {verdict:<28}")

    rec = {"name": name, "final_round": final_round,
           "buggy": b, "fixed": f_sum, "verdict": verdict,
           "buggy_code": buggy_code, "fixed_code": fixed_code}
    results.append(rec)
    if d.get("round0_stats", {}).get("n_confirmed_buggy", 0) > 0 and \
       d.get("round0_stats", {}).get("n_unexpected_pass", 0) == 0:
        truly_fixed_list.append(rec)

# ---- summary ----
print()
from collections import Counter
verdicts = Counter(r["verdict"] for r in results)
print("=== Verdict distribution ===")
for k, v in verdicts.most_common():
    print(f"  {k:<30} {v:>3}")

# ---- print diff for a few representative truly-fixed kernels ----
print()
print("=" * 80)
print("Detailed code-level comparison for 3 representative TRULY_FIXED kernels")
print("=" * 80)

# Pick representative: smallest, medium, largest
truly_fixed_list.sort(key=lambda r: r["buggy"]["total_lines"])
samples = []
if truly_fixed_list:
    samples = [truly_fixed_list[0],
               truly_fixed_list[len(truly_fixed_list)//2],
               truly_fixed_list[-1]]

for r in samples:
    name = r["name"]
    print(f"\n{'#'*70}")
    print(f"# {name}  (round {r['final_round']}, "
          f"{r['buggy']['total_lines']}→{r['fixed']['total_lines']} lines)")
    print(f"{'#'*70}")
    buggy_lines = r["buggy_code"].splitlines(keepends=True)
    fixed_lines = r["fixed_code"].splitlines(keepends=True)
    diff = list(difflib.unified_diff(buggy_lines, fixed_lines,
                                      fromfile=f"{name}_buggy",
                                      tofile=f"{name}_fixed_r{r['final_round']}",
                                      n=2))
    out = "".join(diff)
    # Limit diff output for readability
    if len(out) > 12000:
        out = out[:8000] + f"\n... [{len(out)-8000} chars truncated] ...\n" + out[-3000:]
    print(out)
