"""Strict v2 audit using direct __global__ count on full file content.
For each fixed kernel, compare __global__ count, also check forward() does what."""
import json, re
from pathlib import Path

ROOT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_b_regenerate")
BEST = json.loads(Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/best_kernels.json").read_text())


PYTORCH_FALLBACK_TOKENS = [
    "torch::mm(", "torch::matmul(", "torch::bmm(",
    "at::mm(", "at::matmul(",
    "torch::cumsum", "torch::cumprod",
    "cublasSgemm", "cublasDgemm",
]
PY_FALLBACK_FWD = [
    "torch.matmul(", "torch.cumsum(", "torch.bmm(", "torch.mm(",
    "torch.cumprod(",
]


def analyze(code: str) -> dict:
    n_global = len(re.findall(r"__global__\s+(?:void|\w+\s+\w+)", code))
    has_load_inline = "load_inline" in code
    n_torch_cpp_wrap = sum(code.count(tok) for tok in PYTORCH_FALLBACK_TOKENS)
    # Forward body
    fwd_match = re.search(r"def\s+forward\s*\([^)]*\)\s*:(.*?)(?=\n    def\s|\nclass\s|\Z)",
                          code, re.DOTALL)
    fwd_body = fwd_match.group(1) if fwd_match else ""
    py_fb_in_fwd = sum(fwd_body.count(tok) for tok in PY_FALLBACK_FWD)
    # Check if forward is essentially "return torch.xxx(...)"
    fwd_lines = [ln.strip() for ln in fwd_body.splitlines()
                 if ln.strip() and not ln.strip().startswith(('#', '"', "'"))]
    fwd_is_pure_py_op = (any(re.search(r"return\s+torch\.\w+\(", l) for l in fwd_lines)
                        and not has_load_inline)
    return {
        "n_global": n_global,
        "has_load_inline": has_load_inline,
        "n_torch_cpp_wrap": n_torch_cpp_wrap,
        "py_fallback_in_forward": py_fb_in_fwd,
        "forward_pure_torch": fwd_is_pure_py_op,
        "code_lines": code.count("\n") + 1,
    }


def verdict(b: dict, f: dict) -> str:
    """Compare buggy vs fixed to determine cheating."""
    if not f["has_load_inline"] and f["forward_pure_torch"]:
        return "CHEAT_PYTORCH_OP"
    if f["has_load_inline"] and f["n_global"] == 0 and f["n_torch_cpp_wrap"] > 0:
        return "CHEAT_CPP_WRAPPER"
    if b["n_global"] > 0 and f["n_global"] == 0:
        return "CHEAT_KERNEL_REMOVED"
    if b["n_global"] > 0 and f["n_global"] >= 1:
        return "REAL_FIX"
    return "OTHER"


print(f"{'Kernel':<8} {'R':>2} {'b_glo':>5} {'f_glo':>5} {'f_inl':>5} {'f_cpp_wrap':>10} "
      f"{'f_pure_T':>8} {'verdict':<22}")
print("-" * 95)

rows = []
for f in sorted((ROOT / "details").glob("*.json")):
    d = json.load(open(f))
    name = d.get("kernel_name")
    if not d.get("final_status", "").startswith("fixed"): continue
    fr = d.get("final_round", 0)
    if fr == 0 or name not in BEST: continue
    fixed_p = ROOT / "kernels" / f"{name}_round{fr}.py"
    if not fixed_p.exists(): continue
    buggy_code = Path(BEST[name]["kernel_path"]).read_text(encoding="utf-8")
    fixed_code = fixed_p.read_text(encoding="utf-8")
    b = analyze(buggy_code)
    fs = analyze(fixed_code)
    v = verdict(b, fs)
    r0 = d.get("round0_stats", {})
    pseudo = ""
    if r0.get("n_confirmed_buggy", 0) == 0 and r0.get("n_unexpected_pass", 0) > 0:
        pseudo = "PSEUDO"
    elif r0.get("n_unexpected_pass", 0) >= max(1, r0.get("n_confirmed_buggy", 0)):
        pseudo = "PARTIAL"
    print(f"{name:<8} {fr:>2} {b['n_global']:>5} {fs['n_global']:>5} "
          f"{int(fs['has_load_inline']):>5} {fs['n_torch_cpp_wrap']:>10} "
          f"{int(fs['forward_pure_torch']):>8} {v:<22} {pseudo}")
    rows.append({"name": name, "round": fr, "buggy": b, "fixed": fs,
                 "verdict": v, "pseudo": pseudo})

# Summary
from collections import Counter
c = Counter((r["verdict"], r["pseudo"]) for r in rows)
print("\n=== Summary ===")
for (v, p), n in sorted(c.items()):
    print(f"  {v:25}  pseudo={p:8}  count={n}")

# Save
out = ROOT / "audit_taskB_strict.json"
out.write_text(json.dumps([{**r, "buggy": r["buggy"], "fixed": r["fixed"]} for r in rows],
                          indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved: {out}")
