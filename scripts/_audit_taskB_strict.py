"""Strict Task B audit: classify each fixed kernel into 4 categories.

1. REAL_CUDA_FIX:   custom __global__ kernel(s) preserved + actual fix logic
2. PYTORCH_OP:      forward() directly returns torch.xxx() (no load_inline at all)
3. PYTORCH_WRAPPER: load_inline exists but cuda_source body only calls
                    torch::mm / cublasSgemm / cudnnConv / at::matmul etc.
                    (no original __global__ kernel)
4. TRIVIAL_SHELL:   not enough kernel code

Also: count __global__ kernel functions in cuda_source as a structural measure.
"""
import json, re, ast
from pathlib import Path

TASKB = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_b_regenerate")
DET_DIR = TASKB / "details"
KERN_DIR = TASKB / "kernels"
BEST = json.loads(Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/best_kernels.json").read_text())


PYTORCH_CPP_WRAPPER_KEYWORDS = [
    # PyTorch C++ ops (i.e. wrapper, not custom CUDA kernel)
    "torch::mm(", "torch::matmul(", "torch::bmm(",
    "at::mm(", "at::matmul(", "at::bmm(",
    "at::native::matmul", "at::native::mm",
    "torch::nn::functional::linear",
    "cublasSgemm", "cublasDgemm", "cublasGemmEx",
    "cudnnConvolution", "cudnn",
    "torch::softmax", "torch::layer_norm",
    "torch::cumsum", "torch::cumprod",
]


def extract_cuda_source(py_code: str) -> str:
    """Extract the triple-quoted cuda_sources string."""
    # Look for assignments like `xxx_source = """ ... """`
    pattern = re.compile(
        r'(?:cuda_source|kernel_source|.*_source)\s*=\s*("""|\'\'\')(.*?)\1',
        re.DOTALL,
    )
    bodies = []
    for m in pattern.finditer(py_code):
        body = m.group(2)
        # Heuristic: only consider cuda-like bodies
        if "#include" in body or "__global__" in body or "torch::" in body:
            bodies.append(body)
    return "\n".join(bodies)


def count_global_kernels(cuda_src: str) -> int:
    return len(re.findall(r"__global__\s+\w+", cuda_src))


def count_pytorch_cpp_calls(cuda_src: str) -> int:
    return sum(1 for kw in PYTORCH_CPP_WRAPPER_KEYWORDS if kw in cuda_src)


def has_load_inline(py_code: str) -> bool:
    return "load_inline" in py_code


def forward_returns_torch_op(py_code: str) -> bool:
    """Detect: forward() body contains return torch.xxx(...) and nothing else."""
    m = re.search(r"def\s+forward\s*\([^)]*\)\s*:\s*\n(.*?)(?=\n\s*\S|\nclass|\Z)",
                  py_code, re.DOTALL)
    if not m:
        return False
    body = m.group(1)
    return_match = re.search(r"return\s+torch\.\w+\(", body)
    # body has only the return line (ignoring docstring) → pure torch fallback
    code_lines = [ln.strip() for ln in body.splitlines()
                  if ln.strip() and not ln.strip().startswith(("#", '"""', "'''"))]
    # Strip docstring
    in_doc = False
    real = []
    for ln in body.splitlines():
        st = ln.strip()
        if st.startswith(('"""', "'''")) and st.count('"""') + st.count("'''") % 2 == 1:
            in_doc = not in_doc
            continue
        if in_doc:
            continue
        if not st or st.startswith("#"):
            continue
        real.append(st)
    return (return_match is not None and
            len([l for l in real if not l.startswith(("\"", "'"))]) <= 2)


def classify(py_code: str) -> dict:
    has_inline = has_load_inline(py_code)
    cuda_src = extract_cuda_source(py_code)
    n_global = count_global_kernels(cuda_src)
    n_pyops = count_pytorch_cpp_calls(cuda_src)
    pure_torch_fwd = forward_returns_torch_op(py_code)

    if not has_inline and pure_torch_fwd:
        verdict = "PYTORCH_OP"
    elif has_inline and n_global == 0 and n_pyops > 0:
        verdict = "PYTORCH_WRAPPER"
    elif has_inline and n_global >= 1:
        verdict = "REAL_CUDA_FIX"
    elif has_inline and n_global == 0:
        verdict = "TRIVIAL_SHELL"
    else:
        verdict = "PYTORCH_OP" if pure_torch_fwd else "UNKNOWN"

    return {
        "has_load_inline": has_inline,
        "n_global_kernels": n_global,
        "n_pytorch_cpp_calls": n_pyops,
        "forward_pure_torch": pure_torch_fwd,
        "cuda_src_len": len(cuda_src),
        "verdict": verdict,
    }


# Process all kernel pairs
rows = []
for f in sorted(DET_DIR.glob("*.json")):
    d = json.load(open(f))
    name = d.get("kernel_name", f.stem)
    final_status = d.get("final_status", "")
    if not final_status.startswith("fixed"):
        continue
    final_round = d.get("final_round", 0)
    if final_round == 0 or name not in BEST:
        continue
    r0 = d.get("round0_stats", {})
    is_pseudo = (r0.get("n_confirmed_buggy", 0) == 0 and
                 r0.get("n_unexpected_pass", 0) > 0)
    is_partial_pseudo = (r0.get("n_unexpected_pass", 0) >= r0.get("n_total", 0) * 0.5
                         and r0.get("n_confirmed_buggy", 0) > 0)

    buggy_code = Path(BEST[name]["kernel_path"]).read_text(encoding="utf-8")
    fixed_p = KERN_DIR / f"{name}_round{final_round}.py"
    if not fixed_p.exists():
        continue
    fixed_code = fixed_p.read_text(encoding="utf-8")

    b = classify(buggy_code)
    f_sum = classify(fixed_code)
    rows.append({"name": name, "round": final_round,
                 "buggy": b, "fixed": f_sum, "pseudo": is_pseudo,
                 "partial_pseudo": is_partial_pseudo})


# Print
print(f"{'Kernel':<10} {'rd':>2} {'b_globl':>7} {'f_globl':>7} {'b_inl':>5} {'f_inl':>5} "
      f"{'f_pyops':>7} {'f_pureT':>7} {'verdict_buggy':<18} {'verdict_fixed':<18} {'flag':<15}")
print("-" * 140)
for r in rows:
    flag = ""
    if r["pseudo"]: flag = "PSEUDO_FIX"
    elif r["partial_pseudo"]: flag = "PARTIAL_PSEUDO"
    print(f"{r['name']:<10} {r['round']:>2} "
          f"{r['buggy']['n_global_kernels']:>7} {r['fixed']['n_global_kernels']:>7} "
          f"{int(r['buggy']['has_load_inline']):>5} {int(r['fixed']['has_load_inline']):>5} "
          f"{r['fixed']['n_pytorch_cpp_calls']:>7} {int(r['fixed']['forward_pure_torch']):>7} "
          f"{r['buggy']['verdict']:<18} {r['fixed']['verdict']:<18} {flag:<15}")

from collections import Counter
verdicts_fixed = Counter(r["fixed"]["verdict"] for r in rows
                          if not r["pseudo"] and not r["partial_pseudo"])
print()
print("=== Fixed-version verdict distribution (excluding pseudo/partial_pseudo) ===")
for k, v in verdicts_fixed.most_common():
    print(f"  {k:<25} {v:>3}")

print()
print("=== Strict counts by category ===")
print(f"  REAL_CUDA_FIX (genuinely fixed kernel preserving custom CUDA):")
for r in rows:
    if r["fixed"]["verdict"] == "REAL_CUDA_FIX" and not r["pseudo"] and not r["partial_pseudo"]:
        print(f"    {r['name']}")
print()
print(f"  PYTORCH_WRAPPER (kept load_inline but body only calls torch::mm/cublas):")
for r in rows:
    if r["fixed"]["verdict"] == "PYTORCH_WRAPPER":
        print(f"    {r['name']}")
print()
print(f"  PYTORCH_OP (forward() pure torch.xxx fallback, no load_inline):")
for r in rows:
    if r["fixed"]["verdict"] == "PYTORCH_OP":
        print(f"    {r['name']}")
print()
print(f"  PSEUDO_FIX / PARTIAL_PSEUDO (Phase II data non-reproducible):")
for r in rows:
    if r["pseudo"] or r["partial_pseudo"]:
        print(f"    {r['name']} ({'PSEUDO' if r['pseudo'] else 'PARTIAL'})")
