"""快速检查 L1 数据: 正确 kernel 数量、语言分布、C 类算子可命中情况。"""
import json, re, sys
from pathlib import Path

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
RUN_DIR = KB_ROOT / "runs" / "iter_full_l1_caesar_paper_v2"

# 1. Load eval results
with open(RUN_DIR / "eval_results.json") as f:
    results = json.load(f)

correct = {k: v for k, v in results.items() if v.get("correctness", False)}
print(f"Total problems: {len(results)}")
print(f"Correct kernels: {len(correct)}")
print(f"Correct IDs: {sorted(int(x) for x in correct.keys())}")

# 2. Sample generated kernels - check language and C-category patterns
C_PATTERNS = {
    "tl.max(":       "StabRemove (tl.max)",
    "torch.max(":    "StabRemove (torch.max)",
    ".max(":         "StabRemove (.max)",
    ".to(torch.float32)": "AccDowngrade/CastRemove",
    ".to(tl.float32)":    "AccDowngrade/CastRemove",
    ".float()":      "AccDowngrade/CastRemove (.float)",
    "tl.sum(":       "ReductionReorder (tl.sum)",
    "torch.sum(":    "ReductionReorder (torch.sum)",
    "float('-inf')": "InitModify",
    "float('inf')":  "InitModify",
    "rsqrt":         "ScaleModify",
    "1/math.sqrt":   "ScaleModify",
    "1/torch.sqrt":  "ScaleModify",
}

cuda_count = 0
triton_count = 0
pattern_hits = {}

for pid in sorted(int(x) for x in correct.keys()):
    kernel_file = RUN_DIR / f"level_1_problem_{pid}_sample_0_kernel.py"
    if not kernel_file.exists():
        continue
    code = kernel_file.read_text()
    
    is_cuda = "__global__" in code or "load_inline" in code
    is_triton = "import triton" in code or "@triton.jit" in code
    
    if is_cuda:
        cuda_count += 1
    elif is_triton:
        triton_count += 1
    else:
        cuda_count += 1  # assume cuda if unclear
    
    for pat, name in C_PATTERNS.items():
        if pat in code:
            if name not in pattern_hits:
                pattern_hits[name] = []
            pattern_hits[name].append(pid)

print(f"\n--- Language distribution ---")
print(f"CUDA kernels: {cuda_count}")
print(f"Triton kernels: {triton_count}")

print(f"\n--- C-category pattern occurrences in correct kernels ---")
if not pattern_hits:
    print("  ** NONE ** No C-category patterns found in any kernel!")
else:
    for name, pids in sorted(pattern_hits.items()):
        print(f"  {name}: {len(pids)} kernels -> PIDs {pids[:10]}{'...' if len(pids)>10 else ''}")

# 3. Check a few specific kernels that SHOULD have numerical patterns
interesting = ["23", "24", "33", "34", "35", "36", "37", "48", "49"]
print(f"\n--- Interesting kernels (softmax, norms, reductions) ---")
for pid in interesting:
    is_correct = pid in correct
    kernel_file = RUN_DIR / f"level_1_problem_{pid}_sample_0_kernel.py"
    exists = kernel_file.exists()
    print(f"  P{pid}: correct={is_correct}, kernel_exists={exists}")
    if exists and is_correct:
        code = kernel_file.read_text()
        has_cuda = "__global__" in code
        has_triton = "@triton.jit" in code
        lang = "CUDA" if has_cuda else ("Triton" if has_triton else "PyTorch")
        print(f"         language={lang}, lines={len(code.splitlines())}")
        # Show any C-pattern matches
        for pat, name in C_PATTERNS.items():
            if pat in code:
                print(f"         HIT: {name}")
