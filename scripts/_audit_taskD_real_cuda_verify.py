"""Deep verification of REAL_CUDA_FIX (19): ensure they kept genuine CUDA logic.
Check:
- __global__ count
- load_inline used
- Forward method actually invokes the custom kernel (not PyTorch fallback)
- Compute meaningful logic (not just stub)
"""
import json
import re
from pathlib import Path

CKPT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json")
data = json.load(open(CKPT))

REAL_CUDA = [
    "cuda_agent__L1_T45", "cuda_agent__L2_T2", "cuda_agent__L2_T4", "cuda_agent__L2_T3",
    "cuda_agent__L2_T1", "cuda_agent__L2_T14", "cuda_agent__L2_T17", "cuda_agent__L2_T32",
    "cuda_agent__L2_T35", "cuda_agent__L2_T50", "cuda_agent__L2_T53", "cuda_agent__L2_T59",
    "cuda_agent__L2_T49", "cuda_agent__L2_T76", "cuda_agent__L2_T89", "cuda_agent__L2_T96",
    "cuda_agent__L2_T97", "cuda_agent__L3_T13", "cuda_agent__L1_T84",
]

GLOBAL = re.compile(r"__global__\s+\w[\w\s\*<>]*\(")
KERNEL_CALL = re.compile(r"<<<")  # CUDA kernel call syntax
LOAD_INLINE_FUNC = re.compile(r"functions\s*=\s*\[\s*['\"](\w+)['\"]")
FORWARD_USE_CUSTOM = re.compile(r"my_module\.|self\.\w+_op\.|cuda_module\.|custom_module\.")

print(f"{'kernel_id':<20}{'lines':>6}{'global':>8}{'<<<':>5} {'inline_funcs':<30} forward_calls_custom")
print("-" * 100)
results = []
for k in REAL_CUDA:
    entry = data[k]
    fr = entry.get("fixed_round")
    r = entry.get("rounds", {}).get(str(fr)) or entry.get("rounds", {}).get(fr) or {}
    src = r.get("kernel_source", "")
    lc = len(src.split("\n"))
    n_global = len(GLOBAL.findall(src))
    n_kcall = len(KERNEL_CALL.findall(src))
    inline_funcs = LOAD_INLINE_FUNC.findall(src)
    
    # Did the forward method actually call into the custom kernel?
    m = re.search(r"def forward\(self[^)]*\).*?(?=\n    def |\nclass |\Z)", src, re.DOTALL)
    forward = m.group() if m else ""
    # Look for any function name from inline_funcs ANYWHERE in the file, used in forward
    custom_calls = []
    for fn in inline_funcs:
        # match: .fn(  or fn(
        if re.search(rf"\b{fn}\s*\(", forward):
            custom_calls.append(fn)
    forward_uses_custom = bool(custom_calls)
    
    print(f"{k:<20}{lc:>6}{n_global:>8}{n_kcall:>5} {str(inline_funcs)[:28]:<30} {forward_uses_custom} {custom_calls}")
    results.append((k, lc, n_global, n_kcall, inline_funcs, forward_uses_custom, custom_calls))

print(f"\n=== Summary ===")
print(f"Total REAL_CUDA candidates: {len(results)}")
truly_real = [r for r in results if r[2] >= 1 and r[3] >= 1 and r[5]]
print(f"Truly real (has __global__ AND <<< call AND forward uses it): {len(truly_real)}")
no_kernel_call = [r for r in results if r[2] >= 1 and r[3] == 0]
print(f"Has __global__ but no kernel <<< call (possible dead kernel): {len(no_kernel_call)}")
no_forward_use = [r for r in results if r[5] is False and r[2] >= 1]
print(f"Has CUDA but forward doesn't use it: {len(no_forward_use)}")
