"""Audit the kernel source code itself for shortcut/cheating patterns.
Check for:
- "禁用 TF32" patterns (acceptable but should be reported correctly)
- Custom __global__ kernels (real CUDA implementations)
- load_inline use (custom CUDA via inline)
- PyTorch native fallback patterns (torch.matmul, torch.cumsum, etc. without any custom kernel)
- Wrapper around torch::mm / cublasSgemm
"""
import json
import re
from pathlib import Path
from collections import Counter

CKPT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json")
data = json.load(open(CKPT))

# Patterns
TF32 = re.compile(r"torch\.backends\.cuda\.matmul\.allow_tf32\s*=\s*False")
CUDNN_TF32 = re.compile(r"torch\.backends\.cudnn\.allow_tf32\s*=\s*False")
LOAD_INLINE = re.compile(r"load_inline\s*\(")
GLOBAL_KERNEL = re.compile(r"__global__\s+\w[\w\s]*\(")
EXTENSION = re.compile(r"torch\.utils\.cpp_extension|cpp_extension\.load")
TORCH_MM = re.compile(r"torch::mm\b|torch::matmul\b|at::mm\b|at::matmul\b")
CUBLAS = re.compile(r"cublas\w+\b")
FLOAT64_CAST = re.compile(r"\.to\(torch\.float64\)|\.double\(\)")
FLOAT32_CAST = re.compile(r"\.to\(torch\.float32\)|\.float\(\)")
CONTIGUOUS = re.compile(r"\.contiguous\(\)")

# also: high-level torch ops that may serve as "fallback"
TORCH_OPS = [
    "torch.matmul", "torch.bmm", "torch.einsum", "torch.cumsum", "torch.cumprod",
    "torch.sum", "torch.mean", "torch.nn.functional.conv2d",
    "torch.nn.functional.linear", "torch.nn.functional.softmax",
]

categories = {"TF32_ONLY": [], "FLOAT64": [], "FLOAT32": [],
              "CONTIG_ONLY": [], "REAL_CUDA": [], "OTHER": []}

stats = {"with_load_inline": 0, "with_global_kernel": 0, "with_tf32_disable": 0,
         "with_float64_cast": 0, "with_float32_cast": 0, "with_contig": 0,
         "with_torch_mm_cpp": 0}

for kid, entry in data.items():
    if entry.get("status") != "FIXED":
        continue
    fr = entry.get("fixed_round")
    rounds = entry.get("rounds", {})
    r = rounds.get(str(fr)) or rounds.get(fr) or {}
    src = r.get("kernel_source", "")
    
    has_tf32 = bool(TF32.search(src)) or bool(CUDNN_TF32.search(src))
    has_load_inline = bool(LOAD_INLINE.search(src))
    has_global = bool(GLOBAL_KERNEL.search(src))
    has_ext = bool(EXTENSION.search(src))
    has_torch_mm = bool(TORCH_MM.search(src))
    has_cublas = bool(CUBLAS.search(src))
    has_f64 = bool(FLOAT64_CAST.search(src))
    has_f32 = bool(FLOAT32_CAST.search(src))
    has_contig = bool(CONTIGUOUS.search(src))
    
    if has_tf32: stats["with_tf32_disable"] += 1
    if has_load_inline: stats["with_load_inline"] += 1
    if has_global: stats["with_global_kernel"] += 1
    if has_f64: stats["with_float64_cast"] += 1
    if has_f32: stats["with_float32_cast"] += 1
    if has_contig: stats["with_contig"] += 1
    if has_torch_mm: stats["with_torch_mm_cpp"] += 1
    
    if has_load_inline or has_global or has_ext:
        categories["REAL_CUDA"].append(kid)
    elif has_tf32 and not (has_f64 or has_f32 or has_contig):
        categories["TF32_ONLY"].append(kid)
    elif has_f64:
        categories["FLOAT64"].append(kid)
    elif has_f32:
        categories["FLOAT32"].append(kid)
    elif has_contig:
        categories["CONTIG_ONLY"].append(kid)
    else:
        categories["OTHER"].append(kid)

print("=" * 80)
print("LLM-fixed kernel content audit (90 FIXED kernels)")
print("=" * 80)
print(f"\nFeature counts (a kernel can have multiple):")
for k, v in stats.items():
    print(f"  {k}: {v}  ({v/90*100:.1f}%)")

print(f"\nExclusive categorization:")
for c, items in categories.items():
    print(f"  {c}: {len(items)}")

print(f"\nREAL_CUDA kernels (have __global__ or load_inline):")
for k in categories["REAL_CUDA"]:
    print(f"  {k}")

print(f"\nFLOAT64 (use .double()):")
for k in categories["FLOAT64"]:
    print(f"  {k}")

print(f"\nFLOAT32 only (.float() without TF32):")
for k in categories["FLOAT32"]:
    print(f"  {k}")
    
print(f"\nCONTIG only:")
for k in categories["CONTIG_ONLY"]:
    print(f"  {k}")

print(f"\nOTHER (no recognized fix pattern):")
for k in categories["OTHER"]:
    print(f"  {k}")
