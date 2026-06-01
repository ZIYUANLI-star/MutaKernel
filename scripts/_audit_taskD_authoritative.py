"""Final authoritative classification for the 90 FIXED kernels.
Manually merge UNKNOWN into TORCH_OPS_FALLBACK based on per-kernel checks.
"""
import json
import re
from pathlib import Path

CKPT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json")
data = json.load(open(CKPT))

# Manually verified UNKNOWN→correct class based on reading:
# L1_T59: torch.nn.functional.scaled_dot_product_attention → TORCH_OPS_FALLBACK
# L1_T81: torch.sum → TORCH_OPS_FALLBACK
# L1_T88: torch.cumprod → TORCH_OPS_FALLBACK
# L1_T89: torch.cumsum → TORCH_OPS_FALLBACK
# L3_T41: torch.norm + matmul → TORCH_OPS_FALLBACK
unknowns_resolution = {
    "cuda_agent__L1_T59": "TORCH_OPS_FALLBACK",
    "cuda_agent__L1_T81": "TORCH_OPS_FALLBACK",
    "cuda_agent__L1_T88": "TORCH_OPS_FALLBACK",
    "cuda_agent__L1_T89": "TORCH_OPS_FALLBACK",
    "cuda_agent__L3_T41": "TORCH_OPS_FALLBACK",
}

# Re-run classifier and apply manual resolutions
LOAD_INLINE = re.compile(r"load_inline\s*\(", re.DOTALL)
LOAD_INLINE_FUNCS = re.compile(r"load_inline\s*\([^)]*functions\s*=\s*\[([^\]]+)\]", re.DOTALL)
GLOBAL_KERNEL = re.compile(r"__global__\s+\w[\w\s\*<>]*\(")
NN_MODULE = re.compile(r"\bnn\.(?:Conv1d|Conv2d|Conv3d|ConvTranspose1d|ConvTranspose2d|ConvTranspose3d|"
                       r"Linear|LayerNorm|BatchNorm1d|BatchNorm2d|BatchNorm3d|GroupNorm|"
                       r"MaxPool[123]d|AvgPool[123]d|AdaptiveAvgPool[123]d|MultiheadAttention)\b")
F_OPS = re.compile(r"\b(?:F|torch\.nn\.functional)\.(?:conv[123]d|conv_transpose[123]d|linear|softmax|"
                   r"layer_norm|batch_norm|group_norm|relu|gelu|silu|elu|sigmoid|tanh|max_pool[123]d|"
                   r"avg_pool[123]d|adaptive_avg_pool[123]d|scaled_dot_product_attention)\b")
TORCH_OPS = re.compile(r"\btorch\.(?:matmul|bmm|mm|einsum|cumsum|cumprod|softmax|"
                       r"layer_norm|batch_norm|topk|argmin|argmax|sort|conv_transpose[123]d|"
                       r"logsumexp|clamp|sigmoid|sum|mean|max|min|norm|relu)\b")
TF32 = re.compile(r"allow_tf32\s*=\s*False")

def get_forward(src):
    class_matches = list(re.finditer(r"^class ModelNew\(", src, re.MULTILINE))
    if not class_matches: return ""
    last = class_matches[-1].start()
    m = re.search(r"def forward\(self[^)]*\)[^:]*:(.*?)(?=\n    def |\nclass |\Z)", src[last:], re.DOTALL)
    return m.group(1) if m else ""

results = {}
for kid, entry in data.items():
    if entry.get("status") != "FIXED":
        continue
    fr = entry.get("fixed_round")
    r = entry.get("rounds", {}).get(str(fr)) or entry.get("rounds", {}).get(fr) or {}
    src = r.get("kernel_source", "")
    fwd = get_forward(src)
    
    has_global = bool(GLOBAL_KERNEL.search(src))
    has_inline = bool(LOAD_INLINE.search(src))
    has_tf32 = bool(TF32.search(src))
    
    funcs = []
    for m in LOAD_INLINE_FUNCS.finditer(src):
        for p in m.group(1).split(","):
            p = p.strip().strip("'\"")
            if p: funcs.append(p)
    
    fwd_uses_custom = any(re.search(rf"\b{f}\s*\(", fwd) for f in funcs)
    fwd_uses_nn = bool(re.search(r"self\.\w+\s*\(", fwd)) and bool(NN_MODULE.search(src))
    fwd_uses_torch_ops = bool(F_OPS.search(fwd)) or bool(TORCH_OPS.search(fwd))
    
    if fwd_uses_custom:
        cls = "REAL_CUDA_FIX"
    elif fwd_uses_nn:
        cls = "PYTORCH_NN_FALLBACK"
    elif fwd_uses_torch_ops:
        cls = "TORCH_OPS_FALLBACK"
    elif has_tf32:
        cls = "TF32_ONLY"
    else:
        cls = "UNKNOWN_NEEDS_REVIEW"
    
    # Override
    if kid in unknowns_resolution:
        cls = unknowns_resolution[kid]
    
    # cheat flag
    dead_cuda = (has_global or has_inline) and not fwd_uses_custom
    results[kid] = {"fr": fr, "cls": cls, "tf32": has_tf32, "global": has_global,
                    "inline": has_inline, "dead_cuda": dead_cuda,
                    "level": kid.split("__")[1].split("_T")[0]}

# Stats
from collections import Counter, defaultdict
cls_counter = Counter(v["cls"] for v in results.values())
print("=" * 80)
print("FINAL Classification (90 FIXED kernels)")
print("=" * 80)
for c, n in cls_counter.most_common():
    print(f"  {c}: {n}  ({n/90*100:.1f}%)")

# Per-class by level
print("\nBy level breakdown:")
lvl_break = defaultdict(lambda: Counter())
for k, v in results.items():
    lvl_break[v["cls"]][v["level"]] += 1
for cls, lc in lvl_break.items():
    print(f"  {cls:<24} L1={lc['L1']} L2={lc['L2']} L3={lc['L3']}")

# Per-class by round
print("\nBy fixed_round:")
rnd_break = defaultdict(lambda: Counter())
for k, v in results.items():
    rnd_break[v["cls"]][v["fr"]] += 1
for cls, rb in rnd_break.items():
    print(f"  {cls:<24} R1={rb[1]} R2={rb[2]} R3={rb[3]}")

# Dead CUDA list
print("\nDead CUDA kernels (kept __global__ but forward uses fallback):")
for k, v in sorted(results.items()):
    if v["dead_cuda"]:
        print(f"  {k:<28} class={v['cls']}  fr={v['fr']}  tf32={v['tf32']}")

# Print complete table
print("\n" + "=" * 80)
print("PER-KERNEL CLASSIFICATION TABLE")
print("=" * 80)
for cls in ("REAL_CUDA_FIX", "PYTORCH_NN_FALLBACK", "TORCH_OPS_FALLBACK", "TF32_ONLY", "UNKNOWN_NEEDS_REVIEW"):
    items = sorted([k for k, v in results.items() if v["cls"] == cls])
    print(f"\n## {cls} ({len(items)}):")
    for k in items:
        v = results[k]
        dc = " [DEAD_CUDA]" if v["dead_cuda"] else ""
        tf = " [TF32]" if v["tf32"] else ""
        print(f"   {k:<28} fr={v['fr']}{tf}{dc}")
