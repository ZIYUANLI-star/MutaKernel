"""Final strict classification: examine the ACTUAL forward body for each FIXED kernel.
- If forward calls into a custom CUDA function (load_inline name OR custom op symbol) → REAL_CUDA_FIX
- Else if forward uses nn.Conv*, nn.LayerNorm, etc. → PYTORCH_NN_FALLBACK
- Else if forward uses only torch.x / F.x → TORCH_OPS_FALLBACK
- TF32 disable status reported as a tag, not a class.
"""
import json
import re
from pathlib import Path

CKPT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json")
data = json.load(open(CKPT))

LOAD_INLINE = re.compile(r"load_inline\s*\(", re.DOTALL)
LOAD_INLINE_FUNCS = re.compile(r"load_inline\s*\([^)]*functions\s*=\s*\[([^\]]+)\]", re.DOTALL)
GLOBAL_KERNEL = re.compile(r"__global__\s+\w[\w\s\*<>]*\(")

NN_MODULE = re.compile(r"\bnn\.(?:Conv1d|Conv2d|Conv3d|ConvTranspose1d|ConvTranspose2d|ConvTranspose3d|"
                       r"Linear|LayerNorm|BatchNorm1d|BatchNorm2d|BatchNorm3d|GroupNorm|"
                       r"MaxPool[123]d|AvgPool[123]d|AdaptiveAvgPool[123]d|MultiheadAttention)\b")
F_OPS = re.compile(r"\b(?:F|torch\.nn\.functional)\.(?:conv[123]d|conv_transpose[123]d|linear|softmax|"
                   r"layer_norm|batch_norm|group_norm|relu|gelu|silu|elu|sigmoid|tanh|max_pool[123]d|"
                   r"avg_pool[123]d|adaptive_avg_pool[123]d)\b")
TORCH_OPS = re.compile(r"\btorch\.(?:matmul|bmm|mm|einsum|cumsum|cumprod|softmax|"
                       r"layer_norm|batch_norm|topk|argmin|argmax|sort|conv_transpose[123]d|"
                       r"logsumexp|clamp|sigmoid)\b")
TF32 = re.compile(r"allow_tf32\s*=\s*False")

def get_forward_body(src):
    """Extract the forward method body from class ModelNew."""
    # Find the LAST class ModelNew definition (later defs override earlier in Python)
    # Then within it, find def forward
    class_matches = list(re.finditer(r"^class ModelNew\(", src, re.MULTILINE))
    if not class_matches:
        return ""
    last_class_start = class_matches[-1].start()
    # forward inside the last class
    m = re.search(r"def forward\(self[^)]*\)[^:]*:(.*?)(?=\n    def |\nclass |\Z)",
                  src[last_class_start:], re.DOTALL)
    return m.group(1) if m else ""

results = []
for kid, entry in data.items():
    if entry.get("status") != "FIXED":
        continue
    fr = entry.get("fixed_round")
    r = entry.get("rounds", {}).get(str(fr)) or entry.get("rounds", {}).get(fr) or {}
    src = r.get("kernel_source", "")
    fwd = get_forward_body(src)
    
    has_global = bool(GLOBAL_KERNEL.search(src))
    has_inline = bool(LOAD_INLINE.search(src))
    has_tf32 = bool(TF32.search(src))
    
    # extract inline function names
    funcs = []
    for m in LOAD_INLINE_FUNCS.finditer(src):
        for part in m.group(1).split(","):
            f = part.strip().strip("'\"")
            if f: funcs.append(f)
    
    fwd_uses_custom = any(re.search(rf"\b{f}\s*\(", fwd) for f in funcs)
    fwd_uses_nn = bool(re.search(r"self\.(?:conv|linear|ln|bn|gn|fc|pool|attn|conv_transpose|conv1d|conv2d|conv3d|conv_t)\w*\s*\(", fwd)) or bool(NN_MODULE.search(fwd))
    fwd_uses_torch_ops = bool(F_OPS.search(fwd)) or bool(TORCH_OPS.search(fwd))
    
    # Whole-source check (catch cases where forward calls self.<somename>(x) using a different attr name)
    src_uses_nn_module = bool(NN_MODULE.search(src))
    # forward calls some self.<x>(...) attribute (could be nn module bound in __init__)
    fwd_calls_self_attr = bool(re.search(r"self\.\w+\s*\(", fwd))
    
    if fwd_uses_custom:
        cls = "REAL_CUDA_FIX"
    elif fwd_uses_nn:
        cls = "PYTORCH_NN_FALLBACK"
    elif src_uses_nn_module and fwd_calls_self_attr:
        cls = "PYTORCH_NN_FALLBACK"
    elif fwd_uses_torch_ops:
        cls = "TORCH_OPS_FALLBACK"
    elif has_tf32:
        cls = "TF32_ONLY"
    else:
        cls = "UNKNOWN_NEEDS_REVIEW"
    
    # If has __global__ but forward doesn't use it → mark as PSEUDO (dead kernel)
    cheat_flags = []
    if (has_global or has_inline) and not fwd_uses_custom:
        cheat_flags.append("DEAD_CUDA_KERNEL")
    if fwd_uses_nn and (has_global or has_inline):
        cheat_flags.append("CUDA_DEFINED_BUT_NN_USED")
    
    results.append((kid, fr, cls, has_global, has_inline, has_tf32, fwd_uses_custom,
                    fwd_uses_nn, fwd_uses_torch_ops, cheat_flags))

# Counts per class
from collections import Counter
cls_counter = Counter(r[2] for r in results)
print("=" * 90)
print("Final strict classification of 90 FIXED kernels")
print("=" * 90)
for c, n in cls_counter.most_common():
    print(f"  {c}: {n}")

# Dead CUDA kernels (cheating with stub CUDA + actual PyTorch fallback)
dead_cuda = [r for r in results if "DEAD_CUDA_KERNEL" in r[9]]
print(f"\nCheating mode 'DEAD CUDA KERNEL' (kept __global__ but forward uses PyTorch): {len(dead_cuda)}")
for r in dead_cuda:
    print(f"  {r[0]:<28} class={r[2]} tf32={r[5]} flags={r[9]}")

print("\nFull classification per kernel:")
print(f"{'kernel_id':<28}{'fr':>3} {'class':<25} {'tf32':>5}{'global':>7}{'inline':>7}  flags")
print("-" * 100)
for r in sorted(results, key=lambda x: (x[2], x[0])):
    flags_str = ",".join(r[9]) if r[9] else ""
    print(f"{r[0]:<28}{r[1]:>3} {r[2]:<25} {str(r[5]):<5}{str(r[3]):<7}{str(r[4]):<7}  {flags_str}")
