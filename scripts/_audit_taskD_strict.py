"""Finer-grained classification of 90 FIXED kernels for cheating detection.

Categories (priority order):
1. PYTORCH_NN_FALLBACK: only uses torch.nn.Conv*, nn.LayerNorm, nn.Linear, nn.ConvTranspose*, etc.
   (entire forward = self.module(x))  — likely cheating: gave up CUDA optimization
2. TORCH_OPS_FALLBACK: only uses torch.matmul / torch.bmm / torch.functional.* — wrapper only
3. REAL_CUDA_FIX: contains __global__ or load_inline — kept custom CUDA, real fix
4. TF32_DISABLE: contains tf32 disable, no real CUDA — may still be PyTorch fallback
5. MIXED_TF32+CASTING: TF32 disable + .float()/.contiguous()/etc.

For each FIXED kernel, also compare with the ORIGINAL kernel (round0/baseline) if available
to see how much of the custom CUDA was REMOVED.
"""
import json
import re
from pathlib import Path

CKPT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json")
data = json.load(open(CKPT))

# Patterns - more comprehensive
NN_MODULE = re.compile(r"nn\.(?:Conv1d|Conv2d|Conv3d|ConvTranspose1d|ConvTranspose2d|ConvTranspose3d|"
                       r"Linear|LayerNorm|BatchNorm1d|BatchNorm2d|BatchNorm3d|GroupNorm|RNNCell|"
                       r"LSTMCell|GRUCell|Embedding|MaxPool[123]d|AvgPool[123]d|AdaptiveAvgPool[123]d|"
                       r"AdaptiveMaxPool[123]d|MultiheadAttention)")
TF32 = re.compile(r"allow_tf32\s*=\s*False")
LOAD_INLINE = re.compile(r"load_inline\s*\(")
GLOBAL_KERNEL = re.compile(r"__global__\s+\w[\w\s\*]*\(")
TORCH_F = re.compile(r"torch\.nn\.functional\.|F\.(?:conv[123]d|linear|softmax|layer_norm|"
                     r"batch_norm|group_norm|relu|gelu|silu|elu|sigmoid|tanh)")
TORCH_MATMUL = re.compile(r"torch\.(?:matmul|bmm|mm|einsum|cumsum|cumprod|softmax|"
                          r"layer_norm|batch_norm|topk|argmin|argmax|sort)")

def classify(src):
    has_global = bool(GLOBAL_KERNEL.search(src))
    has_inline = bool(LOAD_INLINE.search(src))
    has_nn_module = bool(NN_MODULE.search(src))
    has_tf32 = bool(TF32.search(src))
    has_f = bool(TORCH_F.search(src))
    has_matmul = bool(TORCH_MATMUL.search(src))
    return {
        "global": has_global, "inline": has_inline,
        "nn_module": has_nn_module, "tf32": has_tf32,
        "F_ops": has_f, "torch_ops": has_matmul,
    }

categories = {
    "REAL_CUDA_FIX": [],         # has __global__ or load_inline → real custom CUDA
    "PYTORCH_NN_FALLBACK": [],   # only nn.X — gave up CUDA optimization
    "TORCH_OPS_FALLBACK": [],    # only torch.x or F.x — wrapper-only
    "TF32_ONLY_FIX": [],         # tf32 disable, no nn modules, no custom CUDA
}

for kid, entry in data.items():
    if entry.get("status") != "FIXED":
        continue
    fr = entry.get("fixed_round")
    r = entry.get("rounds", {}).get(str(fr)) or entry.get("rounds", {}).get(fr) or {}
    src = r.get("kernel_source", "")
    flags = classify(src)
    
    if flags["global"] or flags["inline"]:
        categories["REAL_CUDA_FIX"].append(kid)
    elif flags["nn_module"]:
        categories["PYTORCH_NN_FALLBACK"].append(kid)
    elif flags["tf32"] and not (flags["F_ops"] or flags["torch_ops"]):
        # only tf32, no obvious op — unusual
        categories["TF32_ONLY_FIX"].append(kid)
    elif flags["tf32"]:
        # tf32 + torch ops/F ops — could be the canonical "TF32 fix on torch.matmul" pattern
        categories["TF32_ONLY_FIX"].append(kid)
    else:
        # no tf32, no nn module, no global — only torch ops or F ops
        categories["TORCH_OPS_FALLBACK"].append(kid)

print("=" * 80)
print(f"FIXED kernels strict classification (90 kernels)")
print("=" * 80)
for cat, items in categories.items():
    print(f"\n## {cat}: {len(items)}")
    for k in items:
        # Brief: line count + first non-blank line of body
        entry = data[k]
        fr = entry.get("fixed_round")
        r = entry.get("rounds", {}).get(str(fr)) or entry.get("rounds", {}).get(fr) or {}
        src = r.get("kernel_source", "")
        lc = len(src.split("\n"))
        print(f"   {k:<28} ({lc} lines)")

# Also: count by L1/L2/L3 grouping
import collections
print("\n" + "=" * 80)
print("By difficulty level (L1 / L2 / L3):")
print("=" * 80)
for cat, items in categories.items():
    by_lvl = collections.Counter()
    for k in items:
        m = re.search(r"__(L\d)_", k)
        if m: by_lvl[m.group(1)] += 1
    print(f"  {cat:<25} L1={by_lvl['L1']} L2={by_lvl['L2']} L3={by_lvl['L3']}")
