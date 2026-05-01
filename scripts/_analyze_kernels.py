import json
import re
from collections import Counter

reg_path = "external_benchmarks/cuda_l1/registry.json"
with open(reg_path, encoding="utf-8") as f:
    registry = json.load(f)

print(f"Total kernels in CUDA-L1: {len(registry)}\n")

heavy_keywords = {
    "Conv3d": "3D卷积 - 极高显存",
    "Conv2d": "2D卷积 - 高显存",
    "ConvTranspose": "转置卷积 - 高显存",
    "BatchNorm": "BatchNorm",
    "Linear": "全连接层",
    "MatMul": "矩阵乘法",
    "GEMM": "矩阵乘法(GEMM)",
    "Softmax": "Softmax",
    "LayerNorm": "LayerNorm",
    "ReLU": "ReLU",
    "GELU": "GELU",
    "Sigmoid": "Sigmoid",
    "MaxPool": "最大池化",
    "AvgPool": "平均池化",
    "Dropout": "Dropout",
    "Embedding": "Embedding",
    "LSTM": "LSTM - 高显存",
    "GRU": "GRU",
    "Attention": "Attention - 高显存",
    "Transformer": "Transformer - 极高显存",
}

# Analyze each kernel
risky_kernels = []
kernel_types = Counter()

for entry in registry:
    kid = entry["id"]
    kname = entry.get("kernel_name", "")
    ref_file = entry.get("reference_file", "")
    ksrc = entry.get("kernel_source", "")

    # Try to detect kernel type from name and reference file
    detected = []
    full_text = f"{kname} {ref_file} {ksrc[:500]}"
    for kw, label in heavy_keywords.items():
        if kw.lower() in full_text.lower():
            detected.append(label)
    
    # Check for large tensor dimensions in source
    large_dims = False
    # Look for large numbers in shape definitions
    big_numbers = re.findall(r'\b(\d{3,})\b', ksrc[:1000])
    big_numbers = [int(n) for n in big_numbers if int(n) >= 256]
    
    # Check for 3D/4D/5D operations
    has_3d = "3d" in full_text.lower() or "conv3d" in full_text.lower()
    has_large_batch = any(n >= 1024 for n in big_numbers)
    
    risk = "LOW"
    if has_3d:
        risk = "EXTREME"
    elif "转置卷积" in str(detected) or "Transformer" in str(detected) or "LSTM" in str(detected) or "Attention" in str(detected):
        risk = "HIGH"
    elif "2D卷积" in str(detected) or "矩阵乘法" in str(detected):
        risk = "MEDIUM"
    
    if detected:
        kernel_types.update(detected)
    
    if risk in ("EXTREME", "HIGH"):
        risky_kernels.append((kid, risk, ", ".join(detected) if detected else kname, big_numbers[:5]))

# Print risk analysis
print("=" * 70)
print("HIGH RISK KERNELS (可能吃爆显存)")
print("=" * 70)
print(f"{'Kernel':<25} {'Risk':<10} {'Type':<35} {'Large dims'}")
print("-" * 85)
for kid, risk, types, dims in sorted(risky_kernels, key=lambda x: (0 if x[1]=="EXTREME" else 1, x[0])):
    dim_str = str(dims[:3]) if dims else ""
    print(f"{kid:<25} {risk:<10} {types:<35} {dim_str}")

print(f"\n总计高风险 kernel: {len(risky_kernels)} / {len(registry)}")

print(f"\n{'=' * 50}")
print("Kernel 类型分布:")
print("=" * 50)
for ktype, count in kernel_types.most_common(20):
    print(f"  {ktype:<30}: {count}")

# Also check what's already completed vs what's remaining
cp_path = "第三次实验汇总/results/cuda_l1/checkpoint.json"
try:
    with open(cp_path, encoding="utf-8") as f:
        completed = set(json.load(f).keys())
except Exception:
    completed = set()

print(f"\n{'=' * 50}")
print(f"剩余未测的高风险 kernel:")
print("=" * 50)
for kid, risk, types, dims in sorted(risky_kernels, key=lambda x: x[0]):
    full_kid = f"cuda_l1__{kid}" if not kid.startswith("cuda_l1__") else kid
    status = "DONE" if full_kid in completed else "PENDING"
    if status == "PENDING":
        print(f"  {kid:<25} {risk:<10} {types}")
