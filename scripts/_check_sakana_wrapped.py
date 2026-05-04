import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "external_benchmarks/ai_cuda_engineer/registry.json") as f:
    reg = json.load(f)
e = reg[0]
ks = e["kernel_source"]
print(f"Total: {len(reg)}")
print(f"ID: {e['id']}")
print(f"Has ModelNew: {'class ModelNew' in ks}")
print(f"Has load_inline: {'load_inline' in ks}")
print(f"\nFirst 800 chars:\n{ks[:800]}")
