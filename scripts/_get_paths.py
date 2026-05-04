import json
with open("best_kernels.json") as f:
    d = json.load(f)
for k in ["L1_P1", "L1_P100", "L1_P23"]:
    print(f"{k}: {d.get(k, {}).get('kernel_path', 'NOT FOUND')}")
