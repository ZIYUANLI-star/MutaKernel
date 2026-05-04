import json
d = json.load(open("best_kernels.json"))
l1 = sum(1 for k in d if k.startswith("L1"))
l2 = sum(1 for k in d if k.startswith("L2"))
print(f"Total: {len(d)}, L1: {l1}, L2: {l2}")
for k in sorted(d.keys())[:5]:
    print(f"  {k}: turn={d[k]['turn']}, speedup={d[k]['speedup']:.3f}")
print("  ...")
for k in sorted(d.keys())[-3:]:
    print(f"  {k}: turn={d[k]['turn']}, speedup={d[k]['speedup']:.3f}")
