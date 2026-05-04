import json
cp = "/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第三次实验汇总/results/cuda_l1/checkpoint.json"
with open(cp, encoding="utf-8") as f:
    d = json.load(f)
print(f"Completed kernels: {len(d)}")
for k in sorted(d.keys()):
    s = d[k].get("status", "?")
    print(f"  {k}: {s}")
