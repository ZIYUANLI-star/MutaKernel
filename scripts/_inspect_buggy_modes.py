import json
from collections import Counter
d = json.load(open("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_b_buggy_kernels_from_existing_data.json"))
modes = Counter()
per_kernel_train = {}
for k, evs in d["buggy_kernels"].items():
    for ev in evs:
        modes[ev.get("mode")] += 1
        if "train" in (ev.get("mode") or ""):
            per_kernel_train.setdefault(k, 0)
            per_kernel_train[k] += 1
print("mode counts:", modes.most_common())
print()
print("L1_P48 events:")
for ev in d["buggy_kernels"].get("L1_P48", []):
    print("  ", ev["policy"], ev["seed"], ev["mode"])
print()
print("L1_P97 first 5 events:")
for ev in d["buggy_kernels"].get("L1_P97", [])[:5]:
    print("  ", ev["policy"], ev["seed"], ev["mode"])
print()
print("train-mode kernels:", per_kernel_train)
