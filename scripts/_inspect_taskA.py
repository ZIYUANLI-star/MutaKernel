import json, os, glob
d_dir = "/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details"
files = sorted(glob.glob(os.path.join(d_dir, "*.json")))
print("file count:", len(files))

# Look at first 3 files
for f in files[:3]:
    print("=" * 60)
    print(os.path.basename(f))
    d = json.load(open(f))
    print("  top keys:", list(d.keys()))
    for k in d:
        v = d[k]
        if isinstance(v, list):
            print(f"  {k}: list len={len(v)}")
            if v and isinstance(v[0], dict):
                print(f"    [0].keys: {list(v[0].keys())}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())[:10]}")
        else:
            s = str(v)
            if len(s) > 100: s = s[:100] + "..."
            print(f"  {k}: {s}")

# Count fields
from collections import Counter
field_count = Counter()
killed_field_values = Counter()
killing_round_values = Counter()
final_status_values = Counter()
overall_killed_values = Counter()
for f in files:
    d = json.load(open(f))
    for k in d.keys():
        field_count[k] += 1
    for kfield in ("killed", "overall_killed", "any_killed", "final_killed", "is_killed"):
        if kfield in d:
            killed_field_values[f"{kfield}={d[kfield]}"] += 1
    if "killing_round" in d:
        killing_round_values[d["killing_round"]] += 1
    if "final_status" in d:
        final_status_values[d["final_status"]] += 1

print("\n=== field counts ===")
for k, v in field_count.most_common():
    print(f"  {k}: {v}")

print("\n=== killed-like field values ===")
for k, v in killed_field_values.most_common():
    print(f"  {k}: {v}")

print("\n=== killing_round values ===")
for k, v in killing_round_values.most_common():
    print(f"  {k}: {v}")

print("\n=== final_status values ===")
for k, v in final_status_values.most_common():
    print(f"  {k}: {v}")
