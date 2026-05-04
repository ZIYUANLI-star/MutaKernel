import json

cp = "第三次实验汇总/results/cuda_l1/checkpoint.json"
with open(cp, encoding="utf-8") as f:
    d = json.load(f)

times = []
for kid, data in sorted(d.items()):
    elapsed = data.get("elapsed_s", 0)
    status = data.get("status", "?")
    disc = data.get("total_discrepancies", 0)
    times.append((elapsed, kid, status, disc))

times.sort(reverse=True)

print(f"=== Completed kernel timings (top 20 slowest) ===")
print(f"{'Kernel':<25} {'Time(s)':>8} {'Status':<12} {'Disc':>4}")
print("-" * 55)
for elapsed, kid, status, disc in times[:20]:
    print(f"{kid:<25} {elapsed:>8.1f} {status:<12} {disc:>4}")

print(f"\n=== Summary ===")
all_t = [t[0] for t in times if t[0] > 0]
if all_t:
    print(f"Total kernels: {len(times)}")
    print(f"Min time:  {min(all_t):.1f}s")
    print(f"Max time:  {max(all_t):.1f}s")
    print(f"Mean time: {sum(all_t)/len(all_t):.1f}s")
    print(f"Median:    {sorted(all_t)[len(all_t)//2]:.1f}s")
    over_60 = sum(1 for t in all_t if t > 60)
    over_120 = sum(1 for t in all_t if t > 120)
    over_180 = sum(1 for t in all_t if t > 180)
    print(f"Over  60s: {over_60}")
    print(f"Over 120s: {over_120}")
    print(f"Over 180s: {over_180}")

    # Check individual dimension timing from detail files
    import os
    detail_dir = "第三次实验汇总/results/cuda_l1/details"
    if os.path.isdir(detail_dir):
        dim_names = ["value_stress", "dtype_stress", "training_stress", "repeated_run", "config_stress"]
        print(f"\n=== Per-dimension: checking test_cases for time_ms ===")
        max_per_dim = {d: 0 for d in dim_names}
        for fname in os.listdir(detail_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(detail_dir, fname), encoding="utf-8") as f2:
                detail = json.load(f2)
            for dim in dim_names:
                dim_data = detail.get(dim, {})
                for tc in dim_data.get("test_cases", []):
                    t_ms = tc.get("time_ms", 0)
                    if t_ms and t_ms > max_per_dim[dim]:
                        max_per_dim[dim] = t_ms
        for dim, max_t in max_per_dim.items():
            print(f"  {dim:<20}: max single test = {max_t:.0f}ms ({max_t/1000:.1f}s)")
