"""
Analyze the 56 anomalous kernels: baseline failed but stress detected no discrepancy.
Investigate root causes by examining detail JSON files.
"""
import json
import os
from collections import defaultdict, Counter

base_dir = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel\第三次实验汇总\results"
datasets = ['cuda_l1', 'ai_cuda_engineer', 'tritonbench_g']

print("=" * 80)
print("ANOMALY: Baseline failed but Stress detected no discrepancy")
print("=" * 80)

anomaly_kernels = []

for ds in datasets:
    cp_path = os.path.join(base_dir, ds, 'checkpoint.json')
    with open(cp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for kid, kdata in data.items():
        if kdata.get('status') != 'COMPLETED':
            continue
        bl = kdata.get('baseline', {})
        bf = bl.get('failed', 0)
        td = kdata.get('total_discrepancies', 0)
        if bf > 0 and td == 0:
            anomaly_kernels.append((ds, kid, kdata))

print(f"\nTotal anomaly kernels: {len(anomaly_kernels)}")
print(f"  CUDA-L1: {sum(1 for ds, _, _ in anomaly_kernels if ds == 'cuda_l1')}")
print(f"  AI-CUDA-Engineer: {sum(1 for ds, _, _ in anomaly_kernels if ds == 'ai_cuda_engineer')}")
print(f"  TritonBench-G: {sum(1 for ds, _, _ in anomaly_kernels if ds == 'tritonbench_g')}")

# Analyze pattern
print("\n" + "=" * 80)
print("PATTERN 1: Per-dimension test result breakdown for anomaly kernels")
print("=" * 80)

# Each anomaly kernel: how many tests in each dim were pass, ref_fail, etc.
all_status_counts = defaultdict(int)
ref_fail_dominant = 0  # kernels where most/all stress tests had ref_fail
mostly_pass = 0  # kernels where stress tests show genuine passes

for ds, kid, kdata in anomaly_kernels:
    detail_path = os.path.join(base_dir, ds, 'details', f'{kid}.json')
    if not os.path.exists(detail_path):
        continue
    with open(detail_path, 'r', encoding='utf-8') as f:
        detail = json.load(f)
    
    # Count statuses across all test_cases in all dimensions
    status_count = defaultdict(int)
    for dim in ['value_stress', 'dtype_stress', 'training_stress', 'repeated_run']:
        dim_data = detail.get(dim, {})
        for tc in dim_data.get('test_cases', []):
            status_count[tc.get('status', 'unknown')] += 1
    
    # config_stress has different structure
    cs_data = detail.get('config_stress', {})
    rpb = cs_data.get('raw_results', {})
    for bs, bsd in rpb.items():
        if isinstance(bsd, dict):
            for sr in bsd.get('seeds_tested', []):
                if isinstance(sr, dict):
                    s = sr.get('status', 'unknown')
                    status_count[s] += 1
    
    for s, c in status_count.items():
        all_status_counts[s] += c
    
    # Classify the kernel
    total = sum(status_count.values())
    if total == 0:
        continue
    pass_count = status_count.get('pass', 0)
    ref_fail_count = status_count.get('ref_fail', 0)
    timeout_count = status_count.get('timeout', 0)
    error_count = status_count.get('error', 0)
    orig_diverges_count = status_count.get('orig_diverges_from_ref', 0)
    
    if (ref_fail_count + timeout_count + error_count) >= total * 0.5:
        ref_fail_dominant += 1
    elif pass_count >= total * 0.5:
        mostly_pass += 1

print(f"\n  Aggregate test case status across all anomaly kernels:")
for s, c in sorted(all_status_counts.items(), key=lambda x: -x[1]):
    print(f"    {s}: {c}")

print(f"\n  Per-kernel classification:")
print(f"    Mostly genuine pass (>=50% pass): {mostly_pass}")
print(f"    Mostly ref_fail/timeout/error (>=50%): {ref_fail_dominant}")
print(f"    Other: {len(anomaly_kernels) - mostly_pass - ref_fail_dominant}")

# Look at specific examples - their baseline diff summaries
print("\n" + "=" * 80)
print("PATTERN 2: Baseline diff summaries for sample anomaly kernels")
print("=" * 80)

# We need to look at the baseline diff details. The baseline is called by run_quick_baseline
# but its individual seed results aren't stored in checkpoint. We need to look at each detail
# more carefully - but actually baseline is just stored as passed/failed/errors counts.

# Let's instead look at: of these anomaly kernels, what does each dimension's data look like?
# Print 5 examples per dataset with their dimension stats

for ds in datasets:
    samples = [(kid, kdata) for d, kid, kdata in anomaly_kernels if d == ds][:5]
    if not samples:
        continue
    print(f"\n--- {ds} samples ---")
    
    for kid, kdata in samples:
        detail_path = os.path.join(base_dir, ds, 'details', f'{kid}.json')
        if not os.path.exists(detail_path):
            continue
        with open(detail_path, 'r', encoding='utf-8') as f:
            detail = json.load(f)
        
        bl = kdata.get('baseline', {})
        print(f"\n  {kid}: baseline={bl}")
        
        # For each dimension, print test case status counts
        for dim in ['value_stress', 'dtype_stress', 'training_stress', 'repeated_run']:
            dim_data = detail.get(dim, {})
            tcs = dim_data.get('test_cases', [])
            sc = Counter(tc.get('status', '?') for tc in tcs)
            print(f"    {dim}: {dim_data.get('discrepancies', 0)} disc, {dim_data.get('passes', 0)} pass | test cases: {dict(sc)}")
        
        # For value_stress, show first 3 ref_fail cases
        vs = detail.get('value_stress', {})
        ref_fails = [tc for tc in vs.get('test_cases', []) if tc.get('status') == 'ref_fail']
        if ref_fails:
            print(f"    value_stress ref_fail samples (first 3):")
            for tc in ref_fails[:3]:
                err = tc.get('error', '')[:120]
                print(f"      policy={tc.get('policy')} seed={tc.get('seed')} error={err}")
        
        # config_stress
        cs = detail.get('config_stress', {})
        rpb = cs.get('raw_results', {})
        cs_statuses = []
        for bs, bsd in rpb.items():
            if isinstance(bsd, dict):
                cs_statuses.append(f"bs={bs}: {bsd.get('status', '?')}")
        if cs_statuses:
            print(f"    config_stress: {cs_statuses[:4]}")

# Final pattern: what fraction of anomaly kernels' value_stress tests are ref_fail?
print("\n" + "=" * 80)
print("PATTERN 3: ref_fail dominance in anomaly kernels")
print("=" * 80)

ref_fail_pct_dist = []
for ds, kid, kdata in anomaly_kernels:
    detail_path = os.path.join(base_dir, ds, 'details', f'{kid}.json')
    if not os.path.exists(detail_path):
        continue
    with open(detail_path, 'r', encoding='utf-8') as f:
        detail = json.load(f)
    
    vs = detail.get('value_stress', {})
    tcs = vs.get('test_cases', [])
    if not tcs:
        continue
    ref_fail = sum(1 for tc in tcs if tc.get('status') == 'ref_fail')
    total = len(tcs)
    ref_fail_pct = ref_fail / total
    ref_fail_pct_dist.append((ds, kid, ref_fail_pct, ref_fail, total))

# Buckets
bucket_high = sum(1 for _, _, pct, _, _ in ref_fail_pct_dist if pct >= 0.8)
bucket_med = sum(1 for _, _, pct, _, _ in ref_fail_pct_dist if 0.3 <= pct < 0.8)
bucket_low = sum(1 for _, _, pct, _, _ in ref_fail_pct_dist if pct < 0.3)

print(f"\n  Distribution of value_stress ref_fail percentage among {len(ref_fail_pct_dist)} anomaly kernels:")
print(f"    Very high ref_fail rate (>=80%): {bucket_high}")
print(f"    Medium ref_fail rate (30-80%): {bucket_med}")
print(f"    Low ref_fail rate (<30%): {bucket_low}")

# Top 10 with highest ref_fail
print(f"\n  Top 10 with highest ref_fail rate:")
ref_fail_pct_dist.sort(key=lambda x: -x[2])
for ds, kid, pct, rf, total in ref_fail_pct_dist[:10]:
    print(f"    {kid} ({ds}): {pct*100:.1f}% ref_fail ({rf}/{total})")

# Also analyze which kernels have NO ref_fail and yet pass everything
print(f"\n  Anomaly kernels with NO ref_fail (genuinely pass all stress tests despite baseline fail):")
genuine_pass = [(ds, kid, pct, rf, total) for ds, kid, pct, rf, total in ref_fail_pct_dist if rf == 0]
print(f"    Count: {len(genuine_pass)}")
for ds, kid, pct, rf, total in genuine_pass[:10]:
    print(f"    {kid} ({ds})")
