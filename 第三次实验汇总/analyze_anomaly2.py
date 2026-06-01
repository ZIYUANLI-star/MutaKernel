"""
Detailed root cause classification of 56 anomaly kernels.
"""
import json
import os
from collections import defaultdict, Counter
import re

base_dir = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel\第三次实验汇总\results"
datasets = ['cuda_l1', 'ai_cuda_engineer', 'tritonbench_g']

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

print(f"Total anomaly kernels: {len(anomaly_kernels)}\n")

# Classify by error type
classification = defaultdict(list)

for ds, kid, kdata in anomaly_kernels:
    detail_path = os.path.join(base_dir, ds, 'details', f'{kid}.json')
    if not os.path.exists(detail_path):
        classification['no_detail_file'].append((ds, kid))
        continue
    with open(detail_path, 'r', encoding='utf-8') as f:
        detail = json.load(f)
    
    # Collect all error messages from value_stress + training_stress (the main stress dims)
    error_msgs = []
    for dim in ['value_stress', 'training_stress']:
        dim_data = detail.get(dim, {})
        for tc in dim_data.get('test_cases', []):
            if tc.get('status') == 'ref_fail':
                err = tc.get('error', '')
                if err:
                    error_msgs.append(err)
    
    if not error_msgs:
        classification['no_error_msg'].append((ds, kid))
        continue
    
    # Take the most common error pattern
    main_err = error_msgs[0]
    
    # Classify
    if 'CUDA out of memory' in main_err or 'OOM' in main_err.upper():
        classification['cuda_oom'].append((ds, kid))
    elif "has no attribute 'Model'" in main_err or "has no attribute 'ModelNew'" in main_err:
        classification['module_load_failure'].append((ds, kid))
    elif "No module named" in main_err:
        # Extract module name
        m = re.search(r"No module named '(\w+)'", main_err)
        modname = m.group(1) if m else 'unknown'
        classification[f'missing_module_{modname}'].append((ds, kid))
    elif 'ref NaN/Inf' in main_err:
        classification['ref_nan_fallback'].append((ds, kid))
    elif "is not defined" in main_err:
        classification['kernel_source_namerror'].append((ds, kid))
    elif 'WorkerCrash' in main_err:
        classification['other_worker_crash'].append((ds, kid))
    else:
        classification[f'other'].append((ds, kid))

# Print classification
print("=" * 80)
print("ROOT CAUSE CLASSIFICATION")
print("=" * 80)
total = sum(len(v) for v in classification.values())
print(f"\n{'Cause':<35} {'Count':<10} {'Percentage':<10} Examples")
print("-" * 90)

for cause, kernels in sorted(classification.items(), key=lambda x: -len(x[1])):
    pct = len(kernels) / total * 100
    examples = [k[1] for k in kernels[:3]]
    print(f"  {cause:<33} {len(kernels):<10} {pct:>6.1f}%   {examples}")

print(f"\nTotal classified: {total}")

# Per-dataset
print("\n" + "=" * 80)
print("BY DATASET")
print("=" * 80)

for ds in datasets:
    ds_anomalies = [(c, k) for c, kernels in classification.items() for d, k in kernels if d == ds]
    if not ds_anomalies:
        continue
    print(f"\n--- {ds} ({len(ds_anomalies)} anomaly kernels) ---")
    cause_counts = Counter(c for c, k in ds_anomalies)
    for c, count in cause_counts.most_common():
        print(f"  {c}: {count}")

# What if we exclude these from the analysis - what happens?
print("\n" + "=" * 80)
print("CORRECTED STATISTICS (excluding workerCrash anomalies)")
print("=" * 80)

# True "stress all pass" + "baseline fail" (no ref_fail dominance)
# is essentially zero. Let's look at if baseline failures correlate
# with environment issues vs real defects.

# Baseline tests use seeds 0, 1, 2 with default torch.randn
# Stress tests use much larger seeds (50000+) with extreme distributions
# If baseline fails = the kernel has issues in default setup
# If stress can't run = environment/resource issue at test time

# Look at the 35 cuda_l1 anomaly kernels - all have IDs in range L1_T30-L1_T82 which matches the SKIPPED range
# This suggests these kernels share an environment issue
print("\n  CUDA-L1 anomaly kernel IDs:")
cuda_anomalies = [k for ds, k, _ in anomaly_kernels if ds == 'cuda_l1']
for k in sorted(cuda_anomalies):
    print(f"    {k}")
