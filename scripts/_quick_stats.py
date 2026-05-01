import json, glob, os

base = r'/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第三次实验汇总/results/cuda_l1'
cp_path = os.path.join(base, 'checkpoint.json')

with open(cp_path, 'r') as f:
    data = json.load(f)

total = len(data)
completed = sum(1 for v in data.values() if v.get('status') == 'COMPLETED')
skipped = sum(1 for v in data.values() if v.get('status') == 'SKIPPED')

disc_kernels = []
no_disc = 0
total_disc = 0
baseline_pass_count = 0
baseline_fail_count = 0

for k, v in data.items():
    bp = v.get('baseline', {}).get('passed', 0)
    bf = v.get('baseline', {}).get('failed', 0)
    if bp == 3:
        baseline_pass_count += 1
    if bf >= 2:
        baseline_fail_count += 1

    td = v.get('total_discrepancies', 0)
    total_disc += td
    if td > 0:
        disc_kernels.append({
            'id': k,
            'total': td,
            'dims': v.get('discrepant_dimensions', []),
            'bp': bp, 'bf': bf,
            'vs': v.get('value_stress', {}).get('discrepancies', 0),
            'ds': v.get('dtype_stress', {}).get('discrepancies', 0),
            'ts': v.get('training_stress', {}).get('discrepancies', 0),
            'rr': v.get('repeated_run', {}).get('discrepancies', 0),
            'cs': v.get('config_stress', {}).get('discrepancies', 0),
        })
    else:
        no_disc += 1

times = [v.get('elapsed_s', 0) for v in data.values() if v.get('elapsed_s')]
total_time = sum(times)

print('=== Progress ===')
print('Completed: {}/{}'.format(completed, 241))
print('Skipped: {}'.format(skipped))
print('Total time: {:.1f} hours'.format(total_time / 3600))
print('Avg per kernel: {:.0f}s'.format(total_time / max(completed, 1)))
print()
print('=== Baseline ===')
print('Baseline all pass: {}'.format(baseline_pass_count))
print('Baseline mostly fail: {}'.format(baseline_fail_count))
print()
print('=== Discrepancies ===')
print('Kernels with discrepancies: {}/{}'.format(len(disc_kernels), completed))
print('Kernels without discrepancies: {}'.format(no_disc))
print('Total discrepancy count: {}'.format(total_disc))
print()

dim_counts = {}
for dk in disc_kernels:
    for d in dk['dims']:
        dim_counts[d] = dim_counts.get(d, 0) + 1
print('=== By dimension ===')
for d, c in sorted(dim_counts.items(), key=lambda x: -x[1]):
    print('  {}: {} kernels'.format(d, c))
print()

print('=== Discrepant kernels detail ===')
for dk in sorted(disc_kernels, key=lambda x: -x['total']):
    bl = 'BL_PASS' if dk['bp'] == 3 else 'BL_FAIL({})'.format(dk['bf'])
    print('  {} [{}]: disc={}, dims={}'.format(dk['id'], bl, dk['total'], dk['dims']))
    print('    value={}, dtype={}, train={}, repeat={}, config={}'.format(
        dk['vs'], dk['ds'], dk['ts'], dk['rr'], dk['cs']))

# Check diff_summary in new detail files
print()
print('=== diff_summary check (new files) ===')
detail_dir = os.path.join(base, 'details')
checked = 0
has_summary = 0
for dk in disc_kernels:
    fpath = os.path.join(detail_dir, dk['id'] + '.json')
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        detail = json.load(f)
    for dim in ['value_stress', 'training_stress']:
        for tc in detail.get(dim, {}).get('test_cases', []):
            if tc.get('status') == 'discrepancy':
                checked += 1
                if tc.get('diff_summary', ''):
                    has_summary += 1
print('Discrepancy test cases checked: {}'.format(checked))
print('With diff_summary filled: {}'.format(has_summary))
print('Without diff_summary: {}'.format(checked - has_summary))
