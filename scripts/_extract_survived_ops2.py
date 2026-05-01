import json, os, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

details_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\llm_analysis_results\details'

survived_ops = Counter()
cluster_x_op = Counter()
reason_examples = {}

for f in os.listdir(details_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(details_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    if data.get('status') != 'survived':
        continue

    mid = data.get('mutant_id', '')
    op = data.get('operator_name', '')
    if not op:
        parts = mid.rsplit('__', 2)
        op = parts[1] if len(parts) >= 3 else 'unknown'
    cat = data.get('cluster_label', 'unknown')
    survived_ops[op] += 1
    cluster_x_op[f'{cat} | {op}'] += 1

    key = f'{cat}'
    if key not in reason_examples:
        sr = data.get('survival_reason', '')
        reason_examples[key] = (mid, sr[:200])

print('=== Survived by operator ===')
for op, cnt in survived_ops.most_common():
    print(f'  [{cnt:3d}] {op}')

print()
print('=== Top cluster x operator combos ===')
for combo, cnt in cluster_x_op.most_common(25):
    print(f'  [{cnt:3d}] {combo}')

print()
print('=== Sample survival reasons per category ===')
for cat, (mid, sr) in sorted(reason_examples.items()):
    print(f'\n  [{cat}]')
    print(f'    Example: {mid}')
    print(f'    Reason: {sr}...')
