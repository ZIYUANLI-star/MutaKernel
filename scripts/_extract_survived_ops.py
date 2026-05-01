import json, os, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

details_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\llm_analysis_results\details'

survived_ops = Counter()
cluster_x_op = Counter()

for f in os.listdir(details_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(details_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    if data.get('status') != 'survived':
        continue

    op = data.get('operator', 'unknown')
    cat = data.get('cluster_label', 'unknown')
    survived_ops[op] += 1
    cluster_x_op[f'{cat} | {op}'] += 1

print('=== Survived by operator ===')
for op, cnt in survived_ops.most_common():
    print(f'  [{cnt:3d}] {op}')

print()
print('=== Top 30 cluster x operator combos ===')
for combo, cnt in cluster_x_op.most_common(30):
    print(f'  [{cnt:3d}] {combo}')
