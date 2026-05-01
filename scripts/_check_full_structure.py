import json, os, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

full_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\full_block12_results\details'

total_mutants = 0
status_counter = Counter()
survived_ids = []
fused_survived = []

for f in os.listdir(full_dir):
    if not f.endswith('.json'):
        continue
    kn = f.replace('.json', '')
    with open(os.path.join(full_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    for m in data.get('mutants', []):
        total_mutants += 1
        st = m.get('status', '')
        status_counter[st] += 1
        
        if st == 'survived':
            mid = m.get('id', '')
            op = m.get('operator_name', '')
            mc = m.get('mutated_code', '').lower()
            survived_ids.append({'id': mid, 'kernel': kn, 'op': op})
            
            if any(kw in mc for kw in ['fused', 'fusion', 'fuse_', 'fuse(']):
                fused_survived.append({'id': mid, 'kernel': kn, 'op': op})

print(f'Total mutants: {total_mutants}')
print(f'Status distribution: {dict(status_counter)}')
print(f'Total survived: {len(survived_ids)}')
print(f'Survived with fused keyword: {len(fused_survived)}')
print()

# Show fused kernels
fused_kernels = Counter(e['kernel'] for e in fused_survived)
print('Fused kernels:')
for kn, cnt in fused_kernels.most_common():
    print(f'  {kn}: {cnt} survived mutants')

# Show sample survived id format
print()
print('Sample survived IDs:')
for s in survived_ids[:5]:
    print(f'  kernel={s["kernel"]}, id={s["id"]}, op={s["op"]}')
