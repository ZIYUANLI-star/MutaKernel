import json, os, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

details_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\llm_analysis_results\details'

killed_strategies = []
survival_categories = Counter()

for f in os.listdir(details_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(details_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    status = data.get('status', '')
    if status == 'killed_by_llm':
        ks = data.get('kill_strategy', '')
        killed_strategies.append({
            'id': data.get('mutant_id', ''),
            'op': data.get('operator', ''),
            'strategy': ks[:300]
        })
    elif status == 'survived':
        cat = data.get('cluster_label', '')
        if cat:
            survival_categories[cat] += 1

print('=== KILLED BY LLM (total:', len(killed_strategies), ') ===')
for k in killed_strategies:
    print(f"  {k['id']} ({k['op']})")
    print(f"    strategy: {k['strategy'][:250]}...")
    print()

print()
print('=== SURVIVAL CATEGORIES ===')
for cat, cnt in survival_categories.most_common():
    print(f'  [{cnt:3d}] {cat}')
