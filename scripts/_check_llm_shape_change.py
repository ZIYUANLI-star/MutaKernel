"""检查 LLM 杀死的 31 个变异体中，有多少实际改了 shape"""
import json, os, sys, re
sys.stdout.reconfigure(encoding='utf-8')

details_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\llm_analysis_results\details'

shape_kills = []
value_kills = []

for f in os.listdir(details_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(details_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    if data.get('status') != 'killed_by_llm':
        continue

    mid = data.get('mutant_id', '')
    ks = data.get('kill_strategy', '')
    tcr = data.get('test_construction_rule', {})
    rule_name = tcr.get('rule_name', '') if isinstance(tcr, dict) else ''

    ks_lower = ks.lower()
    changed_shape = False

    shape_keywords = [
        'shape', 'dimension', 'size', 'resize', 'reshape',
        'non-multiple', 'not a multiple', 'not divisible',
        'small input', 'minimal input', '1 element', 'single element',
        'inner_size', 'inner size', 'expand', '3d', '3-d',
        'total_elements', 'total elements',
        'n =', 'm =', 'n=', 'adjusted',
        'force', 'change the input shape',
    ]

    for kw in shape_keywords:
        if kw in ks_lower:
            changed_shape = True
            break

    entry = {
        'id': mid,
        'rule': rule_name,
        'strategy_preview': ks[:180],
    }

    if changed_shape:
        shape_kills.append(entry)
    else:
        value_kills.append(entry)

print(f'=== LLM kills that CHANGED SHAPE: {len(shape_kills)} ===')
for e in shape_kills:
    print(f"  {e['id']}")
    print(f"    rule: {e['rule']}")
    print(f"    strategy: {e['strategy_preview']}...")
    print()

print(f'=== LLM kills with VALUE ONLY (no shape change): {len(value_kills)} ===')
for e in value_kills:
    print(f"  {e['id']}")
    print(f"    rule: {e['rule']}")
    print(f"    strategy: {e['strategy_preview']}...")
    print()

print(f'\n总计: shape改变={len(shape_kills)}, 仅改值={len(value_kills)}, 合计={len(shape_kills)+len(value_kills)}')
