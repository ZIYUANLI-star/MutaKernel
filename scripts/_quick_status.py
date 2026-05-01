import json, os, glob

base = os.path.join(os.path.dirname(__file__), '..', '第二次实验汇总', 'stress_enhance_results')
c = json.load(open(os.path.join(base, 'completed.json')))
if isinstance(c, list):
    completed_ids = set(c)
    print(f"Completed (list): {len(c)}")
elif isinstance(c, dict):
    completed_ids = set(c.keys())
    print(f"Completed (dict): {len(c)}")

details_dir = os.path.join(base, 'details')
detail_files = glob.glob(os.path.join(details_dir, '*.json'))
print(f"Detail files: {len(detail_files)}")

sample = json.load(open(detail_files[0]))
print(f"Sample keys: {list(sample.keys())[:15]}")

killed = 0
survived = 0
llm_kills = []
for df in detail_files:
    d = json.load(open(df))
    mid = os.path.basename(df).replace('.json','')
    fv = d.get('final_verdict', d.get('verdict', ''))
    kb = str(d.get('killed_by', d.get('kill_dimension', '')))
    any_k = d.get('any_killed', False)
    if fv == 'KILLED' or any_k:
        killed += 1
        if 'llm' in kb.lower() or 'suggested' in kb.lower():
            llm_kills.append(mid)
    elif fv == 'SURVIVED' or (not any_k and fv != 'KILLED'):
        survived += 1

print(f"KILLED: {killed}, SURVIVED: {survived}")

llm_kills2 = []
first_kill_modes = {}
for df in detail_files:
    d = json.load(open(df))
    mid = os.path.basename(df).replace('.json','')
    fkm = d.get('first_kill_mode', '')
    if fkm:
        first_kill_modes[fkm] = first_kill_modes.get(fkm, 0) + 1
    if 'llm' in str(fkm).lower() or 'suggest' in str(fkm).lower():
        llm_kills2.append(mid)

print(f"First kill modes: {first_kill_modes}")
print(f"LLM kills (by first_kill_mode): {len(llm_kills2)}")
for m in llm_kills2:
    print(f"  - {m}")
