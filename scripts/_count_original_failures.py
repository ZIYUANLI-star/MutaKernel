import json, glob, os
os.chdir('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/stress_enhance_results/details/')
files = sorted(glob.glob('*.json'))
nonempty = []
total_top = 0
total_main_value = 0
total_main_repeat = 0
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    counts = {}
    if d.get('original_failures'):
        counts['top'] = len(d['original_failures'])
        total_top += counts['top']
    mt = d.get('main_track', {})
    for k in ['value_stress', 'repeated_run']:
        if mt.get(k, {}).get('original_failures'):
            counts[f'mt.{k}'] = len(mt[k]['original_failures'])
            if k == 'value_stress':
                total_main_value += counts[f'mt.{k}']
            else:
                total_main_repeat += counts[f'mt.{k}']
    if counts:
        nonempty.append((f, counts))

print(f'total mutant detail files     : {len(files)}')
print(f'files with non-empty failures : {len(nonempty)}')
print(f'total failure events  top     : {total_top}')
print(f'                      main.vs : {total_main_value}')
print(f'                      main.rr : {total_main_repeat}')
print()
print('--- first 5 examples ---')
for f, c in nonempty[:5]:
    print(f'{f}: {c}')

print()
print('--- inspect first non-empty in detail ---')
if nonempty:
    fname = nonempty[0][0]
    with open(fname) as fp:
        d = json.load(fp)
    if d.get('original_failures'):
        print(f'>> top-level original_failures[0]:')
        print(json.dumps(d['original_failures'][0], indent=2)[:800])
    mt = d.get('main_track', {})
    for k in ['value_stress', 'repeated_run']:
        if mt.get(k, {}).get('original_failures'):
            print(f'>> main_track.{k}.original_failures[0]:')
            print(json.dumps(mt[k]['original_failures'][0], indent=2)[:800])
            break
