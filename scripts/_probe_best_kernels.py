import json
d = json.load(open('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/best_kernels.json'))
print('type:', type(d).__name__)
if isinstance(d, dict):
    keys = list(d.keys())
    print(f'total: {len(keys)}, first 3: {keys[:3]}')
    sample = d[keys[0]]
    print(f'sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample).__name__}')
    print(json.dumps(sample, indent=2)[:800])
elif isinstance(d, list):
    print(f'total: {len(d)}')
    print(json.dumps(d[0], indent=2)[:800])
