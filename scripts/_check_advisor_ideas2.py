"""检查各策略在增强测试中的实际杀伤效果"""
import json, os, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

stress_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\stress_enhance_results\details'

policy_kill_counter = Counter()
policy_tried_counter = Counter()
killed_by_operator = defaultdict(Counter)

for f in os.listdir(stress_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(stress_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    op = data.get('operator_name', '')
    killed = data.get('killed', False)
    kp = data.get('killing_policy', '')

    if killed and kp:
        policy_kill_counter[kp] += 1
        killed_by_operator[kp][op] += 1

    pr = data.get('policy_results', [])
    for entry in pr:
        pol = entry.get('policy', '')
        if pol:
            policy_tried_counter[pol] += 1

print('='*70)
print('各策略执行次数 vs 杀伤次数')
print('='*70)
all_policies = set(list(policy_tried_counter.keys()) + list(policy_kill_counter.keys()))
for pol in sorted(all_policies):
    tried = policy_tried_counter.get(pol, 0)
    killed = policy_kill_counter.get(pol, 0)
    rate = f'{killed/tried*100:.1f}%' if tried > 0 else 'N/A'
    ops = dict(killed_by_operator.get(pol, {}))
    print(f'  {pol:30s}  tried={tried:4d}  killed={killed:2d}  rate={rate:6s}  ops={ops}')

# --- 检查 fused 相关 ---
print()
print('='*70)
print('检查 fused 相关存活变异体')
print('='*70)

full_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\full_block12_results\details'
fused_count = 0
fused_kernels = set()

for f in os.listdir(full_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(full_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    mutants = data.get('mutants', [])
    kn = data.get('kernel_name', f.replace('.json',''))

    for m in mutants:
        if m.get('status') != 'survived':
            continue
        mc = m.get('mutated_code', '')
        if any(kw in mc.lower() for kw in ['fused', 'fusion', 'fuse_', 'fuse(']):
            fused_count += 1
            fused_kernels.add(kn)

print(f'  含 fused/fusion 关键词的存活变异体总数: {fused_count}')
print(f'  涉及的 kernel 数: {len(fused_kernels)}')
print(f'  kernel 列表: {sorted(fused_kernels)[:20]}')

# --- 检查 launch_config 是否 hardcoded ---
print()
print('='*70)
print('B4 launch_config_mutate: 启动配置固定 vs 可变')
print('='*70)

llm_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\llm_analysis_results\details'
for f in os.listdir(llm_dir):
    if not f.endswith('.json'):
        continue
    if 'launch_config' not in f:
        continue
    with open(os.path.join(llm_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    mid = data.get('mutant_id','')
    status = data.get('status','')
    sr = data.get('survival_reason','')
    cl = data.get('cluster_label','')
    print(f'  {mid} [{status}] [{cl}]')
    print(f'    reason: {sr[:250]}')
    print()
