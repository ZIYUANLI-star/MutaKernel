"""检查导师提出的三个策略建议的可行性"""
import json, os, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

# --- Part 1: 检查 sparse 策略的实际杀伤情况 ---
stress_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\stress_enhance_results\details'

sparse_killed = 0
sparse_tried = 0
policy_kill_counter = Counter()

for f in os.listdir(stress_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(stress_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    if data.get('killed'):
        policy_kill_counter[data.get('killing_policy', '')] += 1

    pr = data.get('policy_results', {})
    if 'sparse' in pr:
        sparse_tried += 1
        sp = pr['sparse']
        if isinstance(sp, dict) and sp.get('killed'):
            sparse_killed += 1

print('='*70)
print('Part 1: sparse 策略的当前杀伤情况')
print('='*70)
print(f'  sparse 策略被执行: {sparse_tried} 次')
print(f'  sparse 策略杀死:   {sparse_killed} 个')
print()
print('--- 各策略杀伤排名 ---')
for pol, cnt in policy_kill_counter.most_common():
    print(f'  [{cnt:2d}] {pol}')

# --- Part 2: 检查 B 类存活变异体中 launch_config 相关 ---
print()
print('='*70)
print('Part 2: B 类存活变异体 - launch_config 相关')
print('='*70)

llm_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\llm_analysis_results\details'
b_class_ops = ['index_replace', 'sync_remove', 'mask_boundary', 'launch_config_mutate']

b_survived = Counter()
launch_config_details = []

for f in os.listdir(llm_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(llm_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    mid = data.get('mutant_id', '')
    parts = mid.rsplit('__', 2)
    op = parts[1] if len(parts) >= 3 else ''

    if op in b_class_ops and data.get('status') == 'survived':
        b_survived[op] += 1
        if op == 'launch_config_mutate':
            launch_config_details.append({
                'id': mid,
                'reason': data.get('survival_reason', '')[:200],
                'cluster': data.get('cluster_label', ''),
            })

for op, cnt in b_survived.most_common():
    print(f'  [{cnt:2d}] {op}')

print()
print('--- launch_config_mutate 存活详情 ---')
for d in launch_config_details:
    print(f"  {d['id']} [{d['cluster']}]")
    print(f"    {d['reason']}...")
    print()

# --- Part 3: 检查哪些 kernel 涉及算子融合 ---
print('='*70)
print('Part 3: 算子融合相关 kernel')
print('='*70)

# 看 kernel 名来自哪些 problem
kernel_ops = Counter()
fused_hints = []

full_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\full_block12_results\details'
for f in os.listdir(full_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(full_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    mutants = data.get('mutants', [])
    for m in mutants:
        if m.get('status') == 'survived':
            mc = m.get('mutated_code', '')
            mc_lower = mc.lower()
            if any(kw in mc_lower for kw in ['fused', 'fusion', 'fuse_']):
                fused_hints.append(m.get('mutant_id', ''))

print(f'  含 fused/fusion 关键词的存活变异体: {len(fused_hints)}')
if fused_hints[:10]:
    for fh in fused_hints[:10]:
        print(f'    {fh}')
    if len(fused_hints) > 10:
        print(f'    ... 还有 {len(fused_hints)-10} 个')

# --- Part 4: 稀疏 vs 稠密的互补分析 ---
print()
print('='*70)
print('Part 4: 当前 sparse 策略 vs 导师建议的"稀疏矩阵"测试对比')
print('='*70)
print("""
当前 sparse 策略 (_sparse):
  - 90% 零 + 10% randn*100
  - 目的: 暴露 init_modify (正值掩盖 0.0 初值)

导师建议的"稀疏矩阵 vs 稠密矩阵":
  - 可能指: 构造真正的 structured sparsity
  - 例如: 整行/整列为零, 块稀疏, 对角稀疏
  - 这与当前 random sparse 不同
""")
