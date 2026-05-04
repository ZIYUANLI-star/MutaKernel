"""分析哪些最终存活变异体本可以在更早阶段被识别为等价"""
import json, os, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

LLM_DIR = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\llm_analysis_results\details'

# 收集所有最终存活变异体
survived = []
for f in os.listdir(LLM_DIR):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(LLM_DIR, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    if data.get('status') != 'survived':
        continue

    mid = data.get('mutant_id', '')
    op = data.get('operator_name', '') or mid.rsplit('__', 2)[1] if '__' in mid else ''
    survived.append({
        'id': mid,
        'op': op,
        'killable': data.get('killable'),
        'cluster': data.get('cluster_label', ''),
        'reason': data.get('survival_reason', ''),
    })

# 分析 killable=False 的 (LLM 认为在 fixed-shape 下不可杀 → 候选等价)
unkillable = [s for s in survived if s['killable'] == False]
killable = [s for s in survived if s['killable'] == True]

print(f'最终存活: {len(survived)} 个')
print(f'  killable=False (LLM认为不可杀): {len(unkillable)} 个')
print(f'  killable=True  (理论可杀未杀):  {len(killable)} 个')
print()

# 进一步分类: 哪些 unkillable 的原因是"语义等价"
truly_equiv_keywords = [
    'identical', 'equivalent', 'same result', 'no difference',
    'no effect', 'doesn\'t affect', 'does not affect',
    'never executed', 'dead code', 'unreachable',
    'always true', 'always false', 'tautolog',
    'invariant', 'commutative', 'associative',
    'zero contribution', 'no impact', 'no observable',
]

structural_equiv_keywords = [
    'grid-stride', 'grid stride', 'boundary check',
    'bounds check', 'guard', 'if (idx < size)',
    'blockIdx.y = 1', 'gridDim.y = 1', '1D grid',
    'inner_size = 1', 'inner_size=1',
    'power of two', 'multiple of',
]

tolerance_keywords = [
    'tolerance', 'atol', 'rtol', 'allclose',
    'small difference', 'within', 'precision',
]

categories = {
    'truly_semantic_equiv': [],
    'structural_equiv_under_fixed_shape': [],
    'tolerance_masked': [],
    'other_unkillable': [],
}

for s in unkillable:
    r = s['reason'].lower()
    if any(kw in r for kw in truly_equiv_keywords):
        categories['truly_semantic_equiv'].append(s)
    elif any(kw in r for kw in structural_equiv_keywords):
        categories['structural_equiv_under_fixed_shape'].append(s)
    elif any(kw in r for kw in tolerance_keywords):
        categories['tolerance_masked'].append(s)
    else:
        categories['other_unkillable'].append(s)

print('='*70)
print('killable=False 的 165 个变异体细分:')
print('='*70)

for cat, entries in categories.items():
    ops = Counter(e['op'] for e in entries)
    clusters = Counter(e['cluster'] for e in entries)
    print(f'\n--- {cat}: {len(entries)} 个 ---')
    print(f'  算子: {dict(ops)}')
    print(f'  聚类: {dict(clusters)}')
    if entries:
        print(f'  示例:')
        for e in entries[:3]:
            print(f'    {e["id"]} [{e["cluster"]}]')
            print(f'      {e["reason"][:150]}...')

# 分析哪些可以被更好的静态分析检测出来
print()
print('='*70)
print('关键问题: 哪些可以在 Phase 0 就被检测为等价?')
print('='*70)

# Ineffective Mutation 中 killable=False 的
ineffective_unkillable = [s for s in unkillable if s['cluster'] == 'Ineffective Mutation']
print(f'\n1. Ineffective Mutation + killable=False: {len(ineffective_unkillable)} 个')
print('   这些变异不改变任何执行路径，理论上可通过:')
print('   - AST 级分析（变异位点是否在活跃执行路径上）')
print('   - 数据流分析（变异的值是否影响最终输出）')
print('   - 符号执行（约束求解判断可达性）')

# Algorithmic Invariance 中 killable=False 的
alg_inv_unkillable = [s for s in unkillable if s['cluster'] == 'Algorithmic Invariance']
print(f'\n2. Algorithmic Invariance + killable=False: {len(alg_inv_unkillable)} 个')
print('   算法本身对变异不敏感（如 softmax 平移不变性），可通过:')
print('   - 语义模式匹配（识别 softmax/layernorm 等不变性模式）')
print('   - LLM 静态分析（不运行，仅分析代码语义）')

# 当前 EquivalentDetector 的局限
print(f'\n3. 当前 EquivalentDetector 的覆盖:')
print(f'   - 语法等价: _normalize_source 比较（去注释去空白）')
print(f'   - 统计等价: 100 次 random seed + bitwise identical')
print(f'   - 限制: 仅用 get_inputs() 的默认分布(randn)，不用 stress policies')

# 计算: 如果增强后的等价检测能识别多少
print(f'\n4. 如果在 Phase 0 后增加"增强等价检测":')
print(f'   - 使用 14 个 stress policies 各跑 3 seeds (42次) 做 bitwise 检测')
print(f'   - 对仍然 bitwise identical 的标记为 likely_equivalent')
print(f'   - 预计可提前识别: Ineffective Mutation({len(ineffective_unkillable)}) + ')
print(f'     Algorithmic Invariance({len(alg_inv_unkillable)}) = {len(ineffective_unkillable)+len(alg_inv_unkillable)} 个')
print(f'   - 这些变异体即使经过增强测试+LLM分析也无法被杀死')
