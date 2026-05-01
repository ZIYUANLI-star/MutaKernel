"""
生成存活原因分析文档，包含：
1. 三阶段杀伤汇总
2. LLM 聚类的 7 大存活原因 × 各原因下的变异体列表
3. 算子融合专项分析
4. 未覆盖规则/盲区分析
"""
import json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel')
STRESS_DIR = BASE / 'stress_enhance_results' / 'details'
LLM_DIR = BASE / 'llm_analysis_results' / 'details'
FULL_DIR = BASE / 'full_block12_results' / 'details'
OUT = BASE / 'docs' / 'SurvivalAnalysis.md'

# ─── Phase 0: 全量初始测试数据 ───
initial_survived = {}  # mutant_id -> operator
for f in FULL_DIR.glob('*.json'):
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    for m in data.get('mutants', []):
        if m.get('status') == 'survived':
            mid = m.get('mutant_id', '')
            op = m.get('operator_name', '')
            initial_survived[mid] = op

# ─── Phase 1: 增强测试数据 ───
stress_killed = {}   # mutant_id -> {layer, policy, ...}
stress_survived = {} # mutant_id -> operator

for f in STRESS_DIR.glob('*.json'):
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    mid = data.get('mutant_id', '')
    op = data.get('operator_name', '')
    if data.get('killed'):
        stress_killed[mid] = {
            'operator': op,
            'layer': data.get('killing_layer', ''),
            'policy': data.get('killing_policy', ''),
        }
    else:
        stress_survived[mid] = op

# ─── Phase 2: LLM 分析数据 ───
llm_killed = {}
llm_survived = defaultdict(list)  # cluster_label -> list of entries

for f in LLM_DIR.glob('*.json'):
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    mid = data.get('mutant_id', '')
    op = data.get('operator_name', '')
    if not op:
        parts = mid.rsplit('__', 2)
        op = parts[1] if len(parts) >= 3 else ''

    if data.get('status') == 'killed_by_llm':
        llm_killed[mid] = {
            'operator': op,
            'strategy': data.get('kill_strategy', '')[:200],
            'rule_name': (data.get('test_construction_rule', {}) or {}).get('rule_name', ''),
        }
    elif data.get('status') == 'survived':
        cluster = data.get('cluster_label', 'Uncategorized')
        llm_survived[cluster].append({
            'id': mid,
            'operator': op,
            'kernel': data.get('kernel_name', ''),
            'reason': data.get('survival_reason', ''),
            'killable': data.get('killable', None),
        })

# ─── 算子融合分析 ───
fused_mutant_ids = set()
for f in FULL_DIR.glob('*.json'):
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    for m in data.get('mutants', []):
        if m.get('status') == 'survived':
            mc = m.get('mutated_code', '').lower()
            if any(kw in mc for kw in ['fused', 'fusion', 'fuse_', 'fuse(']):
                fused_mutant_ids.add(m.get('mutant_id', ''))

# ─── 策略杀伤统计 ───
policy_kills = Counter()
for f in STRESS_DIR.glob('*.json'):
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    if data.get('killed'):
        kp = data.get('killing_policy', '')
        if kp:
            policy_kills[kp] += 1

# ─── 未被 LLM 判为 killable 的变异体分析 ───
unkillable_count = 0
killable_but_not_killed = 0
for cluster, entries in llm_survived.items():
    for e in entries:
        if e['killable'] == False:
            unkillable_count += 1
        elif e['killable'] == True:
            killable_but_not_killed += 1

# ─── 生成文档 ───
lines = []
L = lines.append

L('# 存活变异体分析报告')
L('')
L('> 自动生成，基于三阶段实验数据的综合分析。')
L('')

# === Section 1: 三阶段汇总 ===
L('## 一、三阶段杀伤汇总')
L('')
L(f'| 阶段 | 输入变异体 | 杀死 | 存活 | 杀伤率 |')
L(f'|------|----------|------|------|--------|')
L(f'| Phase 0: 初始变异测试 | 1663 | {1663 - len(initial_survived)} | {len(initial_survived)} | {(1663 - len(initial_survived))/1663*100:.1f}% |')
L(f'| Phase 1: 4层增强测试 | {len(initial_survived)} | {len(stress_killed)} | {len(stress_survived)} | {len(stress_killed)/max(1,len(initial_survived))*100:.1f}% |')

llm_input = len(stress_survived)
L(f'| Phase 2: LLM分析 | {llm_input} | {len(llm_killed)} | {sum(len(v) for v in llm_survived.values())} | {len(llm_killed)/max(1,llm_input)*100:.1f}% |')
total_final = sum(len(v) for v in llm_survived.values())
L(f'| **合计** | 1663 | {1663 - total_final} | **{total_final}** | {(1663 - total_final)/1663*100:.1f}% |')
L('')

# === Section 2: Phase 1 策略杀伤详情 ===
L('## 二、Phase 1 增强测试策略杀伤详情')
L('')
L('| 策略 | 杀死数 | 主要针对算子 |')
L('|------|--------|------------|')
for pol, cnt in policy_kills.most_common():
    ops_in_pol = Counter()
    for f2 in STRESS_DIR.glob('*.json'):
        with open(f2, 'r', encoding='utf-8') as fp2:
            d2 = json.load(fp2)
        if d2.get('killed') and d2.get('killing_policy') == pol:
            ops_in_pol[d2.get('operator_name', '')] += 1
    ops_str = ', '.join(f'{o}({c})' for o, c in ops_in_pol.most_common(3))
    L(f'| `{pol}` | {cnt} | {ops_str} |')
L('')

# === Section 3: 最终存活变异体按类别分布 ===
L('## 三、最终存活变异体 — 按 LLM 聚类分类')
L('')
L(f'共 {total_final} 个变异体最终存活，LLM 将其归为 {len(llm_survived)} 个类别：')
L('')

# 先输出总览表
L('| 类别 | 数量 | 占比 | 主要算子 |')
L('|------|------|------|---------|')
for cluster in sorted(llm_survived.keys(), key=lambda c: -len(llm_survived[c])):
    entries = llm_survived[cluster]
    op_counter = Counter(e['operator'] for e in entries)
    ops_str = ', '.join(f'{o}({c})' for o, c in op_counter.most_common(4))
    pct = len(entries) / max(1, total_final) * 100
    L(f'| {cluster} | {len(entries)} | {pct:.1f}% | {ops_str} |')
L('')

# 详细展开每个类别
for cluster in sorted(llm_survived.keys(), key=lambda c: -len(llm_survived[c])):
    entries = llm_survived[cluster]
    L(f'### 3.{list(sorted(llm_survived.keys(), key=lambda c: -len(llm_survived[c]))).index(cluster)+1} {cluster} ({len(entries)} 个)')
    L('')

    # 取第一个作为 representative reason
    rep = entries[0]
    reason_preview = rep['reason'][:300].replace('\n', ' ')
    L(f'**典型存活原因**: {reason_preview}...')
    L('')

    # 按算子分组列出
    by_op = defaultdict(list)
    for e in entries:
        by_op[e['operator']].append(e)

    for op in sorted(by_op.keys(), key=lambda o: -len(by_op[o])):
        op_entries = by_op[op]
        L(f'- **{op}** ({len(op_entries)} 个): ', )
        ids = [e['id'] for e in op_entries]
        # 按 kernel 分组紧凑显示
        by_kernel = defaultdict(list)
        for e in op_entries:
            by_kernel[e['kernel']].append(e['id'])
        parts = []
        for kn in sorted(by_kernel.keys()):
            mids = by_kernel[kn]
            if len(mids) <= 3:
                parts.append(', '.join(f'`{m}`' for m in mids))
            else:
                parts.append(f'{kn}: {len(mids)} 个')
        L('  ' + '; '.join(parts))

    # 标注哪些是融合 kernel
    fused_in_cluster = [e['id'] for e in entries if e['id'] in fused_mutant_ids]
    if fused_in_cluster:
        L(f'- *其中 {len(fused_in_cluster)} 个来自融合 kernel*')
    L('')

# === Section 4: 算子融合专项分析 ===
L('## 四、算子融合（Operator Fusion）专项分析')
L('')

fused_final_survived = []
fused_stress_killed_list = []
for mid in fused_mutant_ids:
    if mid in stress_killed:
        fused_stress_killed_list.append(mid)
    # Check if in llm_survived
    for cluster, entries in llm_survived.items():
        for e in entries:
            if e['id'] == mid:
                fused_final_survived.append({
                    'id': mid,
                    'operator': e['operator'],
                    'kernel': e['kernel'],
                    'cluster': cluster,
                    'reason': e['reason'][:200],
                })

L(f'含 fused/fusion 关键词的存活变异体共 {len(fused_mutant_ids)} 个（初始）。')
L(f'- Phase 1 增强测试杀死: {len(fused_stress_killed_list)} 个')
L(f'- Phase 2 LLM 杀死: {len([m for m in fused_mutant_ids if m in llm_killed])} 个')
L(f'- **最终存活: {len(fused_final_survived)} 个**')
L('')

# 按 cluster 统计融合存活
fused_by_cluster = Counter(e['cluster'] for e in fused_final_survived)
L('融合 kernel 存活变异体的存活原因分布:')
L('')
L('| 存活原因 | 数量 |')
L('|---------|------|')
for cl, cnt in fused_by_cluster.most_common():
    L(f'| {cl} | {cnt} |')
L('')

fused_by_kernel = defaultdict(list)
for e in fused_final_survived:
    fused_by_kernel[e['kernel']].append(e)

L('涉及的融合 kernel 详情:')
L('')
for kn in sorted(fused_by_kernel.keys()):
    elist = fused_by_kernel[kn]
    ops = Counter(e['operator'] for e in elist)
    L(f'- **{kn}** ({len(elist)} 个存活): {dict(ops)}')

L('')

# === Section 5: 未覆盖的规则/盲区 ===
L('## 五、未覆盖的规则与盲区分析')
L('')
L(f'LLM 判定 {unkillable_count} 个变异体为"不可杀死"（killable=False），')
L(f'{killable_but_not_killed} 个为"理论可杀但 3 轮内未成功"（killable=True but survived）。')
L('')

# 按 cluster × killable 统计
L('### 5.1 不可杀死变异体的分布')
L('')
L('| 类别 | 不可杀 | 理论可杀未杀 | killable未知 |')
L('|------|--------|-------------|-------------|')
for cluster in sorted(llm_survived.keys(), key=lambda c: -len(llm_survived[c])):
    entries = llm_survived[cluster]
    unk = sum(1 for e in entries if e['killable'] == False)
    kbl = sum(1 for e in entries if e['killable'] == True)
    na = sum(1 for e in entries if e['killable'] is None)
    L(f'| {cluster} | {unk} | {kbl} | {na} |')
L('')

# 分析"理论可杀但未杀"的是什么
L('### 5.2 "理论可杀但未杀" 变异体（潜在规则盲区）')
L('')
killable_entries = []
for cluster, entries in llm_survived.items():
    for e in entries:
        if e['killable'] == True:
            killable_entries.append({**e, 'cluster': cluster})

killable_by_op = Counter(e['operator'] for e in killable_entries)
L(f'共 {len(killable_entries)} 个，按算子分布:')
L('')
for op, cnt in killable_by_op.most_common():
    L(f'- `{op}`: {cnt} 个')
L('')

L('这些变异体 LLM 认为理论上可以通过特定输入杀死，但 3 轮迭代内未成功。')
L('常见原因: LLM 建议的输入需要改变 shape（在 fixed-shape 约束下无法实现）、')
L('或需要控制 CUDA 运行时行为（非确定性竞态条件、内存布局）。')
L('')

# 按存活原因给出可能的新策略
L('### 5.3 潜在的新增策略建议')
L('')
L('基于 LLM 分析的 `test_construction_rules` 和存活原因，以下为 **不改变 shape** 约束下仍可考虑的策略:')
L('')
L('| 策略名 | 构造方式 | 目标 | 来源 |')
L('|--------|---------|------|------|')
L('| `epsilon_critical_norm` | 控制 L2 范数落入 [1e-9, 1e-2] | epsilon_modify, const_perturb | LLM rule: epsilon_sensitivity_targeting |')
L('| `reduction_identity_extreme` | 整行值 < -1e10 或 > 1e10 | init_modify | LLM rule: reduction_init_identity_extremes |')
L('| `block_sparse` | 整块为零 + 其余为大值 | mask_boundary, arith_replace | 导师建议: 结构化稀疏 |')
L('| `near_relu_zero` | 50% 值 ∈ [-0.01, 0.01] (relu 零点附近) | 融合 kernel 的中间层 | 导师建议: 算子融合启发 |')
L('')

content = '\n'.join(lines)
OUT.write_text(content, encoding='utf-8')
print(f'文档已生成: {OUT}')
print(f'  总行数: {len(lines)}')
