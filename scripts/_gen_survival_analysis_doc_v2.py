"""生成完整的存活变异体分析文档（v2 修正版）"""
import json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel')
STRESS_DIR = BASE / 'stress_enhance_results' / 'details'
LLM_DIR = BASE / 'llm_analysis_results' / 'details'
FULL_DIR = BASE / 'full_block12_results' / 'details'
OUT = BASE / 'docs' / 'SurvivalAnalysis.md'

# ─── Phase 0: 全量初始测试 ───
initial_survived = {}
initial_fused = set()
total_mutants = 0
status_counter = Counter()

for f in FULL_DIR.glob('*.json'):
    kn = f.stem
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    for m in data.get('mutants', []):
        total_mutants += 1
        st = m.get('status', '')
        status_counter[st] += 1
        if st == 'survived':
            mid = m.get('id', '')
            op = m.get('operator_name', '')
            initial_survived[mid] = {'op': op, 'kernel': kn}
            mc = m.get('mutated_code', '').lower()
            if any(kw in mc for kw in ['fused', 'fusion', 'fuse_', 'fuse(']):
                initial_fused.add(mid)

# ─── Phase 1: 增强测试 ───
stress_killed = {}
stress_survived = {}
policy_kills = Counter()
policy_kill_ops = defaultdict(Counter)

for f in STRESS_DIR.glob('*.json'):
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    mid = data.get('mutant_id', '')
    op = data.get('operator_name', '')
    if data.get('killed'):
        kp = data.get('killing_policy', '')
        kl = data.get('killing_layer', '')
        stress_killed[mid] = {'op': op, 'policy': kp, 'layer': kl}
        if kp:
            policy_kills[kp] += 1
            policy_kill_ops[kp][op] += 1
    else:
        stress_survived[mid] = op

# ─── Phase 2: LLM 分析 ───
llm_killed = {}
llm_survived = defaultdict(list)
killable_map = {}

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
            'op': op,
            'strategy': data.get('kill_strategy', '')[:200],
            'rule': (data.get('test_construction_rule') or {}).get('rule_name', ''),
        }
    elif data.get('status') == 'survived':
        cluster = data.get('cluster_label', 'Uncategorized')
        reason = data.get('survival_reason', '')
        killable = data.get('killable', None)
        kn = data.get('kernel_name', '')
        llm_survived[cluster].append({
            'id': mid, 'op': op, 'kernel': kn,
            'reason': reason, 'killable': killable,
        })
        killable_map[mid] = killable

# ─── 生成文档 ───
L = []
def w(s=''):
    L.append(s)

w('# 存活变异体分析报告')
w()
w('> 基于三阶段实验数据（初始变异测试 → 4层增强测试 → LLM迭代分析）的综合分析。')
w()

# ══════════════ Section 1 ══════════════
w('## 一、三阶段杀伤汇总')
w()
killed_p0 = status_counter['killed']
survived_p0 = len(initial_survived)
w('| 阶段 | 输入变异体 | 杀死 | 存活 | 增量杀伤率 |')
w('|------|----------|------|------|-----------|')
w(f'| Phase 0: 初始变异测试 | {total_mutants} (有效) | {killed_p0} | {survived_p0} | — |')
w(f'| Phase 1: 4层增强测试 | {survived_p0} | {len(stress_killed)} | {len(stress_survived)} | {len(stress_killed)/survived_p0*100:.1f}% |')
total_final = sum(len(v) for v in llm_survived.values())
w(f'| Phase 2: LLM 分析 | {len(stress_survived)} | {len(llm_killed)} | {total_final} | {len(llm_killed)/max(1,len(stress_survived))*100:.1f}% |')
w()
w(f'> 另有 {status_counter.get("stillborn",0)} 个 stillborn（编译失败）、{status_counter.get("equivalent",0)} 个 equivalent（静态等价）不计入有效变异体。')
w()

# ══════════════ Section 2 ══════════════
w('## 二、Phase 1 增强测试：各策略杀伤效果')
w()
w('| 策略 | 杀死数 | 主要针对算子 |')
w('|------|--------|------------|')
for pol, cnt in policy_kills.most_common():
    ops = ', '.join(f'{o}({c})' for o, c in policy_kill_ops[pol].most_common(3))
    w(f'| `{pol}` | {cnt} | {ops} |')
w()
w('> 14 个策略中 6 个杀伤为 0：`all_positive`, `alternating_sign`, `denormals`, `head_heavy`, `mixed_extremes`, `near_overflow`。')
w()

# ══════════════ Section 3 ══════════════
w('## 三、最终存活变异体 — 存活原因与变异体对应关系')
w()
w(f'共 **{total_final}** 个变异体最终存活，DeepSeek-R1 将其归为 **{len(llm_survived)}** 个类别：')
w()

# 总览表
w('### 3.0 总览')
w()
w('| 类别 | 数量 | 占比 | 不可杀 | 理论可杀 | 主要算子 |')
w('|------|------|------|--------|---------|---------|')
for cluster in sorted(llm_survived.keys(), key=lambda c: -len(llm_survived[c])):
    entries = llm_survived[cluster]
    op_counter = Counter(e['op'] for e in entries)
    ops_str = ', '.join(f'{o}({c})' for o, c in op_counter.most_common(3))
    pct = len(entries) / max(1, total_final) * 100
    unk = sum(1 for e in entries if e['killable'] == False)
    kbl = sum(1 for e in entries if e['killable'] == True)
    w(f'| {cluster} | {len(entries)} | {pct:.1f}% | {unk} | {kbl} | {ops_str} |')
w()

# 逐类详述
idx = 0
for cluster in sorted(llm_survived.keys(), key=lambda c: -len(llm_survived[c])):
    idx += 1
    entries = llm_survived[cluster]
    w(f'### 3.{idx} {cluster} ({len(entries)} 个)')
    w()

    # 典型原因
    rep_reason = entries[0]['reason'][:400].replace('\n', ' ')
    w(f'**典型存活机制**: {rep_reason}')
    w()

    # 按算子分组
    by_op = defaultdict(list)
    for e in entries:
        by_op[e['op']].append(e)

    for op in sorted(by_op.keys(), key=lambda o: -len(by_op[o])):
        op_ents = by_op[op]
        ids_str = ', '.join(f'`{e["id"]}`' for e in op_ents)
        w(f'- **{op}** ({len(op_ents)} 个): {ids_str}')

    # 融合标注
    fused_in = [e['id'] for e in entries if e['id'] in initial_fused]
    if fused_in:
        w(f'- ⚠ 其中 **{len(fused_in)} 个**来自融合 kernel（含 fused/fusion 代码）')
    w()

# ══════════════ Section 4 ══════════════
w('## 四、算子融合（Operator Fusion）专项分析')
w()

# 哪些 fused 变异体被杀了、哪些存活
fused_stress_killed = [m for m in initial_fused if m in stress_killed]
fused_llm_killed = [m for m in initial_fused if m in llm_killed]
fused_final_survived = []
for cluster, entries in llm_survived.items():
    for e in entries:
        if e['id'] in initial_fused:
            fused_final_survived.append({**e, 'cluster': cluster})

w(f'初始 {survived_p0} 个存活变异体中，**{len(initial_fused)} 个**来自含 fused/fusion 关键词的 kernel（占 {len(initial_fused)/survived_p0*100:.1f}%）。')
w()
w(f'| 阶段 | 杀死 | 存活 |')
w(f'|------|------|------|')
w(f'| Phase 1 增强测试 | {len(fused_stress_killed)} | {len(initial_fused) - len(fused_stress_killed)} |')
remaining_after_stress = len(initial_fused) - len(fused_stress_killed)
w(f'| Phase 2 LLM 分析 | {len(fused_llm_killed)} | {len(fused_final_survived)} |')
w()

fused_by_kernel = defaultdict(list)
for e in fused_final_survived:
    fused_by_kernel[e['kernel']].append(e)

w('### 4.1 最终存活的融合 kernel 变异体')
w()
fused_cluster_counter = Counter(e['cluster'] for e in fused_final_survived)
w('按存活原因:')
w()
w('| 存活原因 | 数量 |')
w('|---------|------|')
for cl, cnt in fused_cluster_counter.most_common():
    w(f'| {cl} | {cnt} |')
w()

w('按 kernel:')
w()
for kn in sorted(fused_by_kernel.keys()):
    elist = fused_by_kernel[kn]
    ops = Counter(e['op'] for e in elist)
    clusters = Counter(e['cluster'] for e in elist)
    w(f'- **{kn}** ({len(elist)} 个): 算子 {dict(ops)}，原因 {dict(clusters)}')
w()

w('### 4.2 算子融合如何掩盖变异？')
w()
w('融合 kernel 将多个 PyTorch 操作合并到一个 CUDA kernel 中，中间结果不经过 global memory。')
w('这导致以下掩盖机制：')
w()
w('1. **中间值截断吸收**: 融合链 `add → relu → mul` 中，add 的微扰可能被 relu 截断（负值→0）')
w('2. **精度差异内部消化**: cast_remove 导致的精度差在融合链传播时，被后续操作的精度要求重新对齐')
w('3. **边界检查冗余**: 融合 kernel 通常有统一的边界检查，mask_boundary 变异被冗余检查抵消')
w()
w('### 4.3 能否杀死融合 kernel 的存活变异体？')
w()
w('策略方向（不改 shape）：')
w()
w('| 策略 | 原理 | 目标算子 | 可行性 |')
w('|------|------|---------|--------|')
w('| `near_relu_zero` | 值域集中在 relu/激活函数的零点附近 [-0.01, 0.01] | const_perturb, arith_replace | 中 |')
w('| `saturation_boundary` | 值域在 sigmoid/tanh 饱和区边界 (±4~±6) | epsilon_modify, const_perturb | 中 |')
w('| `anti_cancellation` | 构造不对称输入使中间值不会被后续操作抵消 | arith_replace, cast_remove | 低 |')
w()
w('**核心困难**: 融合 kernel 的中间计算对外不可观测（封装在 kernel 内部），无法精确控制中间值。')
w('上述策略只能间接影响中间值，效果取决于具体融合结构。')
w()

# ══════════════ Section 5 ══════════════
w('## 五、未覆盖的规则与盲区')
w()

unkillable = sum(1 for mid, k in killable_map.items() if k == False)
killable_not_killed = sum(1 for mid, k in killable_map.items() if k == True)

w(f'LLM 判定结果:')
w(f'- **不可杀死** (killable=False): {unkillable} 个 — 在 fixed-shape 约束下语义等价')
w(f'- **理论可杀但未杀** (killable=True): {killable_not_killed} 个 — **潜在规则盲区**')
w()

w('### 5.1 "理论可杀但未杀"的 53 个变异体')
w()
killable_entries = []
for cluster, entries in llm_survived.items():
    for e in entries:
        if e['killable'] == True:
            killable_entries.append({**e, 'cluster': cluster})

ke_by_cluster = defaultdict(list)
for e in killable_entries:
    ke_by_cluster[e['cluster']].append(e)

for cl in sorted(ke_by_cluster.keys(), key=lambda c: -len(ke_by_cluster[c])):
    elist = ke_by_cluster[cl]
    ops = Counter(e['op'] for e in elist)
    w(f'**{cl}** ({len(elist)} 个): {dict(ops)}')

    # 对每个 cluster，找出该 cluster 中可杀变异体需要什么
    for e in elist[:2]:
        reason_short = e['reason'][:150].replace('\n',' ')
        w(f'  - `{e["id"]}`: {reason_short}...')
    if len(elist) > 2:
        w(f'  - ...还有 {len(elist)-2} 个')
    w()

w('### 5.2 盲区分类与对策')
w()
w('| 盲区类型 | 对应存活类别 | 变异体数 | 当前策略的局限 | 可能的对策 |')
w('|---------|-----------|---------|-------------|----------|')
w('| 值域不够极端 | Numerical Tolerance | 9 | 值虽极端但差异仍在 tolerance 内 | `epsilon_critical_norm`: 精确控制范数使 epsilon 成为主因 |')
w('| OOB 读到零值 | OOB Non-Observability | 14 | 无法控制 OOB 内存内容 | 理论上需要内存布局控制，超出 value-only 范围 |')
w('| grid-stride 吸收 | Kernel Design Resilience | 12 | grid-stride loop 使 block 数无关紧要 | 在 fixed-shape 下无对策（属于设计等价） |')
w('| 测试框架局限 | Test Framework Limitations | 9 | original 和 mutant 都失败 | 需更宽松的 tolerance 策略或分段测试 |')
w('| 竞态未触发 | Race Condition | 5 | 10 次重复不够 | 增加重复次数 / 更大 block_size |')
w('| 算法不变性 | Algorithmic Invariance | 2 | 算法本身对变异不敏感 | 属于真等价变异体 |')
w('| 无效变异 | Ineffective Mutation | 2 | 变异不改变执行路径 | 属于真等价变异体 |')
w()

w('### 5.3 当前可实施的新增策略（不改 shape）')
w()
w('| # | 策略名 | 构造方式 | 目标算子 | 预期增量 | 来源 |')
w('|---|--------|---------|---------|---------|------|')
w('| 1 | `epsilon_critical_norm` | L2 范数 ∈ [1e-9, 1e-2] | epsilon_modify, const_perturb | ~5-10 | LLM rule |')
w('| 2 | `reduction_identity_extreme` | 整行 < -1e10 | init_modify | ~1-3 | LLM rule |')
w('| 3 | `near_activation_zero` | 50% 值 ∈ [-0.01, 0.01] | 融合 kernel 各算子 | ~2-5 | 导师建议 |')
w('| 4 | `block_sparse` | 连续块为零 + 其余大值 | mask_boundary | ~1-3 | 导师建议 |')
w()
w('> **注**: 以上估计基于 LLM 的 killable 判定和存活原因分析。实际效果需重新运行实验验证。')

content = '\n'.join(L)
OUT.write_text(content, encoding='utf-8')
print(f'文档已生成: {OUT}')
print(f'  总行数: {len(L)}')
