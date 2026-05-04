#!/bin/bash
set -e

cd /home/kbuser/projects/MutaKernel
source /home/kbuser/projects/KernelBench-0/.venv/bin/activate
export PYTHONPATH="/home/kbuser/projects/MutaKernel:$PYTHONPATH"

echo "============================================"
echo "  MutaKernel 全流程冒烟测试 (5 Blocks)"
echo "============================================"
echo ""

########################################
# Block 1: MutOperators — 算子注册与变异生成
########################################
echo "=== Block 1: MutOperators ==="
python -c "
import sys
sys.path.insert(0, '.')
from src.mutengine.operators.base import get_all_operators, get_operators_by_category

all_ops = get_all_operators()
print(f'  Total registered operators: {len(all_ops)}')

for cat in ['A', 'B', 'C', 'D']:
    ops = get_operators_by_category(cat)
    names = [o.name for o in ops]
    print(f'  Category {cat} ({len(ops)}): {names}')

assert len(all_ops) >= 16, f'Expected >=16 operators, got {len(all_ops)}'
print('  [PASS] All 16 operators registered')

# Test a C-type operator on sample code
from src.mutengine.operators.ml_semantic import StabRemove, AccDowngrade, EpsilonModify
sample = '''
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float(\"inf\"))
    row_max = tl.max(row, axis=0)
    row = row - tl.max(row, axis=0)
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0) + 1e-6
    result = numerator / denominator
    tl.store(output_ptr + row_idx * n_cols + col_offsets, result, mask=mask)
'''

sr = StabRemove()
sites = sr.find_sites(sample)
print(f'  StabRemove found {len(sites)} sites in sample softmax')
assert len(sites) > 0, 'StabRemove should find sites in softmax kernel'

em = EpsilonModify()
eps_sites = em.find_sites(sample)
print(f'  EpsilonModify found {len(eps_sites)} sites (epsilon literals)')
assert len(eps_sites) > 0, 'EpsilonModify should find 1e-6 in sample'

print('  [PASS] Block 1 OK')
"
echo ""

########################################
# Block 2: MutEngine — 完整变异测试流程
########################################
echo "=== Block 2: MutEngine ==="
python -c "
import sys, logging, time
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING)

from src.bridge.eval_bridge import KernelBenchBridge
from src.mutengine.mutant_runner import MutantRunner
from src.models import MutantStatus

bridge = KernelBenchBridge('/home/kbuser/projects/KernelBench-0')

# Load a kernel
results = bridge.load_eval_results(1)
kernel = None
for k, v in results.items():
    if v.get('correctness'):
        kernel = bridge.load_kernel_info(1, k)
        if kernel is not None:
            break

assert kernel is not None, 'Failed to load any kernel'
print(f'  Loaded: L{kernel.level} P{kernel.problem_id} ({kernel.language})')
print(f'  Code length: {len(kernel.kernel_code)} chars')

# Load runtime components
ref_module, get_inputs_fn, get_init_inputs_fn = bridge.load_runtime_components(kernel)
print(f'  Runtime components loaded OK')

# Generate mutants
runner = MutantRunner(
    atol=1e-2, rtol=1e-2,
    num_test_inputs=3,
    device='cuda',
    seed=42,
    categories=['A', 'B', 'C', 'D'],
)
mutants = runner.generate_mutants(kernel)
print(f'  Generated {len(mutants)} mutants')
assert len(mutants) > 0, 'No mutants generated'

# Show breakdown
from collections import Counter
cat_counts = Counter(m.operator_category for m in mutants)
for cat in sorted(cat_counts.keys()):
    print(f'    Category {cat}: {cat_counts[cat]}')

# Run first 5 mutants
test_mutants = mutants[:5]
print(f'  Testing first {len(test_mutants)} mutants...')
start = time.time()
result = runner.run_all_mutants(kernel, test_mutants, ref_module, get_inputs_fn, get_init_inputs_fn)
elapsed = time.time() - start
print(f'  Done in {elapsed:.1f}s')
print(f'  Results: killed={result.killed}, survived={result.survived}, stillborn={result.stillborn}')
print(f'  Score: {result.mutation_score:.2%}')

for m in test_mutants:
    err = m.error_message[:40] if m.error_message else ''
    print(f'    {m.operator_name:20s} L{m.site.line_start:3d} -> {m.status.value:10s} {err}')

assert result.total > 0, 'No mutants were tested'
print('  [PASS] Block 2 OK')
runner.cleanup()
"
echo ""

########################################
# Block 3: RealismGuard — 真实性验证
########################################
echo "=== Block 3: RealismGuard ==="
python -c "
import sys
sys.path.insert(0, '.')

from src.bridge.eval_bridge import KernelBenchBridge
from src.mutengine.realism_validator import RealismValidator

bridge = KernelBenchBridge('/home/kbuser/projects/KernelBench-0')
validator = RealismValidator()

failed = bridge.list_failed_kernels(1)
correct = bridge.list_correct_kernels(1)
print(f'  L1: {len(failed)} failed, {len(correct)} correct')

analyzed = 0
max_analyze = 15
for entry in failed[:max_analyze]:
    pk = entry['problem_key']
    gen_path = bridge.find_generated_kernel(1, pk)
    ref_path = bridge.find_problem_file(1, pk)
    if gen_path is None or ref_path is None:
        continue
    buggy = gen_path.read_text(encoding='utf-8')
    correct_code = ref_path.read_text(encoding='utf-8')
    validator.analyze_bug_from_diff(
        bug_id=f'L1_{pk}',
        problem_id=int(pk) if pk.isdigit() else 0,
        level=1,
        correct_code=correct_code,
        buggy_code=buggy,
    )
    analyzed += 1

print(f'  Analyzed: {analyzed} bugs')
report = validator.generate_report()
print(f'  Covered by C/D: {report.bugs_covered_by_cd} ({report.coverage_rate_cd:.0%})')
print(f'  Covered by A/B: {report.bugs_covered_by_ab}')
print(f'  Not covered: {report.bugs_not_covered}')
print(f'  Per-operator:')
for op, cnt in sorted(report.per_operator_realism.items(), key=lambda x: -x[1]):
    print(f'    {op}: {cnt}')

assert report.total_bugs_analyzed > 0, 'No bugs analyzed'
print('  [PASS] Block 3 OK')
"
echo ""

########################################
# Block 4: MutRepair — 修复流程验证
########################################
echo "=== Block 4: MutRepair ==="
python -c "
import sys
sys.path.insert(0, '.')

# 4a: Enhanced Input Generator
from src.mutrepair.enhanced_inputs import EnhancedInputGenerator, STRATEGY_MAP
import torch

gen = EnhancedInputGenerator()

# Check strategy coverage
covered_ops = set(STRATEGY_MAP.keys())
print(f'  EnhancedInputGenerator covers {len(covered_ops)} operators')
for op in ['stab_remove', 'acc_downgrade', 'epsilon_modify', 'scale_modify', 'cast_remove']:
    strategies = gen.get_strategies_for_operator(op)
    print(f'    {op}: {strategies}')
    assert len(strategies) > 0, f'No strategies for {op}'

# Test input generation
dummy_inputs = lambda: [torch.randn(4, 8)]
enhanced = gen.generate_enhanced_inputs(dummy_inputs, 'stab_remove', num_per_strategy=2)
print(f'  Generated {len(enhanced)} enhanced input sets for stab_remove')
for strategy, inputs in enhanced[:2]:
    t = inputs[0]
    print(f'    {strategy}: shape={t.shape}, min={t.min():.1f}, max={t.max():.1f}')
assert len(enhanced) > 0, 'No enhanced inputs generated'

# 4b: FeedbackBuilder
from src.mutrepair.feedback_builder import FeedbackBuilder, DEFECT_TAXONOMY

for mode in ['B0', 'B1', 'B2', 'B3', 'ours']:
    fb = FeedbackBuilder(mode=mode)
    prompt = fb.build_prompt(
        kernel_code='def forward(x): return x * 2',
        error_info='Max diff = 1.2e-01' if mode != 'B0' else None,
        failing_input_desc='near_overflow' if mode in ('B2','B3','ours') else None,
        failure_detail='near_overflow: value_mismatch' if mode in ('B2','B3','ours') else None,
        code_location='Line 45: .to(torch.float32)' if mode in ('B3','ours') else None,
    )
    print(f'  {mode:4s} prompt length: {len(prompt):4d} chars')

print(f'  DEFECT_TAXONOMY covers {len(DEFECT_TAXONOMY)} operators')
assert len(DEFECT_TAXONOMY) >= 9, 'Expected at least 9 taxonomy entries'

# 4c: ExperienceStore
from src.mutrepair.experience_store import ExperienceStore
import tempfile
from pathlib import Path
tmpdir = Path(tempfile.mkdtemp()) / 'exp_store'
store = ExperienceStore(tmpdir)
print(f'  ExperienceStore initialized (empty, {len(store.experiences)} experiences)')

print('  [PASS] Block 4 OK')
"
echo ""

########################################
# Block 5: MutEvolve — 规则自适应
########################################
echo "=== Block 5: MutEvolve ==="
python -c "
import sys, tempfile, os
sys.path.insert(0, '.')

from src.mutrepair.experience_store import ExperienceStore, RepairExperience
from src.mutevolve.pattern_miner import PatternMiner, MinedPattern
from src.mutevolve.rule_generator import RuleGenerator, DynamicOperator

# Create mock experience store with synthetic data
tmpdir = tempfile.mkdtemp()
store = ExperienceStore(os.path.join(tmpdir, 'exp.jsonl'))

# Simulate 5 repairs that all add 'x = x.clamp(min=-65504, max=65504)'
for i in range(5):
    exp = RepairExperience(
        kernel_id=f'L1_P{i+10}',
        problem_id=i+10,
        level=1,
        operator_name='cast_remove',
        operator_category='C',
        site_line=10+i,
        diff_lines=['+ x = x.clamp(min=-65504, max=65504)'],
        added_lines=['    x = x.clamp(min=-65504, max=65504)'],
        removed_lines=[],
        rounds=1,
    )
    store.experiences.append(exp)

# Also add 4 repairs with 'output = output.contiguous()'  
for i in range(4):
    exp = RepairExperience(
        kernel_id=f'L1_P{i+20}',
        problem_id=i+20,
        level=1,
        operator_name='layout_assume',
        operator_category='D',
        site_line=20+i,
        diff_lines=['+ output = output.contiguous()'],
        added_lines=['    output = output.contiguous()'],
        removed_lines=[],
        rounds=1,
    )
    store.experiences.append(exp)

print(f'  Created {len(store.experiences)} mock experiences')

# Mine patterns
miner = PatternMiner(store, min_frequency=3)
patterns = miner.mine_patterns()
print(f'  Mined {len(patterns)} new patterns')
for p in patterns:
    print(f'    {p.pattern_id}: freq={p.frequency}, template=\"{p.added_code_template[:50]}\"')

# Generate dynamic operators
gen = RuleGenerator()
new_ops = gen.generate_from_patterns(patterns)
print(f'  Generated {len(new_ops)} dynamic operators')
for op in new_ops:
    print(f'    {op.name}: category={op.category}, regex={op._search_regex is not None}')

# Test a dynamic operator on sample code
if new_ops:
    sample = 'x = x.clamp(min=-65504, max=65504)\ny = x + 1'
    for op in new_ops:
        sites = op.find_sites(sample)
        if sites:
            print(f'    {op.name} found {len(sites)} sites in sample code')
            mutated = op.apply(sample, sites[0])
            print(f'    Mutation: removed \"{sites[0].original_code[:40]}\"')

print('  [PASS] Block 5 OK')
"
echo ""

########################################
# Summary
########################################
echo "============================================"
echo "  ALL 5 BLOCKS PASSED"
echo "============================================"
echo ""
echo "Block 1 (MutOperators):  16 operators registered, C-type detection verified"
echo "Block 2 (MutEngine):     Full pipeline: load → mutate → compile → execute → judge"
echo "Block 3 (RealismGuard):  Real LLM bug analysis → operator coverage mapping"
echo "Block 4 (MutRepair):     Enhanced inputs + 5 feedback modes + experience store"
echo "Block 5 (MutEvolve):     Pattern mining → dynamic operator generation"
