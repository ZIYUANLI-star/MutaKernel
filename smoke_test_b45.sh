#!/bin/bash
set -e

cd /home/kbuser/projects/MutaKernel
source /home/kbuser/projects/KernelBench-0/.venv/bin/activate
export PYTHONPATH="/home/kbuser/projects/MutaKernel:$PYTHONPATH"

echo "=== Block 4: MutRepair ==="
python -c "
import sys
sys.path.insert(0, '.')

from src.mutrepair.enhanced_inputs import EnhancedInputGenerator, STRATEGY_MAP
import torch

gen = EnhancedInputGenerator()
covered_ops = set(STRATEGY_MAP.keys())
print(f'  EnhancedInputGenerator covers {len(covered_ops)} operators')
for op in ['stab_remove', 'acc_downgrade', 'epsilon_modify', 'scale_modify', 'cast_remove']:
    strategies = gen.get_strategies_for_operator(op)
    print(f'    {op}: {strategies}')

dummy_inputs = lambda: [torch.randn(4, 8)]
enhanced = gen.generate_enhanced_inputs(dummy_inputs, 'stab_remove', num_per_strategy=2)
print(f'  Generated {len(enhanced)} enhanced input sets for stab_remove')
for strategy, inputs in enhanced[:2]:
    t = inputs[0]
    print(f'    {strategy}: shape={t.shape}, min={t.min():.1f}, max={t.max():.1f}')

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

from src.mutrepair.experience_store import ExperienceStore
import tempfile
from pathlib import Path
tmpdir = Path(tempfile.mkdtemp()) / 'exp_store'
store = ExperienceStore(tmpdir)
print(f'  ExperienceStore initialized (empty, {len(store.experiences)} experiences)')

print('  [PASS] Block 4 OK')
"
echo ""

echo "=== Block 5: MutEvolve ==="
python -c "
import sys, tempfile
sys.path.insert(0, '.')
from pathlib import Path

from src.mutrepair.experience_store import ExperienceStore, RepairExperience
from src.mutevolve.pattern_miner import PatternMiner
from src.mutevolve.rule_generator import RuleGenerator

tmpdir = Path(tempfile.mkdtemp()) / 'exp_store'
store = ExperienceStore(tmpdir)

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

miner = PatternMiner(store, min_frequency=3)
patterns = miner.mine_patterns()
print(f'  Mined {len(patterns)} new patterns')
for p in patterns:
    print(f'    {p.pattern_id}: freq={p.frequency}, template=\"{p.added_code_template[:50]}\"')

gen = RuleGenerator()
new_ops = gen.generate_from_patterns(patterns)
print(f'  Generated {len(new_ops)} dynamic operators')
for op in new_ops:
    print(f'    {op.name}: category={op.category}, has_regex={op._search_regex is not None}')

if new_ops:
    sample = 'x = x.clamp(min=-65504, max=65504)\ny = x + 1'
    for op in new_ops:
        sites = op.find_sites(sample)
        if sites:
            print(f'    {op.name} found {len(sites)} sites in sample')
            mutated = op.apply(sample, sites[0])
            print(f'    Mutation removes: \"{sites[0].original_code[:40]}\"')

print('  [PASS] Block 5 OK')
"
echo ""
echo "=== ALL BLOCKS PASSED ==="
