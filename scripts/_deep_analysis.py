import json, os, sys, re
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

details_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\llm_analysis_results\details'

# --- Part 1: Kill strategy pattern extraction ---
kill_pattern_counter = Counter()
kill_details = []

for f in os.listdir(details_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(details_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    status = data.get('status', '')
    op = data.get('operator', '')

    if status == 'killed_by_llm':
        ks = data.get('kill_strategy', '')
        tcr = data.get('test_construction_rule', {})
        rule_name = tcr.get('rule_name', '') if isinstance(tcr, dict) else ''
        kill_details.append({
            'id': data.get('mutant_id', ''),
            'op': op,
            'rule': rule_name,
            'strategy_short': ks[:120]
        })

        # Classify the strategy pattern
        ks_lower = ks.lower()
        if 'shape' in ks_lower or 'dimension' in ks_lower or 'size' in ks_lower:
            kill_pattern_counter['Shape/Dimension manipulation'] += 1
        if 'non-multiple' in ks_lower or 'not a multiple' in ks_lower or 'not divisible' in ks_lower:
            kill_pattern_counter['Non-aligned sizes'] += 1
        if 'small' in ks_lower or 'minimal' in ks_lower or 'single element' in ks_lower or '1 element' in ks_lower:
            kill_pattern_counter['Minimal/small inputs'] += 1
        if 'overflow' in ks_lower:
            kill_pattern_counter['Integer overflow'] += 1
        if 'multiple block' in ks_lower or 'force multiple' in ks_lower or 'multi-block' in ks_lower:
            kill_pattern_counter['Force multiple blocks'] += 1
        if 'dtype' in ks_lower or 'precision' in ks_lower or 'cast' in ks_lower:
            kill_pattern_counter['Dtype/precision change'] += 1
        if 'boundary' in ks_lower or 'last element' in ks_lower or 'off-by-one' in ks_lower:
            kill_pattern_counter['Boundary element focus'] += 1
        if 'inner_size' in ks_lower or 'inner size' in ks_lower:
            kill_pattern_counter['Inner dimension > 1'] += 1
        if 'epsilon' in ks_lower or 'norm' in ks_lower:
            kill_pattern_counter['Epsilon-sensitive norm'] += 1

print('='*70)
print('PART 1: Kill Strategy Patterns (from 31 LLM-killed mutants)')
print('='*70)
for pat, cnt in kill_pattern_counter.most_common():
    print(f'  [{cnt:2d}] {pat}')

print()
print('--- Kill rule names ---')
rule_counter = Counter()
for kd in kill_details:
    if kd['rule']:
        rule_counter[kd['rule']] += 1
for r, c in rule_counter.most_common():
    print(f'  [{c}] {r}')

# --- Part 2: Survival reason patterns ---
print()
print('='*70)
print('PART 2: Why 218 mutants survived (detailed patterns)')
print('='*70)

survival_keyword_counter = Counter()
survived_ops = Counter()

for f in os.listdir(details_dir):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(details_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    if data.get('status') != 'survived':
        continue

    op = data.get('operator', '')
    survived_ops[op] += 1
    sr = data.get('survival_reason', '').lower()

    if 'bounds check' in sr or 'boundary check' in sr or 'guard' in sr:
        survival_keyword_counter['Boundary guards absorb mutation'] += 1
    if 'zero' in sr and ('pad' in sr or 'fill' in sr or 'init' in sr):
        survival_keyword_counter['Zero-padding/init masks OOB'] += 1
    if 'shape' in sr and ('fixed' in sr or 'constraint' in sr or 'spec' in sr):
        survival_keyword_counter['Fixed shape hides mutation'] += 1
    if '1d' in sr or 'gridDim.y = 1' in sr or 'gridDim.y=1' in sr or 'blockIdx.y' in sr:
        survival_keyword_counter['1D grid collapses index swap'] += 1
    if 'tolerance' in sr or 'atol' in sr or 'rtol' in sr or 'allclose' in sr:
        survival_keyword_counter['Numerical tolerance absorbs diff'] += 1
    if 'race' in sr or 'non-deterministic' in sr or 'nondeterministic' in sr:
        survival_keyword_counter['Race condition not triggered'] += 1
    if 'equivalent' in sr or 'identical' in sr or 'same result' in sr:
        survival_keyword_counter['Semantically equivalent'] += 1
    if 'overflow' in sr and ('not' in sr or 'doesn' in sr or 'cannot' in sr):
        survival_keyword_counter['Overflow not triggered by inputs'] += 1
    if 'block size' in sr or 'blockdim' in sr or 'block_size' in sr:
        survival_keyword_counter['Block size config masks mutation'] += 1
    if 'grid-stride' in sr or 'grid stride' in sr:
        survival_keyword_counter['Grid-stride loop absorbs extra'] += 1
    if 'warp' in sr:
        survival_keyword_counter['Warp scheduling hides race'] += 1
    if 'inner_size' in sr and ('1' in sr or '= 1' in sr or '==1' in sr):
        survival_keyword_counter['inner_size=1 collapses indexing'] += 1
    if 'vectori' in sr:
        survival_keyword_counter['Vectorization path bypasses scalar'] += 1

print('\n--- Survival patterns ---')
for pat, cnt in survival_keyword_counter.most_common():
    print(f'  [{cnt:3d}] {pat}')

print('\n--- Survived by operator ---')
for op, cnt in survived_ops.most_common():
    print(f'  [{cnt:3d}] {op}')

# --- Part 3: What current stress policies CANNOT do ---
print()
print('='*70)
print('PART 3: Gap Analysis - What stress policies miss')
print('='*70)
print("""
Current 14 policies ONLY manipulate tensor VALUES (keeping shape/dtype fixed).
They CANNOT:
  1. Change tensor shapes/dimensions (non-aligned sizes, prime sizes)
  2. Change tensor dtypes (mixed precision testing)
  3. Control CUDA launch configuration (block size, grid dims)
  4. Manipulate memory layout (contiguity, stride)
  5. Create multi-dimensional grid launches (force gridDim.y > 1)
  6. Control tensor inner dimension sizes (inner_size > 1)

LLM killed 31 mutants primarily by:
  - Changing input SHAPES (non-multiple-of-block-size)
  - Forcing multi-block launches via larger inputs
  - Switching dtypes (float16/float64 instead of float32)
  - Using minimal inputs (1 element → zero grid)
  - Controlling inner_size > 1 for multi-dim indexing
""")
