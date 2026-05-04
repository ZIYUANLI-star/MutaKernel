#!/bin/bash
set -e

VENV="/home/kbuser/projects/KernelBench-0/.venv"
KB="/home/kbuser/projects/KernelBench-0"
MK="/home/kbuser/projects/MutaKernel"

source "$VENV/bin/activate"
cd "$MK"

echo "=== 1. Bridge: load kernel info ==="
python -c "
import sys
sys.path.insert(0, '.')
from src.bridge.eval_bridge import KernelBenchBridge

bridge = KernelBenchBridge('$KB')

# Load first correct kernel from L1
results = bridge.load_eval_results(1)
loaded = 0
for k, v in results.items():
    if v.get('correctness'):
        ki = bridge.load_kernel_info(1, k)
        if ki is not None:
            print(f'Loaded: L{ki.level} P{ki.problem_id} ({ki.problem_name})')
            print(f'  language: {ki.language}')
            print(f'  code len: {len(ki.kernel_code)} chars')
            print(f'  ref path: {ki.reference_module_path}')
            print(f'  src path: {ki.source_path}')
            loaded += 1
            if loaded >= 3:
                break

total_correct = sum(1 for v in results.values() if v.get('correctness'))
print(f'\nTotal correct in L1: {total_correct}')
print(f'Successfully loaded: {loaded}')
"

echo ""
echo "=== 2. Generate mutants on first kernel ==="
python -c "
import sys, ast
sys.path.insert(0, '.')
from src.bridge.eval_bridge import KernelBenchBridge
from src.mutengine.operators import get_all_operators

bridge = KernelBenchBridge('$KB')
results = bridge.load_eval_results(1)

kernel = None
for k, v in results.items():
    if v.get('correctness'):
        kernel = bridge.load_kernel_info(1, k)
        if kernel is not None:
            break

print(f'Kernel: L{kernel.level} P{kernel.problem_id}')
try:
    tree = ast.parse(kernel.kernel_code)
except SyntaxError:
    tree = None

ops = get_all_operators()
total = 0
for op in ops:
    try:
        mutants = op.generate_mutants(kernel.kernel_code, f'L1_P{kernel.problem_id}', tree)
        if mutants:
            print(f'  {op.name:25s} ({op.category}): {len(mutants):4d} mutants')
            total += len(mutants)
    except Exception as e:
        print(f'  {op.name:25s}: ERROR {str(e)[:60]}')
print(f'  TOTAL: {total} mutants')
"

echo ""
echo "=== 3. Run mutation test on 1 kernel (kill/survive) ==="
python -c "
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING)

from src.bridge.eval_bridge import KernelBenchBridge
from src.mutengine.mutant_runner import MutantRunner

bridge = KernelBenchBridge('$KB')
results = bridge.load_eval_results(1)

kernel = None
for k, v in results.items():
    if v.get('correctness'):
        kernel = bridge.load_kernel_info(1, k)
        if kernel is not None:
            break

print(f'Testing: L{kernel.level} P{kernel.problem_id}')

runner = MutantRunner(
    atol=1e-2, rtol=1e-2,
    num_test_inputs=3,
    device='cuda',
    seed=42,
    categories=['A', 'B', 'C', 'D'],
)

mutants = runner.generate_mutants(kernel)
print(f'Generated {len(mutants)} mutants')

# Only test first 10 to keep it fast
test_mutants = mutants[:10]
print(f'Testing first {len(test_mutants)} mutants...')

ref_module, get_inputs_fn, get_init_inputs_fn = bridge.load_runtime_components(kernel)

result = runner.run_all_mutants(
    kernel, test_mutants, ref_module, get_inputs_fn, get_init_inputs_fn
)

print(f'\nResults:')
print(f'  Total tested: {result.total}')
print(f'  Killed:       {result.killed}')
print(f'  Survived:     {result.survived}')
print(f'  Stillborn:    {result.stillborn}')
print(f'  Score:        {result.mutation_score:.2%}')
print()
for m in test_mutants:
    print(f'  {m.operator_name:25s} L{m.site.line_start:3d} -> {m.status.value:10s} {m.error_message[:50] if m.error_message else \"\"}')

runner.cleanup()
"

echo ""
echo "=== 4. Realism validation (quick) ==="
python -c "
import sys
sys.path.insert(0, '.')
from src.bridge.eval_bridge import KernelBenchBridge
from src.mutengine.realism_validator import RealismValidator

bridge = KernelBenchBridge('$KB')
validator = RealismValidator()

# Get failed and correct kernels
failed = bridge.list_failed_kernels(1)
correct = bridge.list_correct_kernels(1)
print(f'L1: {len(failed)} failed, {len(correct)} correct')

analyzed = 0
for entry in failed[:10]:
    pk = entry['problem_key']
    gen = bridge.find_generated_kernel(1, pk)
    if gen is None:
        continue
    try:
        buggy = gen.read_text()
    except:
        continue
    bug = validator.analyze_buggy_kernel_standalone(
        bug_id=f'L1_P{pk}_fail',
        problem_id=int(pk) if pk.isdigit() else 0,
        level=1,
        buggy_code=buggy,
    )
    analyzed += 1

report = validator.generate_report()
print(f'Analyzed: {analyzed} bugs')
print(f'Covered by C/D: {report.bugs_covered_by_cd} ({report.coverage_rate_cd:.0%})')
print(f'Covered by A/B: {report.bugs_covered_by_ab}')
print(f'Not covered: {report.bugs_not_covered}')
print(f'Per-operator:')
for op, cnt in sorted(report.per_operator_realism.items(), key=lambda x: -x[1]):
    print(f'  {op}: {cnt}')
"

echo ""
echo "=== ALL SMOKE TESTS PASSED ==="
