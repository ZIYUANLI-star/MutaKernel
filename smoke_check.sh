#!/bin/bash
set -e

VENV="/home/kbuser/projects/KernelBench-0/.venv"
KB="/home/kbuser/projects/KernelBench-0"
MK="/home/kbuser/projects/MutaKernel"

source "$VENV/bin/activate"

echo "=== Environment ==="
which python
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

echo ""
echo "=== Check KernelBench data ==="
ls "$KB/runs/"
python -c "
import json
with open('$KB/runs/iter_full_l1_caesar_paper_v2/eval_results.json') as f:
    data = json.load(f)
total = len(data)
correct = sum(1 for v in data.values() if v.get('correctness'))
print(f'L1: total={total}, correct={correct}')
keys = list(data.keys())[:3]
for k in keys:
    print(f'  key example: {k}')
"

echo ""
echo "=== Check L1 problem files ==="
ls "$KB/KernelBench/KernelBench/level1/" | head -5

echo ""
echo "=== Check a generated kernel ==="
python -c "
import json, os
with open('$KB/runs/iter_full_l1_caesar_paper_v2/eval_results.json') as f:
    data = json.load(f)
for k, v in data.items():
    if v.get('correctness'):
        print(f'First correct kernel: {k}')
        run_dir = '$KB/runs/iter_full_l1_caesar_paper_v2/level1/' + k
        if os.path.isdir(run_dir):
            files = sorted(os.listdir(run_dir))
            print(f'  dir: {run_dir}')
            print(f'  files: {files[:5]}')
        break
"

echo ""
echo "=== Smoke test: import MutaKernel modules ==="
cd "$MK"
python -c "
import sys
sys.path.insert(0, '.')
from src.mutengine.operators import get_all_operators, get_operators_by_category
ops = get_all_operators()
print(f'Total operators registered: {len(ops)}')
for cat in ['A','B','C','D']:
    c = get_operators_by_category(cat)
    print(f'  Category {cat}: {len(c)} operators')

from src.mutengine.parser.triton_parser import TritonParser
from src.mutengine.parser.cuda_parser import CudaParser
print('Parsers imported OK')

from src.models import MutationTestResult, KernelInfo
print('Models imported OK')
"

echo ""
echo "=== Smoke test: generate mutants on a real kernel ==="
python -c "
import sys, json, os
sys.path.insert(0, '.')

from src.bridge.eval_bridge import KernelBenchBridge
from src.mutengine.operators import get_all_operators

bridge = KernelBenchBridge('$KB')
results = bridge.load_eval_results(1)

# Find first correct kernel
kernel = None
for k, v in results.items():
    if v.get('correctness'):
        ki = bridge.load_kernel_info(1, k)
        if ki is not None:
            kernel = ki
            break

if kernel is None:
    print('ERROR: No correct kernel found')
    exit(1)

print(f'Testing kernel: L{kernel.level} P{kernel.problem_id} ({kernel.problem_name})')
print(f'Language: {kernel.language}')
print(f'Code length: {len(kernel.kernel_code)} chars')
print(f'Code first 3 lines:')
for line in kernel.kernel_code.splitlines()[:3]:
    print(f'  {line}')

import ast
try:
    tree = ast.parse(kernel.kernel_code)
    print(f'AST parse: OK')
except SyntaxError as e:
    tree = None
    print(f'AST parse: FAILED ({e})')

all_ops = get_all_operators()
total_mutants = 0
for op in all_ops:
    try:
        mutants = op.generate_mutants(kernel.kernel_code, f'L1_P{kernel.problem_id}', tree)
        if mutants:
            print(f'  {op.name} ({op.category}): {len(mutants)} mutants')
            total_mutants += len(mutants)
    except Exception as e:
        print(f'  {op.name}: ERROR {e}')

print(f'Total mutants generated: {total_mutants}')
"

echo ""
echo "=== DONE ==="
