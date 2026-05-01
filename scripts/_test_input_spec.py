"""Quick test: verify get_input_spec_from_problem works correctly."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts._pilot_llm20 import get_input_spec_from_problem

test_files = [
    "/home/kbuser/projects/KernelBench-0/KernelBench/level1/100_HingeLoss.py",
    "/home/kbuser/projects/KernelBench-0/KernelBench/level1/23_Softmax.py",
    "/home/kbuser/projects/KernelBench-0/KernelBench/level1/24_LogSoftmax.py",
    "/home/kbuser/projects/KernelBench-0/KernelBench/level1/27_SELU_.py",
    "/home/kbuser/projects/KernelBench-0/KernelBench/level1/28_HardSigmoid.py",
]

for pf in test_files:
    print(f"\n--- {Path(pf).name} ---")
    spec = get_input_spec_from_problem(pf)
    print(spec)
