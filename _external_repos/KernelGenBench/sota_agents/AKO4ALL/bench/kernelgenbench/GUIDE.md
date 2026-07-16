# KernelGenBench Benchmark

Evaluation script for AKO4ALL using the KernelGenBench verification framework. Supports standalone Triton kernels without requiring `class Model` wrapping.

## Setup

When `bench/` contains `kernelgenbench/`, use `bench/kernelgenbench/bench.py` as the benchmark. The kernel format is:

- **Standalone kernel format**: The kernel is a standalone Triton Python file with kernel functions and operator implementations. No `class Model(nn.Module)` wrapping is needed.
- The benchmark tool calls `verify_single.py` which natively supports Triton kernels and compares against PyTorch native implementations.

**Bench command** (for TASK.md step 4):
```
python bench/kernelgenbench/bench.py --solution solution/kernel.py --op <OP> --dataset <DATASET> --verbose
```

The `<OP>` and `<DATASET>` values are provided in the task prompt or TASK.md context.

## Output Format

Each run prints structured lines (parsed by the agent):

```
COMPILED: True
CORRECT: True
RUNTIME: 0.4523
REF_RUNTIME: 1.2301
SPEEDUP: 2.7197x
TESTS: 10/10
```

- **COMPILED** — whether the solution compiled successfully
- **CORRECT** — whether all test cases passed (outputs match PyTorch reference)
- **RUNTIME** — solution kernel mean execution time in milliseconds
- **REF_RUNTIME** — PyTorch native mean execution time in milliseconds
- **SPEEDUP** — `REF_RUNTIME / RUNTIME`
- **TESTS** — number of passed tests / total tests

Exit code: `0` = correct, `1` = incorrect or failed.

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--solution` | (required) | Path to kernel file |
| `--op` | (required) | Operator name (e.g., `aten___index_put_impl_`) |
| `--dataset` | (required) | Dataset name (e.g., `KernelGenBench`) |
| `--verbose` | off | Print detailed debug info |

## Solution File Requirements

- The solution file must be a valid standalone Triton kernel file.
- It should implement the operator specified by `--op`.
- No `class Model`, `get_inputs()`, or `get_init_inputs()` is needed — the verification framework handles test case generation and execution.

## Correctness Validation

- The benchmark runs all parametrized test cases for the operator.
- Each test case compares the kernel output against PyTorch native implementation.
- Tolerances are automatically determined based on dtype (float32: 1e-4, float16/bfloat16: 1e-2).
- All test cases must pass for `CORRECT: True`.
