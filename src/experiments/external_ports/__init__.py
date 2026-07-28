"""E3 external-baseline protocol ports under the unified 32-call budget.

Blueprint Table 2 / §5.1.2: B7 (robust-kbench), B8 (KernelBenchX) and B9
(gpuemu seeded differential fuzzing) are re-implemented as input/context
generators for the unified judging pipeline, each planned to exactly 32
candidate invocations per subject — the same accounting as M-full/M-dir.
B1u is the KernelBench-inputs + unified-oracle anchor row.  Clause-by-clause
alignment checklists live in MutakernelV2/实验/补充实验数据/E3_port对齐清单_*.md.
"""

from .b1u_anchor import B1U_CANDIDATE_RUNS, B1U_STRATEGY, plan_b1u
from .b7_robust_kbench import (
    B7_PORT_STRATEGY,
    filter_input_impact,
    filter_output_axes,
    filter_output_range,
    filter_output_std,
    plan_b7_port,
    run_output_filters,
)
from .b8_kernelbenchx import (
    B8_NATIVE_DTYPE_TOLERANCES,
    B8_PORT_POLICIES,
    B8_PORT_STRATEGY,
    b8_outlier,
    b8_standard,
    b8_uniform,
    plan_b8_port,
)
from .b9_seeded_fuzzing import (
    B9_DEFAULT_TOLERANCES,
    B9_PORT_POLICIES,
    B9_PORT_STRATEGY,
    b9_uniform10,
    calibrate_tolerance,
    compare_to_fp64,
    plan_b9_port,
    run_fp64_cpu_reference,
)
from .budget32 import (
    UNIFIED_CANDIDATE_BUDGET,
    BudgetMismatchError,
    assert_budget_matched,
    fresh_budget_state,
    total_candidate_cost,
)

# One registry for every port-namespace input family the E3 runner must know.
PORT_INPUT_GENERATORS = {**B8_PORT_POLICIES, **B9_PORT_POLICIES}

__all__ = [
    "B1U_CANDIDATE_RUNS",
    "B1U_STRATEGY",
    "B8_NATIVE_DTYPE_TOLERANCES",
    "B8_PORT_POLICIES",
    "B8_PORT_STRATEGY",
    "B7_PORT_STRATEGY",
    "B9_DEFAULT_TOLERANCES",
    "B9_PORT_POLICIES",
    "B9_PORT_STRATEGY",
    "BudgetMismatchError",
    "PORT_INPUT_GENERATORS",
    "UNIFIED_CANDIDATE_BUDGET",
    "assert_budget_matched",
    "b8_outlier",
    "b8_standard",
    "b8_uniform",
    "b9_uniform10",
    "calibrate_tolerance",
    "compare_to_fp64",
    "filter_input_impact",
    "filter_output_axes",
    "filter_output_range",
    "filter_output_std",
    "fresh_budget_state",
    "plan_b1u",
    "plan_b7_port",
    "plan_b8_port",
    "plan_b9_port",
    "run_fp64_cpu_reference",
    "run_output_filters",
    "total_candidate_cost",
]
