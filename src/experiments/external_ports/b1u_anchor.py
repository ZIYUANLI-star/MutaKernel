"""B1u anchor row: KernelBench inputs under the unified oracle.

Blueprint §5.3 (2x2 oracle factorial): B1u keeps the vendored KernelBench
protocol's *inputs* — five IID ``get_inputs()`` draws, eval mode — and swaps
only the oracle for the unified dtype-aware judging pipeline
(``src.validation.compare_outputs``).  It is a descriptive anchor, not a
budget-matched row (raising the budget alone is the separate B3 IID-32 rung
of the strengthening ladder, already registered in
``configs/fse_strategy_matrix.json``).
"""

from __future__ import annotations

from typing import List

from src.experiments.strategy import StrategySpec, TestCaseSpec

B1U_CANDIDATE_RUNS = 5
B1U_SEEDS = (0, 1, 2, 3, 4)

B1U_STRATEGY = StrategySpec(
    name="b1u-kernelbench-inputs-unified-oracle",
    version="1",
    parameters={
        "comparison_role": "oracle_factorial_anchor",
        "sampling_family": "kernelbench_default",
        "oracle": "unified",
        "budget_matched": False,
        "candidate_runs": B1U_CANDIDATE_RUNS,
    },
)


def plan_b1u(subject_id: str) -> List[TestCaseSpec]:
    return [
        TestCaseSpec(
            subject_id=subject_id,
            strategy=B1U_STRATEGY,
            policy="iid",
            seed=seed,
            mode="eval",
            scope="anchor",
            parameters={},
        )
        for seed in B1U_SEEDS
    ]
