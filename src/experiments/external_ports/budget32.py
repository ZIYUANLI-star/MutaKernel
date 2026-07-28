"""Unified candidate-invocation budget for the E3 external-baseline ports.

Blueprint Table 2: every budget-matched row executes *exactly* 32 candidate
invocations per subject — the same accounting used by M-full/M-dir
(``configs/fse_strategy_matrix.json`` ``candidate_run_budget``).  Ports plan
their protocol as :class:`src.experiments.strategy.TestCaseSpec` lists whose
``candidate_run_cost`` sums to the budget; execution charges an immutable
:class:`src.experiments.budget.BudgetState` before every candidate start, so
an over-budget plan can never silently produce a table row.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from src.experiments.budget import BudgetLimit, BudgetState
from src.experiments.strategy import TestCaseSpec

UNIFIED_CANDIDATE_BUDGET = 32


class BudgetMismatchError(ValueError):
    """A budget-matched plan does not cost exactly the unified budget."""


def total_candidate_cost(cases: Iterable[TestCaseSpec]) -> int:
    return sum(case.candidate_run_cost for case in cases)


def assert_budget_matched(
    cases: Sequence[TestCaseSpec],
    budget: int = UNIFIED_CANDIDATE_BUDGET,
) -> Sequence[TestCaseSpec]:
    cost = total_candidate_cost(cases)
    if cost != budget:
        raise BudgetMismatchError(
            f"plan costs {cost} candidate invocations, budget is {budget}"
        )
    return cases


def fresh_budget_state(budget: int = UNIFIED_CANDIDATE_BUDGET) -> BudgetState:
    return BudgetState(limit=BudgetLimit(max_candidate_runs=budget))
