import pytest

from src.experiments.budget import BudgetLimit, BudgetState


def test_candidate_run_limit_is_hard():
    state = BudgetState(BudgetLimit(max_candidate_runs=2))
    state = state.charge(candidate_runs=1, wall_ms=3.0)
    assert state.can_start(candidate_runs=1).allowed
    state = state.charge(candidate_runs=1, wall_ms=2.0)
    assert state.exhausted
    decision = state.can_start(candidate_runs=1)
    assert not decision.allowed
    assert decision.reason == "candidate_runs"
    with pytest.raises(RuntimeError):
        state.charge(candidate_runs=1)


def test_wall_budget_can_overshoot_once_then_stops():
    state = BudgetState(BudgetLimit(max_wall_ms=10.0))
    assert state.can_start(candidate_runs=1).allowed
    state = state.charge(candidate_runs=1, wall_ms=12.0)
    assert state.wall_ms == 12.0
    assert state.exhausted
    assert state.can_start(candidate_runs=1).reason == "wall_ms"


def test_estimated_wall_time_prevents_start():
    state = BudgetState(BudgetLimit(max_wall_ms=10.0), wall_ms=8.0)
    assert state.can_start(candidate_runs=1, estimated_wall_ms=1.5).allowed
    decision = state.can_start(candidate_runs=1, estimated_wall_ms=3.0)
    assert not decision.allowed
    assert decision.reason == "wall_ms"


def test_budget_state_is_immutable_and_pure():
    initial = BudgetState(BudgetLimit(max_candidate_runs=3, max_wall_ms=100.0))
    updated = initial.charge(candidate_runs=1, wall_ms=4.5)
    assert initial.candidate_runs == 0
    assert initial.wall_ms == 0.0
    assert updated.candidate_runs == 1
    assert updated.wall_ms == 4.5
    assert updated.remaining_candidate_runs == 2
    assert updated.remaining_wall_ms == 95.5
