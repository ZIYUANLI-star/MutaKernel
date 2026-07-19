"""Pure budget accounting for compute-matched validation strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BudgetLimit:
    """Maximum candidate invocations and/or parent-observed wall time."""

    max_candidate_runs: Optional[int] = None
    max_wall_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_candidate_runs is not None:
            if isinstance(self.max_candidate_runs, bool) or not isinstance(self.max_candidate_runs, int):
                raise TypeError("max_candidate_runs must be an integer or None")
            if self.max_candidate_runs < 0:
                raise ValueError("max_candidate_runs must be non-negative")
        if self.max_wall_ms is not None:
            if not math.isfinite(self.max_wall_ms) or self.max_wall_ms < 0:
                raise ValueError("max_wall_ms must be finite and non-negative")


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class BudgetState:
    """Immutable budget usage.

    No clock or I/O is consulted here.  A runner measures elapsed wall time and
    passes it to :meth:`charge`.  Candidate runs are a hard pre-execution cap.
    Actual wall duration may overshoot the wall limit by one invocation because
    it is known only after that invocation completes; subsequent starts are
    rejected.
    """

    limit: BudgetLimit
    candidate_runs: int = 0
    wall_ms: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.candidate_runs, bool) or not isinstance(self.candidate_runs, int):
            raise TypeError("candidate_runs must be an integer")
        if self.candidate_runs < 0:
            raise ValueError("candidate_runs must be non-negative")
        if not math.isfinite(self.wall_ms) or self.wall_ms < 0:
            raise ValueError("wall_ms must be finite and non-negative")
        if (
            self.limit.max_candidate_runs is not None
            and self.candidate_runs > self.limit.max_candidate_runs
        ):
            raise ValueError("candidate_runs exceeds its hard limit")

    @property
    def remaining_candidate_runs(self) -> Optional[int]:
        if self.limit.max_candidate_runs is None:
            return None
        return max(0, self.limit.max_candidate_runs - self.candidate_runs)

    @property
    def remaining_wall_ms(self) -> Optional[float]:
        if self.limit.max_wall_ms is None:
            return None
        return max(0.0, self.limit.max_wall_ms - self.wall_ms)

    @property
    def exhausted(self) -> bool:
        candidate_exhausted = (
            self.limit.max_candidate_runs is not None
            and self.candidate_runs >= self.limit.max_candidate_runs
        )
        wall_exhausted = (
            self.limit.max_wall_ms is not None
            and self.wall_ms >= self.limit.max_wall_ms
        )
        return candidate_exhausted or wall_exhausted

    def can_start(
        self,
        *,
        candidate_runs: int = 1,
        estimated_wall_ms: float = 0.0,
    ) -> BudgetDecision:
        if isinstance(candidate_runs, bool) or not isinstance(candidate_runs, int):
            raise TypeError("candidate_runs must be an integer")
        if candidate_runs < 0:
            raise ValueError("candidate_runs must be non-negative")
        if not math.isfinite(estimated_wall_ms) or estimated_wall_ms < 0:
            raise ValueError("estimated_wall_ms must be finite and non-negative")

        remaining_runs = self.remaining_candidate_runs
        if remaining_runs is not None and candidate_runs > remaining_runs:
            return BudgetDecision(False, "candidate_runs")

        remaining_wall = self.remaining_wall_ms
        if remaining_wall is not None:
            if remaining_wall <= 0 and (candidate_runs > 0 or estimated_wall_ms > 0):
                return BudgetDecision(False, "wall_ms")
            if estimated_wall_ms > remaining_wall:
                return BudgetDecision(False, "wall_ms")
        return BudgetDecision(True)

    def charge(self, *, candidate_runs: int = 1, wall_ms: float = 0.0) -> "BudgetState":
        if not math.isfinite(wall_ms) or wall_ms < 0:
            raise ValueError("wall_ms must be finite and non-negative")
        decision = self.can_start(candidate_runs=candidate_runs)
        if not decision.allowed:
            raise RuntimeError(f"budget exhausted: {decision.reason}")
        return BudgetState(
            limit=self.limit,
            candidate_runs=self.candidate_runs + candidate_runs,
            wall_ms=self.wall_ms + wall_ms,
        )
