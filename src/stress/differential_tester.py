"""Differential Tester: Per-dimension result tracking for enhanced testing.

Supports the dual-track (main + config) per-dimension result structure
where each test dimension records its results independently.
Cross-dimension early stopping is disabled (see StressEnhance Plan §5.1.6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PolicyResult:
    """Result of a single policy × seed test round (used by worker comm)."""
    policy_name: str
    original_ok: bool
    mutant_ok: bool
    ref_ok: bool
    error: str = ""
    time_ms: float = 0.0


@dataclass
class StressTestResult:
    """Per-mutant enhanced testing result with per-dimension tracking.

    Each deterministic test dimension records its results independently in
    main_track or config_track dicts.  LLM iterative analysis results are
    stored separately in llm_analysis (Phase 2 Step 3).
    """
    mutant_id: str
    operator_name: str
    operator_category: str
    kernel_name: str
    site_node_type: str = ""
    total_time_ms: float = 0.0
    original_failures: List[str] = field(default_factory=list)

    main_track: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config_track: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    llm_analysis: Dict[str, Any] = field(default_factory=dict)
    _kill_order: List[str] = field(default_factory=list)

    def record_dimension(self, track: str, dimension: str, result: Dict[str, Any]):
        """Record the result of a completed deterministic test dimension."""
        target = self.main_track if track == "main" else self.config_track
        target[dimension] = result
        if result.get("killed"):
            self._kill_order.append(dimension)
        for f in result.get("original_failures", []):
            if f not in self.original_failures:
                self.original_failures.append(f)

    def record_llm_analysis(self, result: Dict[str, Any]):
        """Record LLM iterative analysis result (Phase 2 Step 3)."""
        self.llm_analysis = result
        if result.get("killed"):
            self._kill_order.append("llm_iterative_analysis")

    @property
    def deterministic_killed(self) -> bool:
        """Whether any deterministic dimension (Step 1+2) killed the mutant."""
        for d in self.main_track.values():
            if d.get("killed"):
                return True
        for d in self.config_track.values():
            if d.get("killed"):
                return True
        return False

    @property
    def llm_killed(self) -> bool:
        """Whether LLM iterative analysis (Step 3) killed the mutant."""
        return bool(self.llm_analysis.get("killed"))

    @property
    def any_killed(self) -> bool:
        return self.deterministic_killed or self.llm_killed

    @property
    def first_kill_mode(self) -> Optional[str]:
        return self._kill_order[0] if self._kill_order else None

    def get_kill_summary(self) -> Dict[str, Any]:
        main_killed_by = [k for k, v in self.main_track.items() if v.get("killed")]
        config_killed_by = [k for k, v in self.config_track.items() if v.get("killed")]
        return {
            "deterministic_killed": self.deterministic_killed,
            "llm_killed": self.llm_killed,
            "main_track_killed_by": main_killed_by,
            "config_track_killed_by": config_killed_by,
            "llm_killing_round": self.llm_analysis.get("killing_round", 0),
            "total_dimensions_executed": len(self.main_track) + len(self.config_track),
            "total_dimensions_killed": len(main_killed_by) + len(config_killed_by),
            "final_killed": self.any_killed,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutant_id": self.mutant_id,
            "operator_name": self.operator_name,
            "operator_category": self.operator_category,
            "kernel_name": self.kernel_name,
            "any_killed": self.any_killed,
            "first_kill_mode": self.first_kill_mode,
            "main_track": self.main_track,
            "config_track": self.config_track,
            "llm_iterative_analysis": self.llm_analysis,
            "kill_summary": self.get_kill_summary(),
            "site_node_type": self.site_node_type,
            "total_time_ms": round(self.total_time_ms, 1),
            "original_failures": self.original_failures,
        }


@dataclass
class StressSummary:
    """Aggregate statistics for the enhanced testing experiment."""
    total_tested: int = 0
    killed_count: int = 0
    survived_count: int = 0
    deterministic_kill_count: int = 0
    llm_kill_count: int = 0
    per_dimension_kills: Dict[str, int] = field(default_factory=dict)
    per_policy_kills: Dict[str, int] = field(default_factory=dict)
    multi_dimension_kill_count: int = 0
    llm_rounds_distribution: Dict[int, int] = field(default_factory=dict)
    results: List[StressTestResult] = field(default_factory=list)

    def add_result(self, r: StressTestResult):
        self.results.append(r)
        self.total_tested += 1

        ks = r.get_kill_summary()
        all_killed_dims = ks["main_track_killed_by"] + ks["config_track_killed_by"]

        if r.any_killed:
            self.killed_count += 1
            if r.deterministic_killed:
                self.deterministic_kill_count += 1
            if r.llm_killed:
                self.llm_kill_count += 1
                kr = ks.get("llm_killing_round", 0)
                if kr:
                    self.llm_rounds_distribution[kr] = (
                        self.llm_rounds_distribution.get(kr, 0) + 1)
                self.per_dimension_kills["llm_iterative_analysis"] = (
                    self.per_dimension_kills.get("llm_iterative_analysis", 0) + 1)
            for dim in all_killed_dims:
                self.per_dimension_kills[dim] = self.per_dimension_kills.get(dim, 0) + 1
            for dim_result in list(r.main_track.values()) + list(r.config_track.values()):
                if dim_result.get("killed"):
                    kp = dim_result.get("killing_policy")
                    if kp:
                        self.per_policy_kills[kp] = self.per_policy_kills.get(kp, 0) + 1
            if len(all_killed_dims) > 1:
                self.multi_dimension_kill_count += 1
        else:
            self.survived_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tested": self.total_tested,
            "killed_count": self.killed_count,
            "survived_count": self.survived_count,
            "kill_rate": round(self.killed_count / max(1, self.total_tested), 4),
            "deterministic_kill_count": self.deterministic_kill_count,
            "llm_kill_count": self.llm_kill_count,
            "llm_rounds_distribution": self.llm_rounds_distribution,
            "per_dimension_kills": self.per_dimension_kills,
            "per_policy_kills": self.per_policy_kills,
            "multi_dimension_kill_count": self.multi_dimension_kill_count,
        }
