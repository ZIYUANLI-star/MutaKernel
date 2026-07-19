"""Public data types for candidate-versus-reference validation.

The validation core deliberately uses a three-valued result.  ``PASS`` means
that the observed outputs agree under the configured oracle, ``FAIL`` means a
concrete discrepancy was observed, and ``INCONCLUSIVE`` means that the test
infrastructure could not make a sound comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationStatus(str, Enum):
    """Outcome of one candidate-versus-reference validation run."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Mismatch:
    """One structural or value mismatch reported by the output oracle."""

    path: str
    kind: str
    message: str
    reference: str = ""
    candidate: str = ""


@dataclass
class OracleResult:
    """Result returned by :func:`compare_outputs`."""

    status: ValidationStatus
    compared_leaves: int = 0
    mismatches: List[Mismatch] = field(default_factory=list)
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.status is ValidationStatus.PASS


@dataclass
class PhaseTimings:
    """Wall-clock timings for each validation phase, in milliseconds."""

    state_snapshot_ms: float = 0.0
    state_sync_ms: float = 0.0
    rng_capture_ms: float = 0.0
    input_isolation_ms: float = 0.0
    reference_ms: float = 0.0
    candidate_ms: float = 0.0
    oracle_ms: float = 0.0
    cleanup_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "state_snapshot_ms": self.state_snapshot_ms,
            "state_sync_ms": self.state_sync_ms,
            "rng_capture_ms": self.rng_capture_ms,
            "input_isolation_ms": self.input_isolation_ms,
            "reference_ms": self.reference_ms,
            "candidate_ms": self.candidate_ms,
            "oracle_ms": self.oracle_ms,
            "cleanup_ms": self.cleanup_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class ExecutionError:
    """Serializable exception information from one execution phase."""

    phase: str
    exception_type: str
    message: str


@dataclass
class ValidationResult:
    """End-to-end result from the paired validation executor."""

    status: ValidationStatus
    reason: str
    timings: PhaseTimings
    oracle: Optional[OracleResult] = None
    errors: List[ExecutionError] = field(default_factory=list)
    reference_output: Optional[Any] = field(default=None, repr=False)
    candidate_output: Optional[Any] = field(default=None, repr=False)
    reference_invocations: int = 0
    candidate_invocations: int = 0

    @property
    def passed(self) -> bool:
        return self.status is ValidationStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary without potentially large outputs."""

        return {
            "status": self.status.value,
            "reason": self.reason,
            "timings_ms": self.timings.to_dict(),
            "oracle": None if self.oracle is None else {
                "status": self.oracle.status.value,
                "compared_leaves": self.oracle.compared_leaves,
                "reason": self.oracle.reason,
                "mismatches": [
                    {
                        "path": mismatch.path,
                        "kind": mismatch.kind,
                        "message": mismatch.message,
                        "reference": mismatch.reference,
                        "candidate": mismatch.candidate,
                    }
                    for mismatch in self.oracle.mismatches
                ],
            },
            "errors": [
                {
                    "phase": error.phase,
                    "exception_type": error.exception_type,
                    "message": error.message,
                }
                for error in self.errors
            ],
            "reference_invocations": self.reference_invocations,
            "candidate_invocations": self.candidate_invocations,
        }
