"""Counterexample Search Engine (CSE, M6) — object-agnostic verdict layer.

The execution machinery lives in the workers (scripts/_stress_worker.py,
scripts/_candidate_worker.py) on top of src.validation.  This package holds
the torch-free verdict semantics shared by audit mode and validation mode.

Spec: MutakernelV2/MutakernelV2方法修正/方法V2_06.
"""

from .verdict import (  # noqa: F401
    PASS,
    FAIL,
    INCONCLUSIVE,
    VERDICT_SPEC_VIOLATION,
    VERDICT_EXACT_DIVERGENCE_ONLY,
    VERDICT_INDISTINGUISHED,
    VERDICT_INVALID_INPUT,
    VERDICT_ACCIDENTAL_REPAIR,
    VERDICT_INCONCLUSIVE,
    three_way_verdict,
    two_way_verdict,
    verdict_from_legacy_stress_record,
)
