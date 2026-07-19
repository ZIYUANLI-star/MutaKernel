"""Reproducible experiment infrastructure for MutaKernel.

This package deliberately contains no CUDA or validation logic.  It defines
the stable identities, provenance records, budgets, and timing primitives that
experiment runners use around the validation engine.
"""

from .budget import BudgetDecision, BudgetLimit, BudgetState
from .contract import (
    ContractError,
    assert_case_in_contract,
    validate_call_inputs,
    validate_contract,
)
from .manifest import (
    ArtifactProvenance,
    DuplicateObservationError,
    ObservationLog,
    RunManifest,
    SubjectProvenance,
    canonical_json_bytes,
    sha256_file,
    stable_json_sha256,
)
from .protocol import (
    ALLOWED_SCOPES,
    ANCHOR_STRATEGY_NAME,
    ProtocolError,
    build_experiment_plan,
    plan_from_files,
    write_plan_once,
)
from .strategy import StrategySpec, TestCaseSpec, make_test_id
from .timing import TimingBreakdown

__all__ = [
    "ArtifactProvenance",
    "BudgetDecision",
    "BudgetLimit",
    "BudgetState",
    "DuplicateObservationError",
    "ContractError",
    "ObservationLog",
    "ProtocolError",
    "RunManifest",
    "StrategySpec",
    "SubjectProvenance",
    "TestCaseSpec",
    "TimingBreakdown",
    "ALLOWED_SCOPES",
    "ANCHOR_STRATEGY_NAME",
    "build_experiment_plan",
    "assert_case_in_contract",
    "canonical_json_bytes",
    "make_test_id",
    "plan_from_files",
    "sha256_file",
    "stable_json_sha256",
    "write_plan_once",
    "validate_contract",
    "validate_call_inputs",
]
