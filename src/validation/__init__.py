"""Sound-by-default building blocks for validating generated kernels."""

from .executor import ExecutionConfig, ValidationExecutor, validate_pair
from .inputs import InputIsolationError, clone_call_inputs, clone_tree, describe_input_tree
from .oracle import OracleConfig, Tolerance, compare_outputs
from .state import (
    RNGSnapshot,
    StateSyncError,
    StateSyncReport,
    align_state_keys,
    replay_rng,
    restore_state_dict,
    snapshot_state_dict,
    strict_sync_state_dict,
)
from .types import (
    ExecutionError,
    Mismatch,
    OracleResult,
    PhaseTimings,
    ValidationResult,
    ValidationStatus,
)

__all__ = [
    "ExecutionConfig",
    "ExecutionError",
    "InputIsolationError",
    "Mismatch",
    "OracleConfig",
    "OracleResult",
    "PhaseTimings",
    "RNGSnapshot",
    "StateSyncError",
    "StateSyncReport",
    "Tolerance",
    "ValidationExecutor",
    "ValidationResult",
    "ValidationStatus",
    "align_state_keys",
    "clone_call_inputs",
    "clone_tree",
    "describe_input_tree",
    "compare_outputs",
    "replay_rng",
    "restore_state_dict",
    "snapshot_state_dict",
    "strict_sync_state_dict",
    "validate_pair",
]
