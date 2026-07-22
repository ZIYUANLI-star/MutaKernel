"""RIPR escape-mechanism classification for confirmed validator gaps (M7).

Given the stress case that produced a SPEC_VIOLATION for a probe the baseline
validator accepted, classify *which link of the fault-propagation chain*
(Reachability - Infection - Propagation - Revealability) the baseline failed
at, and therefore which observation axis it was missing.

Decision tree: 方法V2_07 §3.4.  The classifier is deliberately mechanical so
that map derivation is reproducible; conflicts with independent human audit
reason categories are surfaced, never silently overridden.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.stress.policy_metadata import POLICY_TARGET_FAULT_CLASSES

ACTIVATION_FAILURE_VALUE = "ACTIVATION_FAILURE_VALUE"
MASKING_FAILURE_PRECISION = "MASKING_FAILURE_PRECISION"
REACHABILITY_FAILURE_MODE = "REACHABILITY_FAILURE_MODE"
REACHABILITY_FAILURE_CONFIG = "REACHABILITY_FAILURE_CONFIG"
OBSERVATION_FAILURE_NONDETERMINISM = "OBSERVATION_FAILURE_NONDETERMINISM"
ABSORPTION_FAILURE_TOLERANCE = "ABSORPTION_FAILURE_TOLERANCE"

MECHANISM_TO_MISSING_AXIS: Dict[str, str] = {
    ACTIVATION_FAILURE_VALUE: "value_distribution",
    MASKING_FAILURE_PRECISION: "dtype",
    REACHABILITY_FAILURE_MODE: "execution_mode",
    REACHABILITY_FAILURE_CONFIG: "batch_configuration",
    OBSERVATION_FAILURE_NONDETERMINISM: "repetition",
    ABSORPTION_FAILURE_TOLERANCE: "oracle_strictness",
}

MECHANISM_TO_DIMENSION: Dict[str, str] = {
    ACTIVATION_FAILURE_VALUE: "value",
    MASKING_FAILURE_PRECISION: "dtype",
    REACHABILITY_FAILURE_MODE: "training",
    REACHABILITY_FAILURE_CONFIG: "config",
    OBSERVATION_FAILURE_NONDETERMINISM: "repeated",
    ABSORPTION_FAILURE_TOLERANCE: "value",
}

_LOW_PRECISION_DTYPES = frozenset({"float16", "bfloat16", "half", "torch.float16", "torch.bfloat16"})


def dimension_of_case(case: Mapping[str, Any]) -> str:
    """Assign a stress case to its search dimension (方法V2_06 §4)."""
    mode = str(case.get("mode", "eval"))
    parameters = case.get("parameters") or {}
    if mode == "config":
        return "config"
    if mode == "train":
        return "training"
    if mode == "repeated":
        return "repeated"
    if mode == "layout":
        return "layout"
    if str(parameters.get("dtype", "")) in _LOW_PRECISION_DTYPES:
        return "dtype"
    return "value"


def classify_escape(
    killing_case: Mapping[str, Any],
    audit_reason_category: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify why the baseline validator missed the fault this case exposed.

    ``killing_case`` is the CaseConfig-shaped dict of the first
    SPEC_VIOLATION.  ``audit_reason_category`` (from the independent human
    audit, e.g. ``value_insensitive``) is used only as a consistency check.
    """
    mode = str(killing_case.get("mode", "eval"))
    parameters = killing_case.get("parameters") or {}
    policy = str(killing_case.get("policy", ""))

    if mode == "config":
        mechanism = REACHABILITY_FAILURE_CONFIG
    elif mode == "train":
        mechanism = REACHABILITY_FAILURE_MODE
    elif mode == "repeated":
        mechanism = OBSERVATION_FAILURE_NONDETERMINISM
    elif str(parameters.get("dtype", "")) in _LOW_PRECISION_DTYPES:
        mechanism = MASKING_FAILURE_PRECISION
    elif policy in POLICY_TARGET_FAULT_CLASSES:
        mechanism = ACTIVATION_FAILURE_VALUE
    else:
        # An IID/identity input at default context exposed the fault while the
        # baseline still accepted the probe: the divergence existed but was
        # absorbed by the baseline's oracle (tolerance / sample count).
        mechanism = ABSORPTION_FAILURE_TOLERANCE

    result: Dict[str, Any] = {
        "mechanism": mechanism,
        "missing_axis": MECHANISM_TO_MISSING_AXIS[mechanism],
        "derived_dimension": MECHANISM_TO_DIMENSION[mechanism],
    }

    if audit_reason_category:
        expected = _AUDIT_REASON_TO_MECHANISM.get(audit_reason_category)
        result["audit_reason_category"] = audit_reason_category
        result["audit_consistent"] = expected is None or expected == mechanism
    return result


# Observable proxies from the human-audit reason taxonomy (V1 Task A) used
# for the consistency check; ``None``-mapped categories are compatible with
# several mechanisms and never counted as conflicts.
_AUDIT_REASON_TO_MECHANISM: Dict[str, Optional[str]] = {
    "value_insensitive": ACTIVATION_FAILURE_VALUE,
    "requires_config_change": REACHABILITY_FAILURE_CONFIG,
    "path_not_triggered": REACHABILITY_FAILURE_MODE,
    "predicate_unreachable": None,
    "infection_no_propagation": None,
}
