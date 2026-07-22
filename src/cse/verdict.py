"""Five-valued three-way differential verdict (torch-free, M6 core).

Implements the single legal interpretation table of
方法V2_06 §3.2 / 方法V2_00 §3.4:

    A = ok(reference, original, x)   in-contract tolerance check
    B = ok(reference, mutant,   x)
    C = exact(original, mutant, x)   NaN-aware bitwise equality

| A    | B    | C     | verdict                 |
|------|------|-------|-------------------------|
| PASS | FAIL | *     | SPEC_VIOLATION          |
| PASS | PASS | False | EXACT_DIVERGENCE_ONLY   |
| PASS | PASS | True  | INDISTINGUISHED         |
| FAIL | FAIL | *     | INVALID_INPUT           |
| FAIL | PASS | *     | ACCIDENTAL_REPAIR       |
| any INCONCLUSIVE     | INCONCLUSIVE            |

Only SPEC_VIOLATION may enter blind-spot counts, rectification
(false-equivalent) counts, or kill statistics (证据铁律 R2).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
_STATUSES = frozenset({PASS, FAIL, INCONCLUSIVE})

VERDICT_SPEC_VIOLATION = "SPEC_VIOLATION"
VERDICT_EXACT_DIVERGENCE_ONLY = "EXACT_DIVERGENCE_ONLY"
VERDICT_INDISTINGUISHED = "INDISTINGUISHED"
VERDICT_INVALID_INPUT = "INVALID_INPUT"
VERDICT_ACCIDENTAL_REPAIR = "ACCIDENTAL_REPAIR"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


def _check_status(name: str, value: str) -> str:
    if value not in _STATUSES:
        raise ValueError(f"{name} must be one of {sorted(_STATUSES)}, got {value!r}")
    return value


def three_way_verdict(
    a_original_ok: str,
    b_mutant_ok: str,
    c_exact_equal: Optional[bool] = None,
) -> str:
    """Map the (A, B, C) primitives to exactly one five-valued verdict.

    ``c_exact_equal`` may be ``None`` only when it is irrelevant to the row
    (A != PASS or B != PASS).  For the double-PASS row the exact comparison is
    mandatory: without it the round cannot be split into INDISTINGUISHED vs
    EXACT_DIVERGENCE_ONLY and is therefore INCONCLUSIVE (fail-closed).
    """
    a = _check_status("a_original_ok", a_original_ok)
    b = _check_status("b_mutant_ok", b_mutant_ok)

    if a == INCONCLUSIVE or b == INCONCLUSIVE:
        return VERDICT_INCONCLUSIVE
    if a == PASS and b == FAIL:
        return VERDICT_SPEC_VIOLATION
    if a == PASS and b == PASS:
        if c_exact_equal is None:
            return VERDICT_INCONCLUSIVE
        return VERDICT_INDISTINGUISHED if c_exact_equal else VERDICT_EXACT_DIVERGENCE_ONLY
    if a == FAIL and b == FAIL:
        return VERDICT_INVALID_INPUT
    # a == FAIL and b == PASS
    return VERDICT_ACCIDENTAL_REPAIR


def two_way_verdict(candidate_ok: str) -> str:
    """Validation-mode (Mode B) degenerate form: candidate against reference."""
    status = _check_status("candidate_ok", candidate_ok)
    if status == INCONCLUSIVE:
        return VERDICT_INCONCLUSIVE
    return VERDICT_SPEC_VIOLATION if status == FAIL else VERDICT_INDISTINGUISHED


def verdict_from_legacy_stress_record(record: Mapping[str, Any]) -> str:
    """Reinterpret a legacy stress-worker result dict under V2 semantics.

    Legacy fields: ``ref_ok``, ``original_ok``, ``mutant_ok`` (booleans),
    optional ``validation_status`` ("inconclusive" when the normaliser
    neutralised the run) and optional ``observed_bitwise_orig_mut_eq``.

    Used by the P0 historical-impact recount (思路文档 04) and by adapters
    that bridge old detail JSONs into V2 aggregation.  It never *upgrades*
    old evidence: rounds the legacy pipeline already marked unsound stay
    INCONCLUSIVE.
    """
    if str(record.get("validation_status", "")).lower() == "inconclusive":
        return VERDICT_INCONCLUSIVE
    if not record.get("ref_ok", False):
        return VERDICT_INCONCLUSIVE

    a = PASS if record.get("original_ok") else FAIL
    b = PASS if record.get("mutant_ok") else FAIL

    exact: Optional[bool]
    if "observed_bitwise_orig_mut_eq" in record:
        exact = bool(record["observed_bitwise_orig_mut_eq"])
    elif "bitwise_orig_mut_eq" in record:
        exact = bool(record["bitwise_orig_mut_eq"])
    else:
        exact = None
    return three_way_verdict(a, b, exact)
