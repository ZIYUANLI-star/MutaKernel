"""Torch-free classification of INCONCLUSIVE / refusal reasons (E0 lesson 3).

E0 confirmed two *legitimate* refusal families that must be reported as their
own strata instead of drowning in "unknown":

  * ``state_sync_nonbijective`` — the corrected substrate's strict name-exact
    state synchronization refuses candidates whose parameter set is not a
    bijection of the reference (e.g. folded BatchNorm).  Not a subject fault.
  * ``cuda_invalid_configuration`` — CUDA "invalid configuration argument"
    style launch failures from architecture-incompatible launch configs on
    the A800.  Not a subject fault either.

Everything else is bucketed conservatively; ``other_unknown`` is the residue
that a human must look at.  The classifier only ever consumes free-text error
strings already recorded in observations, so it can be re-run offline and is
unit-testable without torch.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

REASON_STATE_SYNC = "state_sync_nonbijective"
REASON_CUDA_INVALID_CONFIG = "cuda_invalid_configuration"
REASON_OOM = "out_of_memory"
REASON_TIMEOUT = "timeout"
REASON_COMPILE = "compile_failure"
REASON_INPUT_GENERATION = "input_generation_failure"
REASON_REFERENCE_FAILURE = "reference_failure"
REASON_OTHER = "other_unknown"

ALL_REASONS = (
    REASON_STATE_SYNC,
    REASON_CUDA_INVALID_CONFIG,
    REASON_OOM,
    REASON_TIMEOUT,
    REASON_COMPILE,
    REASON_INPUT_GENERATION,
    REASON_REFERENCE_FAILURE,
    REASON_OTHER,
)

# Ordered: first match wins.  Markers are matched case-insensitively against
# the concatenated error text of an observation.
_MARKERS = (
    (REASON_STATE_SYNC, (
        "statesyncerror",
        "state_dict keys differ",
        "strict state synchronization failed",
        "strict state_dict load failed",
        "strict state restoration failed",
        "state synchronization",
        "cannot be aligned unambiguously",
        "missing state snapshot for a stateful callable",
    )),
    (REASON_CUDA_INVALID_CONFIG, (
        "invalid configuration argument",
        "invalid device function",
        "no kernel image is available",
        "too many resources requested for launch",
    )),
    (REASON_OOM, (
        "out of memory",
        "cuda oom",
        "cublas_status_alloc_failed",
    )),
    (REASON_TIMEOUT, (
        "timed out",
        "timeout",
        "inconclusive_timeout",
    )),
    (REASON_COMPILE, (
        "compilation:",
        "compilationerror",
        "orig compile",
        "mut compile",
        "candidate_compile",
        "reference_compile",
        "error building extension",
        "ninja: build stopped",
    )),
    (REASON_INPUT_GENERATION, (
        "input_generation",
        "get_inputs_error",
        "initial_input_generation",
    )),
    (REASON_REFERENCE_FAILURE, (
        "ref crash",
        "reference_load",
        "ref nan/inf",
    )),
)


def classify_inconclusive_text(text: str) -> str:
    """Classify one free-text error blob into a refusal reason bucket."""
    lowered = (text or "").lower()
    if not lowered.strip():
        return REASON_OTHER
    for reason, markers in _MARKERS:
        if any(marker in lowered for marker in markers):
            return reason
    return REASON_OTHER


def _collect_text(record: Any, parts: list) -> None:
    if isinstance(record, Mapping):
        for key, value in record.items():
            if key in ("error", "mut_error", "message", "reason", "error_message") and isinstance(value, str):
                parts.append(value)
            else:
                _collect_text(value, parts)
    elif isinstance(record, (list, tuple)):
        for item in record:
            _collect_text(item, parts)


def classify_observation(record: Mapping[str, Any], *, timed_out: Optional[bool] = None) -> str:
    """Classify a worker observation dict (fail-closed to ``other_unknown``).

    ``timed_out`` short-circuits because a killed worker often leaves no
    error text behind.
    """
    if timed_out:
        return REASON_TIMEOUT
    parts: list = []
    _collect_text(record, parts)
    return classify_inconclusive_text("\n".join(parts))


def summarize_reasons(reasons: Iterable[str]) -> dict:
    counts: dict = {reason: 0 for reason in ALL_REASONS}
    for reason in reasons:
        counts[reason if reason in counts else REASON_OTHER] += 1
    return {reason: count for reason, count in counts.items() if count}
