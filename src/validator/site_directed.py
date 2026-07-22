"""Site-directed stress-plan derivation for the online validator (M8, Mode B).

Given a candidate's static SiteFingerprint (方法V2_03 §3.3) and the audited
FaultToStressMap (方法V2_07 §3.5), derive a deterministic, budget-bounded,
ordered list of stress cases:

  * ~``directed_fraction`` of the candidate-call budget goes to cases that
    the map证明 effective against fault classes whose sites are present in
    the candidate, ordered by (closure_rate desc, cost asc);
  * the remaining budget goes to a frozen general fallback sequence — the
    fingerprint only *prioritises*, it never *excludes* (a zero site count is
    absence of evidence, not evidence of absence; 方法V2_08 §3.2).

The plan is a pure function of (subject_id, fingerprint, map, budget,
general_sequence, knobs): reordering-safe, preregistration-compatible and
replayable.  Seeds are derived from stable hashes, never from global RNG.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_DIRECTED_FRACTION = 0.7

# Frozen general fallback sequence (version-controlled; changing it is a plan
# version bump).  At least one case per search dimension.
DEFAULT_GENERAL_SEQUENCE: Sequence[Mapping[str, Any]] = (
    {"policy": "iid", "mode": "eval", "parameters": {}},
    {"policy": "structured_ramp", "mode": "eval", "parameters": {}},
    {"policy": "near_zero", "mode": "eval", "parameters": {}},
    {"policy": "mixed_extremes", "mode": "eval", "parameters": {}},
    {"policy": "boundary_last_element", "mode": "eval", "parameters": {}},
    {"policy": "iid", "mode": "eval", "parameters": {"dtype": "float16"}},
    {"policy": "iid", "mode": "repeated", "parameters": {"repeat_count": 2}},
    {"policy": "iid", "mode": "config", "parameters": {"batch_size": 4}},
    {"policy": "large_magnitude", "mode": "eval", "parameters": {}},
    {"policy": "iid", "mode": "config", "parameters": {"batch_size": 1}},
    {"policy": "iid", "mode": "eval", "parameters": {"dtype": "bfloat16"}},
    {"policy": "reduction_adversarial", "mode": "eval", "parameters": {}},
)


def _canonical_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "policy": str(case.get("policy", "")),
        "mode": str(case.get("mode", "eval")),
        "parameters": dict(case.get("parameters") or {}),
    }


def _case_identity(case: Mapping[str, Any]) -> str:
    return json.dumps(_canonical_case(case), sort_keys=True, separators=(",", ":"))


def _case_cost(case: Mapping[str, Any]) -> int:
    if case.get("mode") == "repeated":
        repeat_count = (case.get("parameters") or {}).get("repeat_count", 2)
        return max(2, int(repeat_count))
    return 1


def derive_seed(subject_id: str, case: Mapping[str, Any], replicate: int = 0) -> int:
    digest = hashlib.sha256(
        f"{subject_id}|{_case_identity(case)}|{replicate}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) % (2**31)


def _directed_case_stream(
    fingerprint: Mapping[str, Any],
    fault_to_stress_map: Mapping[str, Any],
) -> Iterable[Dict[str, Any]]:
    """Yield map-proven cases for present fault classes, best first."""
    entries = {
        entry["fault_class"]: entry
        for entry in fault_to_stress_map.get("entries", [])
    }
    present = [
        fault for fault in fingerprint.get("fault_classes_present", [])
        if fault in entries and entries[fault].get("effective_cases")
    ]
    # Fault classes ordered by their single best closure rate (desc).
    present.sort(
        key=lambda fault: -float(entries[fault]["effective_cases"][0]["closure_rate"])
    )
    # Round-robin over fault classes so no single class monopolises the
    # directed budget; within a class cases keep map order (closure desc).
    iterators = {fault: iter(entries[fault]["effective_cases"]) for fault in present}
    while iterators:
        exhausted = []
        for fault in present:
            if fault not in iterators:
                continue
            try:
                effective = next(iterators[fault])
            except StopIteration:
                exhausted.append(fault)
                continue
            yield {
                "case": _canonical_case(effective["case"]),
                "fault_class": fault,
                "map_closure_rate": float(effective["closure_rate"]),
            }
        for fault in exhausted:
            iterators.pop(fault, None)


def derive_site_directed_plan(
    *,
    subject_id: str,
    fingerprint: Mapping[str, Any],
    fault_to_stress_map: Mapping[str, Any],
    budget_candidate_calls: int,
    general_sequence: Optional[Sequence[Mapping[str, Any]]] = None,
    is_authorized: Optional[Callable[[Mapping[str, Any]], bool]] = None,
    directed_fraction: float = DEFAULT_DIRECTED_FRACTION,
) -> Dict[str, Any]:
    """Derive the ordered, budget-bounded stress plan for one candidate.

    ``is_authorized`` is the contract gate (方法V2_02): unauthorized cases are
    skipped and recorded, never silently swallowed.
    Returns ``{"subject_id", "plan": [planned cases...], "budget": {...},
    "skipped_unauthorized": [...]}`` where every planned case carries
    ``policy/mode/parameters/seed/source`` and its candidate-call cost.
    """
    if budget_candidate_calls < 1:
        raise ValueError("budget_candidate_calls must be >= 1")
    if not 0.0 <= directed_fraction <= 1.0:
        raise ValueError("directed_fraction must be within [0, 1]")

    map_fpv = fingerprint.get("fingerprint_version")
    authorized = is_authorized or (lambda case: True)
    general = [
        _canonical_case(case)
        for case in (general_sequence or DEFAULT_GENERAL_SEQUENCE)
    ]

    directed_budget = math.ceil(directed_fraction * budget_candidate_calls)
    plan: List[Dict[str, Any]] = []
    skipped_unauthorized: List[Dict[str, Any]] = []
    seen: set = set()
    spent = 0
    directed_spent = 0

    def try_add(case: Mapping[str, Any], source: str, budget_cap: int,
                fault_class: Optional[str] = None,
                map_closure_rate: Optional[float] = None) -> int:
        nonlocal spent
        identity = _case_identity(case)
        if identity in seen:
            return 0
        cost = _case_cost(case)
        if spent + cost > budget_cap:
            return 0
        if not authorized(case):
            seen.add(identity)
            skipped_unauthorized.append(_canonical_case(case))
            return 0
        seen.add(identity)
        planned: Dict[str, Any] = dict(_canonical_case(case))
        planned["seed"] = derive_seed(subject_id, case)
        planned["source"] = source
        planned["candidate_run_cost"] = cost
        if fault_class is not None:
            planned["fault_class"] = fault_class
        if map_closure_rate is not None:
            planned["map_closure_rate"] = map_closure_rate
        plan.append(planned)
        spent += cost
        return cost

    for directed in _directed_case_stream(fingerprint, fault_to_stress_map):
        added = try_add(
            directed["case"], "directed", directed_budget,
            fault_class=directed["fault_class"],
            map_closure_rate=directed["map_closure_rate"],
        )
        directed_spent += added
        if spent >= directed_budget:
            break

    for case in general:
        try_add(case, "general", budget_candidate_calls)
        if spent >= budget_candidate_calls:
            break

    return {
        "subject_id": subject_id,
        "fingerprint_version": map_fpv,
        "map_version": fault_to_stress_map.get("map_version"),
        "directed_fraction": directed_fraction,
        "plan": plan,
        "budget": {
            "candidate_calls": budget_candidate_calls,
            "planned_calls": spent,
            "directed_calls": directed_spent,
            "general_calls": spent - directed_spent,
        },
        "skipped_unauthorized": skipped_unauthorized,
    }
