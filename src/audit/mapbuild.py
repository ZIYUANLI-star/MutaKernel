"""FaultToStressMap construction from audited three-way observations (M7).

Aggregates per-probe, per-case verdicts into the versioned artifact that is
the *only* bridge from offline audit mode to the online validator
(方法V2_00 §3.8, 方法V2_07 §3.5).

Input record schema (one dict per executed case; produced by the audit
harness or by an adapter over legacy detail JSONs):

    {
      "probe_id": "L1_P5__epsilon_modify__0",
      "operator": "epsilon_modify",
      "case": {"policy": "near_zero", "mode": "eval", "parameters": {}},
      "verdict": "SPEC_VIOLATION" | "EXACT_DIVERGENCE_ONLY" |
                 "INDISTINGUISHED" | "INVALID_INPUT" |
                 "ACCIDENTAL_REPAIR" | "INCONCLUSIVE",
      "cost_ms": 2100.0,          # optional
      "order": 3                  # optional execution order for first-kill
    }

Only SPEC_VIOLATION contributes to closure evidence (证据铁律 R2).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.cse.verdict import VERDICT_INCONCLUSIVE, VERDICT_INVALID_INPUT, VERDICT_SPEC_VIOLATION
from src.mutengine.fault_classes import OPERATOR_TO_FAULT_CLASS, TAXONOMY_VERSION
from .ripr import classify_escape, dimension_of_case

MAP_SCHEMA_VERSION = "2.0"

# Parameters that define a case's identity in the map (everything else, e.g.
# seeds, is deliberately aggregated away: the map generalises over seeds).
_IDENTITY_PARAMETERS = ("dtype", "batch_size", "repeat_count", "requires_backward", "layout")


def case_key(case: Mapping[str, Any]) -> str:
    parameters = case.get("parameters") or {}
    identity = {
        "policy": case.get("policy", ""),
        "mode": case.get("mode", "eval"),
        "parameters": {
            name: parameters[name]
            for name in _IDENTITY_PARAMETERS
            if name in parameters
        },
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _case_from_key(key: str) -> Dict[str, Any]:
    return json.loads(key)


def build_fault_to_stress_map(
    records: Iterable[Mapping[str, Any]],
    *,
    map_version: str,
    derived_from_run: str,
    operator_to_fault_class: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Aggregate audited observations into a FaultToStressMap.

    ``closure_rate(F, c)`` = |witnessed probes of F killed by c| /
    |witnessed probes of F on which c produced a conclusive verdict|.
    A probe is *witnessed* (non-equivalent) iff at least one case anywhere
    produced SPEC_VIOLATION for it — equivalence-uncertain probes never
    deflate the rates.
    """
    fault_of = dict(operator_to_fault_class or OPERATOR_TO_FAULT_CLASS)

    by_probe: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    probe_operator: Dict[str, str] = {}
    for record in records:
        probe_id = str(record["probe_id"])
        operator = str(record["operator"])
        if operator not in fault_of:
            raise ValueError(f"unknown operator {operator!r} for probe {probe_id}")
        previous = probe_operator.setdefault(probe_id, operator)
        if previous != operator:
            raise ValueError(f"probe {probe_id} reported two operators: {previous}, {operator}")
        by_probe[probe_id].append(record)

    witnessed = {
        probe_id
        for probe_id, probe_records in by_probe.items()
        if any(r["verdict"] == VERDICT_SPEC_VIOLATION for r in probe_records)
    }

    executed: Dict[Tuple[str, str], set] = defaultdict(set)
    killed: Dict[Tuple[str, str], set] = defaultdict(set)
    costs: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    kill_dimensions: Dict[str, set] = defaultdict(set)
    first_kill_case: Dict[str, Mapping[str, Any]] = {}

    for probe_id in witnessed:
        operator = probe_operator[probe_id]
        fault = fault_of[operator]
        ordered = sorted(
            by_probe[probe_id],
            key=lambda r: (r.get("order") is None, r.get("order", 0)),
        )
        for record in ordered:
            key = (fault, case_key(record["case"]))
            verdict = record["verdict"]
            if verdict in (VERDICT_INCONCLUSIVE, VERDICT_INVALID_INPUT):
                continue
            executed[key].add(probe_id)
            if isinstance(record.get("cost_ms"), (int, float)):
                costs[key].append(float(record["cost_ms"]))
            if verdict == VERDICT_SPEC_VIOLATION:
                killed[key].add(probe_id)
                kill_dimensions[probe_id].add(dimension_of_case(record["case"]))
                first_kill_case.setdefault(probe_id, record["case"])

    entries: List[Dict[str, Any]] = []
    faults = sorted({fault_of[probe_operator[p]] for p in witnessed})
    for fault in faults:
        fault_probes = {
            p for p in witnessed if fault_of[probe_operator[p]] == fault
        }

        effective_cases: List[Dict[str, Any]] = []
        for (entry_fault, key), killed_probes in killed.items():
            if entry_fault != fault:
                continue
            executed_probes = executed[(entry_fault, key)]
            sole = sum(
                1
                for p in killed_probes
                if kill_dimensions[p] == {dimension_of_case(_case_from_key(key))}
            )
            entry: Dict[str, Any] = {
                "case": _case_from_key(key),
                "kills": len(killed_probes),
                "executions": len(executed_probes),
                "closure_rate": round(len(killed_probes) / len(executed_probes), 4),
                "sole_detector_count": sole,
            }
            case_costs = costs[(entry_fault, key)]
            if case_costs:
                entry["mean_cost_ms"] = round(sum(case_costs) / len(case_costs), 1)
            effective_cases.append(entry)
        effective_cases.sort(
            key=lambda e: (-e["closure_rate"], e.get("mean_cost_ms", float("inf")), case_key(e["case"]))
        )

        mechanisms = Counter(
            classify_escape(first_kill_case[p])["mechanism"]
            for p in fault_probes
            if p in first_kill_case
        )
        entries.append({
            "fault_class": fault,
            "witnessed_probes": len(fault_probes),
            "escape_mechanisms": [
                {"mechanism": mechanism, "count": count}
                for mechanism, count in mechanisms.most_common()
            ],
            "effective_cases": effective_cases,
            "evidence_counterexamples": sorted(
                str(r.get("counterexample_id"))
                for probe in fault_probes
                for r in by_probe[probe]
                if r["verdict"] == VERDICT_SPEC_VIOLATION and r.get("counterexample_id")
            ),
        })

    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "map_version": map_version,
        "taxonomy_version": TAXONOMY_VERSION,
        "derived_from_run": derived_from_run,
        "operator_to_fault_class": dict(sorted(fault_of.items())),
        "witnessed_probe_count": len(witnessed),
        "entries": entries,
    }
