"""Task-level k-fold cross-fitting for the FaultToStressMap (blueprint §5.6).

Evaluating site direction on C1 with a map built from all of C1 would be
circular.  The controlled evaluation therefore partitions the *source tasks*
(not probes) into k folds; for each fold a map is built from the probes of
the remaining tasks only and evaluated on the held-out fold's probes.  Probes
of one task never straddle folds.

Torch-free; consumes the same record schema as ``src.audit.mapbuild``:

    {"probe_id": ..., "operator": ..., "case": {...}, "verdict": ...,
     "task_id": "L1_P39", ...}

``task_id`` is derived from ``probe_id`` when absent (probe ids are
``<task>__<operator>__<index>``).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.cse.verdict import VERDICT_SPEC_VIOLATION
from .mapbuild import build_fault_to_stress_map, case_key

DEFAULT_FOLDS = 5


def task_of_probe(record: Mapping[str, Any]) -> str:
    task = record.get("task_id")
    if task:
        return str(task)
    probe_id = str(record["probe_id"])
    return probe_id.split("__", 1)[0]


def assign_folds(task_ids: Sequence[str], k: int, *, seed: str = "e1-crossfit-v1") -> Dict[str, int]:
    """Deterministic, content-addressed fold assignment (no RNG state).

    Every task's fold is ``sha256(seed|task_id) % k`` — stable across
    machines and resumable runs, and independent of task enumeration order.
    """
    if k < 2:
        raise ValueError("cross-fitting requires at least two folds")
    return {
        task: int(hashlib.sha256(f"{seed}|{task}".encode("utf-8")).hexdigest(), 16) % k
        for task in task_ids
    }


def _directed_case_ranking(fold_map: Mapping[str, Any]) -> Dict[str, List[str]]:
    """fault_class -> case keys ordered by closure rate (best first)."""
    ranking: Dict[str, List[str]] = {}
    for entry in fold_map.get("entries", []):
        ranking[entry["fault_class"]] = [
            case_key(case["case"]) for case in entry["effective_cases"]
        ]
    return ranking


def crossfit_map_evaluation(
    records: Iterable[Mapping[str, Any]],
    *,
    operator_to_fault_class: Optional[Mapping[str, str]] = None,
    k: int = DEFAULT_FOLDS,
    planned_cases: int = 8,
    map_version: str = "crossfit",
    seed: str = "e1-crossfit-v1",
) -> Dict[str, Any]:
    """Cross-fitted closure evaluation of the fault-to-stress map.

    For every held-out witnessed probe, ask: do the top ``planned_cases``
    directed cases recommended by the *training-folds* map contain a case
    that actually produced SPEC_VIOLATION for this probe?  Pooled over folds
    this yields the §5.6 "closure within first k planned cases" number
    without circularity.
    """
    from src.mutengine.fault_classes import OPERATOR_TO_FAULT_CLASS

    fault_of = dict(operator_to_fault_class or OPERATOR_TO_FAULT_CLASS)
    all_records = [dict(r) for r in records]
    if not all_records:
        raise ValueError("no records to cross-fit")

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        by_task[task_of_probe(record)].append(record)

    folds = assign_folds(sorted(by_task), k, seed=seed)

    per_fold: List[Dict[str, Any]] = []
    pooled_closed = 0
    pooled_witnessed = 0
    for fold_index in range(k):
        train = [
            record
            for task, task_records in by_task.items()
            if folds[task] != fold_index
            for record in task_records
        ]
        held = [
            record
            for task, task_records in by_task.items()
            if folds[task] == fold_index
            for record in task_records
        ]
        if not held:
            per_fold.append({"fold": fold_index, "witnessed": 0, "closed": 0})
            continue
        if not train:
            raise ValueError(f"fold {fold_index} left no training data (k too large)")

        fold_map = build_fault_to_stress_map(
            train,
            map_version=f"{map_version}-fold{fold_index}",
            derived_from_run="crossfit",
            operator_to_fault_class=fault_of,
        )
        ranking = _directed_case_ranking(fold_map)

        killing_cases: Dict[str, set] = defaultdict(set)
        probe_fault: Dict[str, str] = {}
        for record in held:
            probe_id = str(record["probe_id"])
            probe_fault[probe_id] = fault_of[str(record["operator"])]
            if record["verdict"] == VERDICT_SPEC_VIOLATION:
                killing_cases[probe_id].add(case_key(record["case"]))

        witnessed_probes = sorted(killing_cases)
        closed = 0
        for probe_id in witnessed_probes:
            planned = ranking.get(probe_fault[probe_id], [])[:planned_cases]
            if any(case in killing_cases[probe_id] for case in planned):
                closed += 1

        pooled_closed += closed
        pooled_witnessed += len(witnessed_probes)
        per_fold.append({
            "fold": fold_index,
            "held_tasks": sorted(t for t in by_task if folds[t] == fold_index),
            "witnessed": len(witnessed_probes),
            "closed": closed,
        })

    return {
        "schema_version": "1.0",
        "method": "task_level_kfold_crossfit",
        "k": k,
        "fold_seed": seed,
        "planned_cases": planned_cases,
        "fold_assignment": folds,
        "per_fold": per_fold,
        "pooled": {
            "witnessed": pooled_witnessed,
            "closed": pooled_closed,
            "closure_rate": (
                round(pooled_closed / pooled_witnessed, 4) if pooled_witnessed else None
            ),
        },
    }
