#!/usr/bin/env python3
"""Shared core for the E1 offline round-log reuse analyzers (blueprint §5.6,
Table 11 sole-detector column, A13 dual accounting, §5.7 C1-side cost).

Everything here is *offline and read-only*: it consumes the round-level logs
already produced by the E1 drivers (equiv / CSE / baseline observations plus
run manifests) and never re-executes anything on a GPU.  All numbers derived
from partially-finished CSE lanes are interim by construction; rerun the
analyzers after the CSE lanes close for the final-scope figures.

Conservative-evidence doctrine (documented once, applied everywhere):

  * every probe's trial log ends at its first sound divergence (early exit),
    so each witnessed probe carries exactly one recorded witnessing case;
  * "closed" therefore means "the recorded witness case is inside the planned
    top-k" — a *lower bound* on true closure (a planned-but-never-executed
    case might also have killed the probe);
  * conversely "cross-confirmed by >=2 dimensions" from these logs is a lower
    bound (early exit hides later confirmations), and "sole-detector" counts
    are upper bounds.

Torch-free; imports only the torch-free audit/metadata modules.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _resolve_project_root() -> Path:
    """Locate the repo root both locally (scripts/ sibling of src/) and when
    deployed standalone under /root/mk_v2_runs/e1/analysis/scripts/."""
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, Path(os.environ.get("MK_PROJECT_ROOT", ""))):
        if candidate and (candidate / "src").is_dir():
            return candidate
    fallback = Path("/root/mk_v2")
    if (fallback / "src").is_dir():
        return fallback
    raise RuntimeError(
        "cannot locate the MutaKernel project root; set MK_PROJECT_ROOT")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.crossfit import assign_folds, task_of_probe  # noqa: E402
from src.audit.mapbuild import build_fault_to_stress_map, case_key  # noqa: E402
from src.audit.ripr import dimension_of_case  # noqa: E402
from src.cse.verdict import (  # noqa: E402
    VERDICT_INCONCLUSIVE,
    VERDICT_INDISTINGUISHED,
    VERDICT_SPEC_VIOLATION,
)

GRADE_WITNESSED = "WITNESSED_NON_EQUIVALENT"
OUTCOME_FALSIFIED = "FALSIFIED"

# ---------------------------------------------------------------------------
# Blueprint five stress dimensions (Table 11 rows / §5.6 cross-confirmation)
# ---------------------------------------------------------------------------

BLUEPRINT_DIMENSIONS = ("value", "dtype", "training", "repetition", "configuration")

# rirp.dimension_of_case speaks the internal axis names; Table 11 speaks the
# blueprint names.
_RIPR_TO_BLUEPRINT = {
    "value": "value",
    "dtype": "dtype",
    "training": "training",
    "repeated": "repetition",
    "config": "configuration",
    "layout": "configuration",  # layout is appendix material; folded into config
}

# Proxy mapping of the 21 value-stress policies onto the five blueprint
# dimensions.  BASIS (documented per policy): the policy's target fault
# classes in src/stress/policy_metadata.POLICY_TARGET_FAULT_CLASSES are
# matched against the fault classes each *execution-context* dimension
# targets (EXECUTION_CONTEXT_TARGET_FAULT_CLASSES, 方法V2_02 §4.3):
#   dtype   <- {F-PREC-ACC, F-CAST} plus representational-limit regimes
#              (subnormals / machine-epsilon / near-overflow), because those
#              value regimes emulate what a low-precision dtype transform
#              exposes (MASKING_FAILURE_PRECISION in 方法V2_07 §3.4);
#   repetition <- {F-SYNC} plus F-RED-ORD (reduction-order nondeterminism is
#              the canonical repetition-observable mechanism,
#              OBSERVATION_FAILURE_NONDETERMINISM);
#   configuration <- {F-LAUNCH, F-BOUND, F-IDX}: position/boundary-structured
#              inputs emulate configuration/boundary reachability
#              (REACHABILITY_FAILURE_CONFIG);
#   training <- policies whose *only* target classes live in the train
#              context set AND whose mechanism is a training-path property
#              (initialisation / scale), not a generic value magnitude;
#   value   <- everything else: plain extreme/boundary/sign/density value
#              activation (ACTIVATION_FAILURE_VALUE).
# NOTE this is a proxy: strictly (src/audit/ripr.dimension_of_case) every
# executed E1 case is mode=eval/fp32 and thus dimension "value"; the true
# per-dimension Table 11 rows require the audit stress phase on GPU.  Both
# accountings are always reported side by side.
POLICY_DIMENSION_PROXY: Dict[str, Dict[str, str]] = {
    # family 1: extreme value distributions
    "large_magnitude": {"dimension": "value", "rationale": "F-ARITH/F-STAB magnitude activation (plain value regime)"},
    "extreme_magnitude": {"dimension": "dtype", "rationale": "F-STAB/F-CAST at representational limits -> precision-masking proxy"},
    "near_overflow": {"dimension": "dtype", "rationale": "F-STAB/F-CAST near format overflow boundary -> precision-masking proxy"},
    "near_zero": {"dimension": "value", "rationale": "F-EPS/F-SCALE/F-CONST small-value activation (plain value regime)"},
    "denormals": {"dimension": "dtype", "rationale": "F-EPS in the subnormal representation regime -> precision-masking proxy"},
    "near_epsilon": {"dimension": "dtype", "rationale": "F-EPS at machine epsilon -> precision-masking proxy"},
    "mixed_extremes": {"dimension": "dtype", "rationale": "F-PREC-ACC/F-CAST accumulation-precision targets"},
    # family 2: boundary values and sign
    "all_negative": {"dimension": "value", "rationale": "sign activation (F-INIT via values, not training context)"},
    "all_positive": {"dimension": "value", "rationale": "sign activation (F-INIT/F-STAB via values)"},
    "boundary_last_element": {"dimension": "configuration", "rationale": "F-BOUND/F-LAUNCH boundary reachability proxy"},
    "relop_boundary_hit": {"dimension": "value", "rationale": "F-RELOP predicate-boundary value activation"},
    # family 3: sparsity gradient
    "dense_nonzero": {"dimension": "value", "rationale": "F-ARITH dense-value activation"},
    "sparse": {"dimension": "configuration", "rationale": "F-BOUND/F-IDX index-path reachability proxy"},
    "sparse_extreme": {"dimension": "configuration", "rationale": "F-BOUND/F-RELOP extreme-sparsity boundary reachability proxy"},
    # family 4: structured / position-sensitive
    "structured_ramp": {"dimension": "configuration", "rationale": "F-IDX/F-BCAST/F-LAYOUT/F-LAUNCH position reachability proxy"},
    "head_heavy": {"dimension": "configuration", "rationale": "F-IDX position-sensitive reachability proxy"},
    "tail_heavy": {"dimension": "configuration", "rationale": "F-IDX/F-BOUND tail-position reachability proxy"},
    # family 5: reduction adversarial / special behaviour
    "alternating_sign": {"dimension": "repetition", "rationale": "F-RED-ORD reduction-order nondeterminism proxy"},
    "reduction_adversarial": {"dimension": "repetition", "rationale": "F-RED-ORD/F-PREC-ACC reduction-order nondeterminism proxy"},
    "uniform_constant": {"dimension": "training", "rationale": "F-SCALE scaling/normalisation path (train-context target class)"},
    "init_sensitive": {"dimension": "training", "rationale": "F-INIT initialisation sensitivity (train-context target class)"},
    # non-policy rounds
    "iid": {"dimension": "value", "rationale": "IID randn baseline draw (value dimension by definition)"},
    "random": {"dimension": "value", "rationale": "random equivalence round (IID value draw)"},
}


def dimension_of_policy_proxy(policy: str) -> str:
    entry = POLICY_DIMENSION_PROXY.get(policy)
    return entry["dimension"] if entry else "value"


def dimension_strict(case: Mapping[str, Any]) -> str:
    """Strict RIPR accounting mapped onto blueprint dimension names."""
    return _RIPR_TO_BLUEPRINT[dimension_of_case(case)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedup_latest(rows: Iterable[Mapping[str, Any]],
                 key: str = "probe_id",
                 timestamp: str = "finished_at") -> Tuple[List[Dict[str, Any]], int]:
    """Keep the latest record per probe (requeues / lane reshuffles may leave
    an older observation for the same probe in an archived lane file)."""
    best: Dict[str, Dict[str, Any]] = {}
    dropped = 0
    for row in rows:
        pid = str(row[key])
        prev = best.get(pid)
        if prev is None:
            best[pid] = dict(row)
        else:
            dropped += 1
            if str(row.get(timestamp) or "") >= str(prev.get(timestamp) or ""):
                best[pid] = dict(row)
    return list(best.values()), dropped


def load_dataset(e1_dir: Path,
                 cse_files: Optional[Sequence[Path]] = None) -> Dict[str, Any]:
    """Load equiv / CSE / baseline observations plus controls and manifests.

    ``cse_files`` defaults to the *completed* lanes 1 and 2 (interim scope);
    pass every ``cse_observations_lane*.jsonl`` for the final-scope rerun.
    """
    e1_dir = Path(e1_dir)
    inventory: List[Dict[str, Any]] = []

    def _load(path: Path) -> List[Dict[str, Any]]:
        rows = load_jsonl(path)
        inventory.append({"path": str(path), "rows": len(rows)})
        return rows

    equiv_rows, equiv_dupes = dedup_latest(_load(e1_dir / "equiv_observations.jsonl"))

    if cse_files is None:
        cse_files = [e1_dir / "cse_observations_lane1.jsonl",
                     e1_dir / "cse_observations_lane2.jsonl"]
    cse_raw: List[Dict[str, Any]] = []
    for path in cse_files:
        path = Path(path)
        if path.exists():
            cse_raw.extend(_load(path))
        else:
            inventory.append({"path": str(path), "rows": None, "missing": True})
    cse_rows, cse_dupes = dedup_latest(cse_raw)

    baseline_rows = _load(e1_dir / "baseline_observations.jsonl")

    controls = {}
    controls_path = e1_dir / "original_controls.json"
    if controls_path.exists():
        controls = json.loads(controls_path.read_text(encoding="utf-8"))
        inventory.append({"path": str(controls_path), "rows": len(controls)})

    manifests = {}
    for path in sorted(e1_dir.glob("*run_manifest*.json")):
        try:
            manifests[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "equiv": equiv_rows,
        "cse": cse_rows,
        "baseline": baseline_rows,
        "controls": controls,
        "manifests": manifests,
        "inventory": inventory,
        "dedup_dropped": {"equiv": equiv_dupes, "cse": cse_dupes},
    }


# ---------------------------------------------------------------------------
# Trial adaptation (same conventions as run_e1_probe_study --phase map)
# ---------------------------------------------------------------------------

def trial_case(trial: Mapping[str, Any]) -> Dict[str, Any]:
    """Case identity of one trial.  Random rounds collapse onto the IID case
    (the map-phase convention: ``trial.get("policy") or "iid"``)."""
    return {"policy": trial.get("policy") or "iid", "mode": "eval", "parameters": {}}


def trial_verdict(trial: Mapping[str, Any]) -> str:
    status = str(trial.get("status", "")).lower()
    if status == "fail":
        return VERDICT_SPEC_VIOLATION
    if status == "pass":
        return VERDICT_INDISTINGUISHED
    return VERDICT_INCONCLUSIVE


def trial_policy_label(trial: Mapping[str, Any]) -> str:
    """Display label for budget accounting: random rounds stay 'random'
    (A13 contrasts random vs directed), directed rounds use the policy name."""
    if trial.get("round_type") == "random":
        return "random"
    return str(trial.get("policy") or "iid")


def trial_total_ms(trial: Mapping[str, Any]) -> Optional[float]:
    timings = trial.get("timings_ms") or {}
    value = timings.get("total_ms")
    return float(value) if isinstance(value, (int, float)) else None


def build_map_records(equiv_rows: Sequence[Mapping[str, Any]],
                      cse_rows: Sequence[Mapping[str, Any]],
                      baseline_rows: Sequence[Mapping[str, Any]] = (),
                      include_baseline_kills: bool = True) -> List[Dict[str, Any]]:
    """Adapt round logs into mapbuild/crossfit record dicts.

    Mirrors run_e1_probe_study.phase_map: equiv trials + (new) CSE trials +
    baseline IID kills, so map closure rates see the easy kills too.
    """
    records: List[Dict[str, Any]] = []
    for offset, rows in ((0, equiv_rows), (10000, cse_rows)):
        for row in rows:
            for order, trial in enumerate(row.get("trials") or []):
                record = {
                    "probe_id": row["probe_id"],
                    "operator": row["operator_name"],
                    "task_id": task_of_probe(row),
                    "case": trial_case(trial),
                    "verdict": trial_verdict(trial),
                    "order": offset + order,
                }
                cost = trial_total_ms(trial)
                if cost is not None:
                    record["cost_ms"] = cost
                records.append(record)
    if include_baseline_kills:
        for row in baseline_rows:
            if row.get("status") == "killed":
                records.append({
                    "probe_id": row["probe_id"],
                    "operator": row["operator_name"],
                    "task_id": task_of_probe(row),
                    "case": {"policy": "iid", "mode": "eval", "parameters": {}},
                    "verdict": VERDICT_SPEC_VIOLATION,
                    "order": 0,
                })
    return records


# ---------------------------------------------------------------------------
# Witness extraction (blind spots = equiv WITNESSED + CSE FALSIFIED)
# ---------------------------------------------------------------------------

def _first_fail_trial(row: Mapping[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    for index, trial in enumerate(row.get("trials") or []):
        if str(trial.get("status", "")).lower() == "fail":
            return index, trial
    return None, None


def extract_witnesses(equiv_rows: Sequence[Mapping[str, Any]],
                      cse_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One entry per witnessed blind-spot probe with its (single) recorded
    witnessing case, phase-local witness round index (1-based) and source."""
    witnesses: Dict[str, Dict[str, Any]] = {}

    def _add(row: Mapping[str, Any], source: str) -> None:
        index, trial = _first_fail_trial(row)
        if trial is not None:
            case = trial_case(trial)
            label = trial_policy_label(trial)
            round_1based = index + 1
        else:
            # Whole-probe timeout after divergence: fall back to the recorded
            # divergence metadata (same fields as a trial).
            divergence = row.get("divergence") or {}
            case = trial_case(divergence)
            label = trial_policy_label(divergence)
            round_1based = len(row.get("trials") or []) or None
        witnesses[row["probe_id"]] = {
            "probe_id": row["probe_id"],
            "kernel": row.get("kernel"),
            "operator": row["operator_name"],
            "fault_class": row.get("fault_class"),
            "task_id": task_of_probe(row),
            "source": source,
            "witness_case": case,
            "witness_case_key": case_key(case),
            "witness_policy_label": label,
            "witness_round_1based": round_1based,
            "trials_executed": len(row.get("trials") or []),
        }

    for row in equiv_rows:
        if row.get("evidence_grade") == GRADE_WITNESSED:
            _add(row, "equiv")
    for row in cse_rows:
        if row.get("outcome") == OUTCOME_FALSIFIED:
            _add(row, "cse")
    return witnesses


def executed_case_status(equiv_rows: Sequence[Mapping[str, Any]],
                         cse_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    """probe_id -> case_key -> best recorded status ('kill'>'pass'>'inconclusive').

    This is what lets the closure analyzer distinguish "planned case executed
    and passed" (genuinely unclosed) from "planned case never executed"
    (evidence-insufficient; conservative unclosed; rerun candidate).
    """
    rank = {"kill": 2, "pass": 1, "inconclusive": 0}
    out: Dict[str, Dict[str, str]] = defaultdict(dict)
    for rows in (equiv_rows, cse_rows):
        for row in rows:
            pid = row["probe_id"]
            for trial in row.get("trials") or []:
                key = case_key(trial_case(trial))
                verdict = trial_verdict(trial)
                status = ("kill" if verdict == VERDICT_SPEC_VIOLATION
                          else "pass" if verdict == VERDICT_INDISTINGUISHED
                          else "inconclusive")
                previous = out[pid].get(key)
                if previous is None or rank[status] > rank[previous]:
                    out[pid][key] = status
    return dict(out)


# ---------------------------------------------------------------------------
# §5.6 cross-fitted closure (lower bound) + non-cross-fitted upper bound
# ---------------------------------------------------------------------------

def _ranking_of_map(fault_map: Mapping[str, Any]) -> Dict[str, List[str]]:
    return {
        entry["fault_class"]: [case_key(c["case"]) for c in entry["effective_cases"]]
        for entry in fault_map.get("entries", [])
    }


def closure_evaluation(records: Sequence[Mapping[str, Any]],
                       witnesses: Mapping[str, Mapping[str, Any]],
                       executed: Mapping[str, Mapping[str, str]],
                       *,
                       k_folds: int = 5,
                       planned_cases: int = 8,
                       seed: str = "e1-crossfit-v1",
                       curve_max_k: int = 22,
                       map_version: str = "offline-crossfit") -> Dict[str, Any]:
    """Per-witnessed-probe closure under task-level cross-fitting (lower
    bound) and under the non-cross-fitted full map (upper bound).

    Closure criterion (conservative): the probe's *recorded* witness case is
    within the held-out map's first ``planned_cases`` planned cases for its
    fault class.  Unclosed probes are split into
    ``unclosed_evidence_insufficient`` (some planned case never executed on
    the probe, or fault class absent from the training map — small-scale
    verification rerun candidates) and ``unclosed_all_planned_conclusive``
    (every planned case executed with a conclusive non-kill verdict).
    """
    from src.mutengine.fault_classes import OPERATOR_TO_FAULT_CLASS

    by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task_id"])].append(record)
    folds = assign_folds(sorted(by_task), k_folds, seed=seed)

    fold_rankings: Dict[int, Dict[str, List[str]]] = {}
    for fold in range(k_folds):
        train = [r for task, rs in by_task.items() if folds[task] != fold for r in rs]
        if not train:
            raise ValueError(f"fold {fold} left no training data (k too large)")
        fold_map = build_fault_to_stress_map(
            train, map_version=f"{map_version}-fold{fold}",
            derived_from_run="offline-crossfit")
        fold_rankings[fold] = _ranking_of_map(fold_map)

    full_map = build_fault_to_stress_map(
        list(records), map_version=f"{map_version}-full",
        derived_from_run="offline-upper-bound")
    full_ranking = _ranking_of_map(full_map)

    fault_of = OPERATOR_TO_FAULT_CLASS

    def _position(ranking: Mapping[str, List[str]], fault: str, target: str) -> Optional[int]:
        keys = ranking.get(fault, [])
        return keys.index(target) + 1 if target in keys else None

    per_probe: List[Dict[str, Any]] = []
    for pid in sorted(witnesses):
        wit = witnesses[pid]
        fault = fault_of[wit["operator"]]
        fold = folds.get(wit["task_id"])
        ranking = fold_rankings.get(fold, {})
        witness_key = wit["witness_case_key"]

        pos_cf = _position(ranking, fault, witness_key)
        pos_full = _position(full_ranking, fault, witness_key)
        planned = ranking.get(fault, [])[:planned_cases]
        closed = pos_cf is not None and pos_cf <= planned_cases

        planned_status = [
            {"case_key": key,
             "recorded": executed.get(pid, {}).get(key, "not_executed")}
            for key in planned
        ]
        if closed:
            classification = "closed"
        elif not planned:
            classification = "unclosed_evidence_insufficient"
        elif any(s["recorded"] in ("not_executed", "inconclusive")
                 for s in planned_status):
            classification = "unclosed_evidence_insufficient"
        else:
            classification = "unclosed_all_planned_conclusive"

        per_probe.append({
            "probe_id": pid,
            "fault_class": fault,
            "source": wit["source"],
            "fold": fold,
            "witness_case_key": witness_key,
            "witness_policy_label": wit["witness_policy_label"],
            "crossfit_rank_of_witness": pos_cf,
            "full_map_rank_of_witness": pos_full,
            "closed_at_planned_k": closed,
            "classification": classification,
            "planned_case_status": planned_status,
        })

    total = len(per_probe)

    def _curve(rank_field: str) -> List[Dict[str, Any]]:
        points = []
        for k in range(1, curve_max_k + 1):
            closed_k = sum(
                1 for row in per_probe
                if row[rank_field] is not None and row[rank_field] <= k)
            points.append({"k": k, "closed": closed_k,
                           "closure_rate": round(closed_k / total, 4) if total else None})
        return points

    per_fold: List[Dict[str, Any]] = []
    for fold in range(k_folds):
        fold_rows = [r for r in per_probe if r["fold"] == fold]
        per_fold.append({
            "fold": fold,
            "witnessed": len(fold_rows),
            "closed": sum(1 for r in fold_rows if r["closed_at_planned_k"]),
        })

    closed_total = sum(1 for r in per_probe if r["closed_at_planned_k"])
    closed_upper = sum(
        1 for r in per_probe
        if r["full_map_rank_of_witness"] is not None
        and r["full_map_rank_of_witness"] <= planned_cases)
    return {
        "method": "task_level_kfold_crossfit_offline_conservative",
        "k_folds": k_folds,
        "fold_seed": seed,
        "planned_cases": planned_cases,
        "witnessed_total": total,
        "pooled_lower_bound": {
            "closed": closed_total,
            "closure_rate": round(closed_total / total, 4) if total else None,
        },
        "upper_bound_full_map": {
            "closed": closed_upper,
            "closure_rate": round(closed_upper / total, 4) if total else None,
        },
        "per_fold": per_fold,
        "classification_counts": dict(Counter(r["classification"] for r in per_probe)),
        "closure_curve_lower": _curve("crossfit_rank_of_witness"),
        "closure_curve_upper": _curve("full_map_rank_of_witness"),
        "per_probe": per_probe,
    }


# ---------------------------------------------------------------------------
# Table 11 sole-detector column / §5.6 >=2-dimension cross-confirmation
# ---------------------------------------------------------------------------

def witness_dimension_summary(witnesses: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-defect witnessed-dimension sets under both accountings.

    ``proxy``: 21-policy -> blueprint-dimension mapping above.
    ``strict``: RIPR dimension_of_case (all executed E1 cases are eval/fp32
    -> value), reported to make the proxy's status explicit.
    Early-exit logs mean each defect has one recorded witness, so sole-
    detector counts are UPPER bounds and the cross-confirmation rate is a
    LOWER bound.
    """
    per_defect: List[Dict[str, Any]] = []
    for pid in sorted(witnesses):
        wit = witnesses[pid]
        label = wit["witness_policy_label"]
        proxy_dims = {dimension_of_policy_proxy(label)}
        strict_dims = {dimension_strict(wit["witness_case"])}
        per_defect.append({
            "probe_id": pid,
            "fault_class": wit["fault_class"],
            "source": wit["source"],
            "witness_policy": label,
            "dimensions_proxy": sorted(proxy_dims),
            "dimensions_strict": sorted(strict_dims),
        })

    def _summary(field: str) -> Dict[str, Any]:
        sole = Counter()
        witnessed_by = Counter()
        multi = 0
        for row in per_defect:
            dims = row[field]
            for d in dims:
                witnessed_by[d] += 1
            if len(dims) == 1:
                sole[dims[0]] += 1
            else:
                multi += 1
        total = len(per_defect)
        return {
            "defects_total": total,
            "witnessed_by_dimension": {d: witnessed_by.get(d, 0) for d in BLUEPRINT_DIMENSIONS},
            "sole_detector_defects": {d: sole.get(d, 0) for d in BLUEPRINT_DIMENSIONS},
            "sole_detector_total": sum(sole.values()),
            "cross_confirmed_ge2_dims": multi,
            "cross_confirmed_rate": round(multi / total, 4) if total else None,
        }

    return {
        "per_defect": per_defect,
        "proxy": _summary("dimensions_proxy"),
        "strict": _summary("dimensions_strict"),
        "policy_dimension_mapping": {
            name: dict(entry) for name, entry in sorted(POLICY_DIMENSION_PROXY.items())
        },
        "caveats": [
            "early-exit logs record exactly one witnessing case per defect: "
            "sole-detector counts are upper bounds, cross-confirmation is a "
            "lower bound (0 by construction until multi-witness evidence "
            "exists, e.g. from the audit stress phase or verification reruns)",
            "strict accounting: every executed E1 case is mode=eval/fp32, "
            "hence dimension 'value'; per-dimension Table 11 rows need the "
            "GPU audit stress phase (dtype/train/config/repeated transforms)",
        ],
    }


# ---------------------------------------------------------------------------
# Budget-recall replay + A13 dual accounting
# ---------------------------------------------------------------------------

def witness_budget_indices(witnesses: Mapping[str, Mapping[str, Any]],
                           equiv_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Budget position of each witness: rounds spent within its own phase and
    combined across phases (a CSE witness first consumed the probe's full
    equiv-phase budget)."""
    equiv_trials = {row["probe_id"]: len(row.get("trials") or [])
                    for row in equiv_rows}
    out = []
    for pid in sorted(witnesses):
        wit = witnesses[pid]
        phase_index = wit["witness_round_1based"]
        combined = phase_index
        if wit["source"] == "cse" and phase_index is not None:
            combined = phase_index + equiv_trials.get(pid, 0)
        out.append({
            "probe_id": pid,
            "source": wit["source"],
            "witness_policy": wit["witness_policy_label"],
            "phase_round_index": phase_index,
            "combined_round_index": combined,
        })
    return out


def recall_curve(indices: Sequence[Optional[int]], max_budget: int) -> List[Dict[str, Any]]:
    """Cumulative witness recall as the per-probe round budget grows."""
    total = len(indices)
    points = []
    for budget in range(1, max_budget + 1):
        found = sum(1 for i in indices if i is not None and i <= budget)
        points.append({"budget_rounds": budget, "witnesses": found,
                       "recall": round(found / total, 4) if total else None})
    return points


def policy_round_stats(rows: Sequence[Mapping[str, Any]], source: str) -> List[Dict[str, Any]]:
    """A13 dual accounting per policy label ('random' = the IID random rounds).

    hit_rate_per_round = witnesses / rounds executed (denominator includes
    inconclusive rounds: that is the honest per-budgeted-round yield);
    budget shares are reported both by round count and by measured wall time.
    """
    rounds = Counter()
    kills = Counter()
    conclusive = Counter()
    wall_ms = Counter()
    for row in rows:
        for trial in row.get("trials") or []:
            label = trial_policy_label(trial)
            rounds[label] += 1
            verdict = trial_verdict(trial)
            if verdict == VERDICT_SPEC_VIOLATION:
                kills[label] += 1
            if verdict != VERDICT_INCONCLUSIVE:
                conclusive[label] += 1
            cost = trial_total_ms(trial)
            if cost is not None:
                wall_ms[label] += cost
    total_rounds = sum(rounds.values())
    total_ms = sum(wall_ms.values())
    stats = []
    for label in sorted(rounds):
        stats.append({
            "source": source,
            "policy": label,
            "rounds": rounds[label],
            "conclusive_rounds": conclusive[label],
            "witnesses": kills[label],
            "hit_rate_per_round": round(kills[label] / rounds[label], 6),
            "budget_share_rounds": round(rounds[label] / total_rounds, 6) if total_rounds else None,
            "budget_share_wall_ms": round(wall_ms[label] / total_ms, 6) if total_ms else None,
            "wall_ms_total": round(wall_ms[label], 1),
        })
    return stats


# ---------------------------------------------------------------------------
# §5.7 C1-side cost
# ---------------------------------------------------------------------------

MS_PER_HOUR = 3_600_000.0


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (numpy 'linear'), torch/numpy-free."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def cost_stats(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(v) for v in values if isinstance(v, (int, float))]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median_ms": round(percentile(values, 0.5), 1),
        "p95_ms": round(percentile(values, 0.95), 1),
        "mean_ms": round(sum(values) / len(values), 1),
        "total_ms": round(sum(values), 1),
        "total_gpu_hours": round(sum(values) / MS_PER_HOUR, 6),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def provenance(dataset: Mapping[str, Any], scope_label: str) -> Dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "data_scope": scope_label,
        "inputs": dataset["inventory"],
        "dedup_dropped": dataset["dedup_dropped"],
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


DEFAULT_SCOPE_INTERIM = (
    "interim: CSE lanes still in flight; recompute after CSE completion "
    "for the final-scope figures")
