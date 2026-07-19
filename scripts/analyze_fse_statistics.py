#!/usr/bin/env python3
"""Run the deterministic, audit-aware statistical analysis for the FSE study.

The primary estimand is in-contract defect-detection coverage.  Its gold
population contains only subjects independently labelled
``CONFIRMED_IN_CONTRACT_DEFECT``.  A validator counts as detecting such a
subject only when one of that strategy's alarms is independently labelled
``CONFIRMED_IN_CONTRACT_DISCREPANCY``.

No third-party statistics package is required.  Bootstrap randomness is local
to ``random.Random`` and controlled by an explicit seed, so identical inputs
and CLI arguments produce byte-identical JSON output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import ObservationLog, RunManifest


SCHEMA_VERSION = "1.0"
PRIMARY_SCOPE = "in_contract"
GOLD_SUBJECT_LABEL = "CONFIRMED_IN_CONTRACT_DEFECT"
POSITIVE_ALARM_LABEL = "CONFIRMED_IN_CONTRACT_DISCREPANCY"
INCONCLUSIVE_LABEL = "INCONCLUSIVE"
SUBJECT_LABELS = {
    GOLD_SUBJECT_LABEL,
    "EXTENDED_CONTRACT_FAILURE",
    "REFERENCE_OR_ORACLE_FAILURE",
    "INFRASTRUCTURE_FAILURE",
    "NO_DEFECT_FOUND",
    INCONCLUSIVE_LABEL,
}
ALARM_LABELS = {
    POSITIVE_ALARM_LABEL,
    "CONFIRMED_EXTENDED_CONTRACT_DISCREPANCY",
    "INVALID_INPUT",
    "REFERENCE_OR_ORACLE_FAILURE",
    "INFRASTRUCTURE_FAILURE",
    INCONCLUSIVE_LABEL,
}
VALID_STATUSES = {"pass", "fail", "inconclusive"}


class StatisticsError(ValueError):
    """The observation/audit inputs cannot support a sound paired analysis."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StatisticsError(
                    f"invalid {label} JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise StatisticsError(
                    f"{label} at {path}:{line_number} must be an object"
                )
            rows.append(row)
    if not rows:
        raise StatisticsError(f"{label} file is empty: {path}")
    return rows


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StatisticsError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise StatisticsError(f"{label} must be a JSON object")
    return value


def _read_audit_rows(path: Path) -> list[Mapping[str, Any]]:
    """Read compiled analysis labels as JSONL, a JSON list, or an envelope.

    ``compile_human_audit.py`` emits JSONL analysis labels.  Supporting a JSON
    list or ``{"analysis_labels": [...]}`` envelope makes archival packaging
    convenient.  The aggregate audit report alone is deliberately rejected:
    it has no item-level labels and therefore cannot support these statistics.
    """

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise StatisticsError(f"compiled human audit is empty: {path}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _read_jsonl(path, "compiled human audit")

    if isinstance(parsed, list):
        rows = parsed
    elif isinstance(parsed, Mapping) and isinstance(parsed.get("analysis_labels"), list):
        rows = parsed["analysis_labels"]
    elif isinstance(parsed, Mapping) and parsed.get("record_type") in {"subject", "alarm"}:
        rows = [parsed]
    else:
        raise StatisticsError(
            "compiled human audit JSON must contain item-level analysis_labels; "
            "the aggregate audit report is insufficient"
        )
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise StatisticsError("compiled human audit labels must be non-empty objects")
    return list(rows)


def _verify_row_digest(row: Mapping[str, Any]) -> None:
    expected = row.get("analysis_label_sha256")
    if not isinstance(expected, str):
        raise StatisticsError(
            "compiled analysis label has no analysis_label_sha256; use the output "
            "of compile_human_audit.py"
        )
    payload = dict(row)
    del payload["analysis_label_sha256"]
    if _stable_sha256(payload) != expected:
        raise StatisticsError("compiled human audit analysis-label digest mismatch")


def _load_labels(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[Dict[tuple[str, str], str], Dict[str, tuple[str, str, str]]]:
    subject_labels: Dict[tuple[str, str], str] = {}
    alarm_labels: Dict[str, tuple[str, str, str]] = {}
    audit_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        _verify_row_digest(row)
        audit_id = row.get("audit_id")
        if not isinstance(audit_id, str) or not audit_id:
            raise StatisticsError(f"compiled audit row {index} has no audit_id")
        if audit_id in audit_ids:
            raise StatisticsError(f"duplicate compiled audit_id {audit_id}")
        audit_ids.add(audit_id)
        record_type = row.get("record_type")
        scope = row.get("scope", PRIMARY_SCOPE)
        label = row.get("primary_label")
        if scope not in {"in_contract", "extended_contract"}:
            raise StatisticsError(f"audit row {index} has invalid scope {scope!r}")
        if record_type == "subject":
            subject_id = row.get("subject_id")
            if not isinstance(subject_id, str) or not subject_id:
                raise StatisticsError(f"subject audit row {index} has no subject_id")
            if label not in SUBJECT_LABELS:
                raise StatisticsError(f"subject audit row {index} has invalid label")
            key = (subject_id, str(scope))
            if key in subject_labels and subject_labels[key] != label:
                raise StatisticsError(f"conflicting subject labels for {key}")
            subject_labels[key] = str(label)
        elif record_type == "alarm":
            test_id = row.get("test_id")
            subject_id = row.get("subject_id")
            if not isinstance(test_id, str) or not test_id:
                raise StatisticsError(f"alarm audit row {index} has no test_id")
            if not isinstance(subject_id, str) or not subject_id:
                raise StatisticsError(f"alarm audit row {index} has no subject_id")
            if label not in ALARM_LABELS:
                raise StatisticsError(f"alarm audit row {index} has invalid label")
            value = (str(scope), str(label), subject_id)
            if test_id in alarm_labels and alarm_labels[test_id] != value:
                raise StatisticsError(f"conflicting alarm labels for {test_id}")
            alarm_labels[test_id] = value
        else:
            raise StatisticsError(f"audit row {index} has invalid record_type")
    return subject_labels, alarm_labels


def _verify_embedded_digest(
    value: Mapping[str, Any], field: str, label: str
) -> None:
    expected = value.get(field)
    if not isinstance(expected, str):
        raise StatisticsError(f"{label} has no {field}")
    payload = dict(value)
    del payload[field]
    if _stable_sha256(payload) != expected:
        raise StatisticsError(f"{label} {field} mismatch")


def _verify_audit_report(
    report: Optional[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    *,
    run_manifest_sha256: str,
    plan_sha256: str,
    subject_manifest_sha256: str,
    observations_sha256: str,
) -> Dict[str, Any]:
    if report is None:
        return {
            "status": "not_supplied",
            "association_check": (
                "Per-label digests were verified, but no aggregate audit report was "
                "supplied for count/hash cross-checking."
            ),
        }
    _verify_embedded_digest(report, "audit_report_sha256", "audit report")
    if report.get("audit_complete") is not True:
        raise StatisticsError(
            "audit report is incomplete; --allow-partial outputs cannot feed "
            "paper statistics"
        )
    pending = report.get("pending")
    if not isinstance(pending, list) or pending:
        raise StatisticsError("audit report has pending items")
    for field, expected in (
        ("run_manifest_sha256", run_manifest_sha256),
        ("experiment_plan_sha256", plan_sha256),
        ("subject_manifest_sha256", subject_manifest_sha256),
        ("observations_sha256", observations_sha256),
    ):
        if report.get(field) != expected:
            raise StatisticsError(
                f"audit report {field} is not bound to the analyzed experiment"
            )
    final_labels = report.get("final_labels")
    if isinstance(final_labels, bool) or not isinstance(final_labels, int):
        raise StatisticsError("audit report final_labels must be an integer")
    if final_labels != len(audit_rows):
        raise StatisticsError(
            "audit report final_labels does not match compiled analysis-label count"
        )
    expected_collection_digest = report.get("analysis_labels_sha256")
    if not isinstance(expected_collection_digest, str):
        raise StatisticsError("audit report has no analysis_labels_sha256")
    actual_collection_digest = _stable_sha256(list(audit_rows))
    if actual_collection_digest != expected_collection_digest:
        raise StatisticsError(
            "audit report analysis_labels_sha256 does not match the exact "
            "compiled analysis-label collection"
        )
    return {
        "status": "verified",
        "audit_report_sha256": report["audit_report_sha256"],
        "final_labels": final_labels,
        "analysis_labels_sha256": expected_collection_digest,
        "association_check": (
            "The aggregate report digest, final-label count, exact ordered label-set "
            "digest, and every item-level analysis-label digest were verified."
        ),
    }


def _verify_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    _verify_embedded_digest(plan, "plan_sha256", "experiment plan")
    if plan.get("experiment_scope") != PRIMARY_SCOPE:
        raise StatisticsError("primary statistics require an in_contract experiment plan")
    schedule = plan.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise StatisticsError("experiment plan has no schedule")
    if plan.get("schedule_sha256") != _stable_sha256(schedule):
        raise StatisticsError("experiment plan schedule_sha256 mismatch")
    if plan.get("test_case_count") != len(schedule):
        raise StatisticsError("experiment plan test_case_count mismatch")

    strategies = plan.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise StatisticsError("experiment plan has no strategies")
    if plan.get("strategy_count") != len(strategies):
        raise StatisticsError("experiment plan strategy_count mismatch")
    strategy_ids: Dict[str, str] = {}
    strategy_budgets: Dict[str, int] = {}
    for index, strategy in enumerate(strategies, start=1):
        if not isinstance(strategy, Mapping):
            raise StatisticsError(f"plan strategy {index} is not an object")
        name = strategy.get("name")
        strategy_id = strategy.get("strategy_id")
        budget = strategy.get("candidate_runs_per_subject")
        if not isinstance(name, str) or not name:
            raise StatisticsError(f"plan strategy {index} has no name")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise StatisticsError(f"plan strategy {index} has no strategy_id")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise StatisticsError(
                f"plan strategy {name!r} has invalid candidate_runs_per_subject"
            )
        if name in strategy_ids:
            raise StatisticsError(f"duplicate strategy name in plan: {name}")
        strategy_ids[name] = strategy_id
        strategy_budgets[name] = budget

    expected_by_test: Dict[str, Mapping[str, Any]] = {}
    scheduled_subjects: set[str] = set()
    scheduled_pairs: Dict[tuple[str, str], int] = defaultdict(int)
    for index, entry in enumerate(schedule, start=1):
        if not isinstance(entry, Mapping):
            raise StatisticsError(f"plan schedule entry {index} is not an object")
        test_id = entry.get("test_id")
        subject_id = entry.get("subject_id")
        strategy_name = entry.get("strategy_name")
        strategy_id = entry.get("strategy_id")
        candidate_cost = entry.get("candidate_run_cost")
        dataset = entry.get("dataset")
        task_id = entry.get("task_id")
        for field, value in (
            ("test_id", test_id),
            ("subject_id", subject_id),
            ("strategy_name", strategy_name),
            ("strategy_id", strategy_id),
            ("dataset", dataset),
            ("task_id", task_id),
        ):
            if not isinstance(value, str) or not value:
                raise StatisticsError(f"plan schedule entry {index} has no {field}")
        if test_id in expected_by_test:
            raise StatisticsError(f"duplicate test_id in plan schedule: {test_id}")
        if strategy_name not in strategy_ids:
            raise StatisticsError(
                f"plan schedule references unknown strategy {strategy_name!r}"
            )
        if strategy_ids[str(strategy_name)] != strategy_id:
            raise StatisticsError(
                f"plan schedule strategy identity mismatch for {strategy_name!r}"
            )
        if isinstance(candidate_cost, bool) or not isinstance(candidate_cost, int):
            raise StatisticsError(f"plan schedule entry {index} has invalid candidate cost")
        if candidate_cost <= 0:
            raise StatisticsError(f"plan schedule entry {index} has invalid candidate cost")
        if entry.get("scope") != PRIMARY_SCOPE:
            raise StatisticsError("primary plan contains a non-in-contract schedule entry")
        expected_by_test[str(test_id)] = entry
        scheduled_subjects.add(str(subject_id))
        scheduled_pairs[(str(subject_id), str(strategy_name))] += candidate_cost

    if plan.get("subject_count") != len(scheduled_subjects):
        raise StatisticsError("experiment plan subject_count mismatch")
    for subject_id in scheduled_subjects:
        for strategy_name, budget in strategy_budgets.items():
            actual = scheduled_pairs.get((subject_id, strategy_name), 0)
            if actual != budget:
                raise StatisticsError(
                    f"plan candidate budget mismatch for {subject_id}/{strategy_name}: "
                    f"expected {budget}, scheduled {actual}"
                )
    return {
        "expected_by_test": expected_by_test,
        "scheduled_subjects": scheduled_subjects,
        "strategy_ids": strategy_ids,
        "strategy_budgets": strategy_budgets,
    }


def _verify_run_and_observation_binding(
    *,
    run_manifest: Mapping[str, Any],
    run_manifest_path: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    observations: Sequence[Mapping[str, Any]],
    plan_context: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        RunManifest.verify_dict(run_manifest)
    except ValueError as exc:
        raise StatisticsError(f"run manifest integrity check failed: {exc}") from exc
    config = run_manifest.get("config")
    if not isinstance(config, Mapping):
        raise StatisticsError("run manifest has no config object")
    if run_manifest.get("config_sha256") != _stable_sha256(config):
        raise StatisticsError("run manifest config_sha256 mismatch")
    if run_manifest.get("experiment") != "fse-validator-comparison":
        raise StatisticsError("run manifest is not an FSE validator-comparison run")
    if config.get("plan_sha256") != plan.get("plan_sha256"):
        raise StatisticsError("run manifest is bound to another experiment plan")
    if config.get("plan_file_sha256") != _sha256_file(plan_path):
        raise StatisticsError("run manifest plan_file_sha256 mismatch")
    if config.get("subject_manifest_sha256") != plan.get("subject_manifest_sha256"):
        raise StatisticsError("run manifest and plan use different subject manifests")

    manifest_subjects = run_manifest.get("subjects")
    if not isinstance(manifest_subjects, list) or not manifest_subjects:
        raise StatisticsError("run manifest has no subjects")
    subject_metadata: Dict[str, tuple[str, str]] = {}
    for index, subject in enumerate(manifest_subjects, start=1):
        if not isinstance(subject, Mapping):
            raise StatisticsError(f"run manifest subject {index} is not an object")
        subject_id = subject.get("subject_id")
        dataset = subject.get("dataset")
        task_id = subject.get("task_id")
        if any(not isinstance(value, str) or not value for value in (subject_id, dataset, task_id)):
            raise StatisticsError(f"run manifest subject {index} has invalid identity")
        if subject_id in subject_metadata:
            raise StatisticsError(f"duplicate run-manifest subject {subject_id}")
        subject_metadata[str(subject_id)] = (str(dataset), str(task_id))

    runner = config.get("runner")
    if not isinstance(runner, Mapping):
        raise StatisticsError("run manifest has no runner configuration")
    if runner.get("early_stop_on_fail") is not False:
        raise StatisticsError(
            "complete paired statistics reject early_stop_on_fail runs"
        )
    if runner.get("max_wall_ms_per_subject_strategy") is not None:
        raise StatisticsError(
            "complete paired statistics reject wall-censored runs"
        )

    all_subjects = set(plan_context["scheduled_subjects"])
    all_strategies = set(plan_context["strategy_ids"])

    def selected(raw: Any, universe: set[str], label: str) -> set[str]:
        if raw is None:
            return set(universe)
        if not isinstance(raw, list) or any(
            not isinstance(item, str) or not item for item in raw
        ):
            raise StatisticsError(f"run manifest {label} selection is invalid")
        if len(raw) != len(set(raw)):
            raise StatisticsError(f"run manifest {label} selection contains duplicates")
        result = set(raw)
        unknown = result - universe
        if unknown:
            raise StatisticsError(
                f"run manifest {label} selection is unknown to plan: {sorted(unknown)}"
            )
        if not result:
            raise StatisticsError(f"run manifest {label} selection is empty")
        return result

    selected_subjects = selected(
        runner.get("selected_subjects"), all_subjects, "subject"
    )
    selected_strategies = selected(
        runner.get("selected_strategies"), all_strategies, "strategy"
    )
    if not selected_subjects <= set(subject_metadata):
        raise StatisticsError("selected plan subjects are absent from run manifest")

    expected_by_test = {
        test_id: entry
        for test_id, entry in plan_context["expected_by_test"].items()
        if entry["subject_id"] in selected_subjects
        and entry["strategy_name"] in selected_strategies
    }
    observed_by_test = {str(row.get("test_id")): row for row in observations}
    missing = sorted(set(expected_by_test) - set(observed_by_test))
    extra = sorted(set(observed_by_test) - set(expected_by_test))
    if missing or extra:
        raise StatisticsError(
            "observation set is incomplete or stale relative to the selected plan: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)"
        )

    run_manifest_sha256 = run_manifest.get("manifest_sha256")
    observed_pair_budgets: Dict[tuple[str, str], int] = defaultdict(int)
    expected_pair_budgets: Dict[tuple[str, str], int] = defaultdict(int)
    actual_candidate_runs = 0
    for test_id, expected in expected_by_test.items():
        row = observed_by_test[test_id]
        if row.get("run_manifest_sha256") != run_manifest_sha256:
            raise StatisticsError(
                f"observation {test_id} references another run manifest"
            )
        if row.get("run_id") != run_manifest.get("run_id"):
            raise StatisticsError(f"observation {test_id} has a stale run_id")
        for field, expected_value in expected.items():
            if row.get(field) != expected_value:
                raise StatisticsError(
                    f"observation {test_id} disagrees with plan field {field}"
                )
        planned = row.get("planned_candidate_runs")
        if planned != expected["candidate_run_cost"]:
            raise StatisticsError(
                f"observation {test_id} planned candidate cost disagrees with plan"
            )
        actual = row.get("observed_candidate_runs")
        if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
            raise StatisticsError(
                f"observation {test_id} has invalid observed candidate-run count"
            )
        if actual > int(planned):
            raise StatisticsError(
                f"observation {test_id} exceeds its planned candidate-run count"
            )
        if row.get("validation_status") == "pass" and actual != int(planned):
            raise StatisticsError(
                f"passing observation {test_id} did not execute every planned "
                "candidate invocation"
            )
        actual_candidate_runs += actual
        pair = (str(expected["subject_id"]), str(expected["strategy_name"]))
        observed_pair_budgets[pair] += int(planned)
        expected_pair_budgets[pair] += int(expected["candidate_run_cost"])
        identity = subject_metadata[str(expected["subject_id"])]
        if identity != (expected["dataset"], expected["task_id"]):
            raise StatisticsError(
                f"plan identity for subject {expected['subject_id']} disagrees with run manifest"
            )
    if observed_pair_budgets != expected_pair_budgets:
        raise StatisticsError("observation candidate budgets do not match selected plan")

    return {
        "selected_subjects": selected_subjects,
        "selected_strategies": selected_strategies,
        "subject_metadata": subject_metadata,
        "expected_test_ids": set(expected_by_test),
        "observed_by_test": observed_by_test,
        "expected_test_count": len(expected_by_test),
        "expected_candidate_runs": sum(
            int(entry["candidate_run_cost"]) for entry in expected_by_test.values()
        ),
        "actual_candidate_runs": actual_candidate_runs,
        "run_manifest_file_sha256": _sha256_file(run_manifest_path),
        "plan_file_sha256": _sha256_file(plan_path),
    }


def _validate_label_membership(
    *,
    subject_labels: Mapping[tuple[str, str], str],
    alarm_labels: Mapping[str, tuple[str, str, str]],
    binding: Mapping[str, Any],
) -> None:
    for subject_id, scope in subject_labels:
        if scope != PRIMARY_SCOPE or subject_id not in binding["selected_subjects"]:
            raise StatisticsError(
                f"compiled audit contains unknown/stale subject label {subject_id}/{scope}"
            )
    for test_id, (scope, _, subject_id) in alarm_labels.items():
        row = binding["observed_by_test"].get(test_id)
        if row is None or test_id not in binding["expected_test_ids"]:
            raise StatisticsError(
                f"compiled audit contains unknown/stale alarm test_id {test_id}"
            )
        if row.get("validation_status") != "fail":
            raise StatisticsError(f"compiled alarm label {test_id} does not name an alarm")
        if row.get("scope") != scope or row.get("subject_id") != subject_id:
            raise StatisticsError(
                f"compiled alarm label {test_id} identity disagrees with observation"
            )


def _validate_observations(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[Mapping[str, Any]],
    Dict[str, tuple[str, str]],
    Dict[str, str],
    Dict[str, str],
]:
    primary: list[Mapping[str, Any]] = []
    cluster_by_subject: Dict[str, tuple[str, str]] = {}
    strategy_id_by_name: Dict[str, str] = {}
    scope_by_test: Dict[str, str] = {}
    seen_test_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        test_id = row.get("test_id")
        subject_id = row.get("subject_id")
        strategy_name = row.get("strategy_name")
        strategy_id = row.get("strategy_id")
        task_id = row.get("task_id")
        dataset = row.get("dataset")
        scope = row.get("scope")
        status = row.get("validation_status")
        for value, field in (
            (test_id, "test_id"),
            (subject_id, "subject_id"),
            (strategy_name, "strategy_name"),
            (strategy_id, "strategy_id"),
        ):
            if not isinstance(value, str) or not value:
                raise StatisticsError(f"observation {index} has no {field}")
        if test_id in seen_test_ids:
            raise StatisticsError(f"duplicate observation test_id {test_id}")
        seen_test_ids.add(str(test_id))
        if status not in VALID_STATUSES:
            raise StatisticsError(f"observation {index} has invalid validation_status")
        if scope not in {"in_contract", "extended_contract"}:
            raise StatisticsError(f"observation {index} has invalid scope")
        scope_by_test[str(test_id)] = str(scope)
        known_strategy_id = strategy_id_by_name.setdefault(
            str(strategy_name), str(strategy_id)
        )
        if known_strategy_id != strategy_id:
            raise StatisticsError(
                f"strategy name {strategy_name!r} maps to multiple strategy IDs"
            )
        if scope != PRIMARY_SCOPE:
            continue
        if not isinstance(task_id, str) or not task_id:
            raise StatisticsError(f"in-contract observation {index} has no task_id")
        if not isinstance(dataset, str) or not dataset:
            raise StatisticsError(f"in-contract observation {index} has no dataset")
        cluster = (dataset, task_id)
        known_cluster = cluster_by_subject.setdefault(str(subject_id), cluster)
        if known_cluster != cluster:
            raise StatisticsError(
                f"subject {subject_id!r} maps to multiple dataset/task identities"
            )
        for field in ("planned_candidate_runs", "observed_candidate_runs"):
            value = row.get(field, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StatisticsError(f"observation {index} has invalid {field}")
        wall = row.get("parent_wall_ms", 0.0)
        if isinstance(wall, bool) or not isinstance(wall, (int, float)):
            raise StatisticsError(f"observation {index} has invalid parent_wall_ms")
        if not math.isfinite(float(wall)) or float(wall) < 0.0:
            raise StatisticsError(f"observation {index} has invalid parent_wall_ms")
        primary.append(row)
    if not primary:
        raise StatisticsError("no in-contract observations are available")
    return primary, cluster_by_subject, strategy_id_by_name, scope_by_test


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap_ci(
    *,
    values_by_subject: Mapping[str, int],
    cluster_by_subject: Mapping[str, tuple[str, str]],
    replicates: int,
    seed: int,
    confidence_level: float,
) -> Dict[str, Any]:
    if not values_by_subject:
        return {
            "method": "canonical_dataset_task_cluster_percentile_bootstrap",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": confidence_level,
            "task_clusters": 0,
            "lower": None,
            "upper": None,
        }
    clusters: Dict[tuple[str, str], list[str]] = defaultdict(list)
    for subject_id in sorted(values_by_subject):
        try:
            clusters[cluster_by_subject[subject_id]].append(subject_id)
        except KeyError as exc:
            raise StatisticsError(
                f"gold subject {subject_id!r} has no canonical dataset/task cluster"
            ) from exc
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        selected = [rng.choice(cluster_ids) for _ in cluster_ids]
        numerator = 0
        denominator = 0
        for cluster_id in selected:
            for subject_id in clusters[cluster_id]:
                numerator += values_by_subject[subject_id]
                denominator += 1
        estimates.append(numerator / denominator)
    alpha = 1.0 - confidence_level
    return {
        "method": "canonical_dataset_task_cluster_percentile_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "task_clusters": len(cluster_ids),
        "lower": _percentile(estimates, alpha / 2.0),
        "upper": _percentile(estimates, 1.0 - alpha / 2.0),
    }


def _mcnemar_exact(a_detects_b_not: int, b_detects_a_not: int) -> float:
    """Return the two-sided exact McNemar/binomial p-value."""

    if min(a_detects_b_not, b_detects_a_not) < 0:
        raise StatisticsError("McNemar discordant counts must be non-negative")
    discordant = a_detects_b_not + b_detects_a_not
    if discordant == 0:
        return 1.0
    tail = min(a_detects_b_not, b_detects_a_not)
    numerator = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**discordant))


def _holm_adjust(
    hypotheses: Sequence[tuple[str, float]], alpha: float
) -> Dict[str, Dict[str, Any]]:
    ordered = sorted(hypotheses, key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    adjusted: Dict[str, Dict[str, Any]] = {}
    running = 0.0
    for rank, (identifier, raw_p) in enumerate(ordered, start=1):
        value = min(1.0, (family_size - rank + 1) * raw_p)
        running = max(running, value)
        adjusted[identifier] = {
            "holm_rank": rank,
            "raw_p_value": raw_p,
            "holm_adjusted_p_value": running,
            "reject_at_alpha": running <= alpha,
        }
    return adjusted


def _classify_subject_strategy(
    records: Sequence[Mapping[str, Any]],
    alarm_labels: Mapping[str, tuple[str, str, str]],
) -> str:
    for row in records:
        if row["validation_status"] != "fail":
            continue
        label = alarm_labels.get(str(row["test_id"]))
        if label is not None and label[:2] == (PRIMARY_SCOPE, POSITIVE_ALARM_LABEL):
            return "confirmed_detected"
    unresolved_alarm = any(
        row["validation_status"] == "fail"
        and (
            str(row["test_id"]) not in alarm_labels
            or alarm_labels[str(row["test_id"])][1] == INCONCLUSIVE_LABEL
        )
        for row in records
    )
    validator_inconclusive = any(
        row["validation_status"] == "inconclusive" for row in records
    )
    if unresolved_alarm or validator_inconclusive:
        return "unresolved"
    return "not_detected"


def _alarm_precision(
    records: Sequence[Mapping[str, Any]],
    alarm_labels: Mapping[str, tuple[str, str, str]],
) -> Dict[str, Any]:
    alarms = [row for row in records if row["validation_status"] == "fail"]
    confirmed = 0
    rejected = 0
    inconclusive = 0
    unaudited = 0
    for row in alarms:
        label = alarm_labels.get(str(row["test_id"]))
        if label is None:
            unaudited += 1
        elif label[:2] == (PRIMARY_SCOPE, POSITIVE_ALARM_LABEL):
            confirmed += 1
        elif label[1] == INCONCLUSIVE_LABEL:
            inconclusive += 1
        else:
            # A conclusive label other than an in-contract discrepancy does not
            # substantiate an in-contract alarm and belongs in the denominator.
            rejected += 1
    denominator = confirmed + rejected
    return {
        "alarms": len(alarms),
        "conclusively_audited_alarms": denominator,
        "confirmed_in_contract_discrepancies": confirmed,
        "conclusively_rejected_in_contract_alarms": rejected,
        "inconclusive_audit_labels": inconclusive,
        "unaudited_alarms": unaudited,
        "precision": None if denominator == 0 else confirmed / denominator,
        "denominator_policy": (
            "Only conclusive item-level alarm audits are included; unaudited and "
            "INCONCLUSIVE alarms are excluded."
        ),
    }


def _frontier(
    rows: Sequence[Mapping[str, Any]], cost_fields: Sequence[str]
) -> list[str]:
    eligible = [row for row in rows if row["pareto_eligible"]]
    frontier: list[str] = []
    for candidate in eligible:
        dominated = False
        for challenger in eligible:
            if challenger is candidate:
                continue
            no_worse = challenger["coverage"] >= candidate["coverage"] and all(
                challenger[field] <= candidate[field] for field in cost_fields
            )
            strictly_better = (
                challenger["coverage"] > candidate["coverage"]
                or any(challenger[field] < candidate[field] for field in cost_fields)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate["strategy_name"]))
    return sorted(frontier)


def analyze(
    *,
    run_manifest_path: Path,
    plan_path: Path,
    observations_path: Path,
    human_audit_path: Path,
    audit_report_path: Optional[Path] = None,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260719,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Load, authenticate, and analyze one complete manifest-bound experiment."""

    run_manifest_path = run_manifest_path.resolve()
    plan_path = plan_path.resolve()
    observations_path = observations_path.resolve()
    human_audit_path = human_audit_path.resolve()
    audit_report_path = (
        None if audit_report_path is None else audit_report_path.resolve()
    )
    run_manifest = _read_object(run_manifest_path, "run manifest")
    plan = _read_object(plan_path, "experiment plan")
    plan_context = _verify_plan(plan)
    try:
        observations = list(ObservationLog(observations_path).records())
    except ValueError as exc:
        raise StatisticsError(f"observation log integrity check failed: {exc}") from exc
    with observations_path.open("r", encoding="utf-8") as handle:
        physical_records = sum(bool(line.strip()) for line in handle)
    if physical_records != len(observations):
        raise StatisticsError(
            "observation log repeats an identical test_id; each selected plan test "
            "must have exactly one physical record"
        )
    binding = _verify_run_and_observation_binding(
        run_manifest=run_manifest,
        run_manifest_path=run_manifest_path,
        plan=plan,
        plan_path=plan_path,
        observations=observations,
        plan_context=plan_context,
    )
    audit_rows = _read_audit_rows(human_audit_path)
    audit_report = (
        None
        if audit_report_path is None
        else _read_object(audit_report_path, "audit report")
    )
    audit_report_verification = _verify_audit_report(
        audit_report,
        audit_rows,
        run_manifest_sha256=str(run_manifest.get("manifest_sha256")),
        plan_sha256=str(plan.get("plan_sha256")),
        subject_manifest_sha256=str(plan.get("subject_manifest_sha256")),
        observations_sha256=_stable_sha256(
            sorted(observations, key=lambda row: str(row.get("test_id")))
        ),
    )
    return _analyze_verified_records(
        observations=observations,
        audit_rows=audit_rows,
        binding=binding,
        audit_report_verification=audit_report_verification,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        confidence_level=confidence_level,
        alpha=alpha,
        run_manifest_sha256=str(run_manifest["manifest_sha256"]),
        plan_sha256=str(plan["plan_sha256"]),
        observations_sha256=_sha256_file(observations_path),
        audit_sha256=_sha256_file(human_audit_path),
        audit_report_file_sha256=(
            None if audit_report_path is None else _sha256_file(audit_report_path)
        ),
    )


def _analyze_verified_records(
    *,
    observations: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    audit_report_verification: Mapping[str, Any],
    bootstrap_replicates: int = 10_000,
    seed: int = 20260719,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
    run_manifest_sha256: Optional[str] = None,
    plan_sha256: Optional[str] = None,
    observations_sha256: Optional[str] = None,
    audit_sha256: Optional[str] = None,
    audit_report_file_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise StatisticsError("bootstrap_replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise StatisticsError("confidence_level must be between zero and one")
    if not 0.0 < alpha < 1.0:
        raise StatisticsError("alpha must be between zero and one")

    primary, cluster_by_subject, strategy_ids, scope_by_test = _validate_observations(
        observations
    )
    subject_labels, alarm_labels = _load_labels(audit_rows)
    _validate_label_membership(
        subject_labels=subject_labels,
        alarm_labels=alarm_labels,
        binding=binding,
    )
    for test_id, (scope, _, _) in alarm_labels.items():
        if test_id in scope_by_test and scope_by_test[test_id] != scope:
            raise StatisticsError(
                f"alarm audit scope for {test_id} disagrees with its observation"
            )

    observed_subjects = set(cluster_by_subject)
    labelled_gold = {
        subject_id
        for (subject_id, scope), label in subject_labels.items()
        if scope == PRIMARY_SCOPE and label == GOLD_SUBJECT_LABEL
    }
    gold_subjects = sorted(labelled_gold & observed_subjects)
    ignored_gold = sorted(labelled_gold - observed_subjects)
    no_defect_found_count = sum(
        scope == PRIMARY_SCOPE and label == "NO_DEFECT_FOUND"
        for (_, scope), label in subject_labels.items()
    )

    records_by_strategy_subject: Dict[
        str, Dict[str, list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    records_by_strategy: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary:
        strategy_name = str(row["strategy_name"])
        subject_id = str(row["subject_id"])
        records_by_strategy_subject[strategy_name][subject_id].append(row)
        records_by_strategy[strategy_name].append(row)

    strategy_results: list[Dict[str, Any]] = []
    binary_detection: Dict[str, Dict[str, int]] = {}
    classifications: Dict[str, Dict[str, str]] = {}
    global_subject_count = len(observed_subjects)
    for strategy_name in sorted(records_by_strategy):
        per_subject = records_by_strategy_subject[strategy_name]
        classification = {
            subject_id: (
                "not_executed"
                if subject_id not in per_subject
                else _classify_subject_strategy(per_subject[subject_id], alarm_labels)
            )
            for subject_id in gold_subjects
        }
        classifications[strategy_name] = classification
        binary = {
            subject_id: int(value == "confirmed_detected")
            for subject_id, value in classification.items()
        }
        binary_detection[strategy_name] = binary
        confirmed = sum(value == "confirmed_detected" for value in classification.values())
        unresolved = sum(value == "unresolved" for value in classification.values())
        not_detected = sum(value == "not_detected" for value in classification.values())
        not_executed = sum(value == "not_executed" for value in classification.values())
        denominator = len(gold_subjects)
        point = None if denominator == 0 else confirmed / denominator
        upper = None if denominator == 0 else (confirmed + unresolved) / denominator
        strategy_records = records_by_strategy[strategy_name]
        planned_total = sum(
            int(row.get("planned_candidate_runs", 0)) for row in strategy_records
        )
        observed_total = sum(
            int(row.get("observed_candidate_runs", 0)) for row in strategy_records
        )
        wall_total = sum(float(row.get("parent_wall_ms", 0.0)) for row in strategy_records)
        strategy_subjects = set(per_subject)
        # Completeness was already proved against every selected plan test ID
        # and budget in _verify_run_and_observation_binding.  It is not inferred
        # from merely seeing one row for each subject.
        complete = True
        strategy_results.append(
            {
                "strategy_name": strategy_name,
                "strategy_id": strategy_ids[strategy_name],
                "gold_subjects": denominator,
                "confirmed_detected_gold_subjects": confirmed,
                "unresolved_gold_subjects": unresolved,
                "not_detected_gold_subjects": not_detected,
                "not_executed_gold_subjects": not_executed,
                "confirmed_detection_coverage": point,
                "task_cluster_bootstrap_ci": _cluster_bootstrap_ci(
                    values_by_subject=binary,
                    cluster_by_subject=cluster_by_subject,
                    replicates=bootstrap_replicates,
                    seed=seed,
                    confidence_level=confidence_level,
                ),
                "inconclusive_detection_sensitivity_bounds": {
                    "worst_case": point,
                    "best_case": upper,
                    "unresolved_definition": (
                        "At least one validator-inconclusive execution or an unaudited/"
                        "INCONCLUSIVE alarm, with no independently confirmed alarm. "
                        "A strategy that was not executed is not promoted in the best case."
                    ),
                },
                "alarm_level_audited_precision": _alarm_precision(
                    strategy_records, alarm_labels
                ),
                "cost": {
                    "observed_corpus_subjects": len(strategy_subjects),
                    "complete_corpus_subject_coverage": complete,
                    "completeness_basis": "verified selected plan test IDs and budgets",
                    "planned_candidate_runs_total": planned_total,
                    "observed_candidate_runs_total": observed_total,
                    "parent_wall_ms_total": wall_total,
                    "planned_candidate_runs_per_corpus_subject": (
                        None if global_subject_count == 0 else planned_total / global_subject_count
                    ),
                    "observed_candidate_runs_per_corpus_subject": (
                        None if global_subject_count == 0 else observed_total / global_subject_count
                    ),
                    "parent_wall_ms_per_corpus_subject": (
                        None if global_subject_count == 0 else wall_total / global_subject_count
                    ),
                },
            }
        )

    comparisons: list[Dict[str, Any]] = []
    hypotheses: list[tuple[str, float]] = []
    strategy_names = sorted(records_by_strategy)
    for left_index, left in enumerate(strategy_names):
        for right in strategy_names[left_index + 1 :]:
            paired = [
                subject_id
                for subject_id in gold_subjects
                if classifications[left][subject_id] != "not_executed"
                and classifications[right][subject_id] != "not_executed"
            ]
            left_only = sum(
                binary_detection[left][subject_id] == 1
                and binary_detection[right][subject_id] == 0
                for subject_id in paired
            )
            right_only = sum(
                binary_detection[left][subject_id] == 0
                and binary_detection[right][subject_id] == 1
                for subject_id in paired
            )
            identifier = f"{left}__vs__{right}"
            raw_p = _mcnemar_exact(left_only, right_only)
            hypotheses.append((identifier, raw_p))
            comparisons.append(
                {
                    "comparison_id": identifier,
                    "strategy_a": left,
                    "strategy_b": right,
                    "paired_gold_subjects": len(paired),
                    "excluded_missing_strategy_execution": len(gold_subjects) - len(paired),
                    "a_detected_b_not": left_only,
                    "b_detected_a_not": right_only,
                    "discordant_pairs": left_only + right_only,
                    "raw_exact_two_sided_p_value": raw_p,
                    "binary_outcome": (
                        "independently confirmed same-strategy alarm; unresolved outcomes "
                        "are conservatively counted as not detected"
                    ),
                }
            )
    holm = _holm_adjust(hypotheses, alpha)
    for comparison in comparisons:
        comparison.update(holm[comparison["comparison_id"]])

    pareto_rows = []
    for row in strategy_results:
        cost = row["cost"]
        coverage = row["confirmed_detection_coverage"]
        pareto_rows.append(
            {
                "strategy_name": row["strategy_name"],
                "coverage": -1.0 if coverage is None else coverage,
                "planned_candidate_runs_per_corpus_subject": cost[
                    "planned_candidate_runs_per_corpus_subject"
                ],
                "parent_wall_ms_per_corpus_subject": cost[
                    "parent_wall_ms_per_corpus_subject"
                ],
                "pareto_eligible": bool(cost["complete_corpus_subject_coverage"])
                and coverage is not None,
            }
        )
    excluded_pareto = sorted(
        row["strategy_name"] for row in pareto_rows if not row["pareto_eligible"]
    )

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_scope": PRIMARY_SCOPE,
        "input_digests": {
            "run_manifest_sha256": run_manifest_sha256,
            "run_manifest_file_sha256": binding["run_manifest_file_sha256"],
            "plan_sha256": plan_sha256,
            "plan_file_sha256": binding["plan_file_sha256"],
            "observations_sha256": observations_sha256,
            "compiled_human_audit_sha256": audit_sha256,
            "audit_report_file_sha256": audit_report_file_sha256,
        },
        "verified_experiment_completeness": {
            "status": "complete",
            "expected_test_ids": binding["expected_test_count"],
            "observed_test_ids": len(observations),
            "expected_planned_candidate_runs": binding["expected_candidate_runs"],
            "observed_candidate_runs": binding["actual_candidate_runs"],
            "selected_subjects": len(binding["selected_subjects"]),
            "selected_strategies": len(binding["selected_strategies"]),
            "rule": (
                "Every selected plan test_id is present exactly once; all immutable "
                "schedule fields, per-test candidate costs, per-pair budgets, run_id, "
                "and run_manifest_sha256 were verified."
            ),
        },
        "audit_report_verification": dict(audit_report_verification),
        "configuration": {
            "bootstrap_replicates": bootstrap_replicates,
            "seed": seed,
            "confidence_level": confidence_level,
            "familywise_alpha": alpha,
        },
        "corpus": {
            "observations_total": len(observations),
            "in_contract_observations": len(primary),
            "in_contract_subjects_in_observations": global_subject_count,
            "confirmed_in_contract_gold_subjects": len(gold_subjects),
            "confirmed_gold_subjects_absent_from_observations": ignored_gold,
            "canonical_dataset_task_clusters_in_gold_corpus": len(
                {cluster_by_subject[subject_id] for subject_id in gold_subjects}
            ),
            "no_defect_found_subject_labels": no_defect_found_count,
            "gold_population_rule": (
                "Only subjects labelled CONFIRMED_IN_CONTRACT_DEFECT and present "
                "in the in-contract observation corpus enter detection coverage."
            ),
            "no_defect_found_interpretation": (
                "NO_DEFECT_FOUND is an audit outcome under finite evidence, not a "
                "true-negative correctness label; it never enters specificity or the "
                "coverage denominator."
            ),
        },
        "strategies": strategy_results,
        "paired_mcnemar_exact_tests": comparisons,
        "multiple_comparison_control": {
            "method": "Holm step-down familywise-error correction",
            "family_size": len(comparisons),
            "alpha": alpha,
        },
        "cost_detection_pareto_frontier": {
            "eligibility": (
                "Strategy must cover every in-contract corpus subject and have a "
                "non-null confirmed-defect coverage estimate."
            ),
            "excluded_incomplete_strategies": excluded_pareto,
            "joint_candidate_runs_and_wall_time": _frontier(
                pareto_rows,
                (
                    "planned_candidate_runs_per_corpus_subject",
                    "parent_wall_ms_per_corpus_subject",
                ),
            ),
            "candidate_runs_only": _frontier(
                pareto_rows, ("planned_candidate_runs_per_corpus_subject",)
            ),
            "wall_time_only": _frontier(
                pareto_rows, ("parent_wall_ms_per_corpus_subject",)
            ),
            "points": sorted(pareto_rows, key=lambda row: row["strategy_name"]),
        },
        "interpretation": {
            "coverage": (
                "Confirmed defect subjects with at least one independently confirmed "
                "same-strategy alarm divided by confirmed in-contract defect subjects."
            ),
            "bootstrap": (
                "Percentile bootstrap resamples canonical (dataset, task_id) clusters "
                "with replacement and "
                "keeps all subjects belonging to each sampled task cluster."
            ),
            "mcnemar": (
                "Two-sided exact McNemar tests are paired on confirmed-defect subjects; "
                "Holm correction covers every reported strategy pair."
            ),
            "precision": (
                "Alarm precision is conditional on conclusively audited alarms and must "
                "be reported with unaudited/inconclusive counts."
            ),
        },
    }
    payload["analysis_sha256"] = _stable_sha256(payload)
    return payload


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--human-audit", required=True, type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze(
            run_manifest_path=args.run_manifest,
            plan_path=args.plan,
            observations_path=args.observations,
            human_audit_path=args.human_audit,
            audit_report_path=args.audit_report,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            confidence_level=args.confidence_level,
            alpha=args.alpha,
        )
        _write_once(args.output, result)
    except FileExistsError:
        print(f"refusing to overwrite statistical analysis: {args.output}", file=sys.stderr)
        return 2
    except (OSError, StatisticsError, TypeError, ValueError) as exc:
        print(f"failed to analyze FSE statistics: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"output": str(args.output), "analysis_sha256": result["analysis_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
