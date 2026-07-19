#!/usr/bin/env python3
"""Summarize canonical FSE observations without re-executing candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import (
    ObservationLog,
    RunManifest,
    canonical_json_bytes,
    sha256_file,
    stable_json_sha256,
)


SUMMARY_SCHEMA_VERSION = "1.0"
SUBJECT_LABELS = {
    "CONFIRMED_IN_CONTRACT_DEFECT",
    "EXTENDED_CONTRACT_FAILURE",
    "REFERENCE_OR_ORACLE_FAILURE",
    "INFRASTRUCTURE_FAILURE",
    "NO_DEFECT_FOUND",
    "INCONCLUSIVE",
}
ALARM_LABELS = {
    "CONFIRMED_IN_CONTRACT_DISCREPANCY",
    "CONFIRMED_EXTENDED_CONTRACT_DISCREPANCY",
    "INVALID_INPUT",
    "REFERENCE_OR_ORACLE_FAILURE",
    "INFRASTRUCTURE_FAILURE",
    "INCONCLUSIVE",
}


class SummaryError(ValueError):
    """Canonical observations or optional human labels are invalid."""


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SummaryError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SummaryError(f"{label} must be a JSON object")
    return value


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


def _load_labels(
    path: Optional[Path],
) -> Tuple[Dict[Tuple[str, str], str], Dict[str, str]]:
    if path is None:
        return {}, {}
    subject_labels: Dict[Tuple[str, str], str] = {}
    alarm_labels: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SummaryError(f"invalid label JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise SummaryError(f"label at {path}:{line_number} must be an object")
            record_type = value.get("record_type")
            scope = value.get("scope", "in_contract")
            label = value.get("primary_label")
            if scope not in {"in_contract", "extended_contract"}:
                raise SummaryError(f"label at {path}:{line_number} has invalid scope")
            if record_type == "subject":
                subject_id = value.get("subject_id")
                if not isinstance(subject_id, str) or not subject_id:
                    raise SummaryError(
                        f"subject label at {path}:{line_number} has no subject_id"
                    )
                if label not in SUBJECT_LABELS:
                    raise SummaryError(
                        f"subject label at {path}:{line_number} is invalid"
                    )
                key = (subject_id, scope)
                if key in subject_labels and subject_labels[key] != label:
                    raise SummaryError(f"conflicting labels for {subject_id}/{scope}")
                subject_labels[key] = str(label)
            elif record_type == "alarm":
                test_id = value.get("test_id")
                if not isinstance(test_id, str) or not test_id:
                    raise SummaryError(
                        f"alarm label at {path}:{line_number} has no test_id"
                    )
                if label not in ALARM_LABELS:
                    raise SummaryError(
                        f"alarm label at {path}:{line_number} is invalid"
                    )
                if test_id in alarm_labels and alarm_labels[test_id] != label:
                    raise SummaryError(f"conflicting alarm labels for {test_id}")
                alarm_labels[test_id] = str(label)
            else:
                raise SummaryError(
                    f"label at {path}:{line_number} has invalid record_type"
                )
    return subject_labels, alarm_labels


def _status_counts(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {"pass": 0, "fail": 0, "inconclusive": 0}
    for record in records:
        status = record.get("validation_status")
        if status not in counts:
            raise SummaryError(f"observation has invalid validation_status: {status!r}")
        counts[str(status)] += 1
    return counts


def _audit_metrics(
    records: Sequence[Mapping[str, Any]],
    subject_outcomes: Mapping[str, str],
    subject_labels: Mapping[Tuple[str, str], str],
    alarm_labels: Mapping[str, str],
    scope: str,
) -> Optional[Dict[str, Any]]:
    defect_label = (
        "CONFIRMED_IN_CONTRACT_DEFECT"
        if scope == "in_contract"
        else "EXTENDED_CONTRACT_FAILURE"
    )
    confirmed_alarm_label = (
        "CONFIRMED_IN_CONTRACT_DISCREPANCY"
        if scope == "in_contract"
        else "CONFIRMED_EXTENDED_CONTRACT_DISCREPANCY"
    )
    confirmed_defects = {
        subject_id
        for (subject_id, label_scope), label in subject_labels.items()
        if label_scope == scope
        and label == defect_label
        and subject_id in subject_outcomes
    }
    alarms = [record for record in records if record.get("validation_status") == "fail"]
    confirmed_alarm_records = [
        record
        for record in alarms
        if alarm_labels.get(str(record.get("test_id"))) == confirmed_alarm_label
    ]
    confirmed_alarm_subjects = {
        str(record.get("subject_id")) for record in confirmed_alarm_records
    }
    rejected_alarm_labels = {
        "INVALID_INPUT",
        "REFERENCE_OR_ORACLE_FAILURE",
        "INFRASTRUCTURE_FAILURE",
    }
    rejected_alarms = sum(
        alarm_labels.get(str(record.get("test_id"))) in rejected_alarm_labels
        for record in alarms
    )
    inconclusive_alarms = sum(
        alarm_labels.get(str(record.get("test_id"))) == "INCONCLUSIVE"
        for record in alarms
    )
    unaudited_alarms = sum(
        str(record.get("test_id")) not in alarm_labels for record in alarms
    )
    if not confirmed_defects and not alarms:
        return None
    detected_defects = confirmed_defects & confirmed_alarm_subjects
    conclusive_alarm_denominator = len(confirmed_alarm_records) + rejected_alarms
    return {
        "confirmed_defect_corpus_subjects": len(confirmed_defects),
        "confirmed_defects_detected_by_confirmed_alarm": len(detected_defects),
        "confirmed_defect_detection_coverage": (
            None
            if not confirmed_defects
            else len(detected_defects) / len(confirmed_defects)
        ),
        "validator_inconclusive_on_confirmed_defects": sum(
            subject_outcomes[subject_id] == "inconclusive"
            for subject_id in confirmed_defects
        ),
        "alarms": len(alarms),
        "confirmed_alarms": len(confirmed_alarm_records),
        "rejected_alarms": rejected_alarms,
        "inconclusive_alarm_labels": inconclusive_alarms,
        "unaudited_alarms": unaudited_alarms,
        "alarm_precision_on_conclusively_audited_alarms": (
            None
            if conclusive_alarm_denominator == 0
            else len(confirmed_alarm_records) / conclusive_alarm_denominator
        ),
        "note": (
            "NO_DEFECT_FOUND is not treated as a true negative. Detection coverage "
            "requires a confirmed alarm for the same strategy, not merely any alarm "
            "on a subject known to contain some defect."
        ),
    }


def summarize(
    *,
    run_manifest_path: Path,
    observations_path: Path,
    labels_path: Optional[Path] = None,
) -> Dict[str, Any]:
    run_manifest = _read_object(run_manifest_path, "run manifest")
    RunManifest.verify_dict(run_manifest)
    observations = list(ObservationLog(observations_path).records())
    subject_labels, alarm_labels = _load_labels(labels_path)

    groups: Dict[Tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    datasets: Dict[Tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        if observation.get("run_manifest_sha256") != run_manifest.get("manifest_sha256"):
            raise SummaryError("observation references another run manifest")
        strategy = str(observation.get("strategy_name"))
        scope = str(observation.get("scope"))
        if scope not in {"in_contract", "extended_contract"}:
            raise SummaryError(f"observation has invalid scope: {scope!r}")
        groups[(strategy, scope)].append(observation)
        datasets[(strategy, scope, str(observation.get("dataset")))].append(observation)

    summaries = []
    for (strategy, scope), records in sorted(groups.items()):
        records = sorted(records, key=lambda record: int(record["order"]))
        per_subject: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            per_subject[str(record["subject_id"])].append(record)

        subject_outcomes: Dict[str, str] = {}
        first_fail_runs = []
        first_fail_wall = []
        for subject_id, subject_records in per_subject.items():
            statuses = [str(record["validation_status"]) for record in subject_records]
            if "fail" in statuses:
                subject_outcomes[subject_id] = "fail"
            elif "inconclusive" in statuses:
                subject_outcomes[subject_id] = "inconclusive"
            else:
                subject_outcomes[subject_id] = "pass"
            cumulative_runs = 0
            cumulative_wall = 0.0
            for record in subject_records:
                cumulative_runs += int(record.get("planned_candidate_runs", 1))
                cumulative_wall += float(record.get("parent_wall_ms", 0.0))
                if record["validation_status"] == "fail":
                    first_fail_runs.append(cumulative_runs)
                    first_fail_wall.append(cumulative_wall)
                    break

        wall_times = [float(record.get("parent_wall_ms", 0.0)) for record in records]
        planned_runs = sum(int(record.get("planned_candidate_runs", 1)) for record in records)
        observed_runs = sum(int(record.get("observed_candidate_runs", 0)) for record in records)
        failed_subjects = sum(outcome == "fail" for outcome in subject_outcomes.values())
        summary = {
            "strategy": strategy,
            "scope": scope,
            "observations": len(records),
            "status_counts": _status_counts(records),
            "subjects": len(per_subject),
            "subject_outcomes": {
                status: sum(outcome == status for outcome in subject_outcomes.values())
                for status in ("pass", "fail", "inconclusive")
            },
            "planned_candidate_runs": planned_runs,
            "observed_candidate_runs": observed_runs,
            "attempt_applicability": None if planned_runs == 0 else observed_runs / planned_runs,
            "parent_wall_ms": {
                "total": sum(wall_times),
                "median_per_observation": None if not wall_times else statistics.median(wall_times),
                "p95_per_observation": _percentile(wall_times, 0.95),
            },
            "detected_subjects": failed_subjects,
            "parent_wall_ms_per_detected_subject": (
                None if failed_subjects == 0 else sum(wall_times) / failed_subjects
            ),
            "time_to_first_fail": {
                "median_candidate_runs": (
                    None if not first_fail_runs else statistics.median(first_fail_runs)
                ),
                "median_parent_wall_ms": (
                    None if not first_fail_wall else statistics.median(first_fail_wall)
                ),
            },
            "human_audit_metrics": _audit_metrics(
                records,
                subject_outcomes,
                subject_labels,
                alarm_labels,
                scope,
            ),
        }
        summaries.append(summary)

    dataset_summaries = []
    for (strategy, scope, dataset), records in sorted(datasets.items()):
        subject_ids = {str(record["subject_id"]) for record in records}
        detected = {
            str(record["subject_id"])
            for record in records
            if record["validation_status"] == "fail"
        }
        dataset_summaries.append(
            {
                "strategy": strategy,
                "scope": scope,
                "dataset": dataset,
                "observations": len(records),
                "subjects": len(subject_ids),
                "detected_subjects": len(detected),
                "status_counts": _status_counts(records),
            }
        )

    payload: Dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": run_manifest["run_id"],
        "run_manifest_sha256": run_manifest["manifest_sha256"],
        "run_manifest_file_sha256": sha256_file(run_manifest_path),
        "observations_file_sha256": sha256_file(observations_path),
        "labels_file_sha256": None if labels_path is None else sha256_file(labels_path),
        "observation_count": len(observations),
        "strategy_scope_summaries": summaries,
        "dataset_summaries": dataset_summaries,
        "interpretation": {
            "fail": "validator alarm requiring independent confirmation",
            "pass": "no discrepancy observed under executed cases; not a proof",
            "inconclusive": "excluded from binary correctness claims but retained in denominators",
        },
    }
    payload["summary_sha256"] = stable_json_sha256(payload)
    return payload


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = summarize(
            run_manifest_path=args.run_manifest,
            observations_path=args.observations,
            labels_path=args.labels,
        )
        write_once(args.output, result)
    except FileExistsError:
        print(f"refusing to overwrite summary: {args.output}", file=sys.stderr)
        return 2
    except (OSError, SummaryError, TypeError, ValueError) as exc:
        print(f"failed to summarize FSE run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "summary_sha256": result["summary_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
