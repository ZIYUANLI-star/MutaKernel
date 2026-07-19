#!/usr/bin/env python3
"""Build a blinded subject/alarm audit queue and a separate sealed mapping."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import (
    ObservationLog,
    RunManifest,
    canonical_json_bytes,
    stable_json_sha256,
)
from src.experiments.protocol import ProtocolError, validate_frozen_plan


AUDIT_QUEUE_SCHEMA_VERSION = "1.0"


class AuditQueueError(ValueError):
    """Audit inputs are inconsistent or cannot be blinded safely."""


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditQueueError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AuditQueueError(f"{label} must be an object")
    return value


def _verify_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    if not isinstance(expected, str):
        raise AuditQueueError(f"{label} has no {field}")
    payload = dict(value)
    del payload[field]
    if stable_json_sha256(payload) != expected:
        raise AuditQueueError(f"{label} digest mismatch")


def _neutral_id(namespace: str, *parts: str) -> str:
    return "audit-" + stable_json_sha256(
        {"namespace": namespace, "parts": list(parts)}
    )[:24]


def _subject_stratum(subject: Mapping[str, Any]) -> tuple[str, str, str]:
    metadata = subject.get("metadata", {})
    level = "unknown"
    if isinstance(metadata, Mapping):
        raw_level = metadata.get("task_level", metadata.get("level", "unknown"))
        level = str(raw_level)
    return (str(subject.get("dataset")), str(subject.get("language")), level)


def build_queue(
    *,
    subject_manifest: Mapping[str, Any],
    experiment_plan: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    selection_seed: int,
    nonalarm_per_stratum: Optional[int],
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    _verify_digest(subject_manifest, "manifest_sha256", "subject manifest")
    _verify_digest(experiment_plan, "plan_sha256", "experiment plan")
    try:
        validated_schedule = validate_frozen_plan(experiment_plan, subject_manifest)
    except ProtocolError as exc:
        raise AuditQueueError(f"experiment plan is invalid: {exc}") from exc
    subjects = subject_manifest.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise AuditQueueError("subject manifest has no subjects")
    subject_by_id = {str(subject["subject_id"]): subject for subject in subjects}
    if len(subject_by_id) != len(subjects):
        raise AuditQueueError("subject manifest contains duplicate IDs")

    try:
        RunManifest.verify_dict(run_manifest)
    except ValueError as exc:
        raise AuditQueueError(f"run manifest integrity check failed: {exc}") from exc
    run_config = run_manifest.get("config")
    if not isinstance(run_config, Mapping):
        raise AuditQueueError("run manifest has no config object")
    if run_manifest.get("config_sha256") != stable_json_sha256(run_config):
        raise AuditQueueError("run manifest config_sha256 mismatch")
    if run_manifest.get("experiment") != "fse-validator-comparison":
        raise AuditQueueError("run manifest is not an FSE validator-comparison run")
    if run_config.get("plan_sha256") != experiment_plan.get("plan_sha256"):
        raise AuditQueueError("run manifest is bound to another experiment plan")
    if run_config.get("subject_manifest_sha256") != subject_manifest.get(
        "manifest_sha256"
    ):
        raise AuditQueueError("run manifest is bound to another subject manifest")
    run_subjects = run_manifest.get("subjects")
    if not isinstance(run_subjects, list):
        raise AuditQueueError("run manifest has no subject list")
    projected_subjects = [
        {
            key: subject[key]
            for key in (
                "subject_id",
                "dataset",
                "task_id",
                "language",
                "candidate",
                "reference",
                "contract",
                "contract_sha256",
                "metadata",
            )
        }
        for subject in subjects
    ]
    if sorted(run_subjects, key=lambda value: str(value.get("subject_id"))) != sorted(
        projected_subjects, key=lambda value: str(value.get("subject_id"))
    ):
        raise AuditQueueError(
            "run-manifest subjects differ from the frozen subject manifest"
        )

    expected_by_test = {
        str(entry["test_id"]): entry for entry in validated_schedule
    }
    observed_by_test: Dict[str, Mapping[str, Any]] = {}
    for observation in observations:
        test_id = observation.get("test_id")
        if not isinstance(test_id, str) or test_id not in expected_by_test:
            raise AuditQueueError(f"observation references unknown planned test {test_id!r}")
        if test_id in observed_by_test:
            raise AuditQueueError(f"duplicate observation for planned test {test_id}")
        expected = expected_by_test[test_id]
        for field, expected_value in expected.items():
            if observation.get(field) != expected_value:
                raise AuditQueueError(
                    f"observation {test_id} field {field} differs from the plan"
                )
        subject = subject_by_id[str(expected["subject_id"])]
        for field, expected_value in (
            ("run_id", run_manifest.get("run_id")),
            ("run_manifest_sha256", run_manifest.get("manifest_sha256")),
            ("subject_sha256", subject.get("subject_sha256")),
            ("contract_sha256", subject.get("contract_sha256")),
            ("planned_candidate_runs", expected.get("candidate_run_cost")),
        ):
            if observation.get(field) != expected_value:
                raise AuditQueueError(
                    f"observation {test_id} field {field} differs from frozen provenance"
                )
        if observation.get("validation_status") not in {"pass", "fail", "inconclusive"}:
            raise AuditQueueError(f"observation {test_id} has invalid validation_status")
        observed_by_test[test_id] = observation

    alarms = [
        observation
        for observation in observed_by_test.values()
        if observation.get("validation_status") == "fail"
    ]
    alarm_subjects = {str(observation.get("subject_id")) for observation in alarms}
    unknown = alarm_subjects - set(subject_by_id)
    if unknown:
        raise AuditQueueError(f"observations reference unknown subjects: {sorted(unknown)}")

    expected_by_subject: Dict[str, set[str]] = defaultdict(set)
    for test_id, entry in expected_by_test.items():
        expected_by_subject[str(entry["subject_id"])].add(test_id)
    observed_by_subject: Dict[str, set[str]] = defaultdict(set)
    for test_id, observation in observed_by_test.items():
        observed_by_subject[str(observation["subject_id"])].add(test_id)

    completion: Dict[str, str] = {}
    nonalarm_by_stratum: Dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for subject_id, subject in subject_by_id.items():
        expected_ids = expected_by_subject.get(subject_id, set())
        observed_ids = observed_by_subject.get(subject_id, set())
        if not observed_ids:
            completion[subject_id] = "never_executed"
        elif observed_ids != expected_ids:
            completion[subject_id] = "partially_evaluated"
        elif any(
            observed_by_test[test_id]["validation_status"] == "inconclusive"
            for test_id in expected_ids
        ):
            completion[subject_id] = "fully_evaluated_with_inconclusive"
        elif subject_id in alarm_subjects:
            completion[subject_id] = "fully_evaluated_with_alarm"
        else:
            completion[subject_id] = "fully_evaluated_no_alarm"

        if completion[subject_id] == "fully_evaluated_no_alarm":
            nonalarm_by_stratum[_subject_stratum(subject)].append(subject_id)
    selected_nonalarm = set()
    rng = random.Random(selection_seed)
    stratum_samples = []
    for stratum, subject_ids in sorted(nonalarm_by_stratum.items()):
        ordered = sorted(subject_ids)
        rng.shuffle(ordered)
        chosen = (
            ordered
            if nonalarm_per_stratum is None
            else ordered[: min(nonalarm_per_stratum, len(ordered))]
        )
        selected_nonalarm.update(chosen)
        stratum_samples.append(
            {
                "dataset": stratum[0],
                "language": stratum[1],
                "task_level": stratum[2],
                "population": len(ordered),
                "selected": len(chosen),
            }
        )

    queue: list[Dict[str, Any]] = []
    mapping_items: Dict[str, Any] = {}
    unresolved_subjects = {
        subject_id
        for subject_id, status in completion.items()
        if status
        in {
            "never_executed",
            "partially_evaluated",
            "fully_evaluated_with_inconclusive",
        }
    }
    selected_subjects = sorted(
        alarm_subjects | selected_nonalarm | unresolved_subjects
    )
    namespace = subject_manifest["manifest_sha256"]
    for subject_id in selected_subjects:
        subject = subject_by_id[subject_id]
        audit_id = _neutral_id(namespace, "subject", subject_id)
        queue.append(
            {
                "schema_version": AUDIT_QUEUE_SCHEMA_VERSION,
                "audit_id": audit_id,
                "item_kind": "subject",
                "population": "real_kernel",
                "contract_id": subject["contract"]["contract_id"],
                "contract_sha256": subject["contract_sha256"],
                "candidate": dict(subject["candidate"]),
                "reference": dict(subject["reference"]),
                "evidence_status": "SOURCE_AND_CONTRACT_ONLY",
            }
        )
        mapping_items[audit_id] = {
            "item_kind": "subject",
            "subject_id": subject_id,
            "stratum": list(_subject_stratum(subject)),
            "historically_any_alarm": subject_id in alarm_subjects,
            "completion_status": completion[subject_id],
            "selection_role": (
                "alarm_union"
                if subject_id in alarm_subjects
                else (
                    "fully_evaluated_nonalarm_sample"
                    if subject_id in selected_nonalarm
                    else "execution_unresolved"
                )
            ),
        }

    for observation in sorted(alarms, key=lambda item: str(item.get("test_id"))):
        test_id = str(observation.get("test_id"))
        subject_id = str(observation.get("subject_id"))
        if not test_id:
            raise AuditQueueError("alarm observation has no test_id")
        subject = subject_by_id[subject_id]
        audit_id = _neutral_id(namespace, "alarm", test_id)
        worker_result = observation.get("worker_result", {})
        evidence = (
            worker_result.get("evidence")
            if isinstance(worker_result, Mapping)
            else None
        )
        replay_bundle = (
            evidence.get("replay_bundle")
            if isinstance(evidence, Mapping)
            else None
        )
        queue.append(
            {
                "schema_version": AUDIT_QUEUE_SCHEMA_VERSION,
                "audit_id": audit_id,
                "item_kind": "alarm",
                "population": "validator_alarm",
                "contract_id": subject["contract"]["contract_id"],
                "contract_sha256": subject["contract_sha256"],
                "candidate": dict(subject["candidate"]),
                "reference": dict(subject["reference"]),
                "evidence": evidence,
                "evidence_status": (
                    "OPERATOR_ONLY_REQUIRES_BLINDED_EXPORT"
                    if isinstance(replay_bundle, Mapping)
                    else "MISSING_REPLAY_BUNDLE"
                ),
            }
        )
        mapping_items[audit_id] = {
            "item_kind": "alarm",
            "subject_id": subject_id,
            "test_id": test_id,
            "strategy_id": observation.get("strategy_id"),
            "strategy_name": observation.get("strategy_name"),
            "policy": observation.get("policy"),
            "scope": observation.get("scope"),
            "detector_status": observation.get("validation_status"),
        }

    queue.sort(key=lambda item: item["audit_id"])
    for item in queue:
        item["item_sha256"] = stable_json_sha256(item)
    queue_audit_ids = [item["audit_id"] for item in queue]
    sealed: Dict[str, Any] = {
        "schema_version": AUDIT_QUEUE_SCHEMA_VERSION,
        "subject_manifest_sha256": subject_manifest["manifest_sha256"],
        "experiment_plan_sha256": experiment_plan["plan_sha256"],
        "run_manifest_sha256": run_manifest["manifest_sha256"],
        "observed_test_ids_sha256": stable_json_sha256(sorted(observed_by_test)),
        "observations_sha256": stable_json_sha256(
            [observed_by_test[test_id] for test_id in sorted(observed_by_test)]
        ),
        "selection_seed": selection_seed,
        "nonalarm_per_stratum": nonalarm_per_stratum,
        "stratum_samples": stratum_samples,
        "queue_item_count": len(queue),
        "queue_audit_ids": queue_audit_ids,
        "queue_sha256": stable_json_sha256(queue),
        "completion_counts": dict(sorted(
            {
                status: sum(value == status for value in completion.values())
                for status in set(completion.values())
            }.items()
        )),
        "unresolved_subject_count": len(unresolved_subjects),
        "mapping": mapping_items,
    }
    sealed["sealed_mapping_sha256"] = stable_json_sha256(sealed)
    return queue, sealed


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--sealed-mapping", required=True, type=Path)
    parser.add_argument("--selection-seed", type=int, default=20260719)
    parser.add_argument(
        "--nonalarm-per-stratum",
        type=int,
        help="Omit to audit every non-alarm subject; otherwise sample this many per stratum",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nonalarm_per_stratum is not None and args.nonalarm_per_stratum < 0:
        print("--nonalarm-per-stratum must be non-negative", file=sys.stderr)
        return 2
    try:
        if args.queue.exists() or args.sealed_mapping.exists():
            raise FileExistsError(
                "refusing a partial audit write because one or more outputs already exist"
            )
        subjects = _read_object(args.subjects, "subject manifest")
        plan = _read_object(args.plan, "experiment plan")
        run_manifest = _read_object(args.run_manifest, "run manifest")
        observations = list(ObservationLog(args.observations).records())
        with args.observations.open("r", encoding="utf-8") as handle:
            physical_records = sum(bool(line.strip()) for line in handle)
        if physical_records != len(observations):
            raise AuditQueueError(
                "observation log contains duplicate physical test records"
            )
        queue, sealed = build_queue(
            subject_manifest=subjects,
            experiment_plan=plan,
            run_manifest=run_manifest,
            observations=observations,
            selection_seed=args.selection_seed,
            nonalarm_per_stratum=args.nonalarm_per_stratum,
        )
        _write_jsonl_once(args.queue, queue)
        _write_json_once(args.sealed_mapping, sealed)
    except (AuditQueueError, OSError, TypeError, ValueError) as exc:
        print(f"failed to build audit queue: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "queue": str(args.queue),
                "sealed_mapping": str(args.sealed_mapping),
                "items": len(queue),
                "sealed_mapping_sha256": sealed["sealed_mapping_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
