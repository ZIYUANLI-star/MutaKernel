#!/usr/bin/env python3
"""Validate blinded annotations, compute agreement, and unblind final labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import canonical_json_bytes, sha256_file, stable_json_sha256


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
CONFIDENCE_VALUES = {"high", "medium", "low"}


class AuditCompileError(ValueError):
    """Annotations cannot be joined without violating the audit protocol."""


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditCompileError(
                    f"invalid {label} JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, Mapping):
                raise AuditCompileError(f"{label} line {line_number} is not an object")
            rows.append(value)
    return rows


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditCompileError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AuditCompileError(f"{label} must be an object")
    return value


def _verify_embedded_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    if not isinstance(expected, str):
        raise AuditCompileError(f"{label} has no {field}")
    payload = dict(value)
    del payload[field]
    if stable_json_sha256(payload) != expected:
        raise AuditCompileError(f"{label} digest mismatch")


def _cohen_kappa(pairs: Sequence[tuple[str, str]]) -> Optional[float]:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[label] / len(pairs)) * (right_counts[label] / len(pairs))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def compile_audit(
    *,
    queue: Sequence[Mapping[str, Any]],
    sealed_mapping: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
    require_complete: bool,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    _verify_embedded_digest(
        sealed_mapping, "sealed_mapping_sha256", "sealed mapping"
    )
    provenance_fields = (
        "subject_manifest_sha256",
        "experiment_plan_sha256",
        "run_manifest_sha256",
        "observations_sha256",
    )
    for field in provenance_fields:
        digest = sealed_mapping.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise AuditCompileError(
                f"sealed mapping requires lowercase SHA-256 field {field}"
            )
    mapping = sealed_mapping.get("mapping")
    if not isinstance(mapping, Mapping):
        raise AuditCompileError("sealed mapping has no mapping object")
    queue_by_id = {}
    for item in queue:
        audit_id = item.get("audit_id")
        if not isinstance(audit_id, str) or audit_id in queue_by_id:
            raise AuditCompileError("queue audit IDs must be unique strings")
        expected = item.get("item_sha256")
        payload = dict(item)
        payload.pop("item_sha256", None)
        if expected != stable_json_sha256(payload):
            raise AuditCompileError(f"queue item {audit_id} digest mismatch")
        if audit_id not in mapping:
            raise AuditCompileError(f"queue item {audit_id} is absent from sealed mapping")
        queue_by_id[audit_id] = item
    expected_ids = sealed_mapping.get("queue_audit_ids")
    if expected_ids != sorted(queue_by_id):
        raise AuditCompileError("queue audit ID set differs from the sealed mapping")
    if sealed_mapping.get("queue_item_count") != len(queue_by_id):
        raise AuditCompileError("queue item count differs from the sealed mapping")
    if set(mapping) != set(queue_by_id):
        raise AuditCompileError("sealed detector mapping and queue ID sets differ")
    ordered_queue = sorted(queue, key=lambda item: str(item.get("audit_id")))
    if sealed_mapping.get("queue_sha256") != stable_json_sha256(ordered_queue):
        raise AuditCompileError("ordered queue digest differs from the sealed mapping")

    primary: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    adjudicators: Dict[str, Mapping[str, Any]] = {}
    for row in annotations:
        audit_id = row.get("audit_id")
        auditor_id = row.get("auditor_id")
        role = row.get("annotation_role", "primary")
        label = row.get("primary_label")
        confidence = row.get("confidence")
        if audit_id not in queue_by_id:
            raise AuditCompileError(f"annotation references unknown audit_id {audit_id!r}")
        if not isinstance(auditor_id, str) or not auditor_id:
            raise AuditCompileError("annotation requires auditor_id")
        if confidence not in CONFIDENCE_VALUES:
            raise AuditCompileError("annotation confidence is invalid")
        if not isinstance(row.get("rationale"), str) or not row.get("rationale"):
            raise AuditCompileError("annotation requires a rationale")
        if not isinstance(row.get("fault_class"), str) or not row.get("fault_class"):
            raise AuditCompileError("annotation requires fault_class")
        if not isinstance(row.get("contract_clause"), str) or not row.get("contract_clause"):
            raise AuditCompileError("annotation requires contract_clause")
        if not isinstance(row.get("evidence_reproduced"), bool):
            raise AuditCompileError("annotation requires boolean evidence_reproduced")
        item_kind = queue_by_id[audit_id]["item_kind"]
        allowed = SUBJECT_LABELS if item_kind == "subject" else ALARM_LABELS
        if label not in allowed:
            raise AuditCompileError(
                f"label {label!r} is invalid for {item_kind} item {audit_id}"
            )
        if item_kind == "alarm":
            scope = mapping[audit_id].get("scope")
            if scope == "in_contract" and label == "CONFIRMED_EXTENDED_CONTRACT_DISCREPANCY":
                raise AuditCompileError("in-contract alarm cannot receive an extended-contract label")
            if scope == "extended_contract" and label == "CONFIRMED_IN_CONTRACT_DISCREPANCY":
                raise AuditCompileError("extended-contract alarm cannot receive an in-contract label")
        if role == "primary":
            if auditor_id in primary[audit_id]:
                raise AuditCompileError(f"duplicate primary annotation for {audit_id}")
            primary[audit_id][auditor_id] = row
        elif role == "adjudicator":
            if audit_id in adjudicators:
                raise AuditCompileError(f"duplicate adjudication for {audit_id}")
            adjudicators[audit_id] = row
        else:
            raise AuditCompileError("annotation_role must be primary or adjudicator")

    pairs = []
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    final_labels: Dict[str, str] = {}
    pending = []
    disagreements = 0
    auditor_pairs = set()
    for audit_id in sorted(queue_by_id):
        rows = primary.get(audit_id, {})
        if len(rows) != 2:
            pending.append({"audit_id": audit_id, "reason": "requires_two_primary_labels"})
            continue
        ordered = [rows[key] for key in sorted(rows)]
        auditor_pairs.add(tuple(sorted(rows)))
        left = str(ordered[0]["primary_label"])
        right = str(ordered[1]["primary_label"])
        pairs.append((left, right))
        confusion[left][right] += 1
        if left == right:
            if audit_id in adjudicators:
                raise AuditCompileError(
                    f"agreed item {audit_id} must not receive an adjudicator label"
                )
            final_labels[audit_id] = left
        else:
            disagreements += 1
            adjudication = adjudicators.get(audit_id)
            if adjudication is None:
                pending.append({"audit_id": audit_id, "reason": "needs_adjudication"})
            else:
                if adjudication.get("auditor_id") in rows:
                    raise AuditCompileError(
                        f"adjudicator for {audit_id} must be independent of both primary auditors"
                    )
                final_labels[audit_id] = str(adjudication["primary_label"])

    if len(auditor_pairs) > 1:
        raise AuditCompileError(
            "Cohen kappa requires one fixed pair of primary auditors across the queue"
        )

    if require_complete and pending:
        raise AuditCompileError(
            f"audit is incomplete: {len(pending)} item(s) require labels or adjudication"
        )

    analysis_rows = []
    for audit_id, label in sorted(final_labels.items()):
        mapped = mapping[audit_id]
        item_kind = queue_by_id[audit_id]["item_kind"]
        primary_rows = [primary[audit_id][key] for key in sorted(primary[audit_id])]
        adjudication = adjudicators.get(audit_id)
        if item_kind == "subject":
            scope = (
                "extended_contract"
                if label == "EXTENDED_CONTRACT_FAILURE"
                else "in_contract"
            )
            row = {
                "record_type": "subject",
                "audit_id": audit_id,
                "subject_id": mapped["subject_id"],
                "scope": scope,
                "primary_label": label,
            }
        else:
            row = {
                "record_type": "alarm",
                "audit_id": audit_id,
                "subject_id": mapped["subject_id"],
                "test_id": mapped["test_id"],
                "scope": mapped["scope"],
                "primary_label": label,
            }
        row["primary_annotations"] = [
            {
                key: annotation[key]
                for key in (
                    "auditor_id",
                    "primary_label",
                    "confidence",
                    "fault_class",
                    "contract_clause",
                    "evidence_reproduced",
                    "rationale",
                )
            }
            for annotation in primary_rows
        ]
        row["adjudication"] = (
            None
            if adjudication is None
            else {
                key: adjudication[key]
                for key in (
                    "auditor_id",
                    "primary_label",
                    "confidence",
                    "fault_class",
                    "contract_clause",
                    "evidence_reproduced",
                    "rationale",
                )
            }
        )
        # The digest commits the full analysis record, including preserved raw
        # judgements, rather than only the final categorical label.
        row["analysis_label_sha256"] = stable_json_sha256(
            {key: value for key, value in row.items() if key != "analysis_label_sha256"}
        )
        analysis_rows.append(row)

    agreement = sum(left == right for left, right in pairs)
    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "queue_items": len(queue_by_id),
        "paired_items": len(pairs),
        "raw_agreement": None if not pairs else agreement / len(pairs),
        "cohen_kappa": _cohen_kappa(pairs),
        "disagreements": disagreements,
        "adjudicated": sum(
            audit_id in adjudicators for audit_id in final_labels
        ),
        "final_labels": len(final_labels),
        "pending": pending,
        "audit_complete": not pending,
        "primary_auditor_pair": (
            None if not auditor_pairs else list(next(iter(auditor_pairs)))
        ),
        "queue_sha256": sealed_mapping["queue_sha256"],
        "sealed_mapping_sha256": sealed_mapping["sealed_mapping_sha256"],
        "subject_manifest_sha256": sealed_mapping["subject_manifest_sha256"],
        "experiment_plan_sha256": sealed_mapping["experiment_plan_sha256"],
        "run_manifest_sha256": sealed_mapping["run_manifest_sha256"],
        "observations_sha256": sealed_mapping["observations_sha256"],
        "annotations_sha256": stable_json_sha256(list(annotations)),
        "analysis_labels_sha256": stable_json_sha256(analysis_rows),
        "confusion": {
            left: dict(sorted(counts.items()))
            for left, counts in sorted(confusion.items())
        },
        "interpretation": (
            "Agreement is calculated before adjudication. Analysis labels preserve "
            "NO_DEFECT_FOUND as uncertainty rather than treating it as proof of correctness."
        ),
    }
    report["audit_report_sha256"] = stable_json_sha256(report)
    return report, analysis_rows


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--sealed-mapping", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--analysis-labels", required=True, type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Development-only: emit labels with unresolved audit items",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.report.exists() or args.analysis_labels.exists():
            raise FileExistsError(
                "refusing a partial audit write because one or more outputs already exist"
            )
        report, labels = compile_audit(
            queue=_read_jsonl(args.queue, "audit queue"),
            sealed_mapping=_read_object(args.sealed_mapping, "sealed mapping"),
            annotations=_read_jsonl(args.annotations, "annotations"),
            require_complete=not args.allow_partial,
        )
        report["input_files"] = {
            "queue_sha256": sha256_file(args.queue),
            "sealed_mapping_sha256": sha256_file(args.sealed_mapping),
            "annotations_sha256": sha256_file(args.annotations),
        }
        report.pop("audit_report_sha256", None)
        report["audit_report_sha256"] = stable_json_sha256(report)
        _write_json_once(args.report, report)
        _write_jsonl_once(args.analysis_labels, labels)
    except (AuditCompileError, OSError, TypeError, ValueError) as exc:
        print(f"failed to compile human audit: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "report": str(args.report),
                "analysis_labels": str(args.analysis_labels),
                "final_labels": len(labels),
                "pending": len(report["pending"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
