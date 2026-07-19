"""Tests for cost/status aggregation and optional human-label metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from src.experiments import (
    ArtifactProvenance,
    ObservationLog,
    RunManifest,
    SubjectProvenance,
)


PROJECT_ROOT = Path(__file__).parents[2]
SUMMARIZER_PATH = PROJECT_ROOT / "scripts" / "summarize_fse_results.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("fse_summarizer_test", SUMMARIZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_files(tmp_path: Path) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate.py"
    reference = tmp_path / "reference.py"
    candidate.write_text("candidate", encoding="utf-8")
    reference.write_text("reference", encoding="utf-8")
    subjects = tuple(
        SubjectProvenance(
            subject_id=f"s{index}",
            dataset="unit",
            task_id=str(index),
            language="python",
            candidate=ArtifactProvenance.from_file(candidate, root=tmp_path, role="candidate"),
            reference=ArtifactProvenance.from_file(reference, root=tmp_path, role="reference"),
        )
        for index in range(1, 4)
    )
    manifest = RunManifest(
        run_id="summary-test",
        experiment="fse-validator-comparison",
        git_commit="a" * 40,
        git_dirty=False,
        command=("test",),
        config={"plan_sha256": "b" * 64},
        subjects=subjects,
        environment={"test": True},
        created_at_utc="2026-01-01T00:00:00.000000Z",
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest.write_once(manifest_path)
    manifest_hash = manifest.to_dict()["manifest_sha256"]

    observations_path = tmp_path / "observations.jsonl"
    log = ObservationLog(observations_path)
    for order, (subject_id, status, wall, observed_runs) in enumerate(
        (
            ("s1", "fail", 10.0, 1),
            ("s2", "pass", 20.0, 1),
            ("s3", "inconclusive", 30.0, 0),
        ),
        start=1,
    ):
        log.append(
            {
                "test_id": str(order) * 64,
                "order": order,
                "run_manifest_sha256": manifest_hash,
                "subject_id": subject_id,
                "strategy_id": "strategy-id",
                "strategy_name": "strategy",
                "dataset": "unit",
                "scope": "in_contract",
                "validation_status": status,
                "planned_candidate_runs": 1,
                "observed_candidate_runs": observed_runs,
                "parent_wall_ms": wall,
            }
        )
    return manifest_path, observations_path


def test_summary_keeps_inconclusive_and_cost_denominators(tmp_path: Path) -> None:
    module = _load()
    manifest, observations = _run_files(tmp_path)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {
                    "record_type": "subject",
                    "subject_id": "s1",
                    "scope": "in_contract",
                    "primary_label": "CONFIRMED_IN_CONTRACT_DEFECT",
                },
                {
                    "record_type": "subject",
                    "subject_id": "s2",
                    "scope": "in_contract",
                    "primary_label": "NO_DEFECT_FOUND",
                },
                {
                    "record_type": "subject",
                    "subject_id": "s3",
                    "scope": "in_contract",
                    "primary_label": "NO_DEFECT_FOUND",
                },
                {
                    "record_type": "alarm",
                    "test_id": "1" * 64,
                    "scope": "in_contract",
                    "primary_label": "CONFIRMED_IN_CONTRACT_DISCREPANCY",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.summarize(
        run_manifest_path=manifest,
        observations_path=observations,
        labels_path=labels,
    )

    summary = result["strategy_scope_summaries"][0]
    assert summary["status_counts"] == {"pass": 1, "fail": 1, "inconclusive": 1}
    assert summary["planned_candidate_runs"] == 3
    assert summary["observed_candidate_runs"] == 2
    assert summary["attempt_applicability"] == pytest.approx(2 / 3)
    assert summary["detected_subjects"] == 1
    assert summary["parent_wall_ms_per_detected_subject"] == 60.0
    metrics = summary["human_audit_metrics"]
    assert metrics["alarm_precision_on_conclusively_audited_alarms"] == 1.0
    assert metrics["confirmed_defect_detection_coverage"] == 1.0
    assert metrics["validator_inconclusive_on_confirmed_defects"] == 0


def test_conflicting_human_labels_are_rejected(tmp_path: Path) -> None:
    module = _load()
    manifest, observations = _run_files(tmp_path)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "record_type": "subject",
                "subject_id": "s1",
                "primary_label": "CONFIRMED_IN_CONTRACT_DEFECT",
            }
        )
        + "\n"
        + json.dumps(
            {
                "record_type": "subject",
                "subject_id": "s1",
                "primary_label": "NO_DEFECT_FOUND",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(module.SummaryError, match="conflicting labels"):
        module.summarize(
            run_manifest_path=manifest,
            observations_path=observations,
            labels_path=labels,
        )
