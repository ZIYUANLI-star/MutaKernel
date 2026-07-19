"""Tests for provenance-bound, audit-aware FSE statistical analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.experiments import ObservationLog, canonical_json_bytes, sha256_file, stable_json_sha256


PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_fse_statistics.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("fse_statistics_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _audit_row(audit_id: str, **fields: Any) -> dict[str, Any]:
    row = {"audit_id": audit_id, **fields}
    row["analysis_label_sha256"] = stable_json_sha256(row)
    return row


def _fixture(tmp_path: Path) -> dict[str, Any]:
    identities = {
        # s1/s2 and s3 intentionally share task_id across datasets.  The
        # canonical cluster count is two, whereas task_id alone would yield one.
        "s1": ("dataset-alpha", "shared-task"),
        "s2": ("dataset-alpha", "shared-task"),
        "s3": ("dataset-beta", "shared-task"),
        "s4": ("dataset-beta", "task-3"),
    }
    statuses = {
        "A": {"s1": "fail", "s2": "pass", "s3": "inconclusive", "s4": "fail"},
        "B": {"s1": "pass", "s2": "fail", "s3": "fail", "s4": "pass"},
        "C": {"s1": "fail", "s2": "fail", "s3": "pass", "s4": "pass"},
    }
    costs = {"A": (1, 10.0), "B": (2, 20.0), "C": (3, 30.0)}
    schedule: list[dict[str, Any]] = []
    test_ids: dict[tuple[str, str], str] = {}
    order = 0
    for strategy in ("A", "B", "C"):
        for subject_id in ("s1", "s2", "s3", "s4"):
            order += 1
            test_id = f"test-{strategy}-{subject_id}"
            test_ids[(strategy, subject_id)] = test_id
            dataset, task_id = identities[subject_id]
            schedule.append(
                {
                    "test_id": test_id,
                    "order": order,
                    "subject_id": subject_id,
                    "strategy_name": strategy,
                    "strategy_id": f"strategy-{strategy}",
                    "policy": "iid",
                    "seed": 7,
                    "mode": "eval",
                    "scope": "in_contract",
                    "parameters": {},
                    "replicate": 0,
                    "candidate_run_cost": costs[strategy][0],
                    "budget_matched": True,
                    "strategy_candidate_run_budget": costs[strategy][0],
                    "dataset": dataset,
                    "task_id": task_id,
                }
            )
    plan: dict[str, Any] = {
        "schema_version": "1.0",
        "matrix_id": "statistics-test",
        "experiment_scope": "in_contract",
        "subject_manifest_sha256": "b" * 64,
        "subject_count": 4,
        "strategy_count": 3,
        "test_case_count": len(schedule),
        "strategies": [
            {
                "strategy_id": f"strategy-{strategy}",
                "name": strategy,
                "version": "1",
                "budget_matched": True,
                "candidate_runs_per_subject": costs[strategy][0],
                "parameters": {},
            }
            for strategy in ("A", "B", "C")
        ],
        "schedule": schedule,
        "schedule_sha256": stable_json_sha256(schedule),
    }
    plan["plan_sha256"] = stable_json_sha256(plan)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)

    runner_config = {
        "schema_version": "1.0",
        "device": "cpu",
        "timeout_s": 10,
        "max_wall_ms_per_subject_strategy": None,
        "selected_subjects": None,
        "selected_strategies": None,
        "early_stop_on_fail": False,
    }
    config = {
        "schema_version": "1.0",
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": sha256_file(plan_path),
        "subject_manifest_sha256": plan["subject_manifest_sha256"],
        "runner": runner_config,
    }
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": "statistics-run",
        "experiment": "fse-validator-comparison",
        "git": {"commit": "a" * 40, "dirty": False},
        "command": ["unit-test"],
        "config": config,
        "config_sha256": stable_json_sha256(config),
        "subjects": [
            {
                "subject_id": subject_id,
                "dataset": dataset,
                "task_id": task_id,
                "language": "python",
                "candidate": {},
                "reference": {},
                "contract": {},
                "contract_sha256": stable_json_sha256({}),
                "metadata": {},
            }
            for subject_id, (dataset, task_id) in identities.items()
        ],
        "environment": {"test": True},
        "created_at_utc": "2026-07-19T00:00:00.000000Z",
    }
    manifest["manifest_sha256"] = stable_json_sha256(manifest)
    manifest_path = tmp_path / "run_manifest.json"
    _write_json(manifest_path, manifest)

    observation_path = tmp_path / "observations.jsonl"
    log = ObservationLog(observation_path)
    observations: list[dict[str, Any]] = []
    for entry in schedule:
        strategy = entry["strategy_name"]
        subject_id = entry["subject_id"]
        candidate_cost, wall = costs[strategy]
        observation = {
            **entry,
            "run_id": manifest["run_id"],
            "run_manifest_sha256": manifest["manifest_sha256"],
            "validation_status": statuses[strategy][subject_id],
            "planned_candidate_runs": candidate_cost,
            "observed_candidate_runs": candidate_cost,
            "parent_wall_ms": wall,
        }
        assert log.append(observation)
        observations.append(observation)

    labels: list[dict[str, Any]] = [
        _audit_row(
            f"audit-subject-{subject_id}",
            record_type="subject",
            subject_id=subject_id,
            scope="in_contract",
            primary_label="CONFIRMED_IN_CONTRACT_DEFECT",
        )
        for subject_id in ("s1", "s2", "s3")
    ]
    labels.append(
        _audit_row(
            "audit-subject-s4",
            record_type="subject",
            subject_id="s4",
            scope="in_contract",
            primary_label="NO_DEFECT_FOUND",
        )
    )
    for strategy, subject_id, label in (
        ("A", "s1", "CONFIRMED_IN_CONTRACT_DISCREPANCY"),
        ("A", "s4", "INVALID_INPUT"),
        ("B", "s2", "CONFIRMED_IN_CONTRACT_DISCREPANCY"),
        ("B", "s3", "INCONCLUSIVE"),
        ("C", "s1", "CONFIRMED_IN_CONTRACT_DISCREPANCY"),
        ("C", "s2", "CONFIRMED_IN_CONTRACT_DISCREPANCY"),
    ):
        labels.append(
            _audit_row(
                f"audit-alarm-{strategy}-{subject_id}",
                record_type="alarm",
                subject_id=subject_id,
                test_id=test_ids[(strategy, subject_id)],
                scope="in_contract",
                primary_label=label,
            )
        )
    audit_path = tmp_path / "analysis-labels.jsonl"
    audit_path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in labels)
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "queue_items": len(labels),
        "paired_items": len(labels),
        "final_labels": len(labels),
        "pending": [],
        "audit_complete": True,
        "analysis_labels_sha256": stable_json_sha256(labels),
        "run_manifest_sha256": manifest["manifest_sha256"],
        "experiment_plan_sha256": plan["plan_sha256"],
        "subject_manifest_sha256": plan["subject_manifest_sha256"],
        "observations_sha256": stable_json_sha256(
            sorted(log.records(), key=lambda row: row["test_id"])
        ),
    }
    report["audit_report_sha256"] = stable_json_sha256(report)
    report_path = tmp_path / "audit-report.json"
    _write_json(report_path, report)
    return {
        "run_manifest": manifest_path,
        "plan": plan_path,
        "observations": observation_path,
        "audit": audit_path,
        "audit_report": report_path,
        "observation_rows": list(ObservationLog(observation_path).records()),
        "labels": labels,
    }


def _analyze(module: Any, files: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    arguments = {
        "run_manifest_path": files["run_manifest"],
        "plan_path": files["plan"],
        "observations_path": files["observations"],
        "human_audit_path": files["audit"],
        "audit_report_path": files["audit_report"],
        "bootstrap_replicates": 400,
        "seed": 73,
    }
    arguments.update(overrides)
    return module.analyze(**arguments)


def test_analysis_is_plan_complete_clustered_paired_and_deterministic(
    tmp_path: Path,
) -> None:
    module = _load()
    files = _fixture(tmp_path)
    first = _analyze(module, files)
    second = _analyze(module, files)
    assert first == second
    assert first["analysis_sha256"] == second["analysis_sha256"]

    completeness = first["verified_experiment_completeness"]
    assert completeness["status"] == "complete"
    assert completeness["expected_test_ids"] == 12
    assert completeness["expected_planned_candidate_runs"] == 24
    assert completeness["observed_candidate_runs"] == 24
    assert first["audit_report_verification"]["status"] == "verified"

    corpus = first["corpus"]
    assert corpus["confirmed_in_contract_gold_subjects"] == 3
    assert corpus["canonical_dataset_task_clusters_in_gold_corpus"] == 2
    assert corpus["no_defect_found_subject_labels"] == 1
    assert "not a true-negative" in corpus["no_defect_found_interpretation"]

    strategies = {row["strategy_name"]: row for row in first["strategies"]}
    assert strategies["A"]["confirmed_detection_coverage"] == pytest.approx(1 / 3)
    assert strategies["A"]["inconclusive_detection_sensitivity_bounds"][
        "best_case"
    ] == pytest.approx(2 / 3)
    assert strategies["B"]["confirmed_detection_coverage"] == pytest.approx(1 / 3)
    assert strategies["B"]["unresolved_gold_subjects"] == 1
    assert strategies["C"]["confirmed_detection_coverage"] == pytest.approx(2 / 3)
    assert strategies["C"]["task_cluster_bootstrap_ci"]["task_clusters"] == 2
    assert strategies["A"]["alarm_level_audited_precision"]["precision"] == 0.5
    assert strategies["B"]["alarm_level_audited_precision"]["precision"] == 1.0

    comparisons = {
        row["comparison_id"]: row for row in first["paired_mcnemar_exact_tests"]
    }
    assert comparisons["A__vs__B"]["a_detected_b_not"] == 1
    assert comparisons["A__vs__B"]["b_detected_a_not"] == 1
    assert comparisons["A__vs__B"]["holm_adjusted_p_value"] == 1.0
    pareto = first["cost_detection_pareto_frontier"]
    assert pareto["joint_candidate_runs_and_wall_time"] == ["A", "C"]


def test_exact_mcnemar_and_holm_are_standard_two_sided_forms() -> None:
    module = _load()
    assert module._mcnemar_exact(6, 0) == pytest.approx(0.03125)
    assert module._mcnemar_exact(0, 0) == 1.0
    adjusted = module._holm_adjust(
        (("first", 0.01), ("second", 0.03), ("third", 0.04)), 0.05
    )
    assert adjusted["first"]["holm_adjusted_p_value"] == pytest.approx(0.03)
    assert adjusted["second"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert adjusted["third"]["holm_adjusted_p_value"] == pytest.approx(0.06)


def test_cli_requires_provenance_inputs_and_is_byte_deterministic(tmp_path: Path) -> None:
    module = _load()
    files = _fixture(tmp_path)
    first_output = tmp_path / "statistics-1.json"
    second_output = tmp_path / "statistics-2.json"
    common = [
        "--run-manifest",
        str(files["run_manifest"]),
        "--plan",
        str(files["plan"]),
        "--observations",
        str(files["observations"]),
        "--human-audit",
        str(files["audit"]),
        "--audit-report",
        str(files["audit_report"]),
        "--bootstrap-replicates",
        "100",
        "--seed",
        "11",
    ]
    assert module.main([*common, "--output", str(first_output)]) == 0
    assert module.main([*common, "--output", str(second_output)]) == 0
    assert first_output.read_bytes() == second_output.read_bytes()
    payload = json.loads(first_output.read_text(encoding="utf-8"))
    digest_payload = dict(payload)
    digest = digest_payload.pop("analysis_sha256")
    assert module._stable_sha256(digest_payload) == digest


def _rewrite_observations(path: Path, rows: list[Mapping[str, Any]]) -> None:
    log = ObservationLog(path)
    for source in rows:
        row = dict(source)
        row.pop("observation_sha256", None)
        row.pop("schema_version", None)
        assert log.append(row)


def test_missing_plan_test_and_wrong_candidate_budget_are_rejected(tmp_path: Path) -> None:
    module = _load()
    files = _fixture(tmp_path)
    partial = tmp_path / "partial.jsonl"
    _rewrite_observations(partial, files["observation_rows"][:-1])
    with pytest.raises(module.StatisticsError, match="incomplete or stale"):
        _analyze(module, files, observations_path=partial)

    wrong_budget = tmp_path / "wrong-budget.jsonl"
    rows = [dict(row) for row in files["observation_rows"]]
    rows[0]["planned_candidate_runs"] += 1
    _rewrite_observations(wrong_budget, rows)
    with pytest.raises(module.StatisticsError, match="candidate cost disagrees"):
        _analyze(module, files, observations_path=wrong_budget)

    impossible_actual = tmp_path / "impossible-actual.jsonl"
    rows = [dict(row) for row in files["observation_rows"]]
    rows[0]["observed_candidate_runs"] = rows[0]["planned_candidate_runs"] + 1
    _rewrite_observations(impossible_actual, rows)
    with pytest.raises(module.StatisticsError, match="exceeds its planned"):
        _analyze(module, files, observations_path=impossible_actual)


def test_observation_digest_and_run_manifest_binding_are_enforced(tmp_path: Path) -> None:
    module = _load()
    files = _fixture(tmp_path)
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_bytes(files["observations"].read_bytes())
    text = tampered.read_text(encoding="utf-8").replace(
        '"validation_status":"fail"', '"validation_status":"pass"', 1
    )
    tampered.write_text(text, encoding="utf-8")
    with pytest.raises(module.StatisticsError, match="observation log integrity"):
        _analyze(module, files, observations_path=tampered)

    stale = tmp_path / "stale-run.jsonl"
    rows = [dict(row) for row in files["observation_rows"]]
    rows[0]["run_manifest_sha256"] = "f" * 64
    _rewrite_observations(stale, rows)
    with pytest.raises(module.StatisticsError, match="another run manifest"):
        _analyze(module, files, observations_path=stale)

    duplicate = tmp_path / "duplicate.jsonl"
    lines = files["observations"].read_bytes().splitlines(keepends=True)
    duplicate.write_bytes(b"".join(lines) + lines[0])
    with pytest.raises(module.StatisticsError, match="exactly one physical record"):
        _analyze(module, files, observations_path=duplicate)


def test_stale_subject_and_alarm_labels_are_rejected(tmp_path: Path) -> None:
    module = _load()
    files = _fixture(tmp_path)
    stale_subject = tmp_path / "stale-subject.jsonl"
    rows = list(files["labels"])
    rows.append(
        _audit_row(
            "audit-stale-subject",
            record_type="subject",
            subject_id="old-run-subject",
            scope="in_contract",
            primary_label="CONFIRMED_IN_CONTRACT_DEFECT",
        )
    )
    stale_subject.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )
    with pytest.raises(module.StatisticsError, match="unknown/stale subject"):
        _analyze(
            module,
            files,
            human_audit_path=stale_subject,
            audit_report_path=None,
        )

    stale_alarm = tmp_path / "stale-alarm.jsonl"
    rows = list(files["labels"])
    rows.append(
        _audit_row(
            "audit-stale-alarm",
            record_type="alarm",
            subject_id="s1",
            test_id="test-from-another-run",
            scope="in_contract",
            primary_label="CONFIRMED_IN_CONTRACT_DISCREPANCY",
        )
    )
    stale_alarm.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )
    with pytest.raises(module.StatisticsError, match="unknown/stale alarm"):
        _analyze(
            module,
            files,
            human_audit_path=stale_alarm,
            audit_report_path=None,
        )


def test_plan_and_audit_report_digests_are_verified(tmp_path: Path) -> None:
    module = _load()
    files = _fixture(tmp_path)
    bad_plan = tmp_path / "bad-plan.json"
    plan = json.loads(files["plan"].read_text(encoding="utf-8"))
    plan["matrix_id"] = "tampered"
    _write_json(bad_plan, plan)
    with pytest.raises(module.StatisticsError, match="plan_sha256 mismatch"):
        _analyze(module, files, plan_path=bad_plan)

    bad_report = tmp_path / "bad-report.json"
    report = json.loads(files["audit_report"].read_text(encoding="utf-8"))
    report["final_labels"] += 1
    _write_json(bad_report, report)
    with pytest.raises(module.StatisticsError, match="audit_report_sha256 mismatch"):
        _analyze(module, files, audit_report_path=bad_report)

    partial_report = tmp_path / "partial-report.json"
    report = json.loads(files["audit_report"].read_text(encoding="utf-8"))
    report.pop("audit_report_sha256")
    report["audit_complete"] = False
    report["pending"] = [{"audit_id": "missing"}]
    report["audit_report_sha256"] = stable_json_sha256(report)
    _write_json(partial_report, report)
    with pytest.raises(module.StatisticsError, match="audit report is incomplete"):
        _analyze(module, files, audit_report_path=partial_report)

    substituted_labels = tmp_path / "substituted-labels.jsonl"
    rows = json.loads(json.dumps(files["labels"]))
    rows[0]["primary_label"] = "NO_DEFECT_FOUND"
    rows[0]["analysis_label_sha256"] = stable_json_sha256(
        {
            key: value
            for key, value in rows[0].items()
            if key != "analysis_label_sha256"
        }
    )
    substituted_labels.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )
    with pytest.raises(module.StatisticsError, match="exact compiled analysis-label"):
        _analyze(module, files, human_audit_path=substituted_labels)
