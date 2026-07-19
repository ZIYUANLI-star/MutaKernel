import importlib.util
import json
from pathlib import Path

import pytest

from scripts.build_fse_subject_manifest import (
    build_subject_manifest,
    write_new_canonical_json,
)
from src.experiments import ArtifactProvenance, RunManifest, SubjectProvenance
from src.experiments.protocol import plan_from_files
from tests.experiments.contract_fixture import rich_contract


PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "build_human_audit_queue.py"
MATRIX = PROJECT_ROOT / "configs" / "fse_strategy_matrix.json"


def _load():
    spec = importlib.util.spec_from_file_location("audit_queue_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_and_plan(tmp_path: Path, count: int = 3):
    rows = []
    for index in range(count):
        candidate = tmp_path / f"candidate-{index}.py"
        reference = tmp_path / f"reference-{index}.py"
        candidate.write_text("candidate\n", encoding="utf-8")
        reference.write_text("reference\n", encoding="utf-8")
        rows.append(
            {
                "subject_id": f"s{index}",
                "dataset": "unit",
                "task_id": str(index),
                "language": "cuda",
                "candidate_path": candidate.name,
                "reference_path": reference.name,
                "contract": rich_contract(),
                "source": {"kind": "unit"},
                "metadata": {"task_level": "L1"},
            }
        )
    spec_path = tmp_path / "subjects.jsonl"
    spec_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = build_subject_manifest(input_path=spec_path, root=tmp_path)
    manifest_path = tmp_path / "subjects.json"
    write_new_canonical_json(manifest_path, manifest)
    plan = plan_from_files(manifest_path, MATRIX)
    run_subjects = tuple(
        SubjectProvenance(
            subject_id=subject["subject_id"],
            dataset=subject["dataset"],
            task_id=subject["task_id"],
            language=subject["language"],
            candidate=ArtifactProvenance(**subject["candidate"]),
            reference=ArtifactProvenance(**subject["reference"]),
            contract=subject["contract"],
            metadata=subject["metadata"],
        )
        for subject in manifest["subjects"]
    )
    run_manifest = RunManifest(
        run_id="audit-queue-unit",
        experiment="fse-validator-comparison",
        git_commit="a" * 40,
        git_dirty=False,
        command=("unit-test",),
        config={
            "plan_sha256": plan["plan_sha256"],
            "subject_manifest_sha256": manifest["manifest_sha256"],
        },
        subjects=run_subjects,
    ).to_dict()
    return manifest, plan, run_manifest


def _completed_observations(manifest, plan, run_manifest):
    subject_by_id = {
        subject["subject_id"]: subject for subject in manifest["subjects"]
    }
    return [
        {
            **entry,
            "run_id": run_manifest["run_id"],
            "run_manifest_sha256": run_manifest["manifest_sha256"],
            "subject_sha256": subject_by_id[entry["subject_id"]]["subject_sha256"],
            "contract_sha256": subject_by_id[entry["subject_id"]][
                "contract_sha256"
            ],
            "planned_candidate_runs": entry["candidate_run_cost"],
            "validation_status": "pass",
            "worker_result": {},
        }
        for entry in plan["schedule"]
    ]


def test_queue_blinds_detector_metadata_and_samples_completed_nonalarm_subjects(
    tmp_path: Path,
):
    module = _load()
    manifest, plan, run_manifest = _manifest_and_plan(tmp_path)
    observations = _completed_observations(manifest, plan, run_manifest)
    s0 = [item for item in observations if item["subject_id"] == "s0"]
    s0[0]["validation_status"] = "fail"
    s0[0]["worker_result"] = {
        "evidence": {"replay_bundle": {"logical_path": "evidence/neutral"}}
    }
    s0[1]["validation_status"] = "fail"

    queue, sealed = module.build_queue(
        subject_manifest=manifest,
        experiment_plan=plan,
        run_manifest=run_manifest,
        observations=observations,
        selection_seed=7,
        nonalarm_per_stratum=1,
    )

    assert len([item for item in queue if item["item_kind"] == "subject"]) == 2
    assert len([item for item in queue if item["item_kind"] == "alarm"]) == 2
    public_text = str(queue)
    assert s0[0]["strategy_name"] not in public_text
    assert s0[0]["policy"] not in public_text
    assert any(
        item["evidence_status"] == "MISSING_REPLAY_BUNDLE"
        for item in queue
        if item["item_kind"] == "alarm"
    )
    assert any(
        item["evidence_status"] == "OPERATOR_ONLY_REQUIRES_BLINDED_EXPORT"
        for item in queue
        if item["item_kind"] == "alarm"
    )
    assert sealed["queue_item_count"] == len(queue)
    assert sealed["queue_audit_ids"] == [item["audit_id"] for item in queue]
    assert any(
        mapping.get("strategy_name") == s0[0]["strategy_name"]
        for mapping in sealed["mapping"].values()
    )


def test_unresolved_subjects_are_audited_but_not_sampled_as_nonalarm(
    tmp_path: Path,
):
    module = _load()
    manifest, plan, run_manifest = _manifest_and_plan(tmp_path, count=4)
    completed = _completed_observations(manifest, plan, run_manifest)
    observations = [
        item
        for item in completed
        if item["subject_id"] == "s0"
    ]
    # One observation for s1 makes it partial; s2 remains never executed.
    observations.append(
        next(item for item in completed if item["subject_id"] == "s1")
    )
    s3 = [item for item in completed if item["subject_id"] == "s3"]
    s3[0]["validation_status"] = "inconclusive"
    observations.extend(s3)

    queue, sealed = module.build_queue(
        subject_manifest=manifest,
        experiment_plan=plan,
        run_manifest=run_manifest,
        observations=observations,
        selection_seed=7,
        nonalarm_per_stratum=None,
    )

    selected_subject_ids = {
        sealed["mapping"][item["audit_id"]]["subject_id"]
        for item in queue
        if item["item_kind"] == "subject"
    }
    assert selected_subject_ids == {"s0", "s1", "s2", "s3"}
    assert sealed["completion_counts"]["partially_evaluated"] == 1
    assert sealed["completion_counts"]["never_executed"] == 1
    assert sealed["completion_counts"]["fully_evaluated_with_inconclusive"] == 1
    assert sealed["unresolved_subject_count"] == 3
    unresolved_roles = {
        mapping["subject_id"]: mapping["selection_role"]
        for mapping in sealed["mapping"].values()
        if mapping["item_kind"] == "subject"
    }
    assert unresolved_roles["s1"] == "execution_unresolved"
    assert unresolved_roles["s2"] == "execution_unresolved"
    assert unresolved_roles["s3"] == "execution_unresolved"


def test_queue_rejects_observation_from_another_run(tmp_path: Path) -> None:
    module = _load()
    manifest, plan, run_manifest = _manifest_and_plan(tmp_path, count=1)
    observations = _completed_observations(manifest, plan, run_manifest)
    observations[0]["run_manifest_sha256"] = "f" * 64

    with pytest.raises(module.AuditQueueError, match="frozen provenance"):
        module.build_queue(
            subject_manifest=manifest,
            experiment_plan=plan,
            run_manifest=run_manifest,
            observations=observations,
            selection_seed=7,
            nonalarm_per_stratum=None,
        )
