import json
from pathlib import Path

import pytest

from src.experiments.manifest import (
    ArtifactProvenance,
    DuplicateObservationError,
    ManifestIntegrityError,
    ObservationLog,
    RunManifest,
    SubjectProvenance,
    canonical_json_bytes,
    stable_json_sha256,
)


def test_canonical_hash_is_mapping_and_set_order_independent():
    left = {"b": {3, 1, 2}, "a": [1, {"y": 2, "x": 1}]}
    right = {"a": [1, {"x": 1, "y": 2}], "b": {2, 3, 1}}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert stable_json_sha256(left) == stable_json_sha256(right)


def test_canonical_hash_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        stable_json_sha256({"bad": float("nan")})


def test_artifact_provenance_detects_tampering(tmp_path: Path):
    artifact = tmp_path / "subject.py"
    artifact.write_text("print('v1')\n", encoding="utf-8")
    provenance = ArtifactProvenance.from_file(artifact, root=tmp_path, role="candidate")
    assert provenance.logical_path == "subject.py"
    assert provenance.verify(artifact)

    artifact.write_text("print('v2')\n", encoding="utf-8")
    assert not provenance.verify(artifact)


def _make_manifest(tmp_path: Path) -> RunManifest:
    candidate_path = tmp_path / "candidate.py"
    reference_path = tmp_path / "reference.py"
    candidate_path.write_text("candidate\n", encoding="utf-8")
    reference_path.write_text("reference\n", encoding="utf-8")
    subject = SubjectProvenance(
        subject_id="dataset/task-1",
        dataset="dataset",
        task_id="task-1",
        language="cuda",
        candidate=ArtifactProvenance.from_file(candidate_path, root=tmp_path, role="candidate"),
        reference=ArtifactProvenance.from_file(reference_path, root=tmp_path, role="reference"),
        contract={"dtype": "float32", "shapes": [[32, 32]]},
    )
    return RunManifest(
        run_id="run-001",
        experiment="fse-smoke",
        git_commit="a" * 40,
        git_dirty=False,
        command=("python", "scripts/run_fse_experiment.py"),
        config={"budget": {"candidate_runs": 5}},
        subjects=(subject,),
        environment={"gpu": "test-gpu"},
        created_at_utc="2026-07-19T00:00:00.000000Z",
    )


def test_run_manifest_write_once_and_integrity(tmp_path: Path):
    manifest = _make_manifest(tmp_path)
    path = tmp_path / "run_manifest.json"
    manifest.write_once(path)
    manifest.write_once(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    RunManifest.verify_dict(raw)
    raw["config"]["budget"]["candidate_runs"] = 6
    with pytest.raises(ManifestIntegrityError):
        RunManifest.verify_dict(raw)


def test_observation_log_append_resume_and_deduplicate(tmp_path: Path):
    path = tmp_path / "observations.jsonl"
    log = ObservationLog(path)
    first = {"test_id": "test-a", "verdict": "pass", "candidate_runs": 1}
    second = {"test_id": "test-b", "verdict": "fail", "candidate_runs": 1}

    assert log.append(first)
    assert not log.append(first)
    assert log.append(second)
    assert len(log) == 2
    assert list(log.pending(["test-a", "test-c"])) == ["test-c"]

    resumed = ObservationLog(path)
    assert resumed.seen_ids == frozenset({"test-a", "test-b"})
    assert not resumed.append(first)
    assert resumed.get("test-b")["verdict"] == "fail"

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all("observation_sha256" in json.loads(line) for line in lines)


def test_observation_log_rejects_conflicting_duplicate(tmp_path: Path):
    log = ObservationLog(tmp_path / "observations.jsonl")
    assert log.append({"test_id": "same", "verdict": "pass"})
    with pytest.raises(DuplicateObservationError):
        log.append({"test_id": "same", "verdict": "fail"})
