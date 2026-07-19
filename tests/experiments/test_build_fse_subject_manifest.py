import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_fse_subject_manifest import (
    build_subject_manifest,
    main,
    write_new_canonical_json,
)
from src.experiments.manifest import stable_json_sha256
from tests.experiments.contract_fixture import rich_contract


def _write_subject_spec(root: Path, rows):
    path = root / "subjects.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _row(subject_id="s1", candidate="candidate.py", reference="reference.py"):
    return {
        "subject_id": subject_id,
        "dataset": "public-set",
        "task_id": "task-7",
        "language": "cuda",
        "candidate_path": candidate,
        "reference_path": reference,
        "contract": rich_contract(),
        "source": {"repository": "example/repo", "revision": "abc123"},
        "metadata": {"generator": "example-agent"},
    }


def test_subject_manifest_hashes_sources_without_executing_them(tmp_path: Path):
    marker = tmp_path / "executed.txt"
    candidate_bytes = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n".encode()
    (tmp_path / "candidate.py").write_bytes(candidate_bytes)
    (tmp_path / "reference.py").write_text("this is intentionally invalid python !!!\n", encoding="utf-8")
    spec_path = _write_subject_spec(tmp_path, [_row()])

    manifest = build_subject_manifest(input_path=spec_path, root=tmp_path)
    assert not marker.exists()
    assert manifest["subject_count"] == 1
    subject = manifest["subjects"][0]
    assert subject["dataset"] == "public-set"
    assert subject["task_id"] == "task-7"
    assert subject["source"]["dataset"] == "public-set"
    assert subject["candidate"]["sha256"] == hashlib.sha256(candidate_bytes).hexdigest()
    assert subject["candidate"]["logical_path"] == "candidate.py"
    assert len(subject["subject_sha256"]) == 64

    payload = dict(manifest)
    digest = payload.pop("manifest_sha256")
    assert digest == stable_json_sha256(payload)


def test_subjects_are_sorted_and_duplicate_ids_rejected(tmp_path: Path):
    for name in ("candidate.py", "reference.py"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    spec_path = _write_subject_spec(tmp_path, [_row("z"), _row("a")])
    manifest = build_subject_manifest(input_path=spec_path, root=tmp_path)
    assert [subject["subject_id"] for subject in manifest["subjects"]] == ["a", "z"]

    duplicate_path = tmp_path / "duplicates.jsonl"
    duplicate_path.write_text(json.dumps(_row("same")) + "\n" + json.dumps(_row("same")) + "\n")
    with pytest.raises(ValueError, match="unique"):
        build_subject_manifest(input_path=duplicate_path, root=tmp_path)


def test_artifacts_outside_root_are_rejected(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside", encoding="utf-8")
    (root / "reference.py").write_text("reference", encoding="utf-8")
    spec_path = _write_subject_spec(
        root,
        [_row(candidate=str(outside), reference="reference.py")],
    )
    with pytest.raises(ValueError, match="outside provenance root"):
        build_subject_manifest(input_path=spec_path, root=root)


def test_subject_output_and_cli_refuse_overwrite(tmp_path: Path):
    (tmp_path / "candidate.py").write_text("candidate", encoding="utf-8")
    (tmp_path / "reference.py").write_text("reference", encoding="utf-8")
    spec_path = _write_subject_spec(tmp_path, [_row()])
    output = tmp_path / "subject_manifest.json"

    assert main(["--input", str(spec_path), "--output", str(output), "--root", str(tmp_path)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["subject_count"] == 1
    assert main(["--input", str(spec_path), "--output", str(output), "--root", str(tmp_path)]) == 2

    other = tmp_path / "other.json"
    write_new_canonical_json(other, {"x": 1})
    with pytest.raises(FileExistsError):
        write_new_canonical_json(other, {"x": 2})
