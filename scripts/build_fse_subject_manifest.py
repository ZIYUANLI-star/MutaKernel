#!/usr/bin/env python3
"""Build a content-addressed FSE subject manifest without executing sources.

Input is JSONL with one object per subject.  Required fields are
``subject_id``, ``dataset``, ``task_id``, ``language``, ``candidate_path``, and
``reference_path``.  Optional ``contract``, ``source``, and ``metadata`` fields
must be JSON objects.  Source files are opened only as bytes for SHA-256 and
size calculation; they are never imported or evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.manifest import (
    ArtifactProvenance,
    SubjectProvenance,
    canonical_json_bytes,
    stable_json_sha256,
)
from src.experiments.contract import validate_contract


SUBJECT_MANIFEST_SCHEMA_VERSION = "1.0"
REQUIRED_FIELDS = {
    "subject_id",
    "dataset",
    "task_id",
    "language",
    "candidate_path",
    "reference_path",
}
OPTIONAL_FIELDS = {"contract", "source", "metadata"}


def _read_subject_specs(path: Path) -> List[Mapping[str, Any]]:
    specs: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"subject at {path}:{line_number} must be a JSON object")
            specs.append(value)
    if not specs:
        raise ValueError(f"no subjects found in {path}")
    return specs


def _resolve_artifact(root: Path, raw_path: Any, field_name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{field_name} must be a non-empty string")
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} is outside provenance root: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{field_name} is not a regular file: {resolved}")
    return resolved


def _require_string(spec: Mapping[str, Any], field_name: str) -> str:
    value = spec.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_mapping(spec: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = spec.get(field_name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _build_subject(spec: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    missing = REQUIRED_FIELDS - set(spec)
    if missing:
        raise ValueError(f"subject is missing required fields: {sorted(missing)}")
    unknown = set(spec) - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if unknown:
        raise ValueError(f"subject has unknown fields: {sorted(unknown)}")

    subject_id = _require_string(spec, "subject_id")
    dataset = _require_string(spec, "dataset")
    task_id = _require_string(spec, "task_id")
    language = _require_string(spec, "language")
    contract = validate_contract(_require_mapping(spec, "contract"))
    source = _require_mapping(spec, "source")
    metadata = _require_mapping(spec, "metadata")

    candidate_path = _resolve_artifact(root, spec["candidate_path"], "candidate_path")
    reference_path = _resolve_artifact(root, spec["reference_path"], "reference_path")
    candidate = ArtifactProvenance.from_file(candidate_path, root=root, role="candidate")
    reference = ArtifactProvenance.from_file(reference_path, root=root, role="reference")

    provenance = SubjectProvenance(
        subject_id=subject_id,
        dataset=dataset,
        task_id=task_id,
        language=language,
        candidate=candidate,
        reference=reference,
        contract=contract,
        metadata=metadata,
    )
    payload = provenance.to_dict()
    payload["source"] = {
        "dataset": dataset,
        "task_id": task_id,
        **dict(source),
    }
    payload["subject_sha256"] = stable_json_sha256(payload)
    return payload


def build_subject_manifest(*, input_path: Path, root: Path) -> Dict[str, Any]:
    """Hash and validate all subject artifacts without loading them as code."""

    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"provenance root is not a directory: {root}")
    input_path = input_path.resolve()
    try:
        input_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"input specification is outside provenance root: {input_path}") from exc

    specs = _read_subject_specs(input_path)
    subjects = [_build_subject(spec, root) for spec in specs]
    subjects.sort(key=lambda subject: subject["subject_id"])
    subject_ids = [subject["subject_id"] for subject in subjects]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("subject_id values must be unique")

    input_provenance = ArtifactProvenance.from_file(input_path, root=root, role="subject_spec")
    payload: Dict[str, Any] = {
        "schema_version": SUBJECT_MANIFEST_SCHEMA_VERSION,
        "input_spec": input_provenance.to_dict(),
        "subject_count": len(subjects),
        "subjects": subjects,
        "subjects_sha256": stable_json_sha256(subjects),
    }
    payload["manifest_sha256"] = stable_json_sha256(payload)
    return payload


def write_new_canonical_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(data) + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL subject specification")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_subject_manifest(input_path=args.input, root=args.root)
    except (OSError, ValueError, TypeError) as exc:
        print(f"failed to build subject manifest: {exc}", file=sys.stderr)
        return 2
    try:
        write_new_canonical_json(args.output, manifest)
    except FileExistsError:
        print(f"refusing to overwrite existing subject manifest: {args.output}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "manifest_sha256": manifest["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
