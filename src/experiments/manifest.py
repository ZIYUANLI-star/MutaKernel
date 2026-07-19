"""Canonical provenance manifests and append-only observation storage.

The experiment layer uses canonical JSON for identities and SHA-256 hashes.
Run manifests are immutable once written.  Observations are stored as JSONL so
an interrupted run can resume without rewriting completed records.

``ObservationLog`` is intentionally a single-writer abstraction.  Independent
workers should return observations to one orchestrator process, which owns the
log and performs duplicate/conflict checks.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple


MANIFEST_SCHEMA_VERSION = "1.0"
OBSERVATION_SCHEMA_VERSION = "1.0"


class ManifestIntegrityError(ValueError):
    """A persisted manifest or observation does not match its digest."""


class DuplicateObservationError(ValueError):
    """The same test id was associated with two different observations."""


def _canonicalize(value: Any) -> Any:
    """Convert supported Python values to a deterministic JSON value.

    Mapping keys must be strings.  Sets are sorted by their own canonical JSON
    encoding, making their representation independent of hash iteration order.
    Non-finite floats are rejected because JSON encodings for them are neither
    portable nor suitable for stable scientific provenance.
    """

    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            raise ValueError("naive datetimes are not canonical")
        normalized = value.astimezone(dt.timezone.utc)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, dt.date):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not canonical JSON")
        # Normalize negative zero, whose sign is irrelevant for experiment
        # metadata but otherwise yields a distinct JSON byte sequence.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical JSON requires string keys, got {type(key)!r}")
            normalized[key] = _canonicalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: canonical_json_bytes(item))
    raise TypeError(f"unsupported canonical JSON value: {type(value)!r}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique UTF-8 JSON encoding used for experiment hashes."""

    normalized = _canonicalize(value)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def stable_json_sha256(value: Any) -> str:
    """SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Hash the exact bytes of a file without loading it all into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _logical_path(path: Path, root: Optional[Path]) -> str:
    resolved = path.resolve()
    if root is None:
        return path.as_posix()
    root_resolved = root.resolve()
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact {resolved} is outside provenance root {root_resolved}") from exc


@dataclass(frozen=True)
class ArtifactProvenance:
    """Content-addressed provenance for one input artifact."""

    logical_path: str
    sha256: str
    size_bytes: int
    role: str = "input"

    def __post_init__(self) -> None:
        if not self.logical_path:
            raise ValueError("logical_path must not be empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("sha256 must be a lowercase 64-character digest")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not self.role:
            raise ValueError("role must not be empty")

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        root: Path | str | None = None,
        role: str = "input",
    ) -> "ArtifactProvenance":
        artifact_path = Path(path)
        provenance_root = Path(root) if root is not None else None
        return cls(
            logical_path=_logical_path(artifact_path, provenance_root),
            sha256=sha256_file(artifact_path),
            size_bytes=artifact_path.stat().st_size,
            role=role,
        )

    def verify(self, path: Path | str) -> bool:
        candidate = Path(path)
        return candidate.is_file() and candidate.stat().st_size == self.size_bytes and sha256_file(candidate) == self.sha256

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SubjectProvenance:
    """Immutable identity and artifacts for one evaluated kernel subject."""

    subject_id: str
    dataset: str
    task_id: str
    language: str
    candidate: ArtifactProvenance
    reference: ArtifactProvenance
    contract: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("subject_id", "dataset", "task_id", "language"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        # Validate early so invalid metadata cannot enter a persisted manifest.
        canonical_json_bytes(self.contract)
        canonical_json_bytes(self.metadata)

    @property
    def provenance_sha256(self) -> str:
        return stable_json_sha256(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "dataset": self.dataset,
            "task_id": self.task_id,
            "language": self.language,
            "candidate": self.candidate.to_dict(),
            "reference": self.reference.to_dict(),
            "contract": _canonicalize(self.contract),
            "contract_sha256": stable_json_sha256(self.contract),
            "metadata": _canonicalize(self.metadata),
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class RunManifest:
    """Immutable provenance envelope for an experiment run."""

    run_id: str
    experiment: str
    git_commit: str
    git_dirty: bool
    command: Tuple[str, ...]
    config: Mapping[str, Any]
    subjects: Tuple[SubjectProvenance, ...]
    environment: Mapping[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=_utc_now)
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id or not self.experiment or not self.git_commit:
            raise ValueError("run_id, experiment, and git_commit are required")
        if not self.command:
            raise ValueError("command must not be empty")
        ids = [subject.subject_id for subject in self.subjects]
        if len(ids) != len(set(ids)):
            raise ValueError("subject ids must be unique within a run")
        canonical_json_bytes(self.config)
        canonical_json_bytes(self.environment)

    def _payload(self) -> Dict[str, Any]:
        ordered_subjects = sorted(self.subjects, key=lambda subject: subject.subject_id)
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "git": {"commit": self.git_commit, "dirty": self.git_dirty},
            "command": list(self.command),
            "config": _canonicalize(self.config),
            "config_sha256": stable_json_sha256(self.config),
            "subjects": [subject.to_dict() for subject in ordered_subjects],
            "environment": _canonicalize(self.environment),
            "created_at_utc": self.created_at_utc,
        }

    @property
    def manifest_sha256(self) -> str:
        return stable_json_sha256(self._payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        payload["manifest_sha256"] = stable_json_sha256(payload)
        return payload

    def write_once(self, path: Path | str) -> None:
        """Write a manifest once; an identical existing manifest is accepted."""

        target = Path(path)
        encoded = canonical_json_bytes(self.to_dict()) + b"\n"
        if target.exists():
            if target.read_bytes() == encoded:
                return
            raise FileExistsError(f"refusing to overwrite immutable run manifest: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def verify_dict(data: Mapping[str, Any]) -> None:
        expected = data.get("manifest_sha256")
        if not isinstance(expected, str):
            raise ManifestIntegrityError("manifest_sha256 is missing")
        payload = dict(data)
        del payload["manifest_sha256"]
        actual = stable_json_sha256(payload)
        if actual != expected:
            raise ManifestIntegrityError(f"manifest digest mismatch: expected {expected}, got {actual}")


class ObservationLog:
    """Append-only JSONL observations with resume-safe test-id deduplication."""

    def __init__(self, path: Path | str, *, key_field: str = "test_id") -> None:
        if not key_field:
            raise ValueError("key_field must not be empty")
        self.path = Path(path)
        self.key_field = key_field
        self._records: Dict[str, Dict[str, Any]] = {}
        self._digests: Dict[str, str] = {}
        self._lock = threading.RLock()
        if self.path.exists():
            self._load_existing()

    @staticmethod
    def _prepare_record(record: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
        payload = dict(record)
        supplied = payload.pop("observation_sha256", None)
        payload.setdefault("schema_version", OBSERVATION_SCHEMA_VERSION)
        digest = stable_json_sha256(payload)
        if supplied is not None and supplied != digest:
            raise ManifestIntegrityError(
                f"observation digest mismatch: expected {supplied}, got {digest}"
            )
        payload["observation_sha256"] = digest
        return payload, digest

    def _load_existing(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestIntegrityError(
                        f"invalid JSONL at {self.path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise ManifestIntegrityError(
                        f"observation at {self.path}:{line_number} is not an object"
                    )
                prepared, digest = self._prepare_record(raw)
                test_id = prepared.get(self.key_field)
                if not isinstance(test_id, str) or not test_id:
                    raise ManifestIntegrityError(
                        f"missing {self.key_field!r} at {self.path}:{line_number}"
                    )
                previous = self._digests.get(test_id)
                if previous is not None and previous != digest:
                    raise DuplicateObservationError(
                        f"conflicting observations for {test_id!r} in {self.path}"
                    )
                self._records[test_id] = prepared
                self._digests[test_id] = digest

    @property
    def seen_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, test_id: object) -> bool:
        return test_id in self._records

    def get(self, test_id: str) -> Optional[Dict[str, Any]]:
        record = self._records.get(test_id)
        return dict(record) if record is not None else None

    def records(self) -> Iterator[Dict[str, Any]]:
        for record in self._records.values():
            yield dict(record)

    def pending(self, test_ids: Iterable[str]) -> Iterator[str]:
        for test_id in test_ids:
            if test_id not in self._records:
                yield test_id

    def append(self, record: Mapping[str, Any]) -> bool:
        """Append one observation.

        Returns ``False`` when the exact observation already exists (normal
        resume behavior).  Reusing a test id with different content raises
        :class:`DuplicateObservationError`.
        """

        prepared, digest = self._prepare_record(record)
        test_id = prepared.get(self.key_field)
        if not isinstance(test_id, str) or not test_id:
            raise ValueError(f"observation requires a non-empty {self.key_field!r}")

        with self._lock:
            previous = self._digests.get(test_id)
            if previous is not None:
                if previous == digest:
                    return False
                raise DuplicateObservationError(
                    f"test id {test_id!r} already has a different observation"
                )

            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = canonical_json_bytes(prepared).decode("utf-8")
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            self._records[test_id] = prepared
            self._digests[test_id] = digest
            return True
