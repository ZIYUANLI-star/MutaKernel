"""Portable, fail-closed resolution for generated-kernel registries.

Historical ``best_kernels.json`` files recorded author-machine absolute paths.
Those paths are neither portable nor safe to replace with an arbitrary string
operation.  This module gives a registry entry a stable identity relative to a
configured ``runs_root`` and permits legacy rebasing only at an explicit
``runs`` path component.

The canonical portable field is ``kernel_relpath``.  During migration,
``kernel_path`` is also rewritten to the same POSIX relative identity so no
author-machine absolute path remains in the migrated data.  If
``kernel_relpath`` is present it is authoritative: an invalid or missing
portable target is an error, never a reason to fall back to a legacy path.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


PORTABLE_PATH_FIELD = "kernel_relpath"
LEGACY_PATH_FIELD = "kernel_path"

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_DRIVE_RELATIVE_RE = re.compile(r"^[A-Za-z]:")
_PROBLEM_RE = re.compile(r"^problem_(\d+)$")
_TURN_RE = re.compile(r"^turn_(\d+)$")
_REGISTRY_KEY_RE = re.compile(r"^L\d+_P(\d+)$", re.IGNORECASE)


class KernelRegistryError(ValueError):
    """Base class for registry format, path, and collision failures."""


class RegistryFormatError(KernelRegistryError):
    """The registry or one of its entries does not implement the schema."""


class UnsafeKernelPathError(KernelRegistryError):
    """A path is ambiguous, non-portable, or may escape ``runs_root``."""


class KernelPathMissingError(KernelRegistryError):
    """A portable kernel identity has no regular-file target."""


class RegistryCollisionError(KernelRegistryError):
    """Two registry identities resolve to the same portable target."""


class KernelRegistryScanError(KernelRegistryError):
    """One full-registry scan found one or more entry-level failures."""

    def __init__(self, issues: Sequence[KernelRegistryError]) -> None:
        if not issues:
            raise ValueError("KernelRegistryScanError requires at least one issue")
        self.issues: Tuple[KernelRegistryError, ...] = tuple(issues)
        details = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(
            f"kernel registry validation failed with {len(self.issues)} issue(s):\n"
            f"{details}"
        )


@dataclass(frozen=True)
class KernelPathResolution:
    """One registry entry resolved beneath the configured runs root."""

    registry_key: str
    absolute_path: Path
    portable_identity: str
    source_field: str
    rebased_legacy_absolute: bool


@dataclass(frozen=True)
class KernelRegistryScan:
    """Successful full-registry validation and its pure migration result."""

    runs_root: Path
    resolutions: Mapping[str, KernelPathResolution]
    migrated_registry: Mapping[str, Mapping[str, Any]]
    migrated_keys: Tuple[str, ...]

    @property
    def resolved_paths(self) -> Dict[str, Path]:
        return {
            key: resolution.absolute_path
            for key, resolution in self.resolutions.items()
        }


def _normalized_foreign_path(raw: str, *, registry_key: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise RegistryFormatError(
            f"registry entry {registry_key!r} has an empty or non-string kernel path"
        )
    if "\x00" in raw:
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} kernel path contains a NUL byte"
        )
    normalized = raw.replace("\\", "/")
    if not normalized or normalized.strip() != normalized:
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} kernel path has ambiguous whitespace"
        )
    return normalized


def _is_foreign_absolute(normalized: str) -> bool:
    return (
        normalized.startswith("/")
        or normalized.startswith("//")
        or bool(_WINDOWS_DRIVE_RE.match(normalized))
    )


def _portable_components(raw: str, *, registry_key: str) -> Tuple[str, ...]:
    normalized = _normalized_foreign_path(raw, registry_key=registry_key)
    if _is_foreign_absolute(normalized) or _WINDOWS_DRIVE_RELATIVE_RE.match(normalized):
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} portable kernel identity must be relative: "
            f"{raw!r}"
        )
    components = tuple(normalized.split("/"))
    if any(not component for component in components):
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} kernel identity has an empty path component"
        )
    if any(component in {".", "..", "~"} for component in components):
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} kernel identity contains traversal or "
            f"home-expansion components"
        )
    if any(":" in component for component in components):
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} kernel identity contains a drive or URI marker"
        )
    if any(component.casefold() == "runs" for component in components):
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} portable identity must be relative to "
            f"runs_root, not include another 'runs' component"
        )
    return components


def _legacy_components_after_runs(
    raw: str,
    *,
    registry_key: str,
) -> Tuple[str, ...]:
    normalized = _normalized_foreign_path(raw, registry_key=registry_key)
    if not _is_foreign_absolute(normalized):
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} legacy path is not an absolute path: {raw!r}"
        )
    # Remove only the syntactic absolute prefix.  The remaining components are
    # parsed independently of the host OS, so POSIX registries can migrate on
    # Windows and Windows registries can migrate on POSIX.
    components = tuple(component for component in normalized.split("/") if component)
    anchors = [
        index
        for index, component in enumerate(components)
        if component.casefold() == "runs"
    ]
    if len(anchors) != 1:
        qualifier = "no" if not anchors else "multiple"
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} legacy absolute path has {qualifier} "
            f"explicit '/runs/' anchor: {raw!r}"
        )
    suffix = components[anchors[0] + 1 :]
    if not suffix:
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} legacy path ends at its '/runs/' anchor"
        )
    # Re-run the portable checks on the suffix rather than trusting the old
    # absolute path's spelling.
    return _portable_components("/".join(suffix), registry_key=registry_key)


def _validate_kernel_identity(
    components: Tuple[str, ...],
    *,
    registry_key: str,
    entry: Mapping[str, Any],
) -> Tuple[str, int, int]:
    if len(components) < 4 or components[-1] != "kernel.py":
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} must identify "
            f"<run>/.../problem_N/turn_N/kernel.py"
        )

    problem_match = _PROBLEM_RE.fullmatch(components[-3])
    turn_match = _TURN_RE.fullmatch(components[-2])
    if problem_match is None or turn_match is None:
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} must end in "
            f"problem_N/turn_N/kernel.py"
        )
    # At least one component before problem_N is the run identity.  Additional
    # safe components such as the historical 'iterations' directory are kept.
    run_identity = "/".join(components[:-3])
    if not run_identity:
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} has no run identity"
        )

    problem_id = int(problem_match.group(1))
    turn = int(turn_match.group(1))

    if "problem_id" in entry:
        recorded_problem = entry["problem_id"]
        if (
            isinstance(recorded_problem, bool)
            or not isinstance(recorded_problem, (int, str))
            or not str(recorded_problem).isdigit()
            or int(recorded_problem) != problem_id
        ):
            raise RegistryFormatError(
                f"registry entry {registry_key!r} problem_id does not match "
                f"{components[-3]!r}"
            )
    if "turn" in entry:
        recorded_turn = entry["turn"]
        if (
            isinstance(recorded_turn, bool)
            or not isinstance(recorded_turn, (int, str))
            or not str(recorded_turn).isdigit()
            or int(recorded_turn) != turn
        ):
            raise RegistryFormatError(
                f"registry entry {registry_key!r} turn does not match "
                f"{components[-2]!r}"
            )
    key_match = _REGISTRY_KEY_RE.fullmatch(registry_key)
    if key_match is not None and int(key_match.group(1)) != problem_id:
        raise RegistryFormatError(
            f"registry key {registry_key!r} does not match {components[-3]!r}"
        )
    return run_identity, problem_id, turn


def _validated_runs_root(runs_root: Path | str) -> Path:
    root = Path(runs_root)
    try:
        resolved = root.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise KernelPathMissingError(
            f"configured runs_root does not exist or cannot be resolved: {root}"
        ) from exc
    if not resolved.is_dir():
        raise KernelPathMissingError(
            f"configured runs_root is not a directory: {resolved}"
        )
    return resolved


def _resolve_under_root(
    root: Path,
    components: Tuple[str, ...],
    *,
    registry_key: str,
    require_exists: bool,
) -> Path:
    candidate = root.joinpath(*components)
    try:
        resolved = candidate.resolve(strict=require_exists)
    except (FileNotFoundError, OSError) as exc:
        raise KernelPathMissingError(
            f"registry entry {registry_key!r} kernel target is missing or unreadable: "
            f"{'/'.join(components)}"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UnsafeKernelPathError(
            f"registry entry {registry_key!r} resolves outside configured runs_root: "
            f"{'/'.join(components)}"
        ) from exc
    if require_exists and not resolved.is_file():
        raise KernelPathMissingError(
            f"registry entry {registry_key!r} kernel target is not a regular file: "
            f"{'/'.join(components)}"
        )
    return resolved


def resolve_kernel_entry(
    registry_key: str,
    entry: Mapping[str, Any],
    runs_root: Path | str,
    *,
    require_exists: bool = True,
) -> KernelPathResolution:
    """Resolve one registry entry without permitting arbitrary absolute rebases.

    ``kernel_relpath`` is preferred and authoritative when present.  Otherwise
    a relative ``kernel_path`` is already portable.  An absolute
    ``kernel_path`` is accepted only as legacy data with exactly one explicit
    ``runs`` component; only the suffix after that anchor is retained.
    """

    if not isinstance(registry_key, str) or not registry_key:
        raise RegistryFormatError("registry entry key must be a non-empty string")
    if not isinstance(entry, Mapping):
        raise RegistryFormatError(
            f"registry entry {registry_key!r} must be a JSON object"
        )
    root = _validated_runs_root(runs_root)

    rebased = False
    if PORTABLE_PATH_FIELD in entry:
        raw = entry[PORTABLE_PATH_FIELD]
        if not isinstance(raw, str):
            raise RegistryFormatError(
                f"registry entry {registry_key!r} {PORTABLE_PATH_FIELD} must be a string"
            )
        components = _portable_components(raw, registry_key=registry_key)
        source_field = PORTABLE_PATH_FIELD
    else:
        raw = entry.get(LEGACY_PATH_FIELD)
        if not isinstance(raw, str) or not raw:
            raise RegistryFormatError(
                f"registry entry {registry_key!r} requires {PORTABLE_PATH_FIELD!r} "
                f"or {LEGACY_PATH_FIELD!r}"
            )
        normalized = _normalized_foreign_path(raw, registry_key=registry_key)
        if _is_foreign_absolute(normalized):
            components = _legacy_components_after_runs(
                raw,
                registry_key=registry_key,
            )
            source_field = f"{LEGACY_PATH_FIELD}:legacy_absolute"
            rebased = True
        else:
            components = _portable_components(raw, registry_key=registry_key)
            source_field = LEGACY_PATH_FIELD

    _validate_kernel_identity(
        components,
        registry_key=registry_key,
        entry=entry,
    )
    portable_identity = "/".join(components)
    absolute_path = _resolve_under_root(
        root,
        components,
        registry_key=registry_key,
        require_exists=require_exists,
    )
    return KernelPathResolution(
        registry_key=registry_key,
        absolute_path=absolute_path,
        portable_identity=portable_identity,
        source_field=source_field,
        rebased_legacy_absolute=rebased,
    )


def scan_kernel_registry(
    registry: Mapping[str, Any],
    runs_root: Path | str,
    *,
    require_exists: bool = True,
) -> KernelRegistryScan:
    """Validate every entry once and build an in-memory portable migration.

    Entry failures are aggregated so a single scan reports the complete set of
    missing and invalid records.  Any error prevents a partial migration from
    being returned.  Portable identities are compared case-insensitively to
    reject registries that would collide when moved between POSIX and Windows.
    """

    if not isinstance(registry, Mapping) or not registry:
        raise RegistryFormatError("kernel registry must be a non-empty JSON object")
    root = _validated_runs_root(runs_root)
    issues = []
    resolutions: Dict[str, KernelPathResolution] = {}
    migrated: Dict[str, Mapping[str, Any]] = {}
    migrated_keys = []
    identity_owners: Dict[str, str] = {}
    target_owners: Dict[str, str] = {}

    sortable_keys = list(registry.keys())
    if any(not isinstance(key, str) or not key for key in sortable_keys):
        raise RegistryFormatError("all kernel registry keys must be non-empty strings")

    for registry_key in sorted(sortable_keys):
        entry = registry[registry_key]
        try:
            resolution = resolve_kernel_entry(
                registry_key,
                entry,
                root,
                require_exists=require_exists,
            )
        except KernelRegistryError as exc:
            issues.append(exc)
            continue

        identity_key = resolution.portable_identity.casefold()
        target_key = os.path.normcase(str(resolution.absolute_path)).replace("\\", "/").casefold()
        previous_identity = identity_owners.get(identity_key)
        previous_target = target_owners.get(target_key)
        if previous_identity is not None:
            issues.append(
                RegistryCollisionError(
                    f"registry entries {previous_identity!r} and {registry_key!r} "
                    f"have the same portable kernel identity "
                    f"{resolution.portable_identity!r}"
                )
            )
            continue
        if previous_target is not None:
            issues.append(
                RegistryCollisionError(
                    f"registry entries {previous_target!r} and {registry_key!r} "
                    f"resolve to the same kernel file {resolution.absolute_path}"
                )
            )
            continue
        identity_owners[identity_key] = registry_key
        target_owners[target_key] = registry_key
        resolutions[registry_key] = resolution

        assert isinstance(entry, Mapping)
        migrated_entry = copy.deepcopy(dict(entry))
        before_path = migrated_entry.get(LEGACY_PATH_FIELD)
        before_identity = migrated_entry.get(PORTABLE_PATH_FIELD)
        migrated_entry[PORTABLE_PATH_FIELD] = resolution.portable_identity
        migrated_entry[LEGACY_PATH_FIELD] = resolution.portable_identity
        migrated[registry_key] = migrated_entry
        if (
            before_path != resolution.portable_identity
            or before_identity != resolution.portable_identity
        ):
            migrated_keys.append(registry_key)

    if issues:
        raise KernelRegistryScanError(issues)
    return KernelRegistryScan(
        runs_root=root,
        resolutions=resolutions,
        migrated_registry=migrated,
        migrated_keys=tuple(migrated_keys),
    )


def _reject_duplicate_object_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistryFormatError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_kernel_registry(path: Path | str) -> Mapping[str, Any]:
    """Load registry JSON while rejecting duplicate object keys."""

    registry_path = Path(path)
    try:
        raw = registry_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryFormatError(
            f"cannot read kernel registry {registry_path}: {exc}"
        ) from exc
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_object_keys)
    except json.JSONDecodeError as exc:
        raise RegistryFormatError(
            f"invalid kernel registry JSON at {registry_path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping) or not value:
        raise RegistryFormatError(
            f"kernel registry {registry_path} must contain a non-empty JSON object"
        )
    return value


def scan_kernel_registry_file(
    registry_path: Path | str,
    runs_root: Path | str,
    *,
    require_exists: bool = True,
) -> KernelRegistryScan:
    """Load, fully validate, and prepare migration of one registry file."""

    return scan_kernel_registry(
        load_kernel_registry(registry_path),
        runs_root,
        require_exists=require_exists,
    )


__all__ = [
    "KernelPathMissingError",
    "KernelPathResolution",
    "KernelRegistryError",
    "KernelRegistryScan",
    "KernelRegistryScanError",
    "PORTABLE_PATH_FIELD",
    "LEGACY_PATH_FIELD",
    "RegistryCollisionError",
    "RegistryFormatError",
    "UnsafeKernelPathError",
    "load_kernel_registry",
    "resolve_kernel_entry",
    "scan_kernel_registry",
    "scan_kernel_registry_file",
]
