"""Tests for portable, fail-closed generated-kernel registry resolution."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from src.experiments.kernel_registry import (
    KernelPathMissingError,
    KernelRegistryScanError,
    RegistryFormatError,
    UnsafeKernelPathError,
    load_kernel_registry,
    resolve_kernel_entry,
    scan_kernel_registry,
    scan_kernel_registry_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _kernel(
    runs_root: Path,
    relative: str = "paper_run/iterations/problem_001/turn_02/kernel.py",
) -> Path:
    path = runs_root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# generated kernel\n", encoding="utf-8")
    return path


def _entry(path_field: str, path: str, *, problem_id: str = "1", turn: int = 2):
    return {
        "level": "L1",
        "problem_id": problem_id,
        "turn": turn,
        path_field: path,
    }


def test_portable_relative_kernel_path_resolves_under_runs_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    expected = _kernel(runs_root)

    resolution = resolve_kernel_entry(
        "L1_P1",
        _entry(
            "kernel_path",
            "paper_run/iterations/problem_001/turn_02/kernel.py",
        ),
        runs_root,
    )

    assert resolution.absolute_path == expected.resolve()
    assert resolution.portable_identity == (
        "paper_run/iterations/problem_001/turn_02/kernel.py"
    )
    assert resolution.source_field == "kernel_path"
    assert resolution.rebased_legacy_absolute is False


@pytest.mark.parametrize(
    "legacy",
    [
        "/home/author/KernelBench/runs/paper_run/iterations/"
        "problem_001/turn_02/kernel.py",
        r"C:\Users\author\KernelBench\runs\paper_run\iterations\problem_001\turn_02\kernel.py",
    ],
)
def test_legacy_absolute_path_rebases_only_after_explicit_runs_anchor(
    tmp_path: Path,
    legacy: str,
) -> None:
    runs_root = tmp_path / "portable-runs"
    expected = _kernel(runs_root)

    resolution = resolve_kernel_entry(
        "L1_P1",
        _entry("kernel_path", legacy),
        runs_root,
    )

    assert resolution.absolute_path == expected.resolve()
    assert resolution.rebased_legacy_absolute is True
    assert resolution.portable_identity.startswith("paper_run/")


def test_authoritative_portable_identity_is_preferred_to_legacy_path(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    expected = _kernel(runs_root, "new_run/problem_001/turn_02/kernel.py")
    entry = _entry(
        "kernel_path",
        "/old/machine/runs/old_run/problem_001/turn_02/kernel.py",
    )
    entry["kernel_relpath"] = "new_run/problem_001/turn_02/kernel.py"

    resolution = resolve_kernel_entry("L1_P1", entry, runs_root)

    assert resolution.absolute_path == expected.resolve()
    assert resolution.source_field == "kernel_relpath"
    assert resolution.rebased_legacy_absolute is False


def test_missing_authoritative_portable_identity_never_falls_back(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _kernel(runs_root, "legacy_run/problem_001/turn_02/kernel.py")
    entry = _entry(
        "kernel_path",
        "/old/runs/legacy_run/problem_001/turn_02/kernel.py",
    )
    entry["kernel_relpath"] = "missing_run/problem_001/turn_02/kernel.py"

    with pytest.raises(KernelPathMissingError, match="L1_P1.*missing"):
        resolve_kernel_entry("L1_P1", entry, runs_root)


@pytest.mark.parametrize(
    "unsafe",
    [
        "../outside/problem_001/turn_02/kernel.py",
        "run/./problem_001/turn_02/kernel.py",
        "run//problem_001/turn_02/kernel.py",
        "runs/run/problem_001/turn_02/kernel.py",
        "C:relative/problem_001/turn_02/kernel.py",
        "file://run/problem_001/turn_02/kernel.py",
    ],
)
def test_portable_identity_rejects_ambiguous_or_escaping_paths(
    tmp_path: Path,
    unsafe: str,
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    with pytest.raises(UnsafeKernelPathError):
        resolve_kernel_entry(
            "L1_P1",
            _entry("kernel_relpath", unsafe),
            runs_root,
            require_exists=False,
        )


@pytest.mark.parametrize(
    "legacy",
    [
        "/home/author/KernelBench/paper_run/problem_001/turn_02/kernel.py",
        "/home/runs/archive/runs/paper_run/problem_001/turn_02/kernel.py",
        "/home/author/runs/",
    ],
)
def test_legacy_absolute_path_requires_one_unambiguous_runs_anchor(
    tmp_path: Path,
    legacy: str,
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    with pytest.raises(UnsafeKernelPathError, match="runs"):
        resolve_kernel_entry(
            "L1_P1",
            _entry("kernel_path", legacy),
            runs_root,
            require_exists=False,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "run/problem_001/kernel.py",
        "run/problem_001/turn_02/not_kernel.py",
        "run/problem_one/turn_02/kernel.py",
        "run/problem_001/turn_two/kernel.py",
    ],
)
def test_identity_requires_run_problem_turn_kernel_shape(
    tmp_path: Path,
    relative: str,
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    with pytest.raises(UnsafeKernelPathError):
        resolve_kernel_entry(
            "L1_P1",
            _entry("kernel_path", relative),
            runs_root,
            require_exists=False,
        )


@pytest.mark.parametrize(
    "key,problem_id,turn,message",
    [
        ("L1_P9", "1", 2, "registry key"),
        ("L1_P1", "9", 2, "problem_id"),
        ("L1_P1", "1", 9, "turn"),
    ],
)
def test_path_identity_must_match_registry_metadata(
    tmp_path: Path,
    key: str,
    problem_id: str,
    turn: int,
    message: str,
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    with pytest.raises(RegistryFormatError, match=message):
        resolve_kernel_entry(
            key,
            _entry(
                "kernel_path",
                "run/problem_001/turn_02/kernel.py",
                problem_id=problem_id,
                turn=turn,
            ),
            runs_root,
            require_exists=False,
        )


def test_missing_file_and_invalid_runs_root_fail_closed(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    entry = _entry("kernel_path", "run/problem_001/turn_02/kernel.py")

    with pytest.raises(KernelPathMissingError, match="missing"):
        resolve_kernel_entry("L1_P1", entry, runs_root)
    with pytest.raises(KernelPathMissingError, match="runs_root"):
        resolve_kernel_entry("L1_P1", entry, tmp_path / "absent")


def test_symlink_cannot_escape_runs_root(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    runs_root = tmp_path / "runs"
    outside = _kernel(
        tmp_path / "outside",
        "run/problem_001/turn_02/kernel.py",
    )
    link = runs_root / "run" / "problem_001" / "turn_02" / "kernel.py"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("test account cannot create symlinks")

    with pytest.raises(UnsafeKernelPathError, match="outside"):
        resolve_kernel_entry(
            "L1_P1",
            _entry("kernel_path", "run/problem_001/turn_02/kernel.py"),
            runs_root,
        )


def test_full_scan_migrates_without_mutating_input(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    first = _kernel(runs_root, "l1_run/problem_001/turn_02/kernel.py")
    second = _kernel(runs_root, "l2_run/problem_003/turn_04/kernel.py")
    registry = {
        "L1_P1": _entry(
            "kernel_path",
            "/author/runs/l1_run/problem_001/turn_02/kernel.py",
        ),
        "L2_P3": {
            "level": "L2",
            "problem_id": "3",
            "turn": 4,
            "kernel_path": "l2_run/problem_003/turn_04/kernel.py",
        },
    }
    before = copy.deepcopy(registry)

    result = scan_kernel_registry(registry, runs_root)

    assert registry == before
    assert result.resolved_paths == {
        "L1_P1": first.resolve(),
        "L2_P3": second.resolve(),
    }
    assert result.migrated_keys == ("L1_P1", "L2_P3")
    for entry in result.migrated_registry.values():
        assert not Path(entry["kernel_path"]).is_absolute()
        assert entry["kernel_path"] == entry["kernel_relpath"]


def test_full_scan_reports_all_invalid_and_missing_entries(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    registry = {
        "L1_P1": _entry(
            "kernel_path",
            "missing/problem_001/turn_02/kernel.py",
        ),
        "L1_P2": {
            "level": "L1",
            "problem_id": "2",
            "turn": 0,
            "kernel_path": "../escape/problem_002/turn_00/kernel.py",
        },
    }

    with pytest.raises(KernelRegistryScanError) as captured:
        scan_kernel_registry(registry, runs_root)

    assert len(captured.value.issues) == 2
    assert "L1_P1" in str(captured.value)
    assert "L1_P2" in str(captured.value)


@pytest.mark.parametrize("different_case", [False, True])
def test_scan_rejects_portable_identity_collisions(
    tmp_path: Path,
    different_case: bool,
) -> None:
    runs_root = tmp_path / "runs"
    relative = "run/problem_001/turn_02/kernel.py"
    _kernel(runs_root, relative)
    duplicate = (
        "RUN/problem_001/turn_02/kernel.py"
        if different_case
        else relative
    )
    registry = {
        "first": _entry("kernel_path", relative),
        "second": _entry("kernel_path", duplicate),
    }

    # On a case-sensitive filesystem the uppercase target need not exist; the
    # collision is still a portability error, so scan without existence checks.
    with pytest.raises(KernelRegistryScanError, match="same portable"):
        scan_kernel_registry(registry, runs_root, require_exists=False)


def test_loader_and_file_scan_reject_duplicate_json_keys(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        '{"L1_P1":{"kernel_path":"a"},"L1_P1":{"kernel_path":"b"}}',
        encoding="utf-8",
    )

    with pytest.raises(RegistryFormatError, match="duplicate"):
        load_kernel_registry(registry_path)


def test_scan_registry_file_validates_and_migrates(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _kernel(runs_root)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "L1_P1": _entry(
                    "kernel_path",
                    "/legacy/runs/paper_run/iterations/problem_001/turn_02/kernel.py",
                )
            }
        ),
        encoding="utf-8",
    )

    result = scan_kernel_registry_file(registry_path, runs_root)

    assert result.migrated_registry["L1_P1"]["kernel_path"] == (
        "paper_run/iterations/problem_001/turn_02/kernel.py"
    )


def test_repository_legacy_registry_has_a_complete_portable_migration(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    result = scan_kernel_registry_file(
        PROJECT_ROOT / "best_kernels.json",
        runs_root,
        require_exists=False,
    )

    assert result.resolutions
    assert len(result.resolutions) == len(result.migrated_registry)
    assert set(result.migrated_keys) == set(result.resolutions)
    assert all(
        resolution.rebased_legacy_absolute
        for resolution in result.resolutions.values()
    )
    assert all(
        not Path(entry["kernel_path"]).is_absolute()
        for entry in result.migrated_registry.values()
    )
