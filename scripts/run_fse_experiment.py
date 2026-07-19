#!/usr/bin/env python3
"""Run an immutable FSE experiment plan with per-case process isolation.

The runner verifies every manifest and source artifact before execution, writes
one append-only observation per planned test, supports exact resume by stable
``test_id``, and treats timeout/missing worker output as ``INCONCLUSIVE``.  It
never invokes mutation generation or EMD.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKER_PATH = SCRIPT_DIR / "_candidate_worker.py"
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import (
    ArtifactProvenance,
    ObservationLog,
    RunManifest,
    SubjectProvenance,
    canonical_json_bytes,
    sha256_file,
    stable_json_sha256,
)
from src.experiments.protocol import ProtocolError, validate_frozen_plan


RUNNER_SCHEMA_VERSION = "1.0"
WORKER_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "C_INCLUDE_PATH",
        "CC",
        "CXX",
        "NVCC",
        "MAX_JOBS",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_CUDA_ARCH_LIST",
        "SYSTEMROOT",  # required by Windows process creation
        "WINDIR",
    }
)


class RunnerError(RuntimeError):
    """A run cannot proceed without violating its frozen protocol."""


LIVE_ENVIRONMENT_FIELDS = (
    "python",
    "os",
    "dependencies",
    "torch",
    "nvidia_smi",
    "nvcc",
    "environment_variables",
)


def _capture_live_environment() -> Mapping[str, Any]:
    # Import lazily so the runner remains importable in minimal unit-test
    # environments and the exact capture implementation is itself hashed.
    from scripts.capture_fse_environment import capture_environment

    return capture_environment(repo_root=PROJECT_ROOT)


def _verify_live_environment(
    stored: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    allow_dirty: bool,
) -> None:
    stored_git = stored.get("git")
    live_git = live.get("git")
    if not isinstance(stored_git, Mapping) or not isinstance(live_git, Mapping):
        raise RunnerError("stored and live environment manifests require Git state")
    for field in ("commit", "dirty"):
        if live_git.get(field) != stored_git.get(field):
            raise RunnerError(
                f"live Git {field} differs from the captured environment manifest"
            )
    if live_git.get("dirty") is True and not allow_dirty:
        raise RunnerError("refusing a primary run from a dirty Git worktree")

    for field in LIVE_ENVIRONMENT_FIELDS:
        if field not in stored:
            raise RunnerError(f"environment manifest is missing live field {field!r}")
        if live.get(field) != stored.get(field):
            raise RunnerError(
                f"live environment field {field!r} differs from the captured manifest"
            )


def _logical_implementation_path(path: Path, role: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        # Do not persist an author- or CI-specific absolute path.
        return f"external/{role}/{resolved.name}"


def _implementation_artifacts(worker_path: Path) -> list[Dict[str, Any]]:
    paths = [
        ("runner", Path(__file__)),
        ("worker", worker_path),
        ("environment_capture", SCRIPT_DIR / "capture_fse_environment.py"),
        ("policy_bank", PROJECT_ROOT / "src" / "stress" / "policy_bank.py"),
    ]
    paths.extend(
        ("validation_core", path)
        for path in sorted((PROJECT_ROOT / "src" / "validation").glob("*.py"))
    )
    paths.extend(
        ("experiment_protocol", path)
        for path in sorted((PROJECT_ROOT / "src" / "experiments").glob("*.py"))
    )
    artifacts = []
    seen = set()
    for role, path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            raise RunnerError(f"required implementation artifact is missing: {role}")
        artifacts.append(
            {
                "role": role,
                "logical_path": _logical_implementation_path(resolved, role),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return sorted(artifacts, key=lambda item: (item["role"], item["logical_path"]))


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunnerError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RunnerError(f"{label} must be a JSON object")
    return value


def _verify_embedded_digest(
    value: Mapping[str, Any],
    digest_field: str,
    label: str,
) -> None:
    supplied = value.get(digest_field)
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise RunnerError(f"{label} has no valid {digest_field}")
    payload = dict(value)
    del payload[digest_field]
    actual = stable_json_sha256(payload)
    if actual != supplied:
        raise RunnerError(
            f"{label} digest mismatch: expected {supplied}, calculated {actual}"
        )


def _load_verified_inputs(
    *,
    plan_path: Path,
    subject_manifest_path: Path,
    environment_path: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    plan = _read_json_object(plan_path, "experiment plan")
    subjects = _read_json_object(subject_manifest_path, "subject manifest")
    environment = _read_json_object(environment_path, "environment manifest")
    _verify_embedded_digest(plan, "plan_sha256", "experiment plan")
    _verify_embedded_digest(subjects, "manifest_sha256", "subject manifest")
    _verify_embedded_digest(environment, "capture_sha256", "environment manifest")

    if plan.get("schedule_sha256") != stable_json_sha256(plan.get("schedule")):
        raise RunnerError("experiment plan schedule digest mismatch")
    if plan.get("subject_manifest_sha256") != subjects.get("manifest_sha256"):
        raise RunnerError("plan references a different subject manifest")
    if plan.get("subject_manifest_file_sha256") != sha256_file(subject_manifest_path):
        raise RunnerError("subject manifest file bytes differ from the planned artifact")
    if plan.get("subject_count") != subjects.get("subject_count"):
        raise RunnerError("plan and subject manifest counts differ")
    validate_frozen_plan(plan, subjects)
    return plan, subjects, environment


def _artifact_from_dict(value: Mapping[str, Any]) -> ArtifactProvenance:
    return ArtifactProvenance(
        logical_path=str(value["logical_path"]),
        sha256=str(value["sha256"]),
        size_bytes=int(value["size_bytes"]),
        role=str(value.get("role", "input")),
    )


def _subject_from_dict(value: Mapping[str, Any]) -> SubjectProvenance:
    return SubjectProvenance(
        subject_id=str(value["subject_id"]),
        dataset=str(value["dataset"]),
        task_id=str(value["task_id"]),
        language=str(value["language"]),
        candidate=_artifact_from_dict(value["candidate"]),
        reference=_artifact_from_dict(value["reference"]),
        contract=value.get("contract", {}),
        metadata={
            **dict(value.get("metadata", {})),
            "source": dict(value.get("source", {})),
            "subject_sha256": value.get("subject_sha256"),
        },
    )


def _resolve_artifact(root: Path, artifact: ArtifactProvenance) -> Path:
    path = (root / Path(artifact.logical_path)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RunnerError(f"artifact escapes root: {artifact.logical_path}") from exc
    if not artifact.verify(path):
        raise RunnerError(
            f"artifact content mismatch for {artifact.role}: {artifact.logical_path}"
        )
    return path


def _materialize_case(
    scheduled_case: Mapping[str, Any],
    contract: Mapping[str, Any],
    device: str,
) -> Dict[str, Any]:
    case = {
        key: scheduled_case[key]
        for key in ("test_id", "policy", "seed", "mode", "scope", "parameters", "replicate")
    }
    parameters = dict(case.get("parameters", {}))
    adapters = contract.get("input_adapters", {})
    if adapters is not None and not isinstance(adapters, Mapping):
        raise RunnerError("contract input_adapters must be an object")
    adapters = adapters or {}

    if case["policy"] not in {"iid", "identity"}:
        bindings = contract.get("policy_bindings", {})
        bound = bindings.get(case["policy"]) if isinstance(bindings, Mapping) else None
        if not isinstance(bound, list) or not bound:
            raise RunnerError(
                f"contract has no argument binding for policy {case['policy']!r}"
            )
        parameters["policy_arg_indices"] = list(bound)

    if "batch_size" in parameters:
        batch = adapters.get("batch")
        if isinstance(batch, Mapping):
            parameters["batch_arg_indices"] = list(batch.get("arg_indices", ()))
            parameters["batch_dimension"] = batch.get("dimension", 0)
    if parameters.get("layout") == "noncontiguous":
        layout = adapters.get("layout")
        if isinstance(layout, Mapping):
            parameters["layout_arg_indices"] = list(layout.get("arg_indices", ()))
    if "dtype" in parameters:
        dtype_adapter = adapters.get("dtype")
        if isinstance(dtype_adapter, Mapping):
            parameters["dtype_arg_indices"] = list(
                dtype_adapter.get("arg_indices", ())
            )

    case["parameters"] = parameters
    case["device"] = device
    candidate_classes = contract.get("candidate_classes")
    if candidate_classes is not None:
        if not isinstance(candidate_classes, list) or not candidate_classes:
            raise RunnerError("contract candidate_classes must be a non-empty list")
        case["candidate_classes"] = list(candidate_classes)
    return case


def _worker_environment(
    run_directory: Path,
    *,
    cache_namespace: str = "shared",
) -> Dict[str, str]:
    # Never place an externally supplied identifier directly in a path.  A
    # content-derived namespace keeps strategy caches separate without making
    # the worker directory traversal-sensitive.
    safe_namespace = stable_json_sha256(
        {"cache_namespace": str(cache_namespace)}
    )[:16]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in WORKER_ENV_ALLOWLIST
    }
    home = run_directory / "worker_home"
    extensions = run_directory / "torch_extensions" / safe_namespace
    cuda_cache = run_directory / "cuda_cache" / safe_namespace
    for directory in (home, extensions, cuda_cache):
        directory.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "TORCH_EXTENSIONS_DIR": str(extensions),
            "CUDA_CACHE_PATH": str(cuda_cache),
            "PYTHONUNBUFFERED": "1",
            "MUTAKERNEL_ISOLATED_WORKER": "1",
        }
    )
    return environment


def _inconclusive_worker_result(
    reason: str,
    phase: str,
    *,
    parent_wall_ms: float,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "validation_status": "inconclusive",
        "reason": reason,
        "phase": phase,
        "scope": "in_contract",
        "errors": [{"phase": phase, "exception_type": "ParentProcessError", "message": reason}],
        "timings_ms": {"parent_wall_ms": parent_wall_ms},
        "candidate_runs": 0,
        "reference_runs": 0,
        "ref_ok": False,
        "candidate_ok": False,
        "killed": False,
    }


def _run_worker(
    config: Mapping[str, Any],
    *,
    worker_path: Path,
    run_directory: Path,
    timeout_s: float,
    cache_namespace: str = "shared",
) -> Tuple[Dict[str, Any], float]:
    temporary_directory = run_directory / "worker_tmp"
    temporary_directory.mkdir(parents=True, exist_ok=True)
    config_fd, config_name = tempfile.mkstemp(
        prefix="case_", suffix=".json", dir=temporary_directory
    )
    result_fd, result_name = tempfile.mkstemp(
        prefix="result_", suffix=".json", dir=temporary_directory
    )
    os.close(config_fd)
    os.close(result_fd)
    config_path = Path(config_name)
    result_path = Path(result_name)
    config_path.write_bytes(canonical_json_bytes(config) + b"\n")
    # An empty pre-created result file lets the parent distinguish no result.
    result_path.write_bytes(b"")

    configured_test_id = config.get("case", {}).get("test_id")
    if not (
        isinstance(configured_test_id, str)
        and len(configured_test_id) == 64
        and all(character in "0123456789abcdef" for character in configured_test_id)
    ):
        configured_test_id = stable_json_sha256(config)
    log_directory = run_directory / "worker_logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_fd, log_name = tempfile.mkstemp(
        prefix=f"{configured_test_id[:16]}_", suffix=".log", dir=log_directory
    )
    os.close(log_fd)
    temporary_log_path = Path(log_name)
    log_handle = temporary_log_path.open("wb")

    def complete(
        result: Dict[str, Any], parent_wall_ms: float
    ) -> Tuple[Dict[str, Any], float]:
        """Close and freeze process evidence before returning an observation."""

        if not log_handle.closed:
            log_handle.flush()
            os.fsync(log_handle.fileno())
            log_handle.close()
        final_log_path = log_directory / f"{configured_test_id}.log"
        if final_log_path.exists():
            if final_log_path.read_bytes() != temporary_log_path.read_bytes():
                raise RunnerError(
                    f"immutable worker log collision for test {configured_test_id}"
                )
            temporary_log_path.unlink()
        else:
            temporary_log_path.replace(final_log_path)

        evidence: Dict[str, Any] = {
            "worker_log": {
                "logical_path": final_log_path.relative_to(run_directory).as_posix(),
                "sha256": sha256_file(final_log_path),
                "size_bytes": final_log_path.stat().st_size,
            }
        }
        if result.get("validation_status") != "pass":
            bundle_directory = run_directory / "evidence" / configured_test_id
            bundle_directory.mkdir(parents=True, exist_ok=True)
            replay_config = dict(config)
            reference_logical = replay_config.pop("reference_logical_path", None)
            candidate_logical = replay_config.pop("candidate_logical_path", None)
            replay_config.pop("artifact_root", None)
            if isinstance(reference_logical, str):
                replay_config["reference_path"] = reference_logical
            if isinstance(candidate_logical, str):
                replay_config["candidate_path"] = candidate_logical
            bundle_files = {
                "case_config.json": canonical_json_bytes(replay_config) + b"\n",
                "worker_result.json": canonical_json_bytes(result) + b"\n",
                "worker.log": final_log_path.read_bytes(),
            }
            file_records = []
            for filename, payload in sorted(bundle_files.items()):
                destination = bundle_directory / filename
                if destination.exists() and destination.read_bytes() != payload:
                    raise RunnerError(
                        f"immutable evidence collision for {configured_test_id}/{filename}"
                    )
                if not destination.exists():
                    destination.write_bytes(payload)
                file_records.append(
                    {
                        "logical_path": filename,
                        "sha256": sha256_file(destination),
                        "size_bytes": destination.stat().st_size,
                    }
                )
            bundle_manifest = {
                "schema_version": "1.0",
                "bundle_id": configured_test_id,
                "validation_status": result.get("validation_status"),
                "provenance": dict(config.get("provenance", {})),
                "files": file_records,
            }
            artifact_root_value = config.get("artifact_root")
            if isinstance(artifact_root_value, str) and artifact_root_value:
                replay_root = Path(artifact_root_value).resolve()
                try:
                    config_logical = (
                        bundle_directory / "case_config.json"
                    ).resolve().relative_to(replay_root).as_posix()
                    result_logical = (
                        bundle_directory / "replay_result.json"
                    ).resolve().relative_to(replay_root).as_posix()
                except ValueError as exc:
                    raise RunnerError(
                        "replay bundle must be contained by artifact root"
                    ) from exc
                try:
                    worker_command = worker_path.resolve().relative_to(
                        replay_root
                    ).as_posix()
                    portable = True
                except ValueError:
                    # Development/test runners may supply a worker outside the
                    # frozen artifact root. Record that honestly; formal runs
                    # use the repository root and therefore remain portable.
                    worker_command = str(worker_path.resolve())
                    portable = False
                bundle_manifest["replay"] = {
                    "working_directory": "frozen artifact root",
                    "command": [
                        "python3",
                        worker_command,
                        config_logical,
                        result_logical,
                    ],
                    "requires_frozen_sources_and_environment": True,
                    "portable": portable,
                }
            else:
                # Direct unit-level _run_worker callers may not have a frozen
                # artifact root. Their bundle remains diagnostic rather than a
                # portable paper artifact.
                bundle_manifest["replay"] = {
                    "working_directory": str(run_directory.resolve()),
                    "command": [
                        sys.executable,
                        str(worker_path.resolve()),
                        str((bundle_directory / "case_config.json").resolve()),
                        str((bundle_directory / "replay_result.json").resolve()),
                    ],
                    "requires_frozen_sources_and_environment": True,
                    "portable": False,
                }
            bundle_manifest["bundle_sha256"] = stable_json_sha256(bundle_manifest)
            manifest_path = bundle_directory / "manifest.json"
            manifest_payload = canonical_json_bytes(bundle_manifest) + b"\n"
            if manifest_path.exists() and manifest_path.read_bytes() != manifest_payload:
                raise RunnerError(
                    f"immutable evidence manifest collision for {configured_test_id}"
                )
            if not manifest_path.exists():
                manifest_path.write_bytes(manifest_payload)
            evidence["replay_bundle"] = {
                "logical_path": bundle_directory.relative_to(run_directory).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "bundle_sha256": bundle_manifest["bundle_sha256"],
            }
        result["evidence"] = evidence
        return result, parent_wall_ms

    started = time.perf_counter_ns()
    timed_out = False
    return_code = None
    try:
        try:
            process = subprocess.Popen(
                [sys.executable, str(worker_path), str(config_path), str(result_path)],
                cwd=str(PROJECT_ROOT),
                env=_worker_environment(
                    run_directory,
                    cache_namespace=cache_namespace,
                ),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            parent_wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            return complete(
                _inconclusive_worker_result(
                    f"worker process could not start: {type(exc).__name__}: {str(exc)[:300]}",
                    "worker_start",
                    parent_wall_ms=parent_wall_ms,
                ),
                parent_wall_ms,
            )
        try:
            return_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:  # pragma: no cover - GPU runs use Linux; keep local CLI portable.
                process.kill()
            process.wait()
        parent_wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0

        if timed_out:
            return complete(
                _inconclusive_worker_result(
                    f"worker exceeded parent timeout of {timeout_s:g} seconds",
                    "parent_timeout",
                    parent_wall_ms=parent_wall_ms,
                ),
                parent_wall_ms,
            )
        if return_code != 0 or result_path.stat().st_size == 0:
            return complete(
                _inconclusive_worker_result(
                    f"worker exited with code {return_code} without a valid result",
                    "worker_exit",
                    parent_wall_ms=parent_wall_ms,
                ),
                parent_wall_ms,
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return complete(
                _inconclusive_worker_result(
                    f"worker result is unreadable: {type(exc).__name__}: {str(exc)[:300]}",
                    "result_decode",
                    parent_wall_ms=parent_wall_ms,
                ),
                parent_wall_ms,
            )
        if not isinstance(result, dict) or result.get("validation_status") not in {
            "pass",
            "fail",
            "inconclusive",
        }:
            return complete(
                _inconclusive_worker_result(
                    "worker result does not implement the three-valued schema",
                    "result_schema",
                    parent_wall_ms=parent_wall_ms,
                ),
                parent_wall_ms,
            )
        return complete(dict(result), parent_wall_ms)
    finally:
        if not log_handle.closed:
            log_handle.close()
        try:
            temporary_log_path.unlink()
        except FileNotFoundError:
            pass
        for path in (config_path, result_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _create_or_verify_run_manifest(
    *,
    path: Path,
    run_id: str,
    plan: Mapping[str, Any],
    subjects: Sequence[SubjectProvenance],
    environment: Mapping[str, Any],
    command: Sequence[str],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    git = environment.get("git", {})
    commit = git.get("commit") if isinstance(git, Mapping) else None
    dirty = git.get("dirty") if isinstance(git, Mapping) else None
    if not isinstance(commit, str) or not commit:
        raise RunnerError("environment manifest has no Git commit")
    if not isinstance(dirty, bool):
        raise RunnerError("environment manifest has no Git dirty-state boolean")

    if path.exists():
        existing = _read_json_object(path, "run manifest")
        RunManifest.verify_dict(existing)
        existing_config = existing.get("config", {})
        expected_subjects = [
            subject.to_dict() for subject in sorted(subjects, key=lambda item: item.subject_id)
        ]
        if (
            existing.get("run_id") != run_id
            or existing.get("experiment") != "fse-validator-comparison"
            or existing.get("git", {}).get("commit") != commit
            or existing.get("git", {}).get("dirty") != dirty
            or existing_config != config
            or existing.get("environment") != environment
            or existing.get("subjects") != expected_subjects
        ):
            raise RunnerError("existing run manifest does not match this invocation")
        return existing

    manifest = RunManifest(
        run_id=run_id,
        experiment="fse-validator-comparison",
        git_commit=commit,
        git_dirty=dirty,
        command=tuple(command),
        config=dict(config),
        subjects=tuple(subjects),
        environment=environment,
    )
    manifest.write_once(path)
    return manifest.to_dict()


def run_experiment(
    *,
    plan_path: Path,
    subject_manifest_path: Path,
    environment_path: Path,
    artifact_root: Path,
    output_directory: Path,
    run_id: str,
    device: str,
    timeout_s: float,
    max_wall_ms_per_subject_strategy: Optional[float] = None,
    selected_subjects: Optional[Iterable[str]] = None,
    selected_strategies: Optional[Iterable[str]] = None,
    max_tests: Optional[int] = None,
    early_stop_on_fail: bool = False,
    allow_dirty: bool = False,
    worker_path: Path = WORKER_PATH,
    command: Sequence[str] = ("run_fse_experiment.py",),
    runtime_environment_probe: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> Dict[str, int]:
    if not run_id:
        raise RunnerError("run_id must not be empty")
    if timeout_s <= 0:
        raise RunnerError("timeout_s must be positive")
    if max_wall_ms_per_subject_strategy is not None and max_wall_ms_per_subject_strategy <= 0:
        raise RunnerError("wall budget must be positive")
    if max_tests is not None and max_tests <= 0:
        raise RunnerError("max_tests must be positive")

    plan, subject_manifest, environment = _load_verified_inputs(
        plan_path=plan_path.resolve(),
        subject_manifest_path=subject_manifest_path.resolve(),
        environment_path=environment_path.resolve(),
    )
    live_environment = (
        _capture_live_environment()
        if runtime_environment_probe is None
        else runtime_environment_probe()
    )
    if not isinstance(live_environment, Mapping):
        raise RunnerError("runtime environment probe did not return an object")
    _verify_live_environment(environment, live_environment, allow_dirty=allow_dirty)

    artifact_root = artifact_root.resolve()
    if not artifact_root.is_dir():
        raise RunnerError(f"artifact root is not a directory: {artifact_root}")
    output_directory = output_directory.resolve()
    try:
        output_directory.relative_to(artifact_root)
    except ValueError as exc:
        raise RunnerError(
            "output directory must be contained by artifact root so evidence "
            "replay paths remain portable"
        ) from exc
    raw_subjects = subject_manifest.get("subjects", [])
    subjects = [_subject_from_dict(value) for value in raw_subjects]
    subject_by_id = {subject.subject_id: subject for subject in subjects}
    raw_by_id = {str(value["subject_id"]): value for value in raw_subjects}
    artifact_paths: Dict[str, Tuple[Path, Path]] = {}
    for subject in subjects:
        artifact_paths[subject.subject_id] = (
            _resolve_artifact(artifact_root, subject.reference),
            _resolve_artifact(artifact_root, subject.candidate),
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    selected_subject_set = None if selected_subjects is None else set(selected_subjects)
    selected_strategy_set = None if selected_strategies is None else set(selected_strategies)
    runner_config = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "device": device,
        "timeout_s": timeout_s,
        "compilation_cache_scope": "per_strategy_within_run",
        "max_wall_ms_per_subject_strategy": max_wall_ms_per_subject_strategy,
        "selected_subjects": None if selected_subject_set is None else sorted(selected_subject_set),
        "selected_strategies": None if selected_strategy_set is None else sorted(selected_strategy_set),
        "early_stop_on_fail": early_stop_on_fail,
    }
    manifest_config = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": sha256_file(plan_path),
        "subject_manifest_sha256": subject_manifest["manifest_sha256"],
        "subject_manifest_file_sha256": sha256_file(subject_manifest_path),
        "environment_capture_sha256": environment["capture_sha256"],
        "environment_file_sha256": sha256_file(environment_path),
        "implementation_artifacts": _implementation_artifacts(worker_path),
        "runner": runner_config,
    }
    run_manifest = _create_or_verify_run_manifest(
        path=output_directory / "run_manifest.json",
        run_id=run_id,
        plan=plan,
        subjects=subjects,
        environment=environment,
        command=command,
        config=manifest_config,
    )
    observations = ObservationLog(output_directory / "observations.jsonl")

    wall_spent: Dict[Tuple[str, str], float] = defaultdict(float)
    detected_pairs = set()
    for record in observations.records():
        key = (str(record.get("subject_id")), str(record.get("strategy_id")))
        wall_spent[key] += float(record.get("parent_wall_ms", 0.0))
        if record.get("validation_status") == "fail":
            detected_pairs.add(key)

    summary = {
        "executed": 0,
        "resumed": 0,
        "skipped_by_filter": 0,
        "skipped_after_fail": 0,
        "budget_exhausted": 0,
    }
    for scheduled in plan["schedule"]:
        subject_id = str(scheduled["subject_id"])
        strategy_name = str(scheduled["strategy_name"])
        strategy_id = str(scheduled["strategy_id"])
        test_id = str(scheduled["test_id"])
        if selected_subject_set is not None and subject_id not in selected_subject_set:
            summary["skipped_by_filter"] += 1
            continue
        if selected_strategy_set is not None and strategy_name not in selected_strategy_set:
            summary["skipped_by_filter"] += 1
            continue
        if test_id in observations:
            summary["resumed"] += 1
            continue
        if max_tests is not None and summary["executed"] >= max_tests:
            break

        subject = subject_by_id.get(subject_id)
        if subject is None:
            raise RunnerError(f"schedule references unknown subject: {subject_id}")
        key = (subject_id, strategy_id)
        if early_stop_on_fail and key in detected_pairs:
            summary["skipped_after_fail"] += 1
            continue
        if (
            max_wall_ms_per_subject_strategy is not None
            and wall_spent[key] >= max_wall_ms_per_subject_strategy
        ):
            worker_result = _inconclusive_worker_result(
                "equal-wall-time budget was exhausted before this case",
                "wall_budget",
                parent_wall_ms=0.0,
            )
            parent_wall_ms = 0.0
            summary["budget_exhausted"] += 1
        else:
            raw_subject = raw_by_id[subject_id]
            case = _materialize_case(scheduled, raw_subject.get("contract", {}), device)
            reference_path, candidate_path = artifact_paths[subject_id]
            oracle_contract = raw_subject.get("contract", {}).get("oracle", {})
            if not isinstance(oracle_contract, Mapping):
                raise RunnerError(f"subject {subject_id} oracle contract must be an object")
            worker_config = {
                "subject_id": subject_id,
                "reference_path": str(reference_path),
                "candidate_path": str(candidate_path),
                "reference_logical_path": subject.reference.logical_path,
                "candidate_logical_path": subject.candidate.logical_path,
                "artifact_root": str(artifact_root),
                "case": case,
                "device": device,
                "oracle": dict(oracle_contract),
                "contract": dict(raw_subject.get("contract", {})),
                "provenance": {
                    "run_manifest_sha256": run_manifest["manifest_sha256"],
                    "subject_sha256": raw_subject["subject_sha256"],
                    "contract_sha256": raw_subject["contract_sha256"],
                    "environment_capture_sha256": environment["capture_sha256"],
                    "candidate_sha256": subject.candidate.sha256,
                    "reference_sha256": subject.reference.sha256,
                },
            }
            worker_result, parent_wall_ms = _run_worker(
                worker_config,
                worker_path=worker_path,
                run_directory=output_directory,
                timeout_s=timeout_s,
                cache_namespace=strategy_id,
            )
            wall_spent[key] += parent_wall_ms

        worker_result["scope"] = scheduled["scope"]

        observation = {
            **dict(scheduled),
            "run_id": run_id,
            "run_manifest_sha256": run_manifest["manifest_sha256"],
            "subject_sha256": raw_by_id[subject_id]["subject_sha256"],
            "contract_sha256": raw_by_id[subject_id]["contract_sha256"],
            "validation_status": worker_result["validation_status"],
            "scope": scheduled["scope"],
            "parent_wall_ms": parent_wall_ms,
            "planned_candidate_runs": int(scheduled.get("candidate_run_cost", 1)),
            "observed_candidate_runs": int(worker_result.get("candidate_runs", 0)),
            "worker_result": worker_result,
        }
        observations.append(observation)
        if observation["validation_status"] == "fail":
            detected_pairs.add(key)
        summary["executed"] += 1
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--subjects", required=True, type=Path, help="Subject manifest")
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--wall-budget-ms", type=float)
    parser.add_argument("--subject", action="append", dest="selected_subjects")
    parser.add_argument("--strategy", action="append", dest="selected_strategies")
    parser.add_argument("--max-tests", type=int)
    parser.add_argument(
        "--early-stop-on-fail",
        action="store_true",
        help="Stop later cases for a subject/strategy after its first alarm",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_experiment(
            plan_path=args.plan,
            subject_manifest_path=args.subjects,
            environment_path=args.environment,
            artifact_root=args.artifact_root,
            output_directory=args.output_dir,
            run_id=args.run_id,
            device=args.device,
            timeout_s=args.timeout,
            max_wall_ms_per_subject_strategy=args.wall_budget_ms,
            selected_subjects=args.selected_subjects,
            selected_strategies=args.selected_strategies,
            max_tests=args.max_tests,
            early_stop_on_fail=args.early_stop_on_fail,
            allow_dirty=args.allow_dirty,
            command=tuple(sys.argv),
        )
    except (OSError, RunnerError, ProtocolError, TypeError, ValueError) as exc:
        print(f"FSE experiment failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
