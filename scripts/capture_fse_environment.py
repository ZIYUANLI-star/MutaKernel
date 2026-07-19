#!/usr/bin/env python3
"""Capture a non-secret, reproducible FSE experiment environment manifest.

The command never imports or executes benchmark candidates.  Optional system
commands (Git, ``nvidia-smi``, and ``nvcc``) are best effort: missing tools,
timeouts, and non-zero exits are represented in the output instead of aborting
the capture.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.manifest import canonical_json_bytes, stable_json_sha256


ENVIRONMENT_SCHEMA_VERSION = "1.0"

# These names are an explicit non-secret allowlist.  Their normalized values
# affect compilation, numerical determinism, device selection, or allocator
# behaviour and therefore must be captured rather than recording presence only.
ENV_NAME_ALLOWLIST = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_CUDA_ARCH_LIST",
)

KEY_DEPENDENCIES = (
    "torch",
    "triton",
    "numpy",
    "pandas",
    "pytest",
    "openai",
    "tree-sitter",
    "tree-sitter-cuda",
)

CommandRunner = Callable[[Sequence[str], Optional[Path], float], Mapping[str, Any]]
PackageVersion = Callable[[str], str]
ModuleImporter = Callable[[str], Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _short_error(exc: BaseException, limit: int = 500) -> str:
    return str(exc).replace("\r", " ").replace("\n", " ")[:limit]


def run_command(
    argv: Sequence[str],
    cwd: Optional[Path] = None,
    timeout_s: float = 5.0,
) -> Dict[str, Any]:
    """Run a metadata command and always return a structured result."""

    command = [str(part) for part in argv]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "command": command,
            "timeout_s": timeout_s,
            "error": _short_error(exc),
        }
    except (FileNotFoundError, OSError) as exc:
        return {
            "status": "unavailable",
            "command": command,
            "error_type": type(exc).__name__,
            "error": _short_error(exc),
        }

    result: Dict[str, Any] = {
        "status": "ok" if completed.returncode == 0 else "error",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
    }
    if completed.stderr.strip():
        result["stderr"] = completed.stderr.strip()[:1000]
    return result


def _collect_dependencies(package_version: PackageVersion) -> Dict[str, Any]:
    dependencies: Dict[str, Any] = {}
    for package in KEY_DEPENDENCIES:
        try:
            dependencies[package] = {
                "status": "installed",
                "version": package_version(package),
            }
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = {"status": "not_installed"}
        except Exception as exc:  # metadata backends can fail independently
            dependencies[package] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": _short_error(exc),
            }
    return dependencies


def _collect_torch(import_module: ModuleImporter) -> Dict[str, Any]:
    try:
        torch = import_module("torch")
    except ModuleNotFoundError:
        return {"status": "not_installed"}
    except Exception as exc:
        return {
            "status": "import_error",
            "error_type": type(exc).__name__,
            "error": _short_error(exc),
        }

    result: Dict[str, Any] = {
        "status": "available",
        "version": str(getattr(torch, "__version__", "unknown")),
        "compiled_cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "compiled_hip": getattr(getattr(torch, "version", None), "hip", None),
    }
    try:
        cuda_available = bool(torch.cuda.is_available())
        result["cuda_available"] = cuda_available
        result["device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
        devices = []
        if cuda_available:
            for index in range(result["device_count"]):
                try:
                    properties = torch.cuda.get_device_properties(index)
                    capability = torch.cuda.get_device_capability(index)
                    devices.append(
                        {
                            "index": index,
                            "name": str(properties.name),
                            "compute_capability": [int(capability[0]), int(capability[1])],
                            "total_memory_bytes": int(properties.total_memory),
                        }
                    )
                except Exception as exc:
                    devices.append(
                        {
                            "index": index,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": _short_error(exc),
                        }
                    )
        result["devices"] = devices
    except Exception as exc:
        result["cuda_probe"] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": _short_error(exc),
        }
    return result


def _collect_git(
    repo_root: Path,
    command_runner: CommandRunner,
    timeout_s: float,
) -> Dict[str, Any]:
    commit_result = dict(command_runner(["git", "rev-parse", "HEAD"], repo_root, timeout_s))
    status_result = dict(command_runner(["git", "status", "--porcelain"], repo_root, timeout_s))

    result: Dict[str, Any] = {
        "commit": None,
        "dirty": None,
        "commit_probe": {key: value for key, value in commit_result.items() if key != "stdout"},
        # Never persist `git status --porcelain` output because it contains
        # local filenames.  Only its success and derived boolean are needed.
        "dirty_probe": {key: value for key, value in status_result.items() if key != "stdout"},
    }
    if commit_result.get("status") == "ok":
        commit = str(commit_result.get("stdout", "")).strip()
        if re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
            result["commit"] = commit.lower()
        else:
            result["commit_probe"]["status"] = "parse_error"
    if status_result.get("status") == "ok":
        result["dirty"] = bool(str(status_result.get("stdout", "")).strip())
    return result


def _collect_nvidia_smi(
    command_runner: CommandRunner,
    timeout_s: float,
) -> Dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    command_result = dict(command_runner(query, None, timeout_s))
    public_result = {key: value for key, value in command_result.items() if key != "stdout"}
    if command_result.get("status") != "ok":
        return public_result

    rows = []
    try:
        for row in csv.reader(str(command_result.get("stdout", "")).splitlines()):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f"expected four columns, got {len(row)}")
            rows.append(
                {
                    "index": int(row[0].strip()),
                    "name": row[1].strip(),
                    "driver_version": row[2].strip(),
                    "memory_total_mib": int(row[3].strip()),
                }
            )
    except Exception as exc:
        public_result["status"] = "parse_error"
        public_result["error_type"] = type(exc).__name__
        public_result["error"] = _short_error(exc)
        return public_result
    public_result["gpus"] = rows
    return public_result


def _collect_nvcc(command_runner: CommandRunner, timeout_s: float) -> Dict[str, Any]:
    command_result = dict(command_runner(["nvcc", "--version"], None, timeout_s))
    public_result = {key: value for key, value in command_result.items() if key != "stdout"}
    if command_result.get("status") != "ok":
        return public_result
    output = str(command_result.get("stdout", ""))
    match = re.search(r"release\s+([^,\s]+)", output)
    public_result["release"] = match.group(1) if match else None
    public_result["version_line"] = next(
        (line.strip() for line in output.splitlines() if "release" in line.lower()),
        None,
    )
    return public_result


def capture_environment(
    *,
    repo_root: Path,
    timeout_s: float = 5.0,
    command_runner: CommandRunner = run_command,
    package_version: PackageVersion = importlib.metadata.version,
    import_module: ModuleImporter = importlib.import_module,
    environ: Optional[Mapping[str, str]] = None,
    captured_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect environment provenance without raising on optional probes."""

    env = os.environ if environ is None else environ
    payload: Dict[str, Any] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "captured_at_utc": captured_at_utc or _utc_now(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "git": _collect_git(repo_root, command_runner, timeout_s),
        "dependencies": _collect_dependencies(package_version),
        "torch": _collect_torch(import_module),
        "nvidia_smi": _collect_nvidia_smi(command_runner, timeout_s),
        "nvcc": _collect_nvcc(command_runner, timeout_s),
        "environment_variables": {
            name: {
                "is_set": name in env,
                "value": None if name not in env else str(env[name]),
            }
            for name in ENV_NAME_ALLOWLIST
        },
    }
    payload["capture_sha256"] = stable_json_sha256(payload)
    return payload


def write_new_canonical_json(path: Path, data: Mapping[str, Any]) -> None:
    """Durably create a canonical JSON file without overwriting anything."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(data) + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--command-timeout", type=float, default=5.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command_timeout <= 0:
        raise SystemExit("--command-timeout must be positive")
    manifest = capture_environment(
        repo_root=args.repo_root.resolve(),
        timeout_s=args.command_timeout,
    )
    try:
        write_new_canonical_json(args.output, manifest)
    except FileExistsError:
        print(f"refusing to overwrite existing environment manifest: {args.output}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "capture_sha256": manifest["capture_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
