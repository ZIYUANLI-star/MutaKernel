import importlib.metadata
import json
from pathlib import Path

import pytest

from scripts.capture_fse_environment import (
    ENV_NAME_ALLOWLIST,
    capture_environment,
    main,
    run_command,
    write_new_canonical_json,
)
from src.experiments.manifest import stable_json_sha256


def test_missing_command_is_recorded_not_raised():
    result = run_command(["definitely-not-a-real-mutakernel-command"], timeout_s=0.1)
    assert result["status"] == "unavailable"
    assert result["error_type"] == "FileNotFoundError"


def test_cpu_only_capture_records_nonsecret_behavioral_environment_values(tmp_path: Path):
    def fake_runner(argv, cwd, timeout_s):
        if argv[:2] == ["git", "rev-parse"]:
            return {"status": "ok", "returncode": 0, "stdout": "a" * 40}
        if argv[:2] == ["git", "status"]:
            return {"status": "ok", "returncode": 0, "stdout": " M secret-name.txt"}
        return {"status": "unavailable", "command": list(argv), "error": "missing"}

    def fake_version(package):
        if package == "torch":
            return "2.0-test"
        raise importlib.metadata.PackageNotFoundError(package)

    def no_torch(_name):
        raise ModuleNotFoundError("torch")

    environment = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "DEEPSEEK_API_KEY": "must-not-appear",
        "AWS_SECRET_ACCESS_KEY": "must-not-appear",
    }
    captured = capture_environment(
        repo_root=tmp_path,
        command_runner=fake_runner,
        package_version=fake_version,
        import_module=no_torch,
        environ=environment,
        captured_at_utc="2026-07-19T00:00:00.000000Z",
    )

    assert captured["torch"] == {"status": "not_installed"}
    assert captured["git"]["commit"] == "a" * 40
    assert captured["git"]["dirty"] is True
    assert "secret-name.txt" not in json.dumps(captured)
    assert set(captured["environment_variables"]) == set(ENV_NAME_ALLOWLIST)
    assert captured["environment_variables"]["CUDA_VISIBLE_DEVICES"] == {
        "is_set": True,
        "value": "0,1",
    }
    serialized = json.dumps(captured)
    assert "0,1" in serialized
    assert "must-not-appear" not in serialized

    digest = captured["capture_sha256"]
    payload = dict(captured)
    del payload["capture_sha256"]
    assert digest == stable_json_sha256(payload)


def test_environment_output_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "environment.json"
    write_new_canonical_json(path, {"capture_sha256": "x"})
    with pytest.raises(FileExistsError):
        write_new_canonical_json(path, {"capture_sha256": "y"})


def test_environment_cli_runs_without_gpu_and_refuses_second_write(tmp_path: Path):
    output = tmp_path / "captured.json"
    assert main(["--output", str(output), "--repo-root", str(tmp_path), "--command-timeout", "0.1"]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "capture_sha256" in data
    assert main(["--output", str(output), "--repo-root", str(tmp_path), "--command-timeout", "0.1"]) == 2
