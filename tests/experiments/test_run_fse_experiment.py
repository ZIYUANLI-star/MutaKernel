"""CPU integration tests for the manifest-driven FSE runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from src.experiments import canonical_json_bytes, stable_json_sha256
from src.experiments.protocol import plan_from_files, write_plan_once
from tests.experiments.contract_fixture import rich_contract


PROJECT_ROOT = Path(__file__).parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_fse_experiment.py"
SUBJECT_BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_fse_subject_manifest.py"
STRATEGY_MATRIX = PROJECT_ROOT / "configs" / "fse_strategy_matrix.json"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _prepare_inputs(
    tmp_path: Path,
    *,
    candidate_expression: str = "value * 2",
) -> tuple[Path, Path, Path]:
    reference = tmp_path / "reference.py"
    candidate = tmp_path / "candidate.py"
    reference.write_text(
        """
import torch
from torch import nn
class Model(nn.Module):
    def forward(self, value):
        return value * 2
def get_inputs():
    return [torch.arange(6, dtype=torch.float32).reshape(2, 3)]
def get_init_inputs():
    return []
""",
        encoding="utf-8",
    )
    candidate.write_text(
        f"""
from torch import nn
class ModelNew(nn.Module):
    def forward(self, value):
        return {candidate_expression}
""",
        encoding="utf-8",
    )
    spec_path = tmp_path / "subjects.jsonl"
    spec_path.write_text(
        json.dumps(
            {
                "subject_id": "cpu-subject",
                "dataset": "unit",
                "task_id": "1",
                "language": "python",
                "candidate_path": candidate.name,
                "reference_path": reference.name,
                "contract": rich_contract(),
                "source": {"kind": "unit"},
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    builder = _load(SUBJECT_BUILDER_PATH, "subject_builder_for_runner_test")
    subject_manifest = builder.build_subject_manifest(input_path=spec_path, root=tmp_path)
    subject_manifest_path = tmp_path / "subject_manifest.json"
    _write_json(subject_manifest_path, subject_manifest)

    plan = plan_from_files(subject_manifest_path, STRATEGY_MATRIX)
    plan_path = tmp_path / "plan.json"
    write_plan_once(plan_path, plan)

    environment = {
        "schema_version": "1.0",
        "git": {"commit": "a" * 40, "dirty": False},
        "python": {"version": "test"},
        "os": {"system": "test"},
        "dependencies": {},
        "torch": {"status": "test"},
        "nvidia_smi": {"status": "test"},
        "nvcc": {"status": "test"},
        "environment_variables": {},
        "test_environment": True,
    }
    environment["capture_sha256"] = stable_json_sha256(environment)
    environment_path = tmp_path / "environment.json"
    _write_json(environment_path, environment)
    return subject_manifest_path, plan_path, environment_path


def test_runner_executes_and_persists_one_direct_case(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "fse_runner_test")
    subjects, plan, environment = _prepare_inputs(tmp_path)
    output = tmp_path / "run"

    summary = runner.run_experiment(
        plan_path=plan,
        subject_manifest_path=subjects,
        environment_path=environment,
        artifact_root=tmp_path,
        output_directory=output,
        run_id="cpu-smoke",
        device="cpu",
        timeout_s=20,
        selected_strategies=["five-iid-historical-anchor"],
        max_tests=1,
        worker_path=PROJECT_ROOT / "scripts" / "_candidate_worker.py",
        runtime_environment_probe=lambda: json.loads(
            environment.read_text(encoding="utf-8")
        ),
    )

    assert summary["executed"] == 1
    observation = json.loads((output / "observations.jsonl").read_text(encoding="utf-8"))
    assert observation["validation_status"] == "pass"
    assert observation["observed_candidate_runs"] == 1
    assert observation["worker_result"]["phase"] == "validation"
    assert (output / "run_manifest.json").is_file()


def test_runner_resume_never_rewrites_an_existing_observation(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "fse_runner_resume_test")
    subjects, plan, environment = _prepare_inputs(tmp_path)
    output = tmp_path / "run"
    arguments = dict(
        plan_path=plan,
        subject_manifest_path=subjects,
        environment_path=environment,
        artifact_root=tmp_path,
        output_directory=output,
        run_id="cpu-resume",
        device="cpu",
        timeout_s=20,
        selected_strategies=["five-iid-historical-anchor"],
        max_tests=1,
        worker_path=PROJECT_ROOT / "scripts" / "_candidate_worker.py",
        runtime_environment_probe=lambda: json.loads(
            environment.read_text(encoding="utf-8")
        ),
    )

    runner.run_experiment(**arguments)
    before = (output / "observations.jsonl").read_bytes()
    summary = runner.run_experiment(**arguments)
    after = (output / "observations.jsonl").read_bytes()

    assert summary["resumed"] >= 1
    assert summary["executed"] == 1  # the next stable planned case
    assert after.startswith(before)
    assert len(after.splitlines()) == 2


def test_online_mode_stops_a_subject_strategy_after_first_alarm(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "fse_runner_early_stop_test")
    subjects, plan, environment = _prepare_inputs(
        tmp_path,
        candidate_expression="value + 1",
    )
    output = tmp_path / "run"

    summary = runner.run_experiment(
        plan_path=plan,
        subject_manifest_path=subjects,
        environment_path=environment,
        artifact_root=tmp_path,
        output_directory=output,
        run_id="cpu-early-stop",
        device="cpu",
        timeout_s=20,
        selected_strategies=["five-iid-historical-anchor"],
        early_stop_on_fail=True,
        worker_path=PROJECT_ROOT / "scripts" / "_candidate_worker.py",
        runtime_environment_probe=lambda: json.loads(
            environment.read_text(encoding="utf-8")
        ),
    )

    assert summary["executed"] == 1
    assert summary["skipped_after_fail"] == 4
    observation = json.loads((output / "observations.jsonl").read_text(encoding="utf-8"))
    assert observation["validation_status"] == "fail"


def test_parent_timeout_is_inconclusive(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "fse_runner_timeout_test")
    sleepy = tmp_path / "sleepy_worker.py"
    sleepy.write_text("import time; time.sleep(5)\n", encoding="utf-8")

    result, wall_ms = runner._run_worker(
        {
            "artifact_root": str(tmp_path),
            "case": {"test_id": "d" * 64},
            "safe": True,
        },
        worker_path=sleepy,
        run_directory=tmp_path / "run",
        timeout_s=0.05,
    )

    assert result["validation_status"] == "inconclusive"
    assert result["phase"] == "parent_timeout"
    assert result["killed"] is False
    assert wall_ms >= 0
    bundle = result["evidence"]["replay_bundle"]
    bundle_root = tmp_path / "run" / bundle["logical_path"]
    assert (bundle_root / "case_config.json").is_file()
    assert (bundle_root / "worker_result.json").is_file()
    assert (bundle_root / "worker.log").is_file()
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_status"] == "inconclusive"
    assert manifest["replay"]["command"][0] == "python3"
    assert manifest["replay"]["command"][2] == (
        f"run/evidence/{'d' * 64}/case_config.json"
    )


def test_worker_stdout_is_hashed_and_failure_gets_replay_bundle(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "fse_runner_evidence_test")
    worker = tmp_path / "failing_worker.py"
    worker.write_text(
        """
import json
import sys
print("compiler diagnostic")
result = {
    "schema_version": "1.0",
    "validation_status": "fail",
    "reason": "known discrepancy",
    "phase": "validation",
    "candidate_runs": 1,
    "reference_runs": 1,
}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(result))
""",
        encoding="utf-8",
    )
    test_id = "c" * 64

    result, _ = runner._run_worker(
        {
            "artifact_root": str(tmp_path),
            "case": {"test_id": test_id},
            "safe": True,
        },
        worker_path=worker,
        run_directory=tmp_path / "run",
        timeout_s=5,
    )

    assert result["validation_status"] == "fail"
    log_record = result["evidence"]["worker_log"]
    log_path = tmp_path / "run" / log_record["logical_path"]
    assert log_path.read_text(encoding="utf-8").strip() == "compiler diagnostic"
    assert len(log_record["sha256"]) == 64
    bundle_path = tmp_path / "run" / result["evidence"]["replay_bundle"]["logical_path"]
    assert json.loads((bundle_path / "worker_result.json").read_text(encoding="utf-8"))[
        "reason"
    ] == "known discrepancy"


def test_worker_environment_drops_secret_variables(monkeypatch: Any, tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "fse_runner_environment_test")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-boundary")
    monkeypatch.setenv("MUTAKERNEL_SSH_PASSWORD", "must-not-cross-boundary")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/import/path")

    environment = runner._worker_environment(tmp_path / "run")

    assert "OPENAI_API_KEY" not in environment
    assert "MUTAKERNEL_SSH_PASSWORD" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["MUTAKERNEL_ISOLATED_WORKER"] == "1"


def test_worker_compilation_caches_are_isolated_by_strategy(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "fse_runner_cache_test")

    first = runner._worker_environment(tmp_path / "run", cache_namespace="strategy-a")
    first_again = runner._worker_environment(
        tmp_path / "run", cache_namespace="strategy-a"
    )
    second = runner._worker_environment(tmp_path / "run", cache_namespace="strategy-b")

    assert first["TORCH_EXTENSIONS_DIR"] == first_again["TORCH_EXTENSIONS_DIR"]
    assert first["CUDA_CACHE_PATH"] == first_again["CUDA_CACHE_PATH"]
    assert first["TORCH_EXTENSIONS_DIR"] != second["TORCH_EXTENSIONS_DIR"]
    assert first["CUDA_CACHE_PATH"] != second["CUDA_CACHE_PATH"]


def test_contract_adapters_are_materialized_per_subject() -> None:
    runner = _load(RUNNER_PATH, "fse_runner_adapter_test")
    scheduled = {
        "test_id": "b" * 64,
        "policy": "iid",
        "seed": 1,
        "mode": "config",
        "scope": "extended_contract",
        "parameters": {"batch_size": 4, "layout": "noncontiguous"},
        "replicate": 0,
    }
    contract = {
        "input_adapters": {
            "batch": {"arg_indices": [0, 2], "dimension": 1},
            "layout": {"arg_indices": [2]},
        }
    }

    case = runner._materialize_case(scheduled, contract, "cpu")

    assert case["parameters"]["batch_arg_indices"] == [0, 2]
    assert case["parameters"]["batch_dimension"] == 1
    assert case["parameters"]["layout_arg_indices"] == [2]


def test_live_environment_must_match_the_captured_manifest() -> None:
    runner = _load(RUNNER_PATH, "fse_runner_live_environment_test")
    stored = {
        "git": {"commit": "a" * 40, "dirty": False},
        **{field: {"value": field} for field in runner.LIVE_ENVIRONMENT_FIELDS},
    }
    live = json.loads(json.dumps(stored))
    live["git"]["commit"] = "b" * 40

    with pytest.raises(runner.RunnerError, match="live Git commit differs"):
        runner._verify_live_environment(stored, live, allow_dirty=False)


def test_resume_rejects_changed_worker_implementation(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "fse_runner_worker_hash_test")
    subjects, plan, environment = _prepare_inputs(tmp_path)
    output = tmp_path / "run"
    worker = tmp_path / "candidate_worker.py"
    worker.write_bytes((PROJECT_ROOT / "scripts" / "_candidate_worker.py").read_bytes())
    arguments = dict(
        plan_path=plan,
        subject_manifest_path=subjects,
        environment_path=environment,
        artifact_root=tmp_path,
        output_directory=output,
        run_id="worker-hash",
        device="cpu",
        timeout_s=20,
        selected_strategies=["five-iid-historical-anchor"],
        max_tests=1,
        worker_path=worker,
        runtime_environment_probe=lambda: json.loads(
            environment.read_text(encoding="utf-8")
        ),
    )

    runner.run_experiment(**arguments)
    worker.write_text(worker.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    with pytest.raises(runner.RunnerError, match="manifest does not match"):
        runner.run_experiment(**arguments)
