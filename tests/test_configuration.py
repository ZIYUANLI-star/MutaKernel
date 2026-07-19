import os
from pathlib import Path

import config
from src.bridge.eval_bridge import KernelBenchBridge


def test_default_kernelbench_root_is_repository_local():
    configured = os.environ.get("MUTAKERNEL_KERNELBENCH_ROOT")
    expected = (
        Path(configured).expanduser().resolve()
        if configured
        else (config.PROJECT_ROOT / "KernelBench").resolve()
    )
    assert config.KERNELBENCH_ROOT == expected
    assert config.KERNELBENCH_PROBLEMS == expected / "KernelBench"


def test_env_path_expands_and_resolves(monkeypatch, tmp_path):
    target = tmp_path / "data"
    monkeypatch.setenv("MUTAKERNEL_TEST_PATH", str(target))
    assert config._env_path("MUTAKERNEL_TEST_PATH", Path("unused")) == target.resolve()


def test_bridge_can_separate_problem_and_run_roots(tmp_path):
    benchmark_root = tmp_path / "KernelBench-source"
    runs_root = tmp_path / "experiment-runs"
    bridge = KernelBenchBridge(benchmark_root, runs_root=runs_root)

    assert bridge.get_problem_dir(1) == benchmark_root / "KernelBench" / "level1"
    assert bridge.get_run_dir(1) == runs_root / "iter_full_l1_caesar_paper_v2"
