"""MutaKernel 全局配置"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


PROJECT_ROOT = Path(__file__).parent

KERNELBENCH_ROOT = Path("/home/kbuser/projects/KernelBench-0")
KERNELBENCH_PROBLEMS = KERNELBENCH_ROOT / "KernelBench" / "KernelBench"
KERNELBENCH_RUNS = KERNELBENCH_ROOT / "runs"

DEFAULT_RUN_NAME = "iter_full_l{level}_caesar_paper_v2"

RESULTS_DIR = PROJECT_ROOT / "runs"

DEFAULT_ATOL = 1e-02
DEFAULT_RTOL = 1e-02
DEFAULT_NUM_INPUTS = 5

EQUIVALENCE_DETECTION_RUNS = 100

MAX_REPAIR_ROUNDS = 5


@dataclass
class MutationTestConfig:
    """单次变异测试运行的配置"""
    atol: float = DEFAULT_ATOL
    rtol: float = DEFAULT_RTOL
    num_test_inputs: int = DEFAULT_NUM_INPUTS
    num_equivalence_runs: int = EQUIVALENCE_DETECTION_RUNS
    timeout_seconds: float = 30.0
    device: str = "cuda"
    seed: int = 42
    categories: list = field(default_factory=lambda: ["A", "B", "C", "D"])


@dataclass
class RepairConfig:
    """MutRepair 修复配置"""
    max_rounds: int = MAX_REPAIR_ROUNDS
    model_name: str = "deepseek-chat"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    baseline_mode: str = "ours"  # B0, B1, B2, B3, ours


@dataclass
class LLMAttributionConfig:
    """Phase 2: LLM-Assisted Iterative Analysis for survived mutants."""
    model_name: str = "deepseek-reasoner"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 16384
    max_mutants: int = 0  # 0 = unlimited
    max_rounds: int = 3
    suggestions_per_round: int = 5
    verify_timeout: int = 180
    generate_robustness_code: bool = True
    generate_test_rules: bool = True


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    mutation: MutationTestConfig = field(default_factory=MutationTestConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    llm_attribution: LLMAttributionConfig = field(default_factory=LLMAttributionConfig)
    levels: list = field(default_factory=lambda: [1, 2])
    output_dir: Path = RESULTS_DIR
    kernelbench_root: Path = KERNELBENCH_ROOT
