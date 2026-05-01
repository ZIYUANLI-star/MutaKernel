"""MutaKernel 核心数据模型"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


class MutantStatus(Enum):
    PENDING = "pending"
    KILLED = "killed"
    SURVIVED = "survived"
    STILLBORN = "stillborn"
    STRICT_EQUIVALENT = "strict_equivalent"
    CANDIDATE_EQUIVALENT = "candidate_equivalent"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"

    @classmethod
    def _missing_(cls, value):
        """Backward compat: old 'equivalent' maps to CANDIDATE_EQUIVALENT."""
        if value == "equivalent":
            return cls.CANDIDATE_EQUIVALENT
        return None

    @property
    def is_equivalent(self) -> bool:
        return self in (
            MutantStatus.STRICT_EQUIVALENT,
            MutantStatus.CANDIDATE_EQUIVALENT,
        )


class OperatorCategory(Enum):
    A_ARITHMETIC = "A"
    B_GPU_PARALLEL = "B"
    C_ML_SEMANTIC = "C"
    D_LLM_PATTERN = "D"


@dataclass
class MutationSite:
    """代码中一个可变异的位置"""
    line_start: int
    line_end: int
    col_start: int = 0
    col_end: int = 0
    original_code: str = ""
    node_type: str = ""

    def __repr__(self) -> str:
        snippet = self.original_code[:40] + "..." if len(self.original_code) > 40 else self.original_code
        return f"MutationSite(L{self.line_start}-{self.line_end}, '{snippet}')"


@dataclass
class Mutant:
    """一个变异体的完整描述"""
    id: str
    operator_name: str
    operator_category: str
    site: MutationSite
    original_code: str
    mutated_code: str
    description: str = ""
    status: MutantStatus = MutantStatus.PENDING
    error_message: str = ""
    kill_input_seed: Optional[int] = None
    execution_time_ms: float = 0.0
    equiv_detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_code: bool = False) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "operator_name": self.operator_name,
            "operator_category": self.operator_category,
            "description": self.description,
            "status": self.status.value,
            "error_message": self.error_message,
            "kill_input_seed": self.kill_input_seed,
            "execution_time_ms": self.execution_time_ms,
            "site": {
                "line_start": self.site.line_start,
                "line_end": self.site.line_end,
                "original_code": self.site.original_code[:200],
                "node_type": self.site.node_type,
            },
        }
        if include_code:
            if self.original_code:
                d["original_code"] = self.original_code
            if self.mutated_code:
                d["mutated_code"] = self.mutated_code
        if self.equiv_detail:
            d["equiv_detail"] = self.equiv_detail
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Mutant":
        site = MutationSite(
            line_start=d["site"]["line_start"],
            line_end=d["site"]["line_end"],
            original_code=d["site"].get("original_code", ""),
            node_type=d["site"].get("node_type", ""),
        )
        return Mutant(
            id=d["id"],
            operator_name=d["operator_name"],
            operator_category=d["operator_category"],
            site=site,
            original_code=d.get("original_code", ""),
            mutated_code=d.get("mutated_code", ""),
            description=d.get("description", ""),
            status=MutantStatus(d["status"]),
            error_message=d.get("error_message", ""),
            kill_input_seed=d.get("kill_input_seed"),
            execution_time_ms=d.get("execution_time_ms", 0.0),
            equiv_detail=d.get("equiv_detail", {}),
        )


@dataclass
class KernelInfo:
    """一个待测 kernel 的信息"""
    problem_id: int
    level: int
    problem_name: str
    source_path: str
    kernel_code: str
    reference_module_path: str
    language: str = "triton"  # "triton" or "cuda"

    def __repr__(self) -> str:
        return f"KernelInfo(L{self.level}/{self.problem_id}: {self.problem_name})"


@dataclass
class MutationTestResult:
    """单个 kernel 的变异测试结果"""
    kernel: KernelInfo
    mutants: List[Mutant] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.mutants)

    @property
    def killed(self) -> int:
        return sum(1 for m in self.mutants if m.status == MutantStatus.KILLED)

    @property
    def survived(self) -> int:
        return sum(1 for m in self.mutants if m.status == MutantStatus.SURVIVED)

    @property
    def stillborn(self) -> int:
        return sum(1 for m in self.mutants if m.status == MutantStatus.STILLBORN)

    @property
    def strict_equivalent(self) -> int:
        return sum(1 for m in self.mutants if m.status == MutantStatus.STRICT_EQUIVALENT)

    @property
    def candidate_equivalent(self) -> int:
        return sum(1 for m in self.mutants if m.status == MutantStatus.CANDIDATE_EQUIVALENT)

    @property
    def equivalent(self) -> int:
        """Total equivalents (strict + candidate) for backward compat."""
        return self.strict_equivalent + self.candidate_equivalent

    @property
    def mutation_score(self) -> float:
        """Conservative: only exclude STRICT_EQUIVALENT from denominator."""
        denom = self.total - self.stillborn - self.strict_equivalent
        if denom <= 0:
            return 0.0
        return self.killed / denom

    @property
    def mutation_score_optimistic(self) -> float:
        """Optimistic: also exclude CANDIDATE_EQUIVALENT from denominator."""
        denom = self.total - self.stillborn - self.strict_equivalent - self.candidate_equivalent
        if denom <= 0:
            return 0.0
        return self.killed / denom

    @staticmethod
    def _equiv_statuses():
        return (MutantStatus.STILLBORN, MutantStatus.STRICT_EQUIVALENT)

    def score_by_category(self) -> Dict[str, float]:
        """按变异算子类别分别计算杀灭率 (conservative)"""
        result = {}
        for cat in ["A", "B", "C", "D"]:
            cat_mutants = [m for m in self.mutants if m.operator_category == cat]
            denom = len(cat_mutants) - sum(
                1 for m in cat_mutants
                if m.status in self._equiv_statuses()
            )
            killed = sum(1 for m in cat_mutants if m.status == MutantStatus.KILLED)
            result[cat] = killed / denom if denom > 0 else 0.0
        return result

    def score_by_operator(self) -> Dict[str, float]:
        """按变异算子分别计算杀灭率 (conservative)"""
        from collections import defaultdict
        groups: Dict[str, List[Mutant]] = defaultdict(list)
        for m in self.mutants:
            groups[m.operator_name].append(m)
        result = {}
        for op_name, mutants in groups.items():
            denom = len(mutants) - sum(
                1 for m in mutants
                if m.status in self._equiv_statuses()
            )
            killed = sum(1 for m in mutants if m.status == MutantStatus.KILLED)
            result[op_name] = killed / denom if denom > 0 else 0.0
        return result

    def survived_mutants(self) -> List[Mutant]:
        return [m for m in self.mutants if m.status == MutantStatus.SURVIVED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel": {
                "problem_id": self.kernel.problem_id,
                "level": self.kernel.level,
                "problem_name": self.kernel.problem_name,
                "language": self.kernel.language,
            },
            "summary": {
                "total": self.total,
                "killed": self.killed,
                "survived": self.survived,
                "stillborn": self.stillborn,
                "strict_equivalent": self.strict_equivalent,
                "candidate_equivalent": self.candidate_equivalent,
                "equivalent": self.equivalent,
                "mutation_score": round(self.mutation_score, 4),
                "mutation_score_optimistic": round(self.mutation_score_optimistic, 4),
                "score_by_category": {
                    k: round(v, 4) for k, v in self.score_by_category().items()
                },
            },
            "mutants": [
                m.to_dict(include_code=True)
                for m in self.mutants
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: Path) -> "MutationTestResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kernel = KernelInfo(
            problem_id=data["kernel"]["problem_id"],
            level=data["kernel"]["level"],
            problem_name=data["kernel"]["problem_name"],
            source_path="",
            kernel_code="",
            reference_module_path="",
            language=data["kernel"].get("language", "triton"),
        )
        mutants = [Mutant.from_dict(m) for m in data.get("mutants", [])]
        result = MutationTestResult(kernel=kernel, mutants=mutants)
        return result


@dataclass
class RepairResult:
    """一次修复尝试的结果"""
    kernel: KernelInfo
    survived_mutant: Mutant
    baseline_mode: str
    success: bool
    rounds_used: int
    original_test_pass: bool = False
    enhanced_test_pass: bool = False
    speedup_before: float = 0.0
    speedup_after: float = 0.0
    repaired_code: str = ""
    feedback_prompt: str = ""
    error_log: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.kernel.problem_id,
            "level": self.kernel.level,
            "mutant_id": self.survived_mutant.id,
            "operator_name": self.survived_mutant.operator_name,
            "baseline_mode": self.baseline_mode,
            "success": self.success,
            "rounds_used": self.rounds_used,
            "original_test_pass": self.original_test_pass,
            "enhanced_test_pass": self.enhanced_test_pass,
            "speedup_before": self.speedup_before,
            "speedup_after": self.speedup_after,
            "error_log": self.error_log[:500],
        }
