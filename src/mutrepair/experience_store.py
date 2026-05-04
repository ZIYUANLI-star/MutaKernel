"""修复经验存储。

记录每次成功修复的:
- 修复前后的代码 diff
- 涉及的变异算子
- 缺陷类型
- 修复轮次

供 MutEvolve 模块挖掘模式使用。
"""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..models import KernelInfo, Mutant

logger = logging.getLogger(__name__)


@dataclass
class RepairExperience:
    """一次成功修复的经验记录。"""
    kernel_id: str
    problem_id: int
    level: int
    operator_name: str
    operator_category: str
    site_line: int
    diff_lines: List[str]
    added_lines: List[str]
    removed_lines: List[str]
    rounds: int
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class ExperienceStore:
    """修复经验存储管理器。"""

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.experiences: List[RepairExperience] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """加载已有经验记录。"""
        index_path = self.store_path / "experiences.jsonl"
        if not index_path.exists():
            return
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    exp = RepairExperience(**data)
                    self.experiences.append(exp)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Skipping malformed experience: {e}")

    def record_success(
        self,
        kernel: KernelInfo,
        mutant: Mutant,
        original_code: str,
        repaired_code: str,
        rounds: int,
    ) -> None:
        """记录一次成功修复。"""
        import datetime

        diff = list(difflib.unified_diff(
            original_code.splitlines(keepends=True),
            repaired_code.splitlines(keepends=True),
            fromfile="original",
            tofile="repaired",
            lineterm="",
        ))
        added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
        removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]

        exp = RepairExperience(
            kernel_id=f"L{kernel.level}_P{kernel.problem_id}",
            problem_id=kernel.problem_id,
            level=kernel.level,
            operator_name=mutant.operator_name,
            operator_category=mutant.operator_category,
            site_line=mutant.site.line_start,
            diff_lines=diff[:50],
            added_lines=added[:20],
            removed_lines=removed[:20],
            rounds=rounds,
            timestamp=datetime.datetime.now().isoformat(),
        )
        self.experiences.append(exp)

        index_path = self.store_path / "experiences.jsonl"
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(exp.to_dict(), ensure_ascii=False) + "\n")

        logger.info(
            f"Recorded repair experience: {exp.kernel_id} "
            f"({exp.operator_name}, {rounds} rounds)"
        )

    def get_experiences_by_operator(self, operator_name: str) -> List[RepairExperience]:
        return [e for e in self.experiences if e.operator_name == operator_name]

    def get_experiences_by_category(self, category: str) -> List[RepairExperience]:
        return [e for e in self.experiences if e.operator_category == category]

    def get_frequent_patterns(self, min_count: int = 3) -> Dict[str, int]:
        """统计高频修复模式（按添加的代码行聚合）。"""
        from collections import Counter
        pattern_counter: Counter = Counter()
        for exp in self.experiences:
            for line in exp.added_lines:
                stripped = line.strip()
                if len(stripped) > 10:
                    pattern_counter[stripped] += 1
        return {k: v for k, v in pattern_counter.items() if v >= min_count}

    def summary(self) -> Dict:
        """返回经验库摘要。"""
        from collections import Counter
        op_counts = Counter(e.operator_name for e in self.experiences)
        cat_counts = Counter(e.operator_category for e in self.experiences)
        avg_rounds = (
            sum(e.rounds for e in self.experiences) / len(self.experiences)
            if self.experiences else 0
        )
        return {
            "total_experiences": len(self.experiences),
            "by_operator": dict(op_counts),
            "by_category": dict(cat_counts),
            "avg_rounds": round(avg_rounds, 2),
            "frequent_patterns": self.get_frequent_patterns(),
        }
