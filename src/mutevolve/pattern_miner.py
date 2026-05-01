"""修复模式挖掘器。

从 ExperienceStore 中提取高频修复模式，
识别当前变异算子集未覆盖的新缺陷类型。
"""

from __future__ import annotations

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from ..mutrepair.experience_store import ExperienceStore, RepairExperience

logger = logging.getLogger(__name__)


@dataclass
class MinedPattern:
    """一个被挖掘出的修复模式。"""
    pattern_id: str
    pattern_type: str
    description: str
    added_code_template: str
    removed_code_template: str
    frequency: int
    source_operators: List[str]
    source_experiences: List[str]


_KNOWN_PATTERNS = {
    "max_subtract",
    "eps_add",
    "float32_cast",
    "contiguous_call",
    "clamp_range",
}


class PatternMiner:
    """从修复经验中挖掘新的缺陷模式。"""

    def __init__(self, experience_store: ExperienceStore, min_frequency: int = 3):
        self.store = experience_store
        self.min_frequency = min_frequency

    def mine_patterns(self) -> List[MinedPattern]:
        """执行模式挖掘，返回新发现的模式。"""
        raw_patterns = self._extract_raw_patterns()
        filtered = self._filter_known(raw_patterns)
        merged = self._merge_similar(filtered)
        return merged

    def _extract_raw_patterns(self) -> List[Dict]:
        """从经验中提取原始代码模式。"""
        add_patterns: Dict[str, List[RepairExperience]] = defaultdict(list)

        for exp in self.store.experiences:
            for line in exp.added_lines:
                normalized = self._normalize_code(line)
                if len(normalized) > 5:
                    add_patterns[normalized].append(exp)

        frequent = []
        for pattern, exps in add_patterns.items():
            if len(exps) >= self.min_frequency:
                frequent.append({
                    "pattern": pattern,
                    "frequency": len(exps),
                    "experiences": exps,
                    "operators": list(set(e.operator_name for e in exps)),
                })

        frequent.sort(key=lambda x: x["frequency"], reverse=True)
        return frequent

    def _normalize_code(self, line: str) -> str:
        """规范化代码行，去除变量名差异。"""
        s = line.strip()
        s = re.sub(r'\b[a-z_]\w*\b', 'VAR', s)
        s = re.sub(r'\d+\.?\d*(?:e[+-]?\d+)?', 'NUM', s)
        s = re.sub(r'\s+', ' ', s)
        return s

    def _filter_known(self, patterns: List[Dict]) -> List[Dict]:
        """过滤掉已被现有变异算子覆盖的模式。"""
        result = []
        for p in patterns:
            pattern_text = p["pattern"]
            is_known = False
            for known in _KNOWN_PATTERNS:
                if known in pattern_text.lower():
                    is_known = True
                    break
            if not is_known:
                result.append(p)
        return result

    def _merge_similar(self, patterns: List[Dict]) -> List[MinedPattern]:
        """合并相似模式。"""
        result = []
        seen: Set[str] = set()

        for i, p in enumerate(patterns):
            key = p["pattern"][:30]
            if key in seen:
                continue
            seen.add(key)

            mined = MinedPattern(
                pattern_id=f"evolved_{i}",
                pattern_type="code_addition",
                description=f"Frequent repair pattern: {p['pattern'][:60]}",
                added_code_template=p["pattern"],
                removed_code_template="",
                frequency=p["frequency"],
                source_operators=p["operators"],
                source_experiences=[e.kernel_id for e in p["experiences"][:5]],
            )
            result.append(mined)

        return result
