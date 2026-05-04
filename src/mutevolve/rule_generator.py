"""变异规则动态生成器。

根据 PatternMiner 发现的新模式，生成新的变异算子
并注册到算子系统中。
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from ..models import MutationSite
from ..mutengine.operators.base import MutationOperator, _find_pattern_sites, _replace_in_source
from .pattern_miner import MinedPattern

logger = logging.getLogger(__name__)


class DynamicOperator(MutationOperator):
    """基于挖掘模式动态创建的变异算子。

    不使用 __init_subclass__ 自动注册，
    需要通过 RuleGenerator.register() 显式注册。
    """
    name = ""
    category = "E"
    description = ""

    def __init__(self, pattern: MinedPattern):
        self.pattern = pattern
        self.name = pattern.pattern_id
        self.category = "E"
        self.description = pattern.description
        self._search_regex = self._build_search_pattern(pattern)

    def find_sites(self, source: str, tree=None) -> List[MutationSite]:
        if self._search_regex is None:
            return []
        return _find_pattern_sites(source, self._search_regex, self.name)

    def apply(self, source: str, site: MutationSite) -> str:
        return _replace_in_source(source, site, site.original_code, "")

    def _build_search_pattern(self, pattern: MinedPattern) -> Optional[str]:
        """从模式模板构建搜索正则。"""
        template = pattern.added_code_template
        if not template or len(template) < 5:
            return None
        escaped = re.escape(template)
        escaped = escaped.replace(r"VAR", r"\w+")
        escaped = escaped.replace(r"NUM", r"\d+\.?\d*(?:e[+-]?\d+)?")
        return escaped


class RuleGenerator:
    """变异规则动态管理器。"""

    def __init__(self):
        self.dynamic_operators: List[DynamicOperator] = []

    def generate_from_patterns(self, patterns: List[MinedPattern]) -> List[DynamicOperator]:
        """根据挖掘的模式生成新变异算子。"""
        new_operators = []
        for pattern in patterns:
            op = DynamicOperator(pattern)
            if op._search_regex is not None:
                new_operators.append(op)
                logger.info(
                    f"Generated dynamic operator: {op.name} "
                    f"(freq={pattern.frequency}, from {pattern.source_operators})"
                )
        self.dynamic_operators.extend(new_operators)
        return new_operators

    def get_all_dynamic_operators(self) -> List[DynamicOperator]:
        return list(self.dynamic_operators)

    def clear(self) -> None:
        self.dynamic_operators.clear()
