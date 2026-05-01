"""变异算子基类与注册机制"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from ...models import MutationSite, Mutant, OperatorCategory


_OPERATOR_REGISTRY: Dict[str, Type["MutationOperator"]] = {}


class MutationOperator(ABC):
    """所有变异算子的基类。

    子类必须实现:
      - name: 算子名称 (如 "arith_replace")
      - category: 算子类别 (A/B/C/D)
      - description: 一句话描述该算子模拟的缺陷类型
      - find_sites(): 在源码中搜索可应用该算子的位置
      - apply(): 对指定位置施加变异，返回变异后的完整源码
    """

    name: str = ""
    category: str = ""
    description: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            _OPERATOR_REGISTRY[cls.name] = cls

    @abstractmethod
    def find_sites(self, source: str, tree: Optional[ast.Module] = None) -> List[MutationSite]:
        """在源码中找到所有可应用此算子的变异位点。

        Args:
            source: 完整源码字符串
            tree: 已解析的 AST（仅 Python/Triton 代码有效；CUDA 字符串时为 None）

        Returns:
            MutationSite 列表
        """
        ...

    @abstractmethod
    def apply(self, source: str, site: MutationSite) -> str:
        """对给定位点施加变异，返回变异后的完整源码。

        Args:
            source: 完整源码字符串
            site: find_sites() 返回的某一个位点

        Returns:
            变异后的完整源码字符串
        """
        ...

    def generate_mutants(self, source: str, kernel_id: str,
                         tree: Optional[ast.Module] = None) -> List[Mutant]:
        """便捷方法：查找所有位点并生成变异体列表。"""
        sites = self.find_sites(source, tree)
        mutants = []
        for i, site in enumerate(sites):
            mutated = self.apply(source, site)
            if mutated == source:
                continue
            mutant = Mutant(
                id=f"{kernel_id}__{self.name}__{i}",
                operator_name=self.name,
                operator_category=self.category,
                site=site,
                original_code=source,
                mutated_code=mutated,
                description=f"{self.description} @ L{site.line_start}",
            )
            mutants.append(mutant)
        return mutants


def get_all_operators() -> List[MutationOperator]:
    """返回所有已注册算子的实例列表。"""
    return [cls() for cls in _OPERATOR_REGISTRY.values()]


def get_operators_by_category(category: str) -> List[MutationOperator]:
    """返回指定类别的所有算子实例。"""
    return [cls() for cls in _OPERATOR_REGISTRY.values() if cls.category == category]


def _split_lines(source: str) -> List[str]:
    """将源码按行分割，保留换行符。"""
    return source.splitlines(keepends=True)


def _replace_in_source(source: str, site: MutationSite,
                       old_fragment: str, new_fragment: str) -> str:
    """在源码中按行范围替换指定片段。

    这是一个精确替换工具：在 site 指定的行范围内，
    将第一个出现的 old_fragment 替换为 new_fragment。
    """
    lines = _split_lines(source)
    target_region = "".join(lines[site.line_start - 1: site.line_end])
    if old_fragment not in target_region:
        return source
    new_region = target_region.replace(old_fragment, new_fragment, 1)
    result_lines = lines[: site.line_start - 1] + [new_region] + lines[site.line_end:]
    return "".join(result_lines)


def _find_pattern_sites(source: str, pattern: str, node_type: str = "") -> List[MutationSite]:
    """用正则表达式在源码中查找匹配位点。"""
    sites = []
    for i, line in enumerate(source.splitlines(), start=1):
        for m in re.finditer(pattern, line):
            sites.append(MutationSite(
                line_start=i,
                line_end=i,
                col_start=m.start(),
                col_end=m.end(),
                original_code=m.group(0),
                node_type=node_type,
            ))
    return sites


# ---------------------------------------------------------------------------
# CUDA C++ helpers – shared across A/B/C/D operator modules
# ---------------------------------------------------------------------------

def _strip_cuda_comment(line: str) -> str:
    """Strip ``// …`` line comments from a CUDA/C++ code line.

    Respects string literals so ``"http://url"`` is not treated as a comment.
    """
    i = 0
    n = len(line)
    in_char: Optional[str] = None
    while i < n:
        c = line[i]
        if in_char:
            if c == '\\' and i + 1 < n:
                i += 2
                continue
            if c == in_char:
                in_char = None
            i += 1
            continue
        if c in ('"', "'"):
            in_char = c
            i += 1
            continue
        if c == '/' and i + 1 < n and line[i + 1] == '/':
            return line[:i]
        i += 1
    return line


def _cuda_find_pattern_sites(
    source: str, pattern: str, node_type: str,
) -> List[MutationSite]:
    """Search for regex *pattern* across all source lines (CUDA-aware).

    Unlike the Python-oriented ``_find_pattern_sites`` this does **not**
    filter by Python string/comment context — CUDA code lives inside
    triple-quoted strings and would otherwise be invisible.  It does strip
    ``// …`` C++ line comments before matching.
    """
    sites: List[MutationSite] = []
    for i, raw_line in enumerate(source.splitlines(), start=1):
        code = _strip_cuda_comment(raw_line)
        for m in re.finditer(pattern, code):
            sites.append(MutationSite(
                line_start=i,
                line_end=i,
                col_start=m.start(),
                col_end=m.end(),
                original_code=m.group(0),
                node_type=node_type,
            ))
    return sites


def _replace_at_columns(source: str, site: MutationSite, replacement: str) -> str:
    """Replace text at the exact column range described by *site*."""
    lines = source.splitlines(keepends=True)
    if site.line_start != site.line_end:
        start = site.line_start - 1
        end = site.line_end
        region = "".join(lines[start:end])
        if site.original_code not in region:
            return source
        new_region = region.replace(site.original_code, replacement, 1)
        return "".join(lines[:start] + [new_region] + lines[end:])
    row = site.line_start - 1
    line = lines[row]
    if line[site.col_start: site.col_end] != site.original_code:
        return source
    lines[row] = line[: site.col_start] + replacement + line[site.col_end:]
    return "".join(lines)
