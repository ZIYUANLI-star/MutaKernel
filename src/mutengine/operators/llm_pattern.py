"""Category D: LLM error pattern mutation operators (data-driven from TritonBench/KernelBench)."""

from __future__ import annotations

import re
from typing import List, Optional

from ...models import MutationSite
from .base import MutationOperator, _replace_at_columns


def _outside_comment_or_string(line: str, col: int) -> bool:
    """检查 col 位置是否在 Python 注释或字符串之外。

    正确处理三引号字符串 (''' / \"\"\") 和单引号字符串。
    """
    i = 0
    n = len(line)
    in_triple: Optional[str] = None
    in_str: Optional[str] = None
    while i < col and i < n:
        c = line[i]
        if in_triple:
            if line[i:i+3] == in_triple:
                i += 3
                in_triple = None
            else:
                i += 1
            continue
        if in_str:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if line[i:i+3] in ("'''", '"""'):
            in_triple = line[i:i+3]
            i += 3
            continue
        if c == "#":
            return False
        if c in "\"'":
            in_str = c
        i += 1
    return in_triple is None and in_str is None


class BroadcastUnsafe(MutationOperator):
    """D1: 移除显式 broadcast/expand 调用，模拟 LLM 忽略形状对齐的常见错误。

    LLM 在生成 kernel 时经常忘记做显式 broadcast，
    或在需要 expand 的地方直接运算，导致隐式广播产生错误结果。
    此算子通过移除显式 broadcast 操作来模拟这类错误。

    不包含 .reshape() / .view(): 这些是必需的形状变换，删除后几乎
    必定触发 shape mismatch 编译错误 (STILLBORN)，不产生有意义的变异。
    """
    name = "broadcast_unsafe"
    category = "D"
    description = "Remove explicit broadcast/expand to simulate LLM shape-alignment omission"

    _EXPAND_CALL = re.compile(r"\.expand\s*\(")
    _EXPAND_AS = re.compile(r"\.expand_as\s*\(")
    _BROADCAST_TO = re.compile(r"\.broadcast_to\s*\(")
    _TL_BROADCAST = re.compile(r"\btl\.broadcast_to\s*\(")
    _UNSQUEEZE = re.compile(r"\.unsqueeze\s*\(")

    def find_sites(self, source: str, tree=None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for line_no, line in enumerate(source.splitlines(), start=1):
            for rx, node_type in [
                (self._EXPAND_CALL, "d1:expand"),
                (self._EXPAND_AS, "d1:expand_as"),
                (self._BROADCAST_TO, "d1:broadcast_to"),
                (self._TL_BROADCAST, "d1:tl_broadcast_to"),
                (self._UNSQUEEZE, "d1:unsqueeze"),
            ]:
                for m in rx.finditer(line):
                    if not _outside_comment_or_string(line, m.start()):
                        continue
                    open_paren = m.end() - 1
                    depth = 1
                    j = open_paren + 1
                    while j < len(line) and depth > 0:
                        if line[j] == "(":
                            depth += 1
                        elif line[j] == ")":
                            depth -= 1
                        j += 1
                    if depth != 0:
                        continue
                    sites.append(MutationSite(
                        line_start=line_no,
                        line_end=line_no,
                        col_start=m.start(),
                        col_end=j,
                        original_code=line[m.start():j],
                        node_type=node_type,
                    ))
        sites.sort(key=lambda s: (s.line_start, s.col_start))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        nt = site.node_type

        if nt == "d1:tl_broadcast_to":
            m = re.match(r"tl\.broadcast_to\s*\(\s*([^,]+),", site.original_code)
            if m:
                return _replace_at_columns(source, site, m.group(1).strip())
            return source

        if nt in ("d1:unsqueeze", "d1:expand", "d1:expand_as", "d1:broadcast_to"):
            return _replace_at_columns(source, site, "")

        return source


class LayoutAssume(MutationOperator):
    """D2: 移除 .contiguous() 调用，模拟 LLM 假设输入已连续的常见错误。

    LLM 生成的 kernel 常假设张量是 contiguous 的，
    但实际推理中经过 transpose/slice 后可能不是。
    此算子移除显式 .contiguous() 来暴露这类假设。

    只在 layout-sensitive 上下文中施加变异：
    - 前面有 .transpose / .permute / .T / .t() / .mT
    - 前面有切片操作 [...]
    防御性冗余 .contiguous() 调用不生成变异体。
    """
    name = "layout_assume"
    category = "D"
    description = "Remove .contiguous() calls to simulate LLM memory-layout assumption errors"

    _CONTIGUOUS = re.compile(r"\.contiguous\s*\(\s*\)")
    _STRIDE_CHECK = re.compile(r"\.is_contiguous\s*\(\s*\)")
    _LAYOUT_SENSITIVE = re.compile(
        r'\.(?:transpose|permute|mT|T|t)\s*(?:\(|[.\[])|'
        r'\[[^\]]*:[^\]]*\]'
    )

    def find_sites(self, source: str, tree=None) -> List[MutationSite]:
        sites: List[MutationSite] = []
        for line_no, line in enumerate(source.splitlines(), start=1):
            for rx, node_type in [
                (self._CONTIGUOUS, "d2:remove_contiguous"),
                (self._STRIDE_CHECK, "d2:remove_is_contiguous"),
            ]:
                for m in rx.finditer(line):
                    if not _outside_comment_or_string(line, m.start()):
                        continue
                    if node_type == "d2:remove_contiguous":
                        prefix = line[:m.start()]
                        if not self._LAYOUT_SENSITIVE.search(prefix):
                            continue
                    sites.append(MutationSite(
                        line_start=line_no,
                        line_end=line_no,
                        col_start=m.start(),
                        col_end=m.end(),
                        original_code=m.group(0),
                        node_type=node_type,
                    ))
        sites.sort(key=lambda s: (s.line_start, s.col_start))
        return sites

    def apply(self, source: str, site: MutationSite) -> str:
        if site.node_type in ("d2:remove_contiguous", "d2:remove_is_contiguous"):
            return _replace_at_columns(source, site, "")
        return source
