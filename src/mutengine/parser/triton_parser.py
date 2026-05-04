"""Triton kernel 源码解析器。

职责:
1. 从完整 Python 文件中定位 @triton.jit 装饰的 kernel 函数
2. 提取 kernel 函数体的源码范围
3. 返回解析后的 AST 和相关元信息供变异算子使用
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class KernelFunction:
    """一个被识别出的 Triton kernel 函数"""
    name: str
    start_line: int
    end_line: int
    body_start_line: int
    source: str
    decorator: str = ""


@dataclass
class TritonParseResult:
    """Triton 文件解析结果"""
    full_source: str
    tree: Optional[ast.Module]
    kernels: List[KernelFunction] = field(default_factory=list)
    is_triton: bool = False
    wrapper_class: Optional[str] = None
    language: str = "triton"

    @property
    def has_kernels(self) -> bool:
        return len(self.kernels) > 0

    @property
    def primary_kernel(self) -> Optional[KernelFunction]:
        if not self.kernels:
            return None
        return self.kernels[0]


_TRITON_DECORATORS = {
    "triton.jit",
    "triton.autotune",
    "triton.heuristics",
    "@triton.jit",
    "@triton.autotune",
}


def _is_triton_decorator(node: ast.expr) -> bool:
    """检查一个装饰器节点是否为 Triton JIT 相关装饰器。"""
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "triton":
            return node.attr in ("jit", "autotune", "heuristics")
    if isinstance(node, ast.Call):
        return _is_triton_decorator(node.func)
    if isinstance(node, ast.Name):
        return node.id in ("jit", "autotune")
    return False


def _get_decorator_text(node: ast.expr) -> str:
    """从装饰器 AST 节点提取可读文本。"""
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Call):
        return _get_decorator_text(node.func)
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _function_end_line(node: ast.FunctionDef, total_lines: int) -> int:
    """获取函数定义的结束行号。"""
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return end
    max_line = node.lineno
    for child in ast.walk(node):
        cl = getattr(child, "end_lineno", None) or getattr(child, "lineno", 0)
        if cl > max_line:
            max_line = cl
    return max_line


def _detect_wrapper_class(tree: ast.Module) -> Optional[str]:
    """检测是否存在 ModelNew 类（KernelBench 模式）。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("ModelNew", "Model"):
            return node.name
    return None


class TritonParser:
    """Triton kernel 源码解析器。"""

    def parse(self, source: str) -> TritonParseResult:
        result = TritonParseResult(full_source=source, tree=None)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return result

        result.tree = tree
        lines = source.splitlines()
        total_lines = len(lines)

        result.is_triton = self._detect_triton_usage(source, tree)
        result.wrapper_class = _detect_wrapper_class(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            triton_dec = None
            for dec in node.decorator_list:
                if _is_triton_decorator(dec):
                    triton_dec = dec
                    break
            if triton_dec is None:
                continue

            dec_line = getattr(triton_dec, "lineno", node.lineno)
            end_line = _function_end_line(node, total_lines)
            body_start = node.body[0].lineno if node.body else node.lineno + 1

            kernel_lines = lines[dec_line - 1: end_line]
            kernel_source = "\n".join(kernel_lines)

            kf = KernelFunction(
                name=node.name,
                start_line=dec_line,
                end_line=end_line,
                body_start_line=body_start,
                source=kernel_source,
                decorator=_get_decorator_text(triton_dec),
            )
            result.kernels.append(kf)

        result.kernels.sort(key=lambda k: k.start_line)
        return result

    def _detect_triton_usage(self, source: str, tree: ast.Module) -> bool:
        """检测源码是否使用了 Triton。"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "triton" or alias.name.startswith("triton."):
                        return True
            if isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "triton" or node.module.startswith("triton.")):
                    return True
        return "triton" in source and ("tl." in source or "triton.jit" in source)

    def extract_mutatable_source(self, source: str) -> Tuple[str, Optional[ast.Module]]:
        """提取可变异的源码和 AST。

        对于 KernelBench 格式的文件，返回整个文件的源码和 AST，
        因为变异算子需要访问 kernel 函数内部。
        """
        result = self.parse(source)
        return source, result.tree
