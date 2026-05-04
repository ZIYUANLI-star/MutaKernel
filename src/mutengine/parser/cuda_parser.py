"""CUDA kernel 源码解析器。

KernelBench 中的 CUDA kernel 通常以字符串形式嵌入在 Python 代码中,
通过 torch.utils.cpp_extension.load_inline() 加载编译。

职责:
1. 从 Python 文件中提取 CUDA 源码字符串
2. 定位 __global__ kernel 函数
3. 返回解析结果供变异算子使用
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CudaKernelInfo:
    """一个 CUDA kernel 函数的位点信息"""
    name: str
    start_offset: int
    end_offset: int
    source: str


@dataclass
class CudaStringBlock:
    """Python 源码中的一个 CUDA 字符串块"""
    variable_name: str
    string_content: str
    py_line_start: int
    py_line_end: int
    quote_style: str
    kernels: List[CudaKernelInfo] = field(default_factory=list)


@dataclass
class CudaParseResult:
    """CUDA 文件解析结果"""
    full_source: str
    tree: Optional[ast.Module]
    cuda_blocks: List[CudaStringBlock] = field(default_factory=list)
    is_cuda: bool = False
    language: str = "cuda"

    @property
    def has_kernels(self) -> bool:
        return any(b.kernels for b in self.cuda_blocks)

    @property
    def all_cuda_source(self) -> str:
        return "\n".join(b.string_content for b in self.cuda_blocks)


_CUDA_INDICATORS = [
    "__global__",
    "__device__",
    "__shared__",
    "threadIdx",
    "blockIdx",
    "blockDim",
    "gridDim",
    "atomicAdd",
    "__syncthreads",
    "cuda_source",
    "cpp_source",
    "load_inline",
]

_GLOBAL_FUNC = re.compile(
    r"__global__\s+void\s+(\w+)\s*\(",
    re.MULTILINE,
)


def _find_kernel_functions(cuda_source: str) -> List[CudaKernelInfo]:
    """在 CUDA 源码字符串中找到所有 __global__ 函数。"""
    kernels = []
    for m in _GLOBAL_FUNC.finditer(cuda_source):
        name = m.group(1)
        start = m.start()
        depth = 0
        found_open = False
        pos = m.end()
        while pos < len(cuda_source):
            ch = cuda_source[pos]
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
                if found_open and depth == 0:
                    end = pos + 1
                    kernels.append(CudaKernelInfo(
                        name=name,
                        start_offset=start,
                        end_offset=end,
                        source=cuda_source[start:end],
                    ))
                    break
            pos += 1
    return kernels


class CudaParser:
    """CUDA kernel 源码解析器。"""

    def parse(self, source: str) -> CudaParseResult:
        result = CudaParseResult(full_source=source, tree=None)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return result

        result.tree = tree
        result.is_cuda = self._detect_cuda_usage(source)

        if not result.is_cuda:
            return result

        self._extract_cuda_strings(source, tree, result)
        return result

    def _detect_cuda_usage(self, source: str) -> bool:
        """检测源码是否包含 CUDA kernel。"""
        score = sum(1 for indicator in _CUDA_INDICATORS if indicator in source)
        return score >= 2

    def _extract_cuda_strings(self, source: str, tree: ast.Module,
                              result: CudaParseResult) -> None:
        """从 AST 中提取包含 CUDA 代码的字符串赋值。"""
        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, (ast.Constant, ast.JoinedStr)):
                continue
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                continue

            content = node.value.value
            if not self._looks_like_cuda(content):
                continue

            var_name = ""
            if node.targets and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id

            end_line = getattr(node, "end_lineno", node.lineno)
            quote = '"""' if '"""' in lines[node.lineno - 1] else "'''"

            kernels = _find_kernel_functions(content)
            block = CudaStringBlock(
                variable_name=var_name,
                string_content=content,
                py_line_start=node.lineno,
                py_line_end=end_line,
                quote_style=quote,
                kernels=kernels,
            )
            result.cuda_blocks.append(block)

    def _looks_like_cuda(self, content: str) -> bool:
        """检查字符串内容是否像 CUDA 代码。"""
        score = sum(1 for ind in _CUDA_INDICATORS[:8] if ind in content)
        return score >= 1

    def extract_mutatable_source(self, source: str) -> Tuple[str, Optional[ast.Module]]:
        """提取可变异的源码。

        对于 CUDA，变异直接在完整 Python 文件上操作
        （变异算子使用正则匹配嵌入的 CUDA 字符串内部的模式）。
        """
        result = self.parse(source)
        return source, result.tree
