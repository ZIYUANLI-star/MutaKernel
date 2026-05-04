"""变异体运行器：编译并执行变异体，判定 killed/survived/stillborn。

核心流程:
1. 将变异后的源码写入临时文件
2. 动态导入编译
3. 使用相同测试输入运行原始 kernel 和变异体
4. 用 torch.allclose 比较输出
5. 判定状态
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import torch

from ..models import Mutant, MutantStatus, KernelInfo, MutationTestResult
from .operators.base import MutationOperator, get_all_operators, get_operators_by_category

logger = logging.getLogger(__name__)


class CompilationError(Exception):
    pass


class ExecutionError(Exception):
    pass


_LOAD_INLINE_NAME_RE = re.compile(
    r"""(load_inline\s*\([^)]*?name\s*=\s*)(['"])([^'"]+)\2""",
    re.DOTALL,
)

_FUNCTIONS_KWARG_RE = re.compile(
    r"""(load_inline\s*\([^)]*?)(,\s*functions\s*=\s*\[[^\]]*\])""",
    re.DOTALL,
)

_CUDA_SOURCES_RE = re.compile(
    r"""load_inline\s*\([^)]*?cuda_sources\s*=\s*([A-Za-z_]\w*)""",
    re.DOTALL,
)


def _patch_load_inline_pybind_conflict(source: str) -> str:
    """Fix conflict where cuda_sources contains PYBIND11_MODULE but
    functions=[...] is also specified, causing load_inline to generate
    a redundant main.cpp that references nonexistent C++ function names.

    PyTorch's load_inline only checks cpp_sources for PYBIND11_MODULE,
    not cuda_sources. When cuda_sources has its own PYBIND11_MODULE and
    functions= is also set, the auto-generated main.cpp fails to compile.

    Fix: if the cuda_sources variable contains PYBIND11_MODULE, remove
    the functions= kwarg so load_inline skips main.cpp generation."""
    for m in _CUDA_SOURCES_RE.finditer(source):
        var_name = m.group(1)
        var_pattern = re.compile(
            rf"""(?:^|\n)\s*{re.escape(var_name)}\s*=\s*(?:r?(?:'''|\"\"\"|\"|'))""",
        )
        var_match = var_pattern.search(source)
        if var_match:
            start = var_match.start()
            chunk = source[start:start + len(source) - start]
            if "PYBIND11_MODULE" in chunk[:10000]:
                source = _FUNCTIONS_KWARG_RE.sub(r"\1", source)
                break
    return source


def _patch_load_inline_names(source: str, unique_suffix: str) -> str:
    """Rewrite every ``load_inline(name="xxx")`` to ``load_inline(name="xxx_<suffix>")``
    so that each mutant triggers a fresh CUDA JIT compilation instead of
    hitting the cached extension built for the original kernel."""
    def _repl(m: re.Match) -> str:
        prefix, quote, name = m.group(1), m.group(2), m.group(3)
        return f"{prefix}{quote}{name}_{unique_suffix}{quote}"
    return _LOAD_INLINE_NAME_RE.sub(_repl, source)


def _load_module_from_source(source: str, module_name: str, tmp_dir: str) -> Any:
    """将源码写入临时文件并动态导入。"""
    source = _patch_load_inline_pybind_conflict(source)
    source = _patch_load_inline_names(source, module_name)
    filepath = os.path.join(tmp_dir, f"{module_name}.py")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(source)

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise CompilationError(f"Cannot create module spec from {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise CompilationError(str(e)) from e
    return module


def _run_model(model: torch.nn.Module, inputs: List[torch.Tensor],
               device: str) -> torch.Tensor:
    """在指定设备上运行模型并返回输出。"""
    model = model.to(device)
    model.eval()
    moved_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
    with torch.no_grad():
        output = model(*moved_inputs)
    return output


def _compare_outputs(ref_output: Any, mut_output: Any,
                     atol: float, rtol: float) -> bool:
    """比较参考输出和变异体输出。返回 True 表示输出一致（变异体存活）。"""
    if isinstance(ref_output, torch.Tensor) and isinstance(mut_output, torch.Tensor):
        if ref_output.shape != mut_output.shape:
            return False
        if ref_output.dtype != mut_output.dtype:
            mut_output = mut_output.to(ref_output.dtype)
        try:
            return torch.allclose(
                ref_output.float().cpu(),
                mut_output.float().cpu(),
                atol=atol, rtol=rtol,
            )
        except RuntimeError:
            return False

    if isinstance(ref_output, (tuple, list)) and isinstance(mut_output, (tuple, list)):
        if len(ref_output) != len(mut_output):
            return False
        return all(
            _compare_outputs(r, m, atol, rtol)
            for r, m in zip(ref_output, mut_output)
        )

    return ref_output == mut_output


class MutantRunner:
    """变异体运行器。

    负责:
    1. 生成变异体（调用各变异算子）
    2. 编译变异体（动态导入）
    3. 执行变异体（运行并比较输出）
    4. 判定状态（killed / survived / stillborn）
    """

    def __init__(
        self,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        num_test_inputs: int = 5,
        timeout_seconds: float = 30.0,
        device: str = "cuda",
        seed: int = 42,
        categories: Optional[List[str]] = None,
    ):
        self.atol = atol
        self.rtol = rtol
        self.num_test_inputs = num_test_inputs
        self.timeout_seconds = timeout_seconds
        self.device = device
        self.seed = seed
        self.categories = categories or ["A", "B", "C", "D"]
        self._tmp_dir = tempfile.mkdtemp(prefix="mutakernel_")

    def generate_mutants(self, kernel: KernelInfo) -> List[Mutant]:
        """对一个 kernel 生成所有变异体。"""
        import ast
        source = kernel.kernel_code
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None

        operators: List[MutationOperator] = []
        for cat in self.categories:
            operators.extend(get_operators_by_category(cat))

        kernel_id = f"L{kernel.level}_P{kernel.problem_id}"
        all_mutants: List[Mutant] = []

        for op in operators:
            try:
                mutants = op.generate_mutants(source, kernel_id, tree)
                all_mutants.extend(mutants)
            except Exception as e:
                logger.warning(f"Operator {op.name} failed on {kernel_id}: {e}")

        logger.info(
            f"Generated {len(all_mutants)} mutants for {kernel_id} "
            f"({', '.join(f'{cat}:{sum(1 for m in all_mutants if m.operator_category == cat)}' for cat in self.categories)})"
        )
        return all_mutants

    def run_mutant(
        self,
        kernel: KernelInfo,
        mutant: Mutant,
        ref_module: Any,
        get_inputs_fn: Any,
        get_init_inputs_fn: Any,
    ) -> Mutant:
        """编译并运行单个变异体，判定其状态。

        Args:
            kernel: 原始 kernel 信息
            mutant: 待测变异体
            ref_module: 参考模块（包含 Model 类）
            get_inputs_fn: 生成测试输入的函数
            get_init_inputs_fn: 生成初始化参数的函数

        Returns:
            更新了状态的 Mutant 对象
        """
        start_time = time.time()

        try:
            mut_module = _load_module_from_source(
                mutant.mutated_code,
                f"mutant_{mutant.id.replace('-', '_').replace('.', '_')}",
                self._tmp_dir,
            )
        except CompilationError as e:
            mutant.status = MutantStatus.STILLBORN
            mutant.error_message = f"Compilation: {str(e)[:300]}"
            mutant.execution_time_ms = (time.time() - start_time) * 1000
            return mutant

        try:
            ref_model = self._instantiate_model(ref_module, "Model", get_init_inputs_fn)
            mut_model = self._instantiate_model(mut_module, "ModelNew", get_init_inputs_fn)
        except Exception as e:
            mutant.status = MutantStatus.STILLBORN
            mutant.error_message = f"Instantiation: {str(e)[:300]}"
            mutant.execution_time_ms = (time.time() - start_time) * 1000
            return mutant

        killed = False
        for trial in range(self.num_test_inputs):
            seed = self.seed + trial
            torch.manual_seed(seed)
            try:
                inputs = get_inputs_fn()
                ref_output = _run_model(ref_model, inputs, self.device)

                torch.manual_seed(seed)
                inputs2 = get_inputs_fn()
                mut_output = _run_model(mut_model, inputs2, self.device)

                if not _compare_outputs(ref_output, mut_output, self.atol, self.rtol):
                    killed = True
                    mutant.kill_input_seed = seed
                    break
            except Exception as e:
                killed = True
                mutant.error_message = f"Runtime trial {trial}: {str(e)[:200]}"
                mutant.kill_input_seed = seed
                break

        mutant.status = MutantStatus.KILLED if killed else MutantStatus.SURVIVED
        mutant.execution_time_ms = (time.time() - start_time) * 1000
        return mutant

    def run_all_mutants(
        self,
        kernel: KernelInfo,
        mutants: List[Mutant],
        ref_module: Any,
        get_inputs_fn: Any,
        get_init_inputs_fn: Any,
    ) -> MutationTestResult:
        """运行所有变异体并返回汇总结果。"""
        for i, mutant in enumerate(mutants):
            logger.info(
                f"  [{i+1}/{len(mutants)}] {mutant.operator_name} "
                f"@ L{mutant.site.line_start}"
            )
            self.run_mutant(kernel, mutant, ref_module, get_inputs_fn, get_init_inputs_fn)
            logger.info(f"    → {mutant.status.value} ({mutant.execution_time_ms:.0f}ms)")

        result = MutationTestResult(kernel=kernel, mutants=mutants)
        logger.info(
            f"Kernel {kernel.problem_id}: "
            f"score={result.mutation_score:.2%} "
            f"(killed={result.killed}, survived={result.survived}, "
            f"stillborn={result.stillborn}, total={result.total})"
        )
        return result

    def _instantiate_model(self, module: Any, class_name: str,
                           get_init_inputs_fn: Any) -> torch.nn.Module:
        """从模块中实例化模型。"""
        cls = getattr(module, class_name, None)
        if cls is None:
            if class_name == "ModelNew":
                cls = getattr(module, "Model", None)
            if cls is None:
                raise AttributeError(f"Module has no class '{class_name}'")

        init_inputs = get_init_inputs_fn()
        if isinstance(init_inputs, (list, tuple)):
            model = cls(*init_inputs)
        else:
            model = cls()
        return model

    def cleanup(self) -> None:
        """清理临时文件。"""
        import shutil
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass
