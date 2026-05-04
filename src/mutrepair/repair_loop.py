"""修复循环：调用 LLM 生成修复版本并进行双重验证。

流程:
1. 构造修复反馈 prompt
2. 调用 LLM 生成修复版本
3. 双重验证: 原始测试 + 增强数值测试
4. 失败则重试（最多 max_rounds 次）
5. 成功则记录修复经验
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ..models import KernelInfo, Mutant, RepairResult
from .feedback_builder import FeedbackBuilder
from .enhanced_inputs import EnhancedInputGenerator
from .experience_store import ExperienceStore

logger = logging.getLogger(__name__)


def _extract_code_from_response(response: str) -> Optional[str]:
    """从 LLM 响应中提取 Python 代码块。"""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        longest = max(matches, key=len)
        return longest.strip()
    if "import torch" in response or "class ModelNew" in response:
        return response.strip()
    return None


def _load_module_from_source(source: str, module_name: str,
                             tmp_dir: str) -> Any:
    """动态加载模块。"""
    filepath = os.path.join(tmp_dir, f"{module_name}.py")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filepath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_and_compare(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    inputs: List[Any],
    device: str,
    atol: float,
    rtol: float,
) -> Tuple[bool, str]:
    """运行模型并比较输出。返回 (通过, 错误信息)。"""
    try:
        model = model.to(device).eval()
        ref_model = ref_model.to(device).eval()
        moved = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

        with torch.no_grad():
            ref_out = ref_model(*moved)
            test_out = model(*moved)

        if isinstance(ref_out, torch.Tensor) and isinstance(test_out, torch.Tensor):
            if ref_out.shape != test_out.shape:
                return False, f"Shape mismatch: {ref_out.shape} vs {test_out.shape}"
            if not torch.allclose(ref_out.float().cpu(), test_out.float().cpu(),
                                  atol=atol, rtol=rtol):
                diff = (ref_out.float().cpu() - test_out.float().cpu()).abs()
                max_diff = diff.max().item()
                return False, f"Max abs diff: {max_diff:.6e}"
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


class RepairLoop:
    """LLM 驱动的修复循环。

    支持 5 种 baseline 模式（B0, B1, B2, B3, ours）。
    """

    def __init__(
        self,
        llm_caller: Callable[[str], str],
        mode: str = "ours",
        max_rounds: int = 5,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        device: str = "cuda",
        experience_store: Optional[ExperienceStore] = None,
    ):
        """
        Args:
            llm_caller: callable(prompt) -> response_text
            mode: B0 / B1 / B2 / B3 / ours
            max_rounds: 最大修复轮次
        """
        self.llm_caller = llm_caller
        self.mode = mode
        self.max_rounds = max_rounds
        self.atol = atol
        self.rtol = rtol
        self.device = device
        self.feedback_builder = FeedbackBuilder(mode=mode)
        self.enhanced_gen = EnhancedInputGenerator()
        self.experience_store = experience_store
        self._tmp_dir = tempfile.mkdtemp(prefix="mutrepair_")

    def repair(
        self,
        kernel: KernelInfo,
        survived_mutant: Mutant,
        ref_module: Any,
        get_inputs_fn: Callable,
        get_init_inputs_fn: Callable,
        enhanced_failure_info: Optional[Dict[str, Any]] = None,
    ) -> RepairResult:
        """执行修复循环。

        Args:
            kernel: 待修复 kernel
            survived_mutant: 触发修复的存活变异体
            ref_module: 参考模块（含 Model 类）
            get_inputs_fn: 生成标准测试输入
            get_init_inputs_fn: 生成模型初始化参数
            enhanced_failure_info: 增强输入测试的失败信息

        Returns:
            RepairResult
        """
        result = RepairResult(
            kernel=kernel,
            survived_mutant=survived_mutant,
            baseline_mode=self.mode,
            success=False,
            rounds_used=0,
        )

        current_code = kernel.kernel_code
        error_log_parts: List[str] = []

        for round_idx in range(self.max_rounds):
            result.rounds_used = round_idx + 1
            logger.info(f"  Repair round {round_idx + 1}/{self.max_rounds}")

            prompt = self._build_round_prompt(
                current_code, survived_mutant, error_log_parts, enhanced_failure_info
            )
            result.feedback_prompt = prompt

            try:
                response = self.llm_caller(prompt)
                fixed_code = _extract_code_from_response(response)
                if fixed_code is None:
                    error_log_parts.append(f"Round {round_idx+1}: No code extracted")
                    continue
            except Exception as e:
                error_log_parts.append(f"Round {round_idx+1}: LLM error: {str(e)[:200]}")
                continue

            try:
                fixed_module = _load_module_from_source(
                    fixed_code,
                    f"repair_{kernel.problem_id}_{round_idx}",
                    self._tmp_dir,
                )
            except Exception as e:
                error_log_parts.append(f"Round {round_idx+1}: Compile error: {str(e)[:200]}")
                continue

            orig_pass, enhanced_pass = self._dual_verify(
                fixed_module, ref_module,
                get_inputs_fn, get_init_inputs_fn,
                survived_mutant.operator_name,
            )

            result.original_test_pass = orig_pass
            result.enhanced_test_pass = enhanced_pass

            if orig_pass and enhanced_pass:
                result.success = True
                result.repaired_code = fixed_code
                logger.info(f"  Repair succeeded at round {round_idx + 1}")

                if self.experience_store:
                    self.experience_store.record_success(
                        kernel=kernel,
                        mutant=survived_mutant,
                        original_code=kernel.kernel_code,
                        repaired_code=fixed_code,
                        rounds=round_idx + 1,
                    )
                break

            if not orig_pass:
                error_log_parts.append(f"Round {round_idx+1}: Original test regression")
            if not enhanced_pass:
                error_log_parts.append(f"Round {round_idx+1}: Enhanced test still fails")

            current_code = fixed_code

        result.error_log = "\n".join(error_log_parts)
        return result

    def _build_round_prompt(
        self,
        current_code: str,
        mutant: Mutant,
        error_log: List[str],
        enhanced_info: Optional[Dict[str, Any]],
    ) -> str:
        error_info = "\n".join(error_log[-3:]) if error_log else None

        failing_input_desc = None
        failure_detail = None
        if enhanced_info and enhanced_info.get("failing_strategies"):
            fails = enhanced_info["failing_strategies"][:3]
            failing_input_desc = ", ".join(f["strategy"] for f in fails)
            failure_detail = "; ".join(
                f"{f['strategy']}: {f['reason']}" for f in fails
            )

        code_location = None
        if self.mode in ("B3", "ours"):
            code_location = (
                f"Line {mutant.site.line_start}: "
                f"`{mutant.site.original_code[:80]}`"
            )

        return self.feedback_builder.build_prompt(
            kernel_code=current_code,
            error_info=error_info,
            failing_input_desc=failing_input_desc,
            failure_detail=failure_detail,
            code_location=code_location,
            survived_mutant=mutant if self.mode == "ours" else None,
        )

    def _dual_verify(
        self,
        fixed_module: Any,
        ref_module: Any,
        get_inputs_fn: Callable,
        get_init_inputs_fn: Callable,
        operator_name: str,
        num_standard: int = 5,
        num_enhanced_per_strategy: int = 2,
    ) -> Tuple[bool, bool]:
        """双重验证：标准测试 + 增强数值测试。"""
        try:
            init_args = get_init_inputs_fn()
            if isinstance(init_args, (list, tuple)):
                fixed_model = getattr(fixed_module, "ModelNew", getattr(fixed_module, "Model"))(*init_args)
                ref_model = getattr(ref_module, "Model")(*init_args)
            else:
                fixed_model = getattr(fixed_module, "ModelNew", getattr(fixed_module, "Model"))()
                ref_model = getattr(ref_module, "Model")()
        except Exception as e:
            logger.error(f"Model instantiation failed: {e}")
            return False, False

        original_pass = True
        for trial in range(num_standard):
            torch.manual_seed(42 + trial)
            inputs = get_inputs_fn()
            passed, err = _run_and_compare(
                fixed_model, ref_model, inputs, self.device, self.atol, self.rtol
            )
            if not passed:
                original_pass = False
                break

        if not original_pass:
            return False, False

        enhanced_pass = True
        enhanced_inputs = self.enhanced_gen.generate_enhanced_inputs(
            get_inputs_fn, operator_name, num_per_strategy=num_enhanced_per_strategy
        )
        for strategy, inputs in enhanced_inputs:
            passed, err = _run_and_compare(
                fixed_model, ref_model, inputs, self.device, self.atol, self.rtol
            )
            if not passed:
                enhanced_pass = False
                break

        return original_pass, enhanced_pass

    def cleanup(self) -> None:
        import shutil
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass
