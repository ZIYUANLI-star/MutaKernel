"""增强数值输入生成器。

核心约束: Shape 和 dtype 与原始输入完全一致，只改变数值分布。
根据存活变异体的类型，生成针对性的极端数值输入来暴露潜在缺陷。

底层实现复用 src.stress.policy_bank 中的通用策略。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ..stress.policy_bank import STRESS_POLICIES

logger = logging.getLogger(__name__)

STRATEGY_MAP: Dict[str, List[str]] = {
    # --- C-class: ML Numerical Semantics ---
    "epsilon_modify":    ["near_zero", "denormals", "dense_nonzero"],
    "scale_modify":      ["uniform_constant", "structured_ramp"],
    "stab_remove":       ["large_magnitude", "near_overflow"],
    "cast_remove":       ["near_overflow", "large_magnitude"],
    "init_modify":       ["all_negative", "sparse"],
    "acc_downgrade":     ["mixed_extremes", "large_magnitude"],
    "reduction_reorder": ["mixed_extremes", "alternating_sign"],
    "broadcast_unsafe":  ["structured_ramp"],
    "layout_assume":     ["structured_ramp"],
    # --- B-class: GPU Parallel Semantics ---
    "index_replace":     ["structured_ramp", "large_magnitude", "head_heavy", "tail_heavy"],
    "mask_boundary":     ["boundary_last_element", "sparse", "sparse_extreme"],
    "sync_remove":       ["large_magnitude", "mixed_extremes"],
    "launch_config_mutate": ["structured_ramp", "large_magnitude"],
    # --- A-class: Arithmetic ---
    "arith_replace":     ["large_magnitude", "mixed_extremes", "dense_nonzero"],
    "relop_replace":     ["boundary_last_element", "structured_ramp", "sparse_extreme"],
    "const_perturb":     ["near_zero", "large_magnitude"],
}


class EnhancedInputGenerator:
    """增强数值输入生成器。

    根据存活变异体的算子类型，生成针对性输入。
    所有增强输入与原始输入具有完全相同的 shape 和 dtype。
    """

    def __init__(self, base_seed: int = 20000):
        self.base_seed = base_seed

    def get_strategies_for_operator(self, operator_name: str) -> List[str]:
        """获取指定算子对应的增强策略列表。"""
        return STRATEGY_MAP.get(operator_name, ["large_magnitude", "near_zero"])

    def generate_enhanced_inputs(
        self,
        get_inputs_fn: Callable,
        operator_name: str,
        num_per_strategy: int = 3,
    ) -> List[Tuple[str, List[Any]]]:
        """生成增强输入集合。

        Args:
            get_inputs_fn: 原始 get_inputs() 函数
            operator_name: 存活变异体的算子名称
            num_per_strategy: 每个策略生成几组输入

        Returns:
            List[(strategy_name, inputs)]
        """
        strategies = self.get_strategies_for_operator(operator_name)

        torch.manual_seed(self.base_seed)
        template_inputs = get_inputs_fn()

        all_enhanced: List[Tuple[str, List[Any]]] = []

        for strategy in strategies:
            for trial in range(num_per_strategy):
                seed = self.base_seed + hash(strategy) % 10000 + trial
                enhanced = self._apply_strategy(template_inputs, strategy, seed)
                all_enhanced.append((strategy, enhanced))

        return all_enhanced

    def _apply_strategy(self, template_inputs: List[Any],
                        strategy: str, seed: int) -> List[Any]:
        """对一组输入应用增强策略，复用 policy_bank 统一实现。"""
        policy_fn = STRESS_POLICIES.get(strategy)
        if policy_fn is None:
            logger.warning(f"Unknown strategy '{strategy}', returning clone")
            return [x.clone() if isinstance(x, torch.Tensor) else x
                    for x in template_inputs]
        return policy_fn(template_inputs, seed)

    def test_kernel_with_enhanced_inputs(
        self,
        run_fn: Callable,
        ref_fn: Callable,
        get_inputs_fn: Callable,
        operator_name: str,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        num_per_strategy: int = 3,
    ) -> Dict[str, Any]:
        """用增强输入测试 kernel，检查是否暴露缺陷。

        Args:
            run_fn: callable(inputs) -> output, 运行待测 kernel
            ref_fn: callable(inputs) -> output, 运行参考实现
            get_inputs_fn: 生成原始输入的函数
            operator_name: 存活变异体的算子名称
            atol, rtol: torch.allclose 参数

        Returns:
            {
                "has_defect": bool,
                "failing_strategies": [...],
                "total_tests": int,
                "failed_tests": int,
            }
        """
        enhanced_inputs = self.generate_enhanced_inputs(
            get_inputs_fn, operator_name, num_per_strategy
        )

        failures = []
        for strategy, inputs in enhanced_inputs:
            try:
                ref_output = ref_fn(inputs)
                test_output = run_fn(inputs)

                if isinstance(ref_output, torch.Tensor) and isinstance(test_output, torch.Tensor):
                    if ref_output.shape != test_output.shape:
                        failures.append({"strategy": strategy, "reason": "shape_mismatch"})
                        continue
                    if not torch.allclose(
                        ref_output.float().cpu(),
                        test_output.float().cpu(),
                        atol=atol, rtol=rtol,
                    ):
                        failures.append({"strategy": strategy, "reason": "value_mismatch"})
                        continue
            except Exception as e:
                failures.append({"strategy": strategy, "reason": f"error: {str(e)[:200]}"})

        return {
            "has_defect": len(failures) > 0,
            "failing_strategies": failures,
            "total_tests": len(enhanced_inputs),
            "failed_tests": len(failures),
        }
