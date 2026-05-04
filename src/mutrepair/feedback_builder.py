"""修复反馈构造器。

按 PLAN.md 中定义的 5 种 baseline 模式生成不同粒度的修复 prompt:
- B0: 标准迭代修复 (仅编译/运行错误)
- B1: 通用鲁棒性提示 (B0 + 通用建议)
- B2: 增强测试反馈 (失败输入 + 错误现象，无变异分析)
- B3: 位置反馈 (B2 + 代码位置)
- ours: 完整变异分析反馈 (B2 + 变异算子类型 + 缺陷分类 + 修复建议)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import Mutant

DEFECT_TAXONOMY: Dict[str, Dict[str, str]] = {
    "stab_remove": {
        "category": "Numerical Stability",
        "description": "Missing max-subtraction for numerical stabilization (e.g., softmax overflow)",
        "suggestion": "Add `x - x.max()` before `exp()` to prevent overflow on large inputs.",
    },
    "acc_downgrade": {
        "category": "Accumulation Precision",
        "description": "Accumulator using insufficient precision (FP16 instead of FP32)",
        "suggestion": "Use FP32 accumulators for reduction operations, cast back to FP16 only at output.",
    },
    "epsilon_modify": {
        "category": "Division Safety",
        "description": "Missing or incorrect epsilon for division/log stability",
        "suggestion": "Add or verify `eps` parameter (typically 1e-5 to 1e-6) in denominators and log arguments.",
    },
    "scale_modify": {
        "category": "Scaling Correctness",
        "description": "Incorrect or missing scaling factor (e.g., 1/sqrt(d) in attention)",
        "suggestion": "Verify the scaling factor matches the mathematical specification (e.g., 1/sqrt(head_dim)).",
    },
    "cast_remove": {
        "category": "Type Cast Safety",
        "description": "Missing explicit dtype cast causing implicit precision loss",
        "suggestion": "Add explicit `.to(dtype)` or `.float()` before precision-sensitive operations.",
    },
    "reduction_reorder": {
        "category": "Reduction Ordering",
        "description": "Floating-point non-associativity causing inconsistent reduction results",
        "suggestion": "Use higher precision (FP32/FP64) for intermediate accumulation in reductions.",
    },
    "init_modify": {
        "category": "Identity Element",
        "description": "Incorrect initialization for min/max reduction (using finite values instead of ±inf)",
        "suggestion": "Use `float('inf')` or `float('-inf')` as identity elements for min/max reductions.",
    },
    "broadcast_unsafe": {
        "category": "Shape Alignment",
        "description": "Missing explicit broadcast/expand before element-wise operations",
        "suggestion": "Add `.expand()` or `.broadcast_to()` to ensure operand shapes match before computation.",
    },
    "layout_assume": {
        "category": "Memory Layout",
        "description": "Assuming contiguous memory layout without explicit .contiguous() call",
        "suggestion": "Add `.contiguous()` after transpose/slice operations before passing to kernels.",
    },
}


class FeedbackBuilder:
    """根据 baseline 模式生成修复 prompt。"""

    def __init__(self, mode: str = "ours"):
        if mode not in ("B0", "B1", "B2", "B3", "ours"):
            raise ValueError(f"Unknown mode: {mode}. Must be one of B0, B1, B2, B3, ours.")
        self.mode = mode

    def build_prompt(
        self,
        kernel_code: str,
        error_info: Optional[str] = None,
        failing_input_desc: Optional[str] = None,
        failure_detail: Optional[str] = None,
        code_location: Optional[str] = None,
        survived_mutant: Optional[Mutant] = None,
    ) -> str:
        """构造修复 prompt。

        不同 mode 下使用不同粒度的信息:
        - B0: kernel_code + error_info
        - B1: B0 + 通用建议
        - B2: B0 + failing_input_desc + failure_detail
        - B3: B2 + code_location
        - ours: B2 + code_location + mutation analysis + defect taxonomy + suggestion
        """
        parts = [self._header(), self._kernel_section(kernel_code)]

        if self.mode == "B0":
            if error_info:
                parts.append(self._error_section(error_info))
            parts.append(self._b0_instruction())

        elif self.mode == "B1":
            if error_info:
                parts.append(self._error_section(error_info))
            parts.append(self._b1_instruction())

        elif self.mode == "B2":
            if error_info:
                parts.append(self._error_section(error_info))
            if failing_input_desc:
                parts.append(self._input_section(failing_input_desc))
            if failure_detail:
                parts.append(self._failure_detail_section(failure_detail))
            parts.append(self._b2_instruction())

        elif self.mode == "B3":
            if error_info:
                parts.append(self._error_section(error_info))
            if failing_input_desc:
                parts.append(self._input_section(failing_input_desc))
            if failure_detail:
                parts.append(self._failure_detail_section(failure_detail))
            if code_location:
                parts.append(self._location_section(code_location))
            parts.append(self._b3_instruction())

        elif self.mode == "ours":
            if error_info:
                parts.append(self._error_section(error_info))
            if failing_input_desc:
                parts.append(self._input_section(failing_input_desc))
            if failure_detail:
                parts.append(self._failure_detail_section(failure_detail))
            if code_location:
                parts.append(self._location_section(code_location))
            if survived_mutant:
                parts.append(self._mutation_analysis_section(survived_mutant))
            parts.append(self._ours_instruction())

        parts.append(self._footer())
        return "\n\n".join(parts)

    def _header(self) -> str:
        return (
            "You are an expert GPU kernel developer. "
            "Fix the following kernel to be numerically correct under all valid inputs."
        )

    def _kernel_section(self, code: str) -> str:
        return f"## Kernel Code\n\n```python\n{code}\n```"

    def _error_section(self, error_info: str) -> str:
        return f"## Error Information\n\n{error_info}"

    def _input_section(self, desc: str) -> str:
        return f"## Failing Input Description\n\n{desc}"

    def _failure_detail_section(self, detail: str) -> str:
        return f"## Failure Detail\n\n{detail}"

    def _location_section(self, location: str) -> str:
        return f"## Suspected Code Location\n\n{location}"

    def _mutation_analysis_section(self, mutant: Mutant) -> str:
        op_name = mutant.operator_name
        taxonomy = DEFECT_TAXONOMY.get(op_name, {})

        parts = ["## Mutation Analysis"]
        parts.append(f"- **Mutation operator**: `{op_name}`")
        parts.append(f"- **Operator category**: {mutant.operator_category}")
        if taxonomy:
            parts.append(f"- **Defect category**: {taxonomy['category']}")
            parts.append(f"- **Defect description**: {taxonomy['description']}")
        parts.append(f"- **Mutation site**: Line {mutant.site.line_start}")
        if mutant.site.original_code:
            snippet = mutant.site.original_code[:100]
            parts.append(f"- **Original code at site**: `{snippet}`")
        if taxonomy.get("suggestion"):
            parts.append(f"\n### Repair Suggestion\n\n{taxonomy['suggestion']}")

        return "\n".join(parts)

    def _b0_instruction(self) -> str:
        return (
            "## Task\n\n"
            "Fix the kernel so it compiles and produces correct output. "
            "Return the complete fixed code."
        )

    def _b1_instruction(self) -> str:
        return (
            "## Task\n\n"
            "Fix the kernel so it compiles and produces correct output. "
            "Pay special attention to numerical stability and precision correctness. "
            "Ensure proper handling of edge cases in floating-point arithmetic. "
            "Return the complete fixed code."
        )

    def _b2_instruction(self) -> str:
        return (
            "## Task\n\n"
            "Fix the kernel so it produces correct output for both standard and "
            "the failing inputs described above. "
            "Return the complete fixed code."
        )

    def _b3_instruction(self) -> str:
        return (
            "## Task\n\n"
            "Fix the kernel at the suspected location above so it produces correct output "
            "for both standard and the failing inputs. "
            "Return the complete fixed code."
        )

    def _ours_instruction(self) -> str:
        return (
            "## Task\n\n"
            "Based on the mutation analysis above, fix the identified defect in the kernel. "
            "The mutation testing revealed that this code region is vulnerable to the "
            "described defect category. Apply the suggested repair pattern while ensuring "
            "the fix does not break existing correctness or degrade performance. "
            "Return the complete fixed code."
        )

    def _footer(self) -> str:
        return (
            "## Output Format\n\n"
            "Return ONLY the complete fixed Python file. "
            "Do not include explanations outside the code block.\n\n"
            "```python\n# Your fixed code here\n```"
        )
