"""Mutant Realism 验证器。

论文能否站住脚的关键模块。负责证明 C/D 类变异体对应真实 LLM 错误。

验证方法:
1. Pattern Matching: 从真实 LLM 错误样本中提取错误根因，与变异算子对齐
2. Injection Detection: 将真实错误注入正确 kernel，检查变异框架能否覆盖
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BugPattern:
    """一个真实 LLM 错误模式。"""
    bug_id: str
    problem_id: int
    level: int
    root_cause: str
    error_category: str
    diff_summary: str
    matched_operators: List[str] = field(default_factory=list)
    source: str = "unknown"  # "iteration_diff" | "final_fail_diff" | "standalone"


ROOT_CAUSE_TO_OPERATORS: Dict[str, List[str]] = {
    "missing_numerical_stability": ["stab_remove"],
    "overflow_no_max_subtract": ["stab_remove"],
    "precision_loss_fp16_accumulator": ["acc_downgrade"],
    "missing_fp32_cast": ["acc_downgrade", "cast_remove"],
    "epsilon_missing": ["epsilon_modify"],
    "epsilon_wrong_value": ["epsilon_modify"],
    "scale_factor_missing": ["scale_modify"],
    "scale_factor_wrong": ["scale_modify"],
    "missing_type_cast": ["cast_remove"],
    "implicit_type_coercion": ["cast_remove"],
    "reduction_precision": ["reduction_reorder", "acc_downgrade"],
    "wrong_init_value": ["init_modify"],
    "missing_broadcast": ["broadcast_unsafe"],
    "shape_mismatch_no_expand": ["broadcast_unsafe"],
    "contiguous_assumption": ["layout_assume"],
    "wrong_index_dimension": ["index_replace"],
    "off_by_one_boundary": ["mask_boundary"],
    "wrong_arithmetic_op": ["arith_replace"],
    "wrong_comparison_op": ["relop_replace"],
    "wrong_constant": ["const_perturb"],
}

# 正则按优先级排序：更具体的模式在前，避免泛模式抢先匹配。
# diff 分析时先在完整 diff 文本中逐条搜索，首次命中即返回。
#
# 注意：这些正则在 _classify_root_cause 的三轮中使用。
# 第一轮搜完整 diff，第二轮搜 +行（正确版本有），第三轮搜 -行。
# 某些"同算子"的细分（如 epsilon_missing vs epsilon_wrong_value）
# 不影响覆盖率统计，因为它们映射到同一个变异算子。

# 这组正则只在 _classify_root_cause 的 Round 2/3 中使用
# （即搜索已去掉 diff 标记 +/- 的代码行），避免 diff 标记干扰。
# Round 1 对完整 diff 文本使用 DIFF_FULL_TEXT_PATTERNS（更保守）。

DIFF_CHANGED_LINE_PATTERNS: Dict[str, str] = {
    # --- 数值稳定化 ---
    # "x - max_val" / "x[j] - row_m" / "val - max" 减法模式
    # 注意 [\w\]\)] 匹配 x, ], ) 等出现在减号前的字符
    r"[\w\]\)]\s*-\s*max\w*|[\w\]\)]\s*-\s*row_m|[\w\]\)]\s*-\s*m_i":
        "missing_numerical_stability",
    r"\.max\s*\(|tl\.max\s*\(": "overflow_no_max_subtract",

    # --- 累加器精度 ---
    r"(?:accum|dot).*(?:float16|fp16|half)|(?:float16|fp16|half).*(?:accum|dot)":
        "precision_loss_fp16_accumulator",

    # --- 初始值（float('-inf') 或 -INFINITY）---
    r"float\s*\(\s*['\"](-)?inf": "wrong_init_value",
    r"-\s*INFINITY|FLT_MAX|-\s*FLT_MAX": "wrong_init_value",

    # --- reduction 精度 ---
    r"tl\.sum\s*\(|torch\.sum\s*\(|\.sum\s*\(": "reduction_precision",

    # --- FP32 转换 ---
    r"\.float\(\)": "missing_fp32_cast",
    r"(?:torch\.)?float32|(?:tl\.)?float32|\bfp32\b": "missing_fp32_cast",

    # --- epsilon ---
    r"\beps\b|epsilon|1e-[5-8]": "epsilon_missing",

    # --- scale factor ---
    r"1\.0\s*/\s*(?:math\.)?sqrt|\/\s*(?:math\.)?sqrt": "scale_factor_missing",
    r"\bscale\s*=": "scale_factor_wrong",
    r"\brsqrt\w*|(?<!\w)sqrt\w*": "scale_factor_missing",

    # --- 类型转换 ---
    r"\.half\(\)|\.bfloat16\(\)": "missing_type_cast",
    r"\.to\s*\(\s*(?:torch\.)?(?:float16|bfloat16|half)\s*\)": "missing_type_cast",
    r"\.to\s*\(\s*(?:torch\.)?(?:float32)\s*\)": "missing_fp32_cast",
    r"\.to\s*\(": "missing_type_cast",

    # --- implicit_type_coercion ---
    r"(?:int|long)\s*\(.*(?:float|double)\)": "implicit_type_coercion",
    r"(?:float|double)\s*\(.*(?:int|long)\)": "implicit_type_coercion",
    r"\bfloat\s*\(\s*\w+\s*\).*\bfloat\s*\(\s*\w+\s*\)": "implicit_type_coercion",

    # --- 结构性 ---
    r"\.expand\s*\(|\.expand_as\s*\(|\.broadcast": "missing_broadcast",
    r"\.view\s*\(|\.reshape\s*\(": "shape_mismatch_no_expand",
    r"\.contiguous\s*\(": "contiguous_assumption",

    # --- GPU 索引 ---
    r"program_id\s*\(\s*\d\s*\)": "wrong_index_dimension",
    r"threadIdx\.\w|blockIdx\.\w|blockDim\.\w": "wrong_index_dimension",

    # --- 边界条件 ---
    r"(?:off(?:set)?|idx|index)\s*<=|<=\s*(?:N\b|n\b|size|length)":
        "off_by_one_boundary",
}

# 为了兼容旧代码入口，保留这个名字（指向同一组规则）
DIFF_KEYWORDS_TO_ROOT_CAUSE = DIFF_CHANGED_LINE_PATTERNS


@dataclass
class RealismReport:
    """Realism 验证报告。"""
    total_bugs_analyzed: int = 0
    total_root_causes: int = 0
    bugs_covered_by_cd: int = 0
    bugs_covered_by_ab: int = 0
    bugs_not_covered: int = 0
    coverage_rate_cd: float = 0.0
    coverage_rate_all: float = 0.0
    per_operator_realism: Dict[str, int] = field(default_factory=dict)
    per_category_coverage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    uncovered_patterns: List[str] = field(default_factory=list)
    bugs: List[BugPattern] = field(default_factory=list)

    def to_dict(self) -> Dict:
        source_counts: Dict[str, int] = defaultdict(int)
        for b in self.bugs:
            source_counts[b.source] += 1

        return {
            "total_bugs_analyzed": self.total_bugs_analyzed,
            "total_root_causes": self.total_root_causes,
            "bugs_covered_by_cd": self.bugs_covered_by_cd,
            "bugs_covered_by_ab": self.bugs_covered_by_ab,
            "bugs_not_covered": self.bugs_not_covered,
            "coverage_rate_cd": round(self.coverage_rate_cd, 4),
            "coverage_rate_all": round(self.coverage_rate_all, 4),
            "per_operator_realism": self.per_operator_realism,
            "per_source_count": dict(source_counts),
            "uncovered_patterns": self.uncovered_patterns[:20],
            "bugs_sample": [
                {
                    "bug_id": b.bug_id,
                    "root_cause": b.root_cause,
                    "error_category": b.error_category,
                    "matched_operators": b.matched_operators,
                    "source": b.source,
                }
                for b in self.bugs[:80]
            ],
        }


class RealismValidator:
    """Mutant Realism 验证器。"""

    def __init__(self):
        self.bugs: List[BugPattern] = []

    def analyze_bug_from_diff(
        self,
        bug_id: str,
        problem_id: int,
        level: int,
        correct_code: str,
        buggy_code: str,
        source: str = "final_fail_diff",
    ) -> BugPattern:
        """从正确/错误代码对中提取错误模式。

        Args:
            bug_id: 唯一标识
            correct_code: 正确 kernel 源码
            buggy_code: LLM 生成的错误 kernel 源码
            source: 数据来源标记
        """
        diff = list(difflib.unified_diff(
            correct_code.splitlines(),
            buggy_code.splitlines(),
            lineterm="",
        ))
        diff_text = "\n".join(diff)

        root_cause = self._classify_root_cause(diff_text, correct_code, buggy_code)
        matched_ops = ROOT_CAUSE_TO_OPERATORS.get(root_cause, [])

        error_cat = self._categorize_error(root_cause)

        bug = BugPattern(
            bug_id=bug_id,
            problem_id=problem_id,
            level=level,
            root_cause=root_cause,
            error_category=error_cat,
            diff_summary=diff_text[:500],
            matched_operators=matched_ops,
            source=source,
        )
        self.bugs.append(bug)
        return bug

    def analyze_buggy_kernel_standalone(
        self,
        bug_id: str,
        problem_id: int,
        level: int,
        buggy_code: str,
        error_message: str = "",
    ) -> BugPattern:
        """仅基于错误代码和错误信息分析根因（无正确代码对比时使用）。"""
        combined_text = buggy_code + "\n" + error_message
        root_cause = self._classify_root_cause_from_code(combined_text)
        matched_ops = ROOT_CAUSE_TO_OPERATORS.get(root_cause, [])
        error_cat = self._categorize_error(root_cause)

        bug = BugPattern(
            bug_id=bug_id,
            problem_id=problem_id,
            level=level,
            root_cause=root_cause,
            error_category=error_cat,
            diff_summary=error_message[:500],
            matched_operators=matched_ops,
            source="standalone",
        )
        self.bugs.append(bug)
        return bug

    def generate_report(self) -> RealismReport:
        """生成 Realism 验证报告。"""
        report = RealismReport()
        report.total_bugs_analyzed = len(self.bugs)
        report.bugs = list(self.bugs)

        root_causes: Set[str] = set()
        op_hit_count: Dict[str, int] = defaultdict(int)

        cd_operators = set()
        for ops in ROOT_CAUSE_TO_OPERATORS.values():
            for op in ops:
                if op.startswith(("stab_", "acc_", "epsilon_", "scale_",
                                  "cast_", "reduction_", "init_",
                                  "broadcast_", "layout_")):
                    cd_operators.add(op)

        ab_operators = {"arith_replace", "relop_replace", "const_perturb",
                        "index_replace", "sync_remove", "mask_boundary",
                        "launch_config_mutate"}

        covered_cd = 0
        covered_ab = 0
        not_covered = 0

        for bug in self.bugs:
            root_causes.add(bug.root_cause)

            if not bug.matched_operators:
                not_covered += 1
                report.uncovered_patterns.append(bug.root_cause)
                continue

            ops_set = set(bug.matched_operators)
            if ops_set & cd_operators:
                covered_cd += 1
            elif ops_set & ab_operators:
                covered_ab += 1
            else:
                not_covered += 1

            for op in bug.matched_operators:
                op_hit_count[op] += 1

        report.total_root_causes = len(root_causes)
        report.bugs_covered_by_cd = covered_cd
        report.bugs_covered_by_ab = covered_ab
        report.bugs_not_covered = not_covered
        report.per_operator_realism = dict(op_hit_count)

        total = len(self.bugs)
        report.coverage_rate_cd = covered_cd / total if total > 0 else 0.0
        report.coverage_rate_all = (covered_cd + covered_ab) / total if total > 0 else 0.0

        return report

    def save_report(self, output_path: Path) -> None:
        """保存报告到 JSON。"""
        report = self.generate_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Realism report saved to {output_path}")

    def _classify_root_cause(self, diff_text: str,
                             correct_code: str, buggy_code: str) -> str:
        """基于 diff 内容推断错误根因。

        三轮尝试 + 兜底:
        1. 在 -行（正确版本有的代码）中用领域正则匹配
           — 正确代码中的关键模式（如 "x - max_val"）是最强信号
        2. 在 +行（错误版本有的代码）中匹配
        3. 在全部变更行中匹配
        4. 兜底: 泛模式分析（算术/比较/常量差异）
        """
        diff_lines = diff_text.splitlines()

        # "-" 行 = 正确版本有、错误版本没有的代码（去掉 diff 标记）
        removed_lines = [l[1:] for l in diff_lines
                         if l.startswith("-") and not l.startswith("---")]
        # "+" 行 = 错误版本有、正确版本没有的代码
        added_lines = [l[1:] for l in diff_lines
                       if l.startswith("+") and not l.startswith("+++")]

        # 按 pattern 优先级遍历（字典顺序），每个 pattern 扫全部变更行。
        # 这保证高优先级 pattern（如数值稳定化）总是先于低优先级 pattern
        # （如 wrong_init_value），无论它们出现在 diff 的哪一行。
        all_changed_lines = removed_lines + added_lines
        for pattern, cause in DIFF_CHANGED_LINE_PATTERNS.items():
            for line in all_changed_lines:
                if re.search(pattern, line, re.IGNORECASE):
                    return cause

        # Round 4: 兜底泛模式 — 对比 +/- 行的结构差异
        all_changed = "\n".join(all_changed_lines)
        if not all_changed.strip():
            return "unknown"

        # 4a: 比较运算符变更
        for r_line in removed_lines:
            for a_line in added_lines:
                rs, as_ = r_line.strip(), a_line.strip()
                if not rs or not as_:
                    continue
                r_no_cmp = re.sub(r"<=|>=|==|!=|<|>", "CMP", rs)
                a_no_cmp = re.sub(r"<=|>=|==|!=|<|>", "CMP", as_)
                if r_no_cmp == a_no_cmp and "CMP" in r_no_cmp:
                    return "wrong_comparison_op"

        # 4b: 常量值变更（代码结构相同，只有数字不同）
        for r_line in removed_lines:
            for a_line in added_lines:
                rs, as_ = r_line.strip(), a_line.strip()
                if not rs or not as_:
                    continue
                r_no_num = re.sub(r"\b\d+\.?\d*\b", "NUM", rs)
                a_no_num = re.sub(r"\b\d+\.?\d*\b", "NUM", as_)
                if r_no_num == a_no_num and "NUM" in r_no_num:
                    r_nums = re.findall(r"\b\d+\.?\d*\b", rs)
                    a_nums = re.findall(r"\b\d+\.?\d*\b", as_)
                    if r_nums != a_nums:
                        return "wrong_constant"

        # 4c: 算术运算符变更（最泛的兜底）
        for r_line in removed_lines:
            for a_line in added_lines:
                rs, as_ = r_line.strip(), a_line.strip()
                if not rs or not as_:
                    continue
                r_no_op = re.sub(r"[\+\-\*/]", "OP", rs)
                a_no_op = re.sub(r"[\+\-\*/]", "OP", as_)
                if r_no_op == a_no_op and "OP" in r_no_op:
                    return "wrong_arithmetic_op"

        if re.search(r"[\+\-\*/]", all_changed):
            return "wrong_arithmetic_op"

        return "unknown"

    def _classify_root_cause_from_code(self, text: str) -> str:
        """仅从代码文本推断根因。"""
        for pattern, cause in DIFF_KEYWORDS_TO_ROOT_CAUSE.items():
            if re.search(pattern, text, re.IGNORECASE):
                return cause
        return "unknown"

    def _categorize_error(self, root_cause: str) -> str:
        """将根因映射到高层错误类别。"""
        numerical = {"missing_numerical_stability", "overflow_no_max_subtract",
                     "precision_loss_fp16_accumulator", "missing_fp32_cast",
                     "epsilon_missing", "epsilon_wrong_value",
                     "scale_factor_missing", "scale_factor_wrong",
                     "missing_type_cast", "implicit_type_coercion",
                     "reduction_precision", "wrong_init_value"}
        structural = {"missing_broadcast", "shape_mismatch_no_expand",
                      "contiguous_assumption"}
        logic = {"wrong_index_dimension", "off_by_one_boundary",
                 "wrong_arithmetic_op", "wrong_comparison_op", "wrong_constant"}

        if root_cause in numerical:
            return "numerical_semantic"
        if root_cause in structural:
            return "structural"
        if root_cause in logic:
            return "logic"
        return "unknown"
