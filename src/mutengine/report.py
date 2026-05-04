"""变异测试报告生成器。

输出:
1. 每个 kernel 的变异测试详细报告 (JSON)
2. 按 operator/category 的汇总统计
3. 全局汇总表 (CSV/Markdown)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from ..models import MutantStatus, MutationTestResult

logger = logging.getLogger(__name__)

CATEGORY_NAMES = {
    "A": "Arithmetic (baseline)",
    "B": "GPU Parallel Semantics",
    "C": "ML Numerical Semantics",
    "D": "LLM Error Patterns",
}


class MutationReporter:
    """变异测试结果报告生成器。"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_kernel_result(self, result: MutationTestResult) -> Path:
        """保存单个 kernel 的详细结果。"""
        filename = f"L{result.kernel.level}_P{result.kernel.problem_id}.json"
        path = self.output_dir / "details" / filename
        result.save(path)
        logger.info(f"Saved result to {path}")
        return path

    def generate_summary(self, results: List[MutationTestResult]) -> Dict[str, Any]:
        """生成全局汇总统计 (V2: strict/candidate equivalent split)。"""
        summary: Dict[str, Any] = {
            "total_kernels": len(results),
            "total_mutants": 0,
            "total_killed": 0,
            "total_survived": 0,
            "total_stillborn": 0,
            "total_strict_equivalent": 0,
            "total_candidate_equivalent": 0,
            "total_equivalent": 0,
            "overall_mutation_score": 0.0,
            "overall_mutation_score_optimistic": 0.0,
            "by_category": {},
            "by_operator": {},
            "by_kernel": [],
        }

        cat_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "killed": 0, "survived": 0, "stillborn": 0,
                      "strict_equivalent": 0, "candidate_equivalent": 0}
        )
        op_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "killed": 0, "survived": 0, "stillborn": 0,
                      "strict_equivalent": 0, "candidate_equivalent": 0}
        )

        for r in results:
            summary["total_mutants"] += r.total
            summary["total_killed"] += r.killed
            summary["total_survived"] += r.survived
            summary["total_stillborn"] += r.stillborn
            summary["total_strict_equivalent"] += r.strict_equivalent
            summary["total_candidate_equivalent"] += r.candidate_equivalent
            summary["total_equivalent"] += r.equivalent

            summary["by_kernel"].append({
                "kernel": f"L{r.kernel.level}_P{r.kernel.problem_id}",
                "name": r.kernel.problem_name,
                "total": r.total,
                "killed": r.killed,
                "survived": r.survived,
                "stillborn": r.stillborn,
                "strict_equivalent": r.strict_equivalent,
                "candidate_equivalent": r.candidate_equivalent,
                "score": round(r.mutation_score, 4),
                "score_optimistic": round(r.mutation_score_optimistic, 4),
            })

            for m in r.mutants:
                cat = m.operator_category
                cat_stats[cat]["total"] += 1
                if m.status == MutantStatus.STRICT_EQUIVALENT:
                    cat_stats[cat]["strict_equivalent"] += 1
                elif m.status == MutantStatus.CANDIDATE_EQUIVALENT:
                    cat_stats[cat]["candidate_equivalent"] += 1
                elif m.status.value in cat_stats[cat]:
                    cat_stats[cat][m.status.value] += 1

                op_stats[m.operator_name]["total"] += 1
                if m.status == MutantStatus.STRICT_EQUIVALENT:
                    op_stats[m.operator_name]["strict_equivalent"] += 1
                elif m.status == MutantStatus.CANDIDATE_EQUIVALENT:
                    op_stats[m.operator_name]["candidate_equivalent"] += 1
                elif m.status.value in op_stats[m.operator_name]:
                    op_stats[m.operator_name][m.status.value] += 1

        for cat, stats in cat_stats.items():
            denom = stats["total"] - stats["stillborn"] - stats["strict_equivalent"]
            score = stats["killed"] / denom if denom > 0 else 0.0
            denom_opt = denom - stats["candidate_equivalent"]
            score_opt = stats["killed"] / denom_opt if denom_opt > 0 else 0.0
            summary["by_category"][cat] = {
                **stats,
                "score": round(score, 4),
                "score_optimistic": round(score_opt, 4),
                "name": CATEGORY_NAMES.get(cat, cat),
            }

        for op, stats in op_stats.items():
            denom = stats["total"] - stats["stillborn"] - stats["strict_equivalent"]
            score = stats["killed"] / denom if denom > 0 else 0.0
            denom_opt = denom - stats["candidate_equivalent"]
            score_opt = stats["killed"] / denom_opt if denom_opt > 0 else 0.0
            summary["by_operator"][op] = {
                **stats,
                "score": round(score, 4),
                "score_optimistic": round(score_opt, 4),
            }

        t = summary
        denom_cons = t["total_mutants"] - t["total_stillborn"] - t["total_strict_equivalent"]
        t["overall_mutation_score"] = round(
            t["total_killed"] / denom_cons if denom_cons > 0 else 0.0, 4
        )
        denom_opt = denom_cons - t["total_candidate_equivalent"]
        t["overall_mutation_score_optimistic"] = round(
            t["total_killed"] / denom_opt if denom_opt > 0 else 0.0, 4
        )

        return summary

    def save_summary(self, results: List[MutationTestResult]) -> Path:
        """生成并保存汇总报告。"""
        summary = self.generate_summary(results)

        json_path = self.output_dir / "summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        md_path = self.output_dir / "summary.md"
        self._last_results = results
        self._write_markdown_report(summary, md_path)
        self._last_results = None

        logger.info(f"Summary saved to {json_path} and {md_path}")
        return json_path

    def _write_markdown_report(self, summary: Dict[str, Any], path: Path) -> None:
        """生成 Markdown 格式的汇总报告 (V2: strict/candidate split)。"""
        s = summary
        lines = [
            "# MutaKernel Mutation Testing Report",
            "",
            "## Overall",
            "",
            f"- Total kernels: {s['total_kernels']}",
            f"- Total mutants: {s['total_mutants']}",
            f"- Killed: {s['total_killed']}",
            f"- Survived: {s['total_survived']}",
            f"- Stillborn: {s['total_stillborn']}",
            f"- Strict Equivalent: {s['total_strict_equivalent']}",
            f"- Candidate Equivalent: {s['total_candidate_equivalent']}",
            f"- **Conservative Score: {s['overall_mutation_score']:.2%}** "
            f"(excl. strict equiv)",
            f"- **Optimistic Score: {s['overall_mutation_score_optimistic']:.2%}** "
            f"(excl. strict + candidate)",
            "",
            "## By Category",
            "",
            "| Category | Name | Total | Killed | Survived | Stillborn "
            "| Strict Eq | Cand Eq | Score | Score (Opt) |",
            "|----------|------|-------|--------|----------|---------- "
            "|-----------|---------|-------|-------------|",
        ]

        for cat in sorted(s["by_category"].keys()):
            cs = s["by_category"][cat]
            lines.append(
                f"| {cat} | {cs['name']} | {cs['total']} | {cs['killed']} | "
                f"{cs['survived']} | {cs['stillborn']} | "
                f"{cs['strict_equivalent']} | {cs['candidate_equivalent']} | "
                f"{cs['score']:.2%} | {cs['score_optimistic']:.2%} |"
            )

        lines.extend([
            "",
            "## By Operator",
            "",
            "| Operator | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |",
            "|----------|-------|--------|----------|-----------|---------|-------|-------------|",
        ])

        for op in sorted(s["by_operator"].keys()):
            os_ = s["by_operator"][op]
            lines.append(
                f"| {op} | {os_['total']} | {os_['killed']} | {os_['survived']} | "
                f"{os_['strict_equivalent']} | {os_['candidate_equivalent']} | "
                f"{os_['score']:.2%} | {os_['score_optimistic']:.2%} |"
            )

        lines.extend([
            "",
            "## By Kernel",
            "",
            "| Kernel | Name | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |",
            "|--------|------|-------|--------|----------|-----------|---------|-------|-------------|",
        ])

        for k in s["by_kernel"]:
            name_short = k["name"][:30]
            lines.append(
                f"| {k['kernel']} | {name_short} | {k['total']} | "
                f"{k['killed']} | {k['survived']} | "
                f"{k.get('strict_equivalent', 0)} | {k.get('candidate_equivalent', 0)} | "
                f"{k['score']:.2%} | {k.get('score_optimistic', k['score']):.2%} |"
            )

        # Equivalent mutant details (Equiv Evidence)
        equiv_mutants = []
        for r in self._last_results or []:
            for m in r.mutants:
                if m.status in (MutantStatus.STRICT_EQUIVALENT,
                                MutantStatus.CANDIDATE_EQUIVALENT):
                    equiv_mutants.append((r, m))

        if equiv_mutants:
            lines.extend([
                "",
                "## Equivalent Mutant Details",
                "",
                "| Kernel | Mutant | Operator | Level | Evidence |",
                "|--------|--------|----------|-------|----------|",
            ])
            for r, m in equiv_mutants:
                level = m.status.value.replace("_", " ").title()
                evidence = m.error_message[:80] if m.error_message else "—"
                lines.append(
                    f"| {r.kernel.problem_name[:20]} | {m.id[:25]} | "
                    f"{m.operator_name} | {level} | {evidence} |"
                )

        # LLM review placeholder
        llm_stats = s.get("llm_review", {})
        if llm_stats:
            lines.extend([
                "",
                "## LLM Equivalence Review",
                "",
                f"- Reviewed: {llm_stats.get('reviewed', 0)}",
                f"- Confirmed equivalent: {llm_stats.get('confirmed', 0)}",
                f"- Reverted to SURVIVED: {llm_stats.get('reverted', 0)}",
                f"- Killed by LLM input: {llm_stats.get('killed', 0)}",
            ])

        lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # StressEnhance report
    # ------------------------------------------------------------------

    def save_stress_report(self, stress_summary: dict) -> Path:
        """Save a combined native + stress-enhanced visibility report."""
        md_path = self.output_dir / "stress_report.md"
        lines = [
            "# MutaKernel StressEnhance Report",
            "",
            "## Visibility Comparison",
            "",
            f"- Native C-category visibility: "
            f"{stress_summary.get('native_c_visibility', 0):.2%}",
            f"- Stress-enhanced C-category visibility: "
            f"{stress_summary.get('stress_c_visibility', 0):.2%}",
            f"- Visibility lift: "
            f"{stress_summary.get('visibility_lift', 0):.1%}",
            "",
            "## Attribution Distribution",
            "",
            "| Type | Count | Description |",
            "|------|-------|-------------|",
        ]
        dist = stress_summary.get("attribution_distribution", {})
        type_desc = {
            "type1_absorbed": "Algorithmically absorbed (math-equivalent)",
            "type2_config_blind": "Configuration-blind (needs precision mode switch)",
            "type3_input_blind": "Input-distribution-blind (killed by stress)",
            "type4_genuine_vuln": "Genuine original vulnerability",
            "unresolved": "Unresolved",
        }
        for key, desc in type_desc.items():
            cnt = dist.get(key, 0)
            lines.append(f"| {key} | {cnt} | {desc} |")

        lines.extend([
            "",
            "## Per-Policy Kill Effectiveness",
            "",
            "| Policy | Kills |",
            "|--------|-------|",
        ])
        for pol, cnt in sorted(
            stress_summary.get("per_policy_kills", {}).items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"| {pol} | {cnt} |")

        lines.append("")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Stress report saved to {md_path}")
        return md_path
