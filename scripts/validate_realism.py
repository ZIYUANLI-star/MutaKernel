"""Realism 验证脚本：证明 C/D 类变异体对应真实 LLM 错误。

数据源（按优先级）:
1. 迭代历史配对（主力）: 正确 kernel 在迭代过程中编译通过但运行错误的版本
   vs 同问题 speedup 最高的正确版本 → diff 分析
2. 最终失败 kernel（补充）: eval_results 中最终判定失败的 kernel
   → 如有同问题正确版本则 diff，否则 standalone

用法:
    python scripts/validate_realism.py --levels 1 2
    python scripts/validate_realism.py --levels 1 --no-iterations  # 仅用旧数据源
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import KERNELBENCH_ROOT, RESULTS_DIR
from src.bridge.eval_bridge import KernelBenchBridge
from src.mutengine.realism_validator import RealismValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_realism")


def _analyze_iteration_pairs(
    bridge: KernelBenchBridge,
    validator: RealismValidator,
    levels: list[int],
) -> int:
    """数据源 1: 从迭代历史中提取「编译通过但运行错误 vs 最佳正确」配对。

    Returns:
        成功分析的 bug 样本数。
    """
    total = 0
    for level in levels:
        logger.info(f"\n=== Iteration Pairs: Level {level} ===")
        pairs = bridge.list_iteration_pairs(level)
        if not pairs:
            logger.info("  No iteration pairs found, skipping")
            continue

        for pair in pairs:
            bug_id = f"L{pair.level}_P{pair.problem_id}_iter_t{pair.failed_turn}"

            try:
                buggy_code = Path(pair.failed_kernel_path).read_text(encoding="utf-8")
                correct_code = Path(pair.correct_kernel_path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                logger.debug(f"  Cannot read kernel files for {bug_id}: {e}")
                continue

            if buggy_code.strip() == correct_code.strip():
                continue

            bug = validator.analyze_bug_from_diff(
                bug_id=bug_id,
                problem_id=pair.problem_id,
                level=pair.level,
                correct_code=correct_code,
                buggy_code=buggy_code,
                source="iteration_diff",
            )
            logger.info(
                f"  [{bug_id}] root_cause={bug.root_cause} "
                f"ops={bug.matched_operators}"
            )
            total += 1

    logger.info(f"\nIteration pairs total: {total} bugs analyzed")
    return total


def _analyze_final_failures(
    bridge: KernelBenchBridge,
    validator: RealismValidator,
    levels: list[int],
) -> int:
    """数据源 2: 从 eval_results 中分析最终判定失败的 kernel。

    Returns:
        成功分析的 bug 样本数。
    """
    total = 0
    for level in levels:
        logger.info(f"\n=== Final Failures: Level {level} ===")
        failed_list = bridge.list_failed_kernels(level)
        correct_list = bridge.list_correct_kernels(level)
        logger.info(f"  Failed: {len(failed_list)}, Correct: {len(correct_list)}")

        correct_by_problem: dict = {}
        for entry in correct_list:
            pid = bridge._extract_problem_id(entry["problem_key"])
            if pid is not None:
                correct_by_problem[pid] = entry

        for entry in failed_list:
            problem_key = entry["problem_key"]
            pid = bridge._extract_problem_id(problem_key)
            if pid is None:
                continue

            bug_id = f"L{level}_P{pid}_final"

            gen_path = bridge.find_generated_kernel(level, problem_key)
            if gen_path is None:
                continue

            try:
                buggy_code = gen_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            if pid in correct_by_problem:
                correct_entry = correct_by_problem[pid]
                correct_path = bridge.find_generated_kernel(
                    level, correct_entry["problem_key"],
                )
                if correct_path is not None:
                    try:
                        correct_code = correct_path.read_text(encoding="utf-8")
                        bug = validator.analyze_bug_from_diff(
                            bug_id=bug_id,
                            problem_id=pid,
                            level=level,
                            correct_code=correct_code,
                            buggy_code=buggy_code,
                            source="final_fail_diff",
                        )
                        logger.info(
                            f"  [{bug_id}] root_cause={bug.root_cause} "
                            f"ops={bug.matched_operators}"
                        )
                        total += 1
                        continue
                    except Exception as e:
                        logger.debug(f"  Diff analysis failed: {e}")

            bug = validator.analyze_buggy_kernel_standalone(
                bug_id=bug_id,
                problem_id=pid,
                level=level,
                buggy_code=buggy_code,
            )
            logger.info(
                f"  [{bug_id}] root_cause={bug.root_cause} "
                f"ops={bug.matched_operators}"
            )
            total += 1

    logger.info(f"\nFinal failures total: {total} bugs analyzed")
    return total


def _print_report(report, report_path: Path) -> None:
    """打印并保存报告。"""
    logger.info(f"\n{'='*60}")
    logger.info(f"  REALISM REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"Total bugs analyzed: {report.total_bugs_analyzed}")

    source_counts = {}
    for b in report.bugs:
        source_counts[b.source] = source_counts.get(b.source, 0) + 1
    logger.info(f"By data source:")
    for src, cnt in sorted(source_counts.items()):
        logger.info(f"  {src}: {cnt}")

    logger.info(f"\nCoverage:")
    logger.info(f"  C/D operators: {report.bugs_covered_by_cd} "
                f"({report.coverage_rate_cd:.1%})")
    logger.info(f"  A/B operators: {report.bugs_covered_by_ab}")
    logger.info(f"  Not covered:   {report.bugs_not_covered}")
    logger.info(f"  Overall:       {report.coverage_rate_all:.1%}")

    logger.info(f"\nPer-operator realism:")
    for op, count in sorted(report.per_operator_realism.items(),
                            key=lambda x: x[1], reverse=True):
        logger.info(f"  {op}: {count} bugs")

    if report.uncovered_patterns:
        from collections import Counter
        uc = Counter(report.uncovered_patterns)
        logger.info(f"\nUncovered root causes:")
        for cause, cnt in uc.most_common(10):
            logger.info(f"  {cause}: {cnt}")

    logger.info(f"\nReport saved to {report_path}")


def run_validation(args: argparse.Namespace) -> None:
    """执行 Realism 验证。"""
    bridge = KernelBenchBridge(args.kernelbench_root)
    validator = RealismValidator()

    if not args.no_iterations:
        _analyze_iteration_pairs(bridge, validator, args.levels)

    _analyze_final_failures(bridge, validator, args.levels)

    output_dir = Path(args.output_dir) / "realism"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = validator.generate_report()

    report_path = output_dir / "realism_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    _print_report(report, report_path)


def main():
    parser = argparse.ArgumentParser(description="MutaKernel Realism Validation")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--kernelbench-root", default=str(KERNELBENCH_ROOT))
    parser.add_argument(
        "--no-iterations", action="store_true",
        help="Skip iteration history, only use final eval_results",
    )

    args = parser.parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
