"""结果分析脚本：读取各 RQ 的实验结果，生成汇总表格和图表数据。

用法:
    python scripts/analyze_results.py --rq1-dir runs/rq1_xxx
    python scripts/analyze_results.py --rq1-dir runs/rq1_xxx --rq2-dir runs/rq2_xxx
    python scripts/analyze_results.py --rq3-dirs runs/rq3_config1 runs/rq3_config2 ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analyze_results")


def analyze_rq1(rq1_dir: Path) -> Dict[str, Any]:
    """分析 RQ1 结果。"""
    summary_path = rq1_dir / "summary.json"
    if not summary_path.exists():
        logger.error(f"summary.json not found in {rq1_dir}")
        return {}

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    logger.info("\n" + "=" * 60)
    logger.info("RQ1: Mutation Testing Adequacy Results")
    logger.info("=" * 60)
    logger.info(f"Total kernels: {summary['total_kernels']}")
    logger.info(f"Total mutants: {summary['total_mutants']}")
    logger.info(f"Overall mutation score: {summary['overall_mutation_score']:.2%}")

    logger.info("\nBy Category:")
    logger.info(f"  {'Cat':<5} {'Name':<30} {'Total':<8} {'Killed':<8} {'Survived':<10} {'Score':<8}")
    logger.info("  " + "-" * 70)
    for cat in sorted(summary.get("by_category", {}).keys()):
        s = summary["by_category"][cat]
        logger.info(
            f"  {cat:<5} {s['name']:<30} {s['total']:<8} {s['killed']:<8} "
            f"{s['survived']:<10} {s['score']:.2%}"
        )

    logger.info("\nBy Operator (top 10 by survived):")
    ops = summary.get("by_operator", {})
    sorted_ops = sorted(ops.items(), key=lambda x: x[1].get("survived", 0), reverse=True)
    for op_name, s in sorted_ops[:10]:
        logger.info(
            f"  {op_name:<25} survived={s['survived']:<5} "
            f"killed={s['killed']:<5} score={s['score']:.2%}"
        )

    c_survived = sum(
        s["survived"] for cat, s in summary.get("by_category", {}).items()
        if cat == "C"
    )
    total_survived = summary.get("total_survived", 0)
    logger.info(f"\nC-type survived: {c_survived} / {total_survived} total survived")

    return summary


def analyze_rq2(rq2_dir: Path) -> Dict[str, Any]:
    """分析 RQ2 修复对比结果。"""
    summary_path = rq2_dir / "repair_summary.json"
    if not summary_path.exists():
        logger.error(f"repair_summary.json not found in {rq2_dir}")
        return {}

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    logger.info("\n" + "=" * 60)
    logger.info("RQ2: Repair Comparison Results")
    logger.info("=" * 60)
    logger.info(f"  {'Mode':<8} {'Total':<8} {'Success':<10} {'Rate':<10} {'Avg Rounds':<12}")
    logger.info("  " + "-" * 50)
    for mode in ["B0", "B1", "B2", "B3", "ours"]:
        s = summary.get(mode, {})
        if s:
            logger.info(
                f"  {mode:<8} {s['total']:<8} {s['success']:<10} "
                f"{s['success_rate']:.2%}{'':4} {s['avg_rounds']:<12}"
            )

    ours = summary.get("ours", {})
    b2 = summary.get("B2", {})
    b3 = summary.get("B3", {})
    if ours and b2:
        delta_b2 = ours.get("success_rate", 0) - b2.get("success_rate", 0)
        logger.info(f"\n  Ours vs B2 (enhanced-test-only): +{delta_b2:.2%}")
    if ours and b3:
        delta_b3 = ours.get("success_rate", 0) - b3.get("success_rate", 0)
        logger.info(f"  Ours vs B3 (location-only): +{delta_b3:.2%}")

    return summary


def analyze_rq3(rq3_dirs: List[Path]) -> Dict[str, Any]:
    """分析 RQ3 多配置对比结果。"""
    logger.info("\n" + "=" * 60)
    logger.info("RQ3: Test Configuration Comparison")
    logger.info("=" * 60)

    config_scores = {}
    for d in rq3_dirs:
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        config_name = d.name.split("_")[1] if "_" in d.name else d.name
        config_scores[config_name] = summary.get("overall_mutation_score", 0)
        logger.info(f"  {config_name}: mutation_score={summary['overall_mutation_score']:.2%}")

        by_cat = summary.get("by_category", {})
        for cat in sorted(by_cat.keys()):
            logger.info(f"    Cat {cat}: {by_cat[cat]['score']:.2%}")

    return config_scores


def main():
    parser = argparse.ArgumentParser(description="MutaKernel Results Analyzer")
    parser.add_argument("--rq1-dir", type=Path, help="RQ1 results directory")
    parser.add_argument("--rq2-dir", type=Path, help="RQ2 results directory")
    parser.add_argument("--rq3-dirs", type=Path, nargs="*", help="RQ3 config directories")
    parser.add_argument("--realism-dir", type=Path, help="Realism validation directory")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")

    args = parser.parse_args()

    combined = {}

    if args.rq1_dir:
        combined["rq1"] = analyze_rq1(args.rq1_dir)

    if args.rq2_dir:
        combined["rq2"] = analyze_rq2(args.rq2_dir)

    if args.rq3_dirs:
        combined["rq3"] = analyze_rq3(args.rq3_dirs)

    if args.realism_dir:
        realism_path = args.realism_dir / "realism_report.json"
        if realism_path.exists():
            with open(realism_path) as f:
                combined["realism"] = json.load(f)
            logger.info(f"\nRealism: coverage_cd={combined['realism']['coverage_rate_cd']:.1%}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        logger.info(f"\nCombined results saved to {args.output}")


if __name__ == "__main__":
    main()
