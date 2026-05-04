"""RQ2 实验脚本：变异分析引导的反馈修复对比实验。

对比 5 种修复策略: B0, B1, B2, B3, ours

用法:
    python scripts/run_repair.py --mutation-results runs/rq1_xxx --modes B0 B1 B2 B3 ours
    python scripts/run_repair.py --mutation-results runs/rq1_xxx --modes ours --max-rounds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import KERNELBENCH_ROOT, RESULTS_DIR, RepairConfig
from src.models import KernelInfo, Mutant, MutantStatus, MutationTestResult, RepairResult
from src.bridge.eval_bridge import KernelBenchBridge
from src.mutrepair.enhanced_inputs import EnhancedInputGenerator
from src.mutrepair.feedback_builder import FeedbackBuilder
from src.mutrepair.repair_loop import RepairLoop
from src.mutrepair.experience_store import ExperienceStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_repair")


def create_llm_caller(config: RepairConfig) -> Callable[[str], str]:
    """创建 LLM 调用函数。"""
    from openai import OpenAI

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.api_base,
    )

    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=8192,
        )
        return response.choices[0].message.content or ""

    return call_llm


def load_mutation_results(results_dir: Path) -> List[MutationTestResult]:
    """从 RQ1 结果目录加载变异测试结果。"""
    details_dir = results_dir / "details"
    if not details_dir.exists():
        logger.error(f"Details directory not found: {details_dir}")
        return []

    results = []
    for f in sorted(details_dir.glob("*.json")):
        try:
            result = MutationTestResult.load(f)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    return results


def find_repair_candidates(
    results: List[MutationTestResult],
    bridge: KernelBenchBridge,
    enhanced_gen: EnhancedInputGenerator,
) -> List[Dict[str, Any]]:
    """筛选需要修复的 kernel。

    只修复：增强输入下原始 kernel 也失败的 kernel（有真实缺陷）。
    """
    candidates = []

    for result in results:
        survived = result.survived_mutants()
        if not survived:
            continue

        kernel = result.kernel
        ki = bridge.load_kernel_info(kernel.level, kernel.problem_name)
        if ki is None:
            continue

        try:
            ref_module, get_inputs_fn, get_init_inputs_fn = \
                bridge.load_runtime_components(ki)
        except Exception as e:
            logger.warning(f"Cannot load runtime for {ki}: {e}")
            continue

        for mutant in survived:
            candidates.append({
                "kernel": ki,
                "mutant": mutant,
                "ref_module": ref_module,
                "get_inputs_fn": get_inputs_fn,
                "get_init_inputs_fn": get_init_inputs_fn,
            })

    logger.info(f"Found {len(candidates)} repair candidates")
    return candidates


def run_repair_experiment(args: argparse.Namespace) -> None:
    """运行 RQ2 修复对比实验。"""
    bridge = KernelBenchBridge(args.kernelbench_root)
    enhanced_gen = EnhancedInputGenerator()

    mutation_results = load_mutation_results(Path(args.mutation_results))
    if not mutation_results:
        logger.error("No mutation results found")
        return

    candidates = find_repair_candidates(mutation_results, bridge, enhanced_gen)
    if not candidates:
        logger.error("No repair candidates found")
        return

    repair_config = RepairConfig(
        max_rounds=args.max_rounds,
        model_name=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    llm_caller = create_llm_caller(repair_config)

    output_dir = Path(args.output_dir) / f"rq2_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List[Dict]] = {}

    for mode in args.modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running repair mode: {mode}")
        logger.info(f"{'='*60}")

        exp_store = ExperienceStore(output_dir / f"experience_{mode}")

        repair_loop = RepairLoop(
            llm_caller=llm_caller,
            mode=mode,
            max_rounds=args.max_rounds,
            device=args.device,
            experience_store=exp_store,
        )

        mode_results = []
        for i, cand in enumerate(candidates):
            logger.info(
                f"[{i+1}/{len(candidates)}] {cand['kernel']} "
                f"mutant={cand['mutant'].operator_name}"
            )
            try:
                result = repair_loop.repair(
                    kernel=cand["kernel"],
                    survived_mutant=cand["mutant"],
                    ref_module=cand["ref_module"],
                    get_inputs_fn=cand["get_inputs_fn"],
                    get_init_inputs_fn=cand["get_init_inputs_fn"],
                )
                mode_results.append(result.to_dict())
                logger.info(
                    f"  → {'SUCCESS' if result.success else 'FAILED'} "
                    f"({result.rounds_used} rounds)"
                )
            except Exception as e:
                logger.error(f"  → ERROR: {e}")
                mode_results.append({
                    "problem_id": cand["kernel"].problem_id,
                    "error": str(e)[:200],
                })

        repair_loop.cleanup()

        all_results[mode] = mode_results

        mode_path = output_dir / f"repair_{mode}.json"
        with open(mode_path, "w", encoding="utf-8") as f:
            json.dump(mode_results, f, indent=2, ensure_ascii=False)

    summary = _compute_repair_summary(all_results)
    summary_path = output_dir / "repair_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_dir}")
    _print_comparison_table(summary)


def _compute_repair_summary(all_results: Dict[str, List[Dict]]) -> Dict:
    """计算各模式的修复汇总统计。"""
    summary = {}
    for mode, results in all_results.items():
        total = len(results)
        success = sum(1 for r in results if r.get("success", False))
        orig_pass = sum(1 for r in results if r.get("original_test_pass", False))
        enhanced_pass = sum(1 for r in results if r.get("enhanced_test_pass", False))
        avg_rounds = (
            sum(r.get("rounds_used", 0) for r in results) / total
            if total > 0 else 0
        )

        summary[mode] = {
            "total": total,
            "success": success,
            "success_rate": round(success / total, 4) if total > 0 else 0,
            "original_test_pass": orig_pass,
            "enhanced_test_pass": enhanced_pass,
            "avg_rounds": round(avg_rounds, 2),
        }
    return summary


def _print_comparison_table(summary: Dict) -> None:
    """打印对比表。"""
    logger.info("\n=== Repair Comparison ===")
    logger.info(f"{'Mode':<8} {'Total':<8} {'Success':<10} {'Rate':<10} {'Avg Rounds':<12}")
    logger.info("-" * 50)
    for mode, s in summary.items():
        logger.info(
            f"{mode:<8} {s['total']:<8} {s['success']:<10} "
            f"{s['success_rate']:.2%}{'':4} {s['avg_rounds']:<12}"
        )


def main():
    parser = argparse.ArgumentParser(description="MutaKernel Repair Experiment (RQ2)")
    parser.add_argument("--mutation-results", required=True,
                        help="Path to RQ1 mutation test results directory")
    parser.add_argument("--modes", nargs="+", default=["B0", "B1", "B2", "B3", "ours"])
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--kernelbench-root", default=str(KERNELBENCH_ROOT))

    args = parser.parse_args()
    run_repair_experiment(args)


if __name__ == "__main__":
    main()
