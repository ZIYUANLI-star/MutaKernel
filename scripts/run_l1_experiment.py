"""L1 正式实验：对所有 L1 正确 kernel 运行变异测试 (RQ1)。

特点:
- 每个 kernel 完成后立即保存结果
- 定期输出进度摘要
- 异常容错，单个 kernel 失败不影响整体
- 设置 TORCH_CUDA_ARCH_LIST 加速编译
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bridge.eval_bridge import KernelBenchBridge
from src.mutengine.mutant_runner import MutantRunner
from src.mutengine.report import MutationReporter
from src.models import MutationTestResult, MutantStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("L1_experiment")

KERNELBENCH_ROOT = "/home/kbuser/projects/KernelBench-0"
OUTPUT_DIR = Path(__file__).parent.parent / "runs" / "rq1_l1_formal"


def print_progress(results: list[MutationTestResult], elapsed: float, total_kernels: int):
    """打印当前进度摘要。"""
    done = len(results)
    if done == 0:
        return

    total_mutants = sum(r.total for r in results)
    total_killed = sum(r.killed for r in results)
    total_survived = sum(r.survived for r in results)
    total_stillborn = sum(r.stillborn for r in results)
    denom = total_mutants - total_stillborn
    score = total_killed / denom if denom > 0 else 0.0

    cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"t": 0, "k": 0, "s": 0, "sb": 0})
    for r in results:
        for m in r.mutants:
            cat_stats[m.operator_category]["t"] += 1
            if m.status == MutantStatus.KILLED:
                cat_stats[m.operator_category]["k"] += 1
            elif m.status == MutantStatus.SURVIVED:
                cat_stats[m.operator_category]["s"] += 1
            elif m.status == MutantStatus.STILLBORN:
                cat_stats[m.operator_category]["sb"] += 1

    avg_time = elapsed / done
    eta = avg_time * (total_kernels - done)

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"  PROGRESS: {done}/{total_kernels} kernels ({done/total_kernels:.0%})")
    logger.info(f"  Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")
    logger.info(f"  Mutants: {total_mutants} (killed={total_killed}, survived={total_survived}, stillborn={total_stillborn})")
    logger.info(f"  Overall Score: {score:.2%}")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        d = s["t"] - s["sb"]
        cs = s["k"] / d if d > 0 else 0
        logger.info(f"    Category {cat}: {s['t']} mutants, score={cs:.2%} (killed={s['k']}, survived={s['s']}, stillborn={s['sb']})")
    logger.info(f"{'='*60}")
    logger.info(f"")


def save_checkpoint(results: list[MutationTestResult], output_dir: Path):
    """保存当前累计结果到 checkpoint。"""
    reporter = MutationReporter(output_dir)
    reporter.save_summary(results)


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge = KernelBenchBridge(KERNELBENCH_ROOT)
    kernels = bridge.load_all_correct_kernels(levels=[1])
    logger.info(f"L1: {len(kernels)} correct kernels loaded")

    already_done = set()
    details_dir = output_dir / "details"
    if details_dir.exists():
        for f in details_dir.glob("L1_P*.json"):
            try:
                pid = int(f.stem.split("_P")[1])
                already_done.add(pid)
            except (ValueError, IndexError):
                pass
    if already_done:
        logger.info(f"Resuming: {len(already_done)} kernels already completed")

    runner = MutantRunner(
        atol=1e-2,
        rtol=1e-2,
        num_test_inputs=5,
        device="cuda",
        seed=42,
        categories=["A", "B", "C", "D"],
    )

    reporter = MutationReporter(output_dir)
    all_results: list[MutationTestResult] = []

    for f in sorted((details_dir or Path()).glob("L1_P*.json")) if details_dir.exists() else []:
        try:
            result = MutationTestResult.load(f)
            all_results.append(result)
        except Exception:
            pass

    start_time = time.time()
    total = len(kernels)

    for i, kernel in enumerate(kernels):
        if kernel.problem_id in already_done:
            continue

        logger.info(f"[{len(all_results)+1}/{total}] Testing L1 P{kernel.problem_id}: {kernel.problem_name}")
        kernel_start = time.time()

        try:
            mutants = runner.generate_mutants(kernel)
            if not mutants:
                logger.warning(f"  No mutants generated, skipping")
                continue

            logger.info(f"  Generated {len(mutants)} mutants")
            ref_module, get_inputs_fn, get_init_inputs_fn = bridge.load_runtime_components(kernel)
            result = runner.run_all_mutants(kernel, mutants, ref_module, get_inputs_fn, get_init_inputs_fn)
            reporter.save_kernel_result(result)
            all_results.append(result)

            kernel_elapsed = time.time() - kernel_start
            logger.info(
                f"  Done in {kernel_elapsed:.0f}s: "
                f"score={result.mutation_score:.2%} "
                f"(killed={result.killed}, survived={result.survived}, stillborn={result.stillborn})"
            )

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

        if len(all_results) % 5 == 0:
            print_progress(all_results, time.time() - start_time, total)
            save_checkpoint(all_results, output_dir)

    total_elapsed = time.time() - start_time
    logger.info(f"\n{'#'*60}")
    logger.info(f"  EXPERIMENT COMPLETE")
    logger.info(f"  Total time: {total_elapsed/60:.1f} min")
    logger.info(f"{'#'*60}")

    print_progress(all_results, total_elapsed, total)
    save_checkpoint(all_results, output_dir)
    runner.cleanup()


if __name__ == "__main__":
    main()
