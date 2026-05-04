"""RQ1 / RQ3 实验脚本：对 KernelBench 正确 kernel 运行变异测试。

用法:
    python scripts/run_mutation_test.py --levels 1 2 --categories A B C D
    python scripts/run_mutation_test.py --levels 1 --problem-ids 1 5 10
    python scripts/run_mutation_test.py --config rq3  # RQ3 多配置对比
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    KERNELBENCH_ROOT, RESULTS_DIR,
    MutationTestConfig, ExperimentConfig,
)
from src.models import KernelInfo, MutationTestResult
from src.bridge.eval_bridge import KernelBenchBridge
from src.mutengine.mutant_runner import MutantRunner
from src.mutengine.report import MutationReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_mutation_test")

RQ3_CONFIGS = {
    "config1": MutationTestConfig(atol=1e-2, rtol=1e-2, num_test_inputs=5),
    "config2": MutationTestConfig(atol=1e-4, rtol=1e-4, num_test_inputs=5),
    "config3": MutationTestConfig(atol=1e-2, rtol=1e-2, num_test_inputs=100),
    "config4": MutationTestConfig(atol=1e-2, rtol=1e-2, num_test_inputs=5),
}


def run_single_kernel(
    kernel: KernelInfo,
    bridge: KernelBenchBridge,
    runner: MutantRunner,
) -> MutationTestResult:
    """对单个 kernel 运行变异测试。"""
    logger.info(f"=== Processing {kernel} ===")

    mutants = runner.generate_mutants(kernel)
    if not mutants:
        logger.warning(f"No mutants generated for {kernel}")
        return MutationTestResult(kernel=kernel, mutants=[])

    ref_module, get_inputs_fn, get_init_inputs_fn = bridge.load_runtime_components(kernel)

    result = runner.run_all_mutants(
        kernel, mutants, ref_module, get_inputs_fn, get_init_inputs_fn
    )
    return result


def run_rq1(args: argparse.Namespace) -> None:
    """RQ1: 对所有正确 kernel 运行变异测试。"""
    bridge = KernelBenchBridge(args.kernelbench_root)
    config = MutationTestConfig(
        categories=args.categories,
        seed=args.seed,
    )
    runner = MutantRunner(
        atol=config.atol,
        rtol=config.rtol,
        num_test_inputs=config.num_test_inputs,
        device=args.device,
        seed=config.seed,
        categories=config.categories,
    )

    output_dir = Path(args.output_dir) / f"rq1_{int(time.time())}"
    reporter = MutationReporter(output_dir)

    all_results: list[MutationTestResult] = []

    for level in args.levels:
        kernels = bridge.load_all_correct_kernels(levels=[level])
        if args.problem_ids:
            kernels = [k for k in kernels if k.problem_id in args.problem_ids]

        logger.info(f"Level {level}: {len(kernels)} kernels to test")

        for kernel in kernels:
            try:
                result = run_single_kernel(kernel, bridge, runner)
                reporter.save_kernel_result(result)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed on {kernel}: {e}", exc_info=True)

    reporter.save_summary(all_results)
    runner.cleanup()
    logger.info(f"Results saved to {output_dir}")


def run_rq3(args: argparse.Namespace) -> None:
    """RQ3: 不同测试配置对比。"""
    bridge = KernelBenchBridge(args.kernelbench_root)

    for config_name, config in RQ3_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"RQ3 Config: {config_name}")
        logger.info(f"  atol={config.atol}, rtol={config.rtol}, num_inputs={config.num_test_inputs}")
        logger.info(f"{'='*60}")

        runner = MutantRunner(
            atol=config.atol,
            rtol=config.rtol,
            num_test_inputs=config.num_test_inputs,
            device=args.device,
            seed=config.seed,
            categories=args.categories,
        )

        output_dir = Path(args.output_dir) / f"rq3_{config_name}_{int(time.time())}"
        reporter = MutationReporter(output_dir)

        all_results = []
        for level in args.levels:
            kernels = bridge.load_all_correct_kernels(levels=[level])
            if args.problem_ids:
                kernels = [k for k in kernels if k.problem_id in args.problem_ids]

            for kernel in kernels:
                try:
                    result = run_single_kernel(kernel, bridge, runner)
                    reporter.save_kernel_result(result)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Failed on {kernel}: {e}")

        reporter.save_summary(all_results)
        runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="MutaKernel Mutation Testing")
    parser.add_argument("--config", choices=["rq1", "rq3"], default="rq1")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--categories", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--problem-ids", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--kernelbench-root", default=str(KERNELBENCH_ROOT))

    args = parser.parse_args()

    if args.config == "rq1":
        run_rq1(args)
    elif args.config == "rq3":
        run_rq3(args)


if __name__ == "__main__":
    main()
