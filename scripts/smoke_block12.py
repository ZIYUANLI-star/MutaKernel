#!/usr/bin/env python3
"""Block 1 + Block 2 联合冒烟测试。

完整调用框架模块，验证 MutOperators → MutEngine 流水线:
  Block 1: MutantRunner.generate_mutants() → 16 个算子 find_sites + apply
  Block 2: MutantRunner.run_all_mutants() → JIT 编译 + 执行 + allclose 判定
           EquivalentDetector → 语法归一化 + 统计 bitwise 检测
           MutationReporter → JSON + Markdown 报告

选取 3 个 kernel，每个算子采样 1 个变异体，等价检测缩减为 20 轮。
"""
import gc
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models import KernelInfo, Mutant, MutantStatus, MutationTestResult
from src.mutengine.mutant_runner import (
    MutantRunner,
    _load_module_from_source,
    _run_model,
)
from src.mutengine.equivalent_detector import EquivalentDetector
from src.mutengine.report import MutationReporter
from src.bridge.eval_bridge import _load_module_from_path

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

SMOKE_KERNELS = 3
MAX_PER_OP = 1
EQUIV_RUNS = 20
DEVICE = "cuda"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("smoke_block12")


def find_problem_file(problem_dir: Path, problem_id) -> Path | None:
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def detect_language(source: str) -> str:
    indicators = ["__global__", "__device__", "load_inline", "cuda_source"]
    return "cuda" if sum(1 for ind in indicators if ind in source) >= 2 else "triton"


def build_run_fn(module, class_name, get_init_inputs_fn, device):
    """Build a callable(inputs)->output for equivalence detection."""
    cls = getattr(module, class_name, None)
    if cls is None and class_name == "ModelNew":
        cls = getattr(module, "Model", None)
    if cls is None:
        raise AttributeError(f"No class {class_name}")
    init_args = get_init_inputs_fn()
    model = cls(*init_args) if isinstance(init_args, (list, tuple)) else cls()
    model = model.to(device).eval()

    def _run(inputs):
        moved = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        with torch.no_grad():
            return model(*moved)

    return _run


def main():
    random.seed(42)
    t_total = time.time()

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    keys = sorted(best_kernels.keys())[:SMOKE_KERNELS]

    print(f"\n{'='*70}")
    print(f"  Block 1 + Block 2  Smoke Test")
    print(f"{'='*70}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Kernels      : {len(keys)}")
    print(f"  Sample/op    : {MAX_PER_OP}")
    print(f"  Equiv runs   : {EQUIV_RUNS}")
    print(f"  atol/rtol    : 1e-2 / 1e-2")
    print()

    # ===== Block 2: 初始化框架组件 =====
    runner = MutantRunner(
        atol=1e-2, rtol=1e-2,
        num_test_inputs=3,
        device=DEVICE, seed=42,
        categories=["A", "B", "C", "D"],
    )
    detector = EquivalentDetector(num_runs=EQUIV_RUNS, device=DEVICE)
    report_dir = PROJECT_ROOT / "smoke_results"
    reporter = MutationReporter(output_dir=report_dir)

    all_results: list[MutationTestResult] = []

    for ki, key in enumerate(keys):
        info = best_kernels[key]
        level_str = info["level"]
        pid = info["problem_id"]
        kpath = Path(info["kernel_path"])
        speedup = info.get("speedup", 0)
        turn = info.get("turn", "?")

        level_int = int(level_str[1]) if isinstance(level_str, str) else int(level_str)
        problem_dir = PROBLEM_DIRS.get(level_str)
        if not problem_dir:
            log.warning(f"SKIP {key}: unknown level {level_str}")
            continue
        pfile = find_problem_file(problem_dir, pid)
        if pfile is None or not kpath.exists():
            log.warning(f"SKIP {key}: missing file(s)")
            continue

        source = kpath.read_text(encoding="utf-8", errors="replace")

        # Block 2 数据模型: KernelInfo
        kernel = KernelInfo(
            problem_id=int(pid),
            level=level_int,
            problem_name=key,
            source_path=str(kpath),
            kernel_code=source,
            reference_module_path=str(pfile),
            language=detect_language(source),
        )

        # Block 2 eval_bridge 工具: 加载参考模块
        ref_mod = _load_module_from_path(str(pfile), f"ref_{key}_{ki}")
        get_inputs = ref_mod.get_inputs
        get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])

        print(f"\n{'─'*70}")
        print(f"  [{ki+1}/{len(keys)}] {key}  lang={kernel.language}  "
              f"turn={turn}  speedup={speedup:.3f}")
        print(f"{'─'*70}")

        # ===== Block 1 (via Block 2): generate_mutants =====
        t0 = time.time()
        all_mutants = runner.generate_mutants(kernel)
        gen_s = time.time() - t0
        print(f"  [Block1] Generated {len(all_mutants)} mutants ({gen_s:.1f}s)")

        by_cat: dict[str, int] = defaultdict(int)
        by_op: dict[str, list[Mutant]] = defaultdict(list)
        for m in all_mutants:
            by_cat[m.operator_category] += 1
            by_op[m.operator_name].append(m)

        for cat in sorted(by_cat):
            ops_in_cat = [
                f"{op}={len(ms)}"
                for op, ms in sorted(by_op.items())
                if ms and ms[0].operator_category == cat
            ]
            print(f"    {cat}: {by_cat[cat]} total  ({', '.join(ops_in_cat)})")

        # 采样
        sampled: list[Mutant] = []
        for op_name, muts in by_op.items():
            sampled.extend(random.sample(muts, min(MAX_PER_OP, len(muts))))
        print(f"  [Sample] {len(sampled)} mutants ({MAX_PER_OP}/op)")

        # ===== Block 2: run_all_mutants =====
        t0 = time.time()
        result = runner.run_all_mutants(
            kernel, sampled, ref_mod, get_inputs, get_init_inputs,
        )
        run_s = time.time() - t0

        print(f"\n  [Block2:Runner] Completed in {run_s:.1f}s")
        print(f"    killed={result.killed}  survived={result.survived}  "
              f"stillborn={result.stillborn}  score={result.mutation_score:.2%}")

        for m in sampled:
            tag = "✗ KILLED" if m.status == MutantStatus.KILLED else (
                "○ SURVIVED" if m.status == MutantStatus.SURVIVED else
                "✦ STILLBORN"
            )
            print(f"    {m.operator_name:24s} L{m.site.line_start:>3d}  "
                  f"{tag}  {m.execution_time_ms:.0f}ms")

        # ===== Block 2: EquivalentDetector =====
        survived = result.survived_mutants()
        if survived:
            print(f"\n  [Block2:Equiv] Checking {len(survived)} survived mutants...")

            run_original_fn = build_run_fn(ref_mod, "Model", get_init_inputs, DEVICE)

            run_mutant_fns: dict[str, any] = {}
            for m in survived:
                try:
                    mut_mod = _load_module_from_source(
                        m.mutated_code,
                        f"eqchk_{m.id.replace('-','_').replace('.','_')}",
                        runner._tmp_dir,
                    )
                    run_mutant_fns[m.id] = build_run_fn(
                        mut_mod, "ModelNew", get_init_inputs, DEVICE,
                    )
                except Exception as e:
                    log.warning(f"    Cannot recompile {m.id}: {e}")

            detector.classify_survived_mutants(
                sampled, run_original_fn, run_mutant_fns, get_inputs,
            )

            for m in survived:
                if m.status.is_equivalent:
                    print(f"    {m.operator_name:24s} → {m.status.value} ({m.error_message})")
                else:
                    print(f"    {m.operator_name:24s} → truly survived")

            print(f"    Conservative score: {result.mutation_score:.2%}")
            print(f"    Optimistic score:   {result.mutation_score_optimistic:.2%}")
        else:
            print(f"\n  [Block2:Equiv] No survived mutants — skip")

        # ===== Block 2: Reporter =====
        reporter.save_kernel_result(result)
        all_results.append(result)

        # GPU cleanup
        stale = [k for k in sys.modules
                 if k.startswith("mutant_") or k.startswith("ref_") or k.startswith("eqchk_")]
        for k in stale:
            del sys.modules[k]
        gc.collect()
        torch.cuda.empty_cache()

        mem = torch.cuda.memory_allocated() / 1e6
        print(f"\n  [Cleanup] GPU mem: {mem:.0f} MB")

    # ===== Block 2: Summary report =====
    print(f"\n{'='*70}")
    print(f"  SUMMARY  (Block 2: MutationReporter)")
    print(f"{'='*70}")

    summary_path = reporter.save_summary(all_results)
    summary = reporter.generate_summary(all_results)

    print(f"  Overall Mutation Score : {summary['overall_mutation_score']:.2%}")
    print(f"  Total mutants         : {summary['total_mutants']}")
    print(f"  Killed                : {summary['total_killed']}")
    print(f"  Survived              : {summary['total_survived']}")
    print(f"  Stillborn             : {summary['total_stillborn']}")
    print(f"  Equivalent            : {summary['total_equivalent']}")

    print(f"\n  By Category:")
    print(f"  {'Cat':<5} {'Name':<30} {'Kill':>5} {'Surv':>5} {'SB':>5} {'EQ':>5} {'Score':>8}")
    print(f"  {'-'*63}")
    for cat in sorted(summary.get("by_category", {})):
        s = summary["by_category"][cat]
        print(f"  {cat:<5} {s['name']:<30} {s['killed']:>5} {s['survived']:>5} "
              f"{s['stillborn']:>5} {s['equivalent']:>5} {s['score']:>7.1%}")

    print(f"\n  By Operator:")
    print(f"  {'Operator':<26} {'Kill':>5} {'Surv':>5} {'SB':>5} {'Score':>8}")
    print(f"  {'-'*49}")
    for op in sorted(summary.get("by_operator", {})):
        s = summary["by_operator"][op]
        print(f"  {op:<26} {s['killed']:>5} {s['survived']:>5} "
              f"{s['stillborn']:>5} {s['score']:>7.1%}")

    elapsed = time.time() - t_total
    print(f"\n  Reports: {report_dir}")
    print(f"  Summary: {summary_path}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    runner.cleanup()
    print(f"\n  Smoke test DONE.\n")


if __name__ == "__main__":
    main()
