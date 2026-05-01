#!/usr/bin/env python3
"""Block 1 + Block 2 C/D 算子冒烟测试。

选取 C/D 类算子丰富的 kernel 验证完整流水线:
  - L2_P24: C1, C2, C4, C5, C7
  - L1_P41: C7, D1, D2
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

TARGET_KEYS = ["L2_P24", "L1_P41"]
MAX_PER_OP = 1
EQUIV_RUNS = 20
DEVICE = "cuda"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("smoke_cd")


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

    print(f"\n{'='*70}")
    print(f"  Block 1 + Block 2  C/D Operator Smoke Test")
    print(f"{'='*70}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Targets      : {TARGET_KEYS}")
    print(f"  Sample/op    : {MAX_PER_OP}")
    print(f"  Equiv runs   : {EQUIV_RUNS}")
    print(f"  atol/rtol    : 1e-2 / 1e-2")
    print()

    runner = MutantRunner(
        atol=1e-2, rtol=1e-2,
        num_test_inputs=3,
        device=DEVICE, seed=42,
        categories=["A", "B", "C", "D"],
    )
    detector = EquivalentDetector(num_runs=EQUIV_RUNS, device=DEVICE)
    report_dir = PROJECT_ROOT / "smoke_cd_results"
    reporter = MutationReporter(output_dir=report_dir)

    all_results: list[MutationTestResult] = []

    for ki, key in enumerate(TARGET_KEYS):
        if key not in best_kernels:
            print(f"  SKIP {key}: not in best_kernels.json")
            continue

        info = best_kernels[key]
        level_str = info["level"]
        pid = info["problem_id"]
        kpath = Path(info["kernel_path"])
        speedup = info.get("speedup", 0)
        turn = info.get("turn", "?")

        level_int = int(level_str[1]) if isinstance(level_str, str) else int(level_str)
        problem_dir = PROBLEM_DIRS.get(level_str)
        if not problem_dir:
            continue
        pfile = find_problem_file(problem_dir, pid)
        if pfile is None or not kpath.exists():
            log.warning(f"SKIP {key}: missing file(s)")
            continue

        source = kpath.read_text(encoding="utf-8", errors="replace")

        kernel = KernelInfo(
            problem_id=int(pid),
            level=level_int,
            problem_name=key,
            source_path=str(kpath),
            kernel_code=source,
            reference_module_path=str(pfile),
            language=detect_language(source),
        )

        ref_mod = _load_module_from_path(str(pfile), f"ref_{key}_{ki}")
        get_inputs = ref_mod.get_inputs
        get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])

        print(f"\n{'─'*70}")
        print(f"  [{ki+1}/{len(TARGET_KEYS)}] {key}  lang={kernel.language}  "
              f"turn={turn}  speedup={speedup:.3f}")
        print(f"{'─'*70}")

        # ===== Block 1: generate_mutants =====
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

        # 采样：C/D 类全部保留（每算子 MAX_PER_OP），A/B 类也采样 1 个作对照
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
            cat = m.operator_category
            tag = "\u2717 KILLED" if m.status == MutantStatus.KILLED else (
                "\u25cb SURVIVED" if m.status == MutantStatus.SURVIVED else
                "\u2726 STILLBORN"
            )
            err = ""
            if m.error_message and m.status == MutantStatus.KILLED:
                err = f"  [{m.error_message[:80]}]"
            print(f"    [{cat}] {m.operator_name:24s} L{m.site.line_start:>3d}  "
                  f"{tag}  {m.execution_time_ms:.0f}ms{err}")

        # ===== Block 2: EquivalentDetector =====
        survived = result.survived_mutants()
        if survived:
            print(f"\n  [Block2:Equiv] Checking {len(survived)} survived mutants "
                  f"({EQUIV_RUNS} bitwise runs)...")

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
                    print(f"    {m.operator_name:24s} \u2192 {m.status.value} "
                          f"({m.error_message[:60]})")
                else:
                    print(f"    {m.operator_name:24s} \u2192 truly survived")

            print(f"    Conservative score: {result.mutation_score:.2%}")
            print(f"    Optimistic score:   {result.mutation_score_optimistic:.2%}")
        else:
            print(f"\n  [Block2:Equiv] No survived mutants \u2014 skip")

        # ===== Block 2: Reporter =====
        reporter.save_kernel_result(result)
        all_results.append(result)

        stale = [k for k in sys.modules
                 if k.startswith("mutant_") or k.startswith("ref_") or k.startswith("eqchk_")]
        for k in stale:
            del sys.modules[k]
        gc.collect()
        torch.cuda.empty_cache()

        mem = torch.cuda.memory_allocated() / 1e6
        print(f"\n  [Cleanup] GPU mem: {mem:.0f} MB")

    # ===== Summary =====
    print(f"\n{'='*70}")
    print(f"  C/D SMOKE TEST SUMMARY")
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
    print(f"  {'Operator':<26} {'Kill':>5} {'Surv':>5} {'SB':>5} {'EQ':>5} {'Score':>8}")
    print(f"  {'-'*55}")
    for op in sorted(summary.get("by_operator", {})):
        s = summary["by_operator"][op]
        print(f"  {op:<26} {s['killed']:>5} {s['survived']:>5} "
              f"{s['stillborn']:>5} {s.get('equivalent',0):>5} {s['score']:>7.1%}")

    elapsed = time.time() - t_total
    print(f"\n  Reports: {report_dir}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    runner.cleanup()
    print(f"\n  C/D smoke test DONE.\n")


if __name__ == "__main__":
    main()
