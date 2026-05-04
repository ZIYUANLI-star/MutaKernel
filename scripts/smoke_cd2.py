#!/usr/bin/env python3
"""Block 1 + Block 2 C/D 算子冒烟测试 (第二轮)。

选取 C3 (EpsilonModify) 丰富的 kernel 以验证"难 kill"场景:
  - L1_P34: C2=8, C3=2, C4=2, C5=8
  - L1_P98: C2=2, C3=4, C5=2
  - L1_P29: C2=3, C5=5, D2=2
  - L2_P41: C2=4, C3=2, C4=3, C5=4

每个 C/D 算子采样 2 个变异体，A/B 采样 1 个作对照。
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

TARGET_KEYS = ["L1_P34", "L1_P98", "L1_P29", "L2_P41"]
MAX_PER_OP_CD = 2
MAX_PER_OP_AB = 1
EQUIV_RUNS = 20
DEVICE = "cuda"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("smoke_cd2")


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
    print(f"  Block 1 + Block 2  C/D Smoke Test (Round 2)")
    print(f"{'='*70}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Targets      : {TARGET_KEYS}")
    print(f"  C/D sample   : {MAX_PER_OP_CD}/op,  A/B sample: {MAX_PER_OP_AB}/op")
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
    report_dir = PROJECT_ROOT / "smoke_cd2_results"
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

        # Block 1: generate_mutants
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

        # C/D 算子采样 2 个，A/B 采样 1 个
        sampled: list[Mutant] = []
        for op_name, muts in by_op.items():
            cat = muts[0].operator_category if muts else "A"
            limit = MAX_PER_OP_CD if cat in ("C", "D") else MAX_PER_OP_AB
            sampled.extend(random.sample(muts, min(limit, len(muts))))
        print(f"  [Sample] {len(sampled)} mutants "
              f"(C/D: {MAX_PER_OP_CD}/op, A/B: {MAX_PER_OP_AB}/op)")

        # Block 2: run_all_mutants
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
            site_code = m.site.original_code[:35].replace("\n", " ")
            err = ""
            if m.error_message and m.status in (MutantStatus.KILLED, MutantStatus.STILLBORN):
                err = f"  [{m.error_message[:60]}]"
            print(f"    [{cat}] {m.operator_name:24s} L{m.site.line_start:>3d}  "
                  f"{tag}  {m.execution_time_ms:.0f}ms  '{site_code}'{err}")

        # Block 2: EquivalentDetector
        survived = result.survived_mutants()
        if survived:
            print(f"\n  [Block2:Equiv] Checking {len(survived)} survived mutants "
                  f"({EQUIV_RUNS} bitwise runs)...")

            try:
                run_original_fn = build_run_fn(ref_mod, "Model", get_init_inputs, DEVICE)
            except Exception as e:
                log.warning(f"    Cannot build ref runner: {e}")
                run_original_fn = None

            if run_original_fn:
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

            strict_cnt = cand_cnt = truly_cnt = 0
            for m in survived:
                if m.status == MutantStatus.STRICT_EQUIVALENT:
                    strict_cnt += 1
                    print(f"    [{m.operator_category}] {m.operator_name:24s} "
                          f"\u2192 STRICT_EQ  ({m.error_message[:60]})")
                elif m.status == MutantStatus.CANDIDATE_EQUIVALENT:
                    cand_cnt += 1
                    print(f"    [{m.operator_category}] {m.operator_name:24s} "
                          f"\u2192 CANDIDATE_EQ  ({m.error_message[:60]})")
                else:
                    truly_cnt += 1
                    print(f"    [{m.operator_category}] {m.operator_name:24s} "
                          f"\u2192 TRULY SURVIVED")

            print(f"    Summary: strict_eq={strict_cnt}, cand_eq={cand_cnt}, "
                  f"truly_survived={truly_cnt}")
            print(f"    Conservative score: {result.mutation_score:.2%}")
            print(f"    Optimistic score:   {result.mutation_score_optimistic:.2%}")
        else:
            print(f"\n  [Block2:Equiv] No survived mutants")

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
    print(f"  C/D SMOKE TEST ROUND 2 SUMMARY")
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

    print(f"\n  By Operator (C/D focus):")
    print(f"  {'Operator':<26} {'Cat':>3} {'Kill':>5} {'Surv':>5} {'SB':>5} {'EQ':>5} {'Score':>8}")
    print(f"  {'-'*58}")
    for op in sorted(summary.get("by_operator", {})):
        s = summary["by_operator"][op]
        from src.mutengine.operators.base import get_all_operators
        cat_map = {o.name: o.category for o in get_all_operators()}
        cat = cat_map.get(op, "?")
        marker = " <<" if cat in ("C", "D") else ""
        print(f"  {op:<26} {cat:>3} {s['killed']:>5} {s['survived']:>5} "
              f"{s['stillborn']:>5} {s.get('equivalent',0):>5} {s['score']:>7.1%}{marker}")

    print(f"\n  By Kernel:")
    print(f"  {'Kernel':<12} {'Kill':>5} {'Surv':>5} {'SB':>5} {'EQ':>5} {'Score':>8}")
    print(f"  {'-'*45}")
    for k in summary.get("by_kernel", []):
        print(f"  {k['kernel']:<12} {k['killed']:>5} {k['survived']:>5} "
              f"{k['stillborn']:>5} {k['equivalent']:>5} {k['score']:>7.1%}")

    elapsed = time.time() - t_total
    print(f"\n  Reports: {report_dir}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    runner.cleanup()
    print(f"\n  C/D smoke test round 2 DONE.\n")


if __name__ == "__main__":
    main()
