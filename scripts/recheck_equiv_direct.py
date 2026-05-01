#!/usr/bin/env python3
"""Phase 0 增强等价检测 — 单进程直接运行版。

直接在进程内编译和比较 original_kernel vs mutant_kernel，
按 kernel 分组使原始 kernel 只编译一次。

Usage:
    python scripts/recheck_equiv_direct.py [--max-mutants N]
"""
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.mutengine.mutant_runner import _load_module_from_source, CompilationError
from src.bridge.eval_bridge import _load_module_from_path
from src.stress.policy_bank import STRESS_POLICIES

FIRST_EXP_DIR = PROJECT_ROOT / "第一次实验汇总" / "full_block12_results"
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
OUTPUT_DIR = PROJECT_ROOT / "第二次实验汇总"

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

EQUIV_RUNS = 100
DEVICE = "cuda"

EQUIV_STRESS_POLICIES = [
    "large_magnitude", "near_zero", "structured_ramp",
    "all_negative", "sparse", "boundary_last_element",
]


def P(msg):
    print(msg, flush=True)


def find_problem_file(problem_dir, problem_id):
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def _bitwise_identical(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return torch.equal(a, b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_bitwise_identical(x, y) for x, y in zip(a, b))
    return a == b


MUTANT_TIMEOUT = 60
WARMUP_TIMEOUT = 15

_thread_pool = ThreadPoolExecutor(max_workers=1)


def _run_forward(model, inputs):
    with torch.no_grad():
        out = model(*inputs)
    torch.cuda.synchronize()
    return out


def _warmup_ok(model, inputs, timeout=WARMUP_TIMEOUT):
    """Run a single forward pass in a thread; return True if it completes within timeout."""
    future = _thread_pool.submit(_run_forward, model, inputs)
    try:
        future.result(timeout=timeout)
        return True
    except FutureTimeout:
        return False
    except Exception:
        return False


def check_equiv(orig_model, mut_model, get_inputs, base_seed, equiv_runs, device):
    """Run bitwise comparison: random inputs + stress policies.
    Uses a warmup with thread-pool timeout to detect slow GPU kernels.
    """
    torch.manual_seed(base_seed)
    warmup_inputs = get_inputs()
    moved = [x.to(device) if isinstance(x, torch.Tensor) else x
             for x in warmup_inputs]
    if not _warmup_ok(mut_model, moved, WARMUP_TIMEOUT):
        del moved, warmup_inputs
        return None, 0, "slow_kernel"

    del moved, warmup_inputs

    t_start = time.time()
    for i in range(equiv_runs):
        if time.time() - t_start > MUTANT_TIMEOUT:
            return None, i, "timeout"
        seed = base_seed + i
        torch.manual_seed(seed)
        inputs = get_inputs()
        moved = [x.to(device) if isinstance(x, torch.Tensor) else x
                 for x in inputs]
        with torch.no_grad():
            orig_out = orig_model(*moved)
            mut_out = mut_model(*moved)
        torch.cuda.synchronize()
        if not _bitwise_identical(orig_out, mut_out):
            return False, i, "random"
        del orig_out, mut_out, moved, inputs

    for policy_name in EQUIV_STRESS_POLICIES:
        if time.time() - t_start > MUTANT_TIMEOUT:
            return None, -1, "timeout"
        policy_fn = STRESS_POLICIES.get(policy_name)
        if policy_fn is None:
            continue
        for si in range(2):
            seed = base_seed + equiv_runs + si
            torch.manual_seed(seed)
            template = get_inputs()
            stress_inputs = policy_fn(template, seed)
            moved = [x.to(device) if isinstance(x, torch.Tensor) else x
                     for x in stress_inputs]
            try:
                with torch.no_grad():
                    orig_out = orig_model(*moved)
                    mut_out = mut_model(*moved)
                torch.cuda.synchronize()
                if not _bitwise_identical(orig_out, mut_out):
                    return False, -1, policy_name
            except Exception:
                pass
            finally:
                del moved, stress_inputs, template

    return True, -1, ""


def load_survived_grouped():
    details_dir = FIRST_EXP_DIR / "details"
    groups = defaultdict(list)
    for jf in sorted(details_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        kernel_name = data["kernel"]["problem_name"]
        for m in data.get("mutants", []):
            if m["status"] == "survived":
                groups[kernel_name].append(m)
    return dict(groups)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mutants", type=int, default=0)
    args = parser.parse_args()

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="equiv_direct_")

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    groups = load_survived_grouped()
    total_mutants = sum(len(v) for v in groups.values())

    P(f"\n{'='*70}")
    P(f"  Phase 0 增强等价检测（第二次实验）")
    P(f"{'='*70}")
    P(f"  Kernel 数: {len(groups)}")
    P(f"  总存活变异体: {total_mutants}")
    P(f"  比较: original_kernel vs mutant_kernel (bitwise)")
    P(f"  随机: {EQUIV_RUNS} 轮 + 压力策略 {len(EQUIV_STRESS_POLICIES)}x2 轮")
    P(f"  GPU: {torch.cuda.get_device_name(0)}")
    P(f"{'='*70}\n")

    completed_file = OUTPUT_DIR / "equiv_recheck_completed.json"
    completed = set()
    if completed_file.exists():
        completed = set(json.loads(completed_file.read_text()))

    all_results = []
    results_file = OUTPUT_DIR / "equiv_recheck_results.json"
    if results_file.exists():
        try:
            all_results = json.loads(results_file.read_text(encoding="utf-8"))
        except Exception:
            all_results = []

    stats = defaultdict(int)
    op_stats = defaultdict(lambda: {"equivalent": 0, "survived": 0, "timeout": 0})
    processed = 0

    for ki, (kernel_name, mutants) in enumerate(sorted(groups.items())):
        remaining = [m for m in mutants if m["id"] not in completed]
        if not remaining:
            continue

        if args.max_mutants > 0 and processed >= args.max_mutants:
            break

        bk = best_kernels.get(kernel_name)
        if not bk:
            P(f"  [{ki+1}/{len(groups)}] {kernel_name}: skip (not in best_kernels)")
            continue

        kernel_path = Path(bk["kernel_path"])
        if not kernel_path.exists():
            P(f"  [{ki+1}/{len(groups)}] {kernel_name}: skip (kernel missing)")
            continue

        level_key = bk["level"]
        problem_dir = PROBLEM_DIRS.get(level_key)
        if not problem_dir:
            continue

        problem_file = find_problem_file(problem_dir, bk["problem_id"])
        if not problem_file:
            P(f"  [{ki+1}/{len(groups)}] {kernel_name}: skip (problem missing)")
            continue

        kernel_code = kernel_path.read_text(encoding="utf-8")

        P(f"\n  [{ki+1}/{len(groups)}] {kernel_name}: "
          f"{len(remaining)} mutants")

        ref_mod = _load_module_from_path(str(problem_file), f"ref_eq_{kernel_name}")
        get_inputs = ref_mod.get_inputs
        get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
        init_args = get_init_inputs()

        t_orig = time.time()
        orig_hash = hashlib.md5(kernel_code.encode()).hexdigest()[:10]
        try:
            orig_mod = _load_module_from_source(
                kernel_code, f"eqo_{orig_hash}", tmp_dir)
        except Exception as e:
            P(f"    Original compile FAILED: {str(e)[:100]}")
            for m in remaining:
                all_results.append({"mutant_id": m["id"], "result": "error",
                                    "error": f"orig_compile: {str(e)[:100]}",
                                    "operator": m["operator_name"],
                                    "category": m["operator_category"]})
                stats["error"] += 1
                completed.add(m["id"])
            continue

        orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
        orig_model = (orig_cls(*init_args) if isinstance(init_args, (list, tuple))
                      else orig_cls())
        orig_model = orig_model.to(DEVICE).eval()
        P(f"    Original compiled ({time.time()-t_orig:.1f}s)")

        for mi, m in enumerate(remaining):
            if args.max_mutants > 0 and processed >= args.max_mutants:
                break

            mutant_id = m["id"]
            op_name = m["operator_name"]
            op_cat = m["operator_category"]
            mutated_code = m.get("mutated_code", "")

            if not mutated_code:
                all_results.append({"mutant_id": mutant_id, "result": "error",
                                    "error": "no mutated_code",
                                    "operator": op_name, "category": op_cat})
                stats["error"] += 1
                completed.add(mutant_id)
                processed += 1
                continue

            t0 = time.time()
            safe_id = mutant_id.replace("-", "_").replace(".", "_")

            try:
                mut_mod = _load_module_from_source(
                    mutated_code, f"eqm_{safe_id}", tmp_dir)
            except Exception as e:
                P(f"    [{mi+1}/{len(remaining)}] {mutant_id}: "
                  f"COMPILE_ERROR ({time.time()-t0:.1f}s)")
                all_results.append({"mutant_id": mutant_id, "result": "error",
                                    "error": f"mut_compile: {str(e)[:100]}",
                                    "operator": op_name, "category": op_cat})
                stats["error"] += 1
                completed.add(mutant_id)
                processed += 1
                continue

            mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
            mut_model = (mut_cls(*init_args) if isinstance(init_args, (list, tuple))
                         else mut_cls())
            mut_model = mut_model.to(DEVICE).eval()

            compile_time = time.time() - t0
            t_check = time.time()

            try:
                is_equiv, fail_iter, fail_policy = check_equiv(
                    orig_model, mut_model, get_inputs, 10000, EQUIV_RUNS, DEVICE)
            except Exception as e:
                P(f"    [{mi+1}/{len(remaining)}] {mutant_id}: "
                  f"CHECK_ERROR: {str(e)[:80]}")
                all_results.append({"mutant_id": mutant_id, "result": "error",
                                    "error": str(e)[:200],
                                    "operator": op_name, "category": op_cat})
                stats["error"] += 1
                completed.add(mutant_id)
                processed += 1
                del mut_model
                torch.cuda.empty_cache()
                continue

            check_time = time.time() - t_check
            total_time = time.time() - t0

            if is_equiv is None:
                tag = f"TIMEOUT ({MUTANT_TIMEOUT}s)"
                all_results.append({"mutant_id": mutant_id, "result": "timeout",
                                    "operator": op_name, "category": op_cat,
                                    "time_ms": total_time * 1000})
                stats["timeout"] += 1
            elif is_equiv:
                tag = "EQUIVALENT"
                all_results.append({"mutant_id": mutant_id, "result": "equivalent",
                                    "operator": op_name, "category": op_cat,
                                    "time_ms": total_time * 1000})
                stats["equivalent"] += 1
                op_stats[op_name]["equivalent"] += 1
            else:
                tag = f"SURVIVED (diverged at iter {fail_iter}, policy={fail_policy})"
                all_results.append({"mutant_id": mutant_id, "result": "survived",
                                    "operator": op_name, "category": op_cat,
                                    "diverge_iter": fail_iter,
                                    "diverge_policy": fail_policy,
                                    "time_ms": total_time * 1000})
                stats["survived"] += 1
                op_stats[op_name]["survived"] += 1

            P(f"    [{mi+1}/{len(remaining)}] {mutant_id}: {tag} "
              f"(compile={compile_time:.1f}s check={check_time:.1f}s)")

            completed.add(mutant_id)
            processed += 1

            del mut_model
            torch.cuda.empty_cache()

        del orig_model
        torch.cuda.empty_cache()
        gc.collect()

        with open(completed_file, "w") as f:
            json.dump(sorted(completed), f)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        eq_ct = sum(1 for r in all_results
                    if r.get("result") == "equivalent"
                    and r["mutant_id"] in {m["id"] for m in remaining})
        sv_ct = sum(1 for r in all_results
                    if r.get("result") == "survived"
                    and r["mutant_id"] in {m["id"] for m in remaining})
        P(f"    Kernel total: {eq_ct} equivalent, {sv_ct} survived")

    elapsed = time.time() - t_start

    summary = {
        "total_checked": processed,
        "previously_survived": total_mutants,
        "newly_equivalent": stats["equivalent"],
        "truly_survived": stats["survived"],
        "error": stats["error"],
        "equiv_runs": EQUIV_RUNS,
        "stress_policies": len(EQUIV_STRESS_POLICIES),
        "comparison": "original_kernel vs mutant_kernel (bitwise)",
        "elapsed_min": elapsed / 60,
        "by_operator": {k: dict(v) for k, v in op_stats.items()},
    }

    summary_file = OUTPUT_DIR / "equiv_recheck_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    P(f"\n{'='*70}")
    P(f"  增强等价检测完成")
    P(f"{'='*70}")
    P(f"  已检测: {processed}")
    P(f"  新发现等价: {stats['equivalent']}")
    P(f"  真正存活: {stats['survived']}")
    P(f"  超时: {stats['timeout']}")
    P(f"  错误: {stats['error']}")
    P(f"  耗时: {elapsed/60:.1f} 分钟")
    P(f"")
    P(f"  第一次实验等价: 234")
    P(f"  第二次新增等价: {stats['equivalent']}")
    P(f"  总等价变异体:  {234 + stats['equivalent']}")


if __name__ == "__main__":
    main()
