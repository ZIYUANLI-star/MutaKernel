#!/usr/bin/env python3
"""Phase 0 增强等价检测 — 第二次实验。

按 kernel 分组批处理：同一 kernel 的所有存活变异体在一个子进程中完成，
原始 kernel 只编译一次，大幅减少总编译开销。

核心修正: 旧版等价检测比较 PyTorch ref vs mutant，新版比较
original_kernel vs mutant_kernel（同类 GPU kernel，bitwise 有意义）。

Usage:
    python scripts/recheck_equiv.py [--max-mutants N]
"""
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIRST_EXP_DIR = PROJECT_ROOT / "第一次实验汇总" / "full_block12_results"
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
OUTPUT_DIR = PROJECT_ROOT / "第二次实验汇总"
BATCH_WORKER = SCRIPT_DIR / "_equiv_batch_worker.py"

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

EQUIV_RUNS = 100
BATCH_TIMEOUT = 3600
DEVICE = "cuda"


def P(msg):
    print(msg, flush=True)


def find_problem_file(problem_dir: Path, problem_id) -> Path | None:
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def _run_batch_worker(cfg: dict, timeout: int) -> list | None:
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="eqbcfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="eqbres_")
    os.close(cfg_fd)
    os.close(res_fd)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(BATCH_WORKER), cfg_path, res_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            if stdout:
                P(stdout.decode("utf-8", errors="replace").rstrip())
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            return None

        if os.path.exists(res_path) and os.path.getsize(res_path) > 2:
            with open(res_path) as f:
                return json.load(f)
        return None

    except Exception:
        return None

    finally:
        for p in [cfg_path, res_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def load_survived_grouped():
    """从第一次实验中加载存活变异体，按 kernel 分组。"""
    details_dir = FIRST_EXP_DIR / "details"
    if not details_dir.exists():
        P(f"ERROR: {details_dir} not found")
        return {}

    groups = defaultdict(list)
    for jf in sorted(details_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        kernel_meta = data["kernel"]
        kernel_name = kernel_meta["problem_name"]
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

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    groups = load_survived_grouped()
    total_mutants = sum(len(v) for v in groups.values())

    P(f"\n{'='*70}")
    P(f"  Phase 0 增强等价检测（第二次实验 — 批处理模式）")
    P(f"{'='*70}")
    P(f"  Kernel 数: {len(groups)}")
    P(f"  总存活变异体: {total_mutants}")
    P(f"  随机输入: {EQUIV_RUNS} 轮 + 压力策略 6x2=12 轮")
    P(f"  比较方式: original_kernel vs mutant_kernel (bitwise)")
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
    op_stats = defaultdict(lambda: {"equivalent": 0, "survived": 0})

    processed = 0
    for ki, (kernel_name, mutants) in enumerate(sorted(groups.items())):
        remaining = [m for m in mutants if m["id"] not in completed]
        if not remaining:
            P(f"  [{ki+1}/{len(groups)}] {kernel_name}: "
              f"{len(mutants)} mutants all done, skip")
            continue

        if args.max_mutants > 0 and processed >= args.max_mutants:
            break

        bk = best_kernels.get(kernel_name)
        if not bk:
            P(f"  [{ki+1}/{len(groups)}] {kernel_name}: not in best_kernels, skip")
            for m in remaining:
                all_results.append({"mutant_id": m["id"], "result": "error",
                                    "error": "not in best_kernels",
                                    "operator": m["operator_name"],
                                    "category": m["operator_category"]})
                stats["error"] += 1
            continue

        kernel_path = Path(bk["kernel_path"])
        if not kernel_path.exists():
            P(f"  [{ki+1}/{len(groups)}] {kernel_name}: kernel file missing")
            for m in remaining:
                all_results.append({"mutant_id": m["id"], "result": "error",
                                    "error": "kernel file missing",
                                    "operator": m["operator_name"],
                                    "category": m["operator_category"]})
                stats["error"] += 1
            continue

        level_key = bk["level"]
        problem_dir = PROBLEM_DIRS.get(level_key)
        if not problem_dir:
            P(f"  [{ki+1}/{len(groups)}] {kernel_name}: unknown level")
            continue

        problem_file = find_problem_file(problem_dir, bk["problem_id"])
        if not problem_file:
            P(f"  [{ki+1}/{len(groups)}] {kernel_name}: problem file not found")
            continue

        kernel_code = kernel_path.read_text(encoding="utf-8")

        mutant_cfgs = []
        for m in remaining:
            mc = m.get("mutated_code", "")
            if not mc:
                all_results.append({"mutant_id": m["id"], "result": "error",
                                    "error": "no mutated_code",
                                    "operator": m["operator_name"],
                                    "category": m["operator_category"]})
                stats["error"] += 1
                continue
            mutant_cfgs.append({
                "mutant_id": m["id"],
                "mutated_code": mc,
                "operator_name": m["operator_name"],
                "operator_category": m["operator_category"],
            })

        if not mutant_cfgs:
            continue

        P(f"\n  [{ki+1}/{len(groups)}] {kernel_name}: "
          f"{len(mutant_cfgs)} mutants in batch")

        batch_cfg = {
            "problem_file": str(problem_file),
            "kernel_code": kernel_code,
            "mutants": [{"mutant_id": mc["mutant_id"],
                         "mutated_code": mc["mutated_code"]}
                        for mc in mutant_cfgs],
            "device": DEVICE,
            "equiv_runs": EQUIV_RUNS,
            "base_seed": 10000,
        }

        timeout = max(BATCH_TIMEOUT, len(mutant_cfgs) * 600)
        batch_results = _run_batch_worker(batch_cfg, timeout=timeout)

        op_lookup = {mc["mutant_id"]: mc for mc in mutant_cfgs}

        if batch_results is None:
            P(f"    -> BATCH TIMEOUT/CRASH")
            for mc in mutant_cfgs:
                all_results.append({"mutant_id": mc["mutant_id"],
                                    "result": "timeout",
                                    "operator": mc["operator_name"],
                                    "category": mc["operator_category"]})
                stats["timeout"] += 1
        else:
            for br in batch_results:
                mid = br["mutant_id"]
                mc = op_lookup.get(mid, {})
                op = mc.get("operator_name", "")
                cat = mc.get("operator_category", "")

                if br.get("error"):
                    all_results.append({"mutant_id": mid, "result": "error",
                                        "error": br["error"][:200],
                                        "operator": op, "category": cat,
                                        "time_ms": br.get("time_ms", 0)})
                    stats["error"] += 1
                elif br.get("is_equivalent"):
                    all_results.append({"mutant_id": mid, "result": "equivalent",
                                        "operator": op, "category": cat,
                                        "time_ms": br.get("time_ms", 0)})
                    stats["equivalent"] += 1
                    op_stats[op]["equivalent"] += 1
                else:
                    all_results.append({"mutant_id": mid, "result": "survived",
                                        "operator": op, "category": cat,
                                        "time_ms": br.get("time_ms", 0)})
                    stats["survived"] += 1
                    op_stats[op]["survived"] += 1

                completed.add(mid)
                processed += 1

        with open(completed_file, "w") as f:
            json.dump(sorted(completed), f)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        eq_this = sum(1 for br in (batch_results or []) if br.get("is_equivalent"))
        sv_this = sum(1 for br in (batch_results or []) if not br.get("is_equivalent")
                      and not br.get("error"))
        P(f"    Batch done: {eq_this} equivalent, {sv_this} survived")

        gc.collect()

    elapsed = time.time() - t_start

    summary = {
        "total_checked": processed,
        "previously_survived": total_mutants,
        "newly_equivalent": stats["equivalent"],
        "truly_survived": stats["survived"],
        "error": stats["error"],
        "timeout": stats["timeout"],
        "equiv_runs": EQUIV_RUNS,
        "stress_policies": 6,
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
    P(f"  错误/超时: {stats['error']} / {stats['timeout']}")
    P(f"  耗时: {elapsed/60:.1f} 分钟")
    P(f"")
    P(f"  第一次实验等价: 234")
    P(f"  第二次新增等价: {stats['equivalent']}")
    P(f"  总等价变异体:  {234 + stats['equivalent']}")


if __name__ == "__main__":
    main()
