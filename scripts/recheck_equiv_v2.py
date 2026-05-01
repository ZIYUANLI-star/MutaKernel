#!/usr/bin/env python3
"""Phase 0 增强等价检测 — 子进程隔离版 (v2)。

每个变异体在独立子进程中检测，主进程用 subprocess.communicate(timeout)
进行硬超时。这是唯一能可靠处理 CUDA kernel 挂起的方案。

核心修正: 旧版比较 PyTorch ref vs mutant，新版比较
original_kernel vs mutant_kernel (bitwise)。

Usage:
    python scripts/recheck_equiv_v2.py [--max-mutants N] [--timeout 60]
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIRST_EXP_DIR = PROJECT_ROOT / "第一次实验汇总" / "full_block12_results"
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
OUTPUT_DIR = PROJECT_ROOT / "第二次实验汇总"
WORKER = SCRIPT_DIR / "_equiv_check_one.py"

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

EQUIV_RUNS = 100
DEVICE = "cuda"
DEFAULT_TIMEOUT = 600


def P(msg):
    print(msg, flush=True)


def find_problem_file(problem_dir, problem_id):
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def run_one_check(cfg: dict, timeout: int) -> dict:
    """在子进程中运行单个变异体的等价检测，有硬超时保护。"""
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="eqcfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="eqres_")
    os.close(cfg_fd)
    os.close(res_fd)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(WORKER), cfg_path, res_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            return {"is_equivalent": False, "error": f"timeout({timeout}s)"}

        if os.path.exists(res_path) and os.path.getsize(res_path) > 2:
            with open(res_path) as f:
                return json.load(f)
        return {"is_equivalent": False, "error": "no_result_file"}

    except Exception as e:
        return {"is_equivalent": False, "error": str(e)[:200]}
    finally:
        for p in [cfg_path, res_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def load_survived_grouped():
    """从第一次实验中加载存活变异体，按 kernel 分组。"""
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
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    groups = load_survived_grouped()
    total_mutants = sum(len(v) for v in groups.values())

    P(f"\n{'='*70}")
    P(f"  Phase 0 增强等价检测（第二次实验 v2 — 子进程隔离）")
    P(f"{'='*70}")
    P(f"  Kernel 数: {len(groups)}")
    P(f"  总存活变异体: {total_mutants}")
    P(f"  随机: {EQUIV_RUNS} 轮 + 压力策略 6x2=12 轮")
    P(f"  比较: original_kernel vs mutant_kernel (bitwise)")
    P(f"  超时: {args.timeout}s / 变异体")
    P(f"{'='*70}\n")

    completed_file = OUTPUT_DIR / "equiv_recheck_completed.json"
    completed = set()
    if completed_file.exists():
        try:
            completed = set(json.loads(completed_file.read_text()))
        except Exception:
            completed = set()

    all_results = []
    results_file = OUTPUT_DIR / "equiv_recheck_results.json"
    if results_file.exists():
        try:
            all_results = json.loads(results_file.read_text(encoding="utf-8"))
        except Exception:
            all_results = []

    stats = defaultdict(int)
    op_stats = defaultdict(lambda: {"candidate_equivalent": 0, "survived": 0, "timeout": 0, "error": 0})
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
            for m in remaining:
                all_results.append({"mutant_id": m["id"], "result": "error",
                                    "error": "not in best_kernels",
                                    "operator": m["operator_name"],
                                    "category": m["operator_category"]})
                stats["error"] += 1
                completed.add(m["id"])
            _save(completed_file, completed, results_file, all_results)
            continue

        kernel_path = Path(bk["kernel_path"])
        if not kernel_path.exists():
            P(f"  [{ki+1}/{len(groups)}] {kernel_name}: skip (kernel missing)")
            for m in remaining:
                all_results.append({"mutant_id": m["id"], "result": "error",
                                    "error": "kernel file missing",
                                    "operator": m["operator_name"],
                                    "category": m["operator_category"]})
                stats["error"] += 1
                completed.add(m["id"])
            _save(completed_file, completed, results_file, all_results)
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

        P(f"\n  [{ki+1}/{len(groups)}] {kernel_name}: {len(remaining)} mutants")

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
                op_stats[op_name]["error"] += 1
                completed.add(mutant_id)
                processed += 1
                continue

            t0 = time.time()
            cfg = {
                "problem_file": str(problem_file),
                "kernel_code": kernel_code,
                "mutant_id": mutant_id,
                "mutated_code": mutated_code,
                "device": DEVICE,
                "equiv_runs": EQUIV_RUNS,
                "base_seed": 10000,
            }

            r = run_one_check(cfg, timeout=args.timeout)
            elapsed = time.time() - t0

            if r.get("error"):
                tag = f"ERROR: {r['error'][:60]}"
                result_type = "timeout" if "timeout" in r["error"] else "error"
                all_results.append({"mutant_id": mutant_id, "result": result_type,
                                    "error": r["error"][:200],
                                    "operator": op_name, "category": op_cat,
                                    "time_ms": elapsed * 1000})
                stats[result_type] += 1
                op_stats[op_name][result_type] += 1
            elif r.get("is_equivalent"):
                tag = "CANDIDATE_EQUIVALENT"
                all_results.append({"mutant_id": mutant_id, "result": "candidate_equivalent",
                                    "operator": op_name, "category": op_cat,
                                    "time_ms": r.get("time_ms", elapsed * 1000),
                                    "compile_ms": r.get("compile_ms", 0)})
                stats["candidate_equivalent"] += 1
                op_stats[op_name]["candidate_equivalent"] += 1
            else:
                div_info = ""
                if "diverged_at" in r:
                    div_info = f" @iter={r['diverged_at']} policy={r.get('diverged_policy','')}"
                tag = f"SURVIVED{div_info}"
                all_results.append({"mutant_id": mutant_id, "result": "survived",
                                    "operator": op_name, "category": op_cat,
                                    "diverged_at": r.get("diverged_at"),
                                    "diverged_policy": r.get("diverged_policy", ""),
                                    "time_ms": r.get("time_ms", elapsed * 1000)})
                stats["survived"] += 1
                op_stats[op_name]["survived"] += 1

            P(f"    [{mi+1}/{len(remaining)}] {mutant_id}: {tag} ({elapsed:.1f}s)")

            completed.add(mutant_id)
            processed += 1

            if processed % 5 == 0 or mi == len(remaining) - 1:
                _save(completed_file, completed, results_file, all_results)

    _save(completed_file, completed, results_file, all_results)

    elapsed_total = time.time() - t_start
    summary = {
        "total_checked": processed,
        "previously_survived": total_mutants,
        "newly_candidate_equivalent": stats["candidate_equivalent"],
        "truly_survived": stats["survived"],
        "timeout": stats["timeout"],
        "error": stats["error"],
        "equiv_runs": EQUIV_RUNS,
        "stress_policies": 6,
        "timeout_seconds": args.timeout,
        "comparison": "original_kernel vs mutant_kernel (bitwise)",
        "elapsed_min": elapsed_total / 60,
        "by_operator": {k: dict(v) for k, v in op_stats.items()},
    }
    summary_file = OUTPUT_DIR / "equiv_recheck_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    P(f"\n{'='*70}")
    P(f"  增强等价检测完成")
    P(f"{'='*70}")
    P(f"  已检测: {processed}")
    P(f"  新发现等价: {stats['candidate_equivalent']}")
    P(f"  真正存活: {stats['survived']}")
    P(f"  超时: {stats['timeout']}")
    P(f"  错误: {stats['error']}")
    P(f"  耗时: {elapsed_total/60:.1f} 分钟")
    P(f"{'='*70}")


def _save(completed_file, completed, results_file, all_results):
    with open(completed_file, "w") as f:
        json.dump(sorted(completed), f)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
