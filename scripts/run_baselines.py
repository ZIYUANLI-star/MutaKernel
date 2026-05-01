#!/usr/bin/env python3
"""Baseline experiments for ICSE evaluation.

Baseline 1 (--mode random20):
  Re-test all survived mutants with 20 random torch.randn inputs (seed 42-61).
  No distribution change — proves "more runs" alone is insufficient.

Baseline 2 (--mode random_stress):
  For each survived mutant, randomly pick 3 stress policies (unguided)
  and test. Proves diagnosis-guided augmentation > blind stress.

Usage:
  python scripts/run_baselines.py --mode random20
  python scripts/run_baselines.py --mode random_stress
"""
import argparse
import gc
import json
import logging
import os
import random
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

import torch

from src.models import KernelInfo
from src.mutengine.mutant_runner import MutantRunner
from src.stress.policy_bank import get_all_policy_names

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

BLOCK12_RESULT_DIR = PROJECT_ROOT / "full_block12_results"
BASELINE_RESULT_DIR = PROJECT_ROOT / "baseline_results"
WORKER_SCRIPT = SCRIPT_DIR / "_stress_worker.py"
RANDOM20_WORKER = SCRIPT_DIR / "_baseline_random20_worker.py"

TIMEOUT = 120
DEVICE = "cuda"
ATOL = 1e-2
RTOL = 1e-2

N_SEEDS_PER_POLICY = 3   # Same per-policy seed budget as main experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("baselines")


def P(msg):
    print(msg, flush=True)


def find_problem_file(problem_dir: Path, problem_id) -> Path | None:
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def detect_language(source: str) -> str:
    indicators = ["__global__", "__device__", "load_inline", "cuda_source"]
    return "cuda" if sum(1 for ind in indicators if ind in source) >= 2 else "triton"


def gpu_cleanup():
    stale = [k for k in sys.modules if k.startswith(("mutant_", "ref_", "stress_", "build_", "b1_"))]
    for k in stale:
        del sys.modules[k]
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def load_all_survived(result_dir: Path):
    details_dir = result_dir / "details"
    if not details_dir.exists():
        return []
    survived = []
    for jf in sorted(details_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        kernel_name = data["kernel"]["problem_name"]
        for m in data.get("mutants", []):
            if m["status"] == "survived":
                survived.append((kernel_name, data["kernel"], m))
    return survived


def reconstruct_mutant_code(kernel_info: KernelInfo, mutant_meta: dict) -> str | None:
    runner = MutantRunner(categories=["A", "B", "C", "D"])
    all_mutants = runner.generate_mutants(kernel_info)
    target_id = mutant_meta["id"]
    target_op = mutant_meta["operator_name"]
    target_line = mutant_meta["site"]["line_start"]
    for m in all_mutants:
        if m.id == target_id:
            return m.mutated_code
        if (m.operator_name == target_op
                and m.site.line_start == target_line
                and m.site.original_code == mutant_meta["site"].get("original_code", "")):
            return m.mutated_code
    return None


def _run_worker(cfg: dict, timeout: int) -> dict | None:
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="blcfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="blres_")
    os.close(cfg_fd)
    os.close(res_fd)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT), cfg_path, res_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
        try:
            proc.communicate(timeout=timeout)
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


def _resolve_mutant(kernel_name, kernel_meta, mutant_meta, best_kernels):
    """Resolve kernel paths and mutated code for a survived mutant."""
    level_key = f"L{kernel_meta['level']}"
    problem_dir = PROBLEM_DIRS.get(level_key)
    if not problem_dir:
        return None

    bk_info = best_kernels.get(kernel_name)
    if not bk_info:
        return None

    kernel_path = Path(bk_info["kernel_path"])
    if not kernel_path.exists():
        return None

    problem_file = find_problem_file(problem_dir, kernel_meta["problem_id"])
    if not problem_file:
        return None

    kernel_code = kernel_path.read_text(encoding="utf-8")

    mutated_code = mutant_meta.get("mutated_code", "")
    if not mutated_code:
        ki = KernelInfo(
            problem_id=kernel_meta["problem_id"],
            level=kernel_meta["level"],
            problem_name=kernel_name,
            source_path=str(kernel_path),
            kernel_code=kernel_code,
            reference_module_path=str(problem_file),
            language=kernel_meta.get("language", detect_language(kernel_code)),
        )
        mutated_code = reconstruct_mutant_code(ki, mutant_meta)
        if not mutated_code:
            return None

    return {
        "problem_file": str(problem_file),
        "kernel_code": kernel_code,
        "mutated_code": mutated_code,
    }


# ---------------------------------------------------------------------------
# Baseline 1: 20x random inputs (standard torch.randn)
# ---------------------------------------------------------------------------

def run_baseline_random20(survived_list, best_kernels, output_dir: Path):
    """For each mutant, test with 20 different random seeds using get_inputs()."""
    P(f"\n{'='*70}")
    P(f"  Baseline 1: Random 20x (torch.randn, seed 42-61)")
    P(f"{'='*70}")
    P(f"  Mutants to test: {len(survived_list)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_file = output_dir / "completed.json"
    completed = set()
    if completed_file.exists():
        completed = set(json.loads(completed_file.read_text()))

    n_seeds = 20
    base_seed = 42
    total_killed = 0
    results = []

    for idx, (kernel_name, kernel_meta, mutant_meta) in enumerate(survived_list):
        mutant_id = mutant_meta["id"]
        if mutant_id in completed:
            continue

        resolved = _resolve_mutant(kernel_name, kernel_meta, mutant_meta, best_kernels)
        if resolved is None:
            P(f"  [{idx+1}] {mutant_id} -- could not resolve, skip")
            continue

        P(f"  [{idx+1}/{len(survived_list)}] {mutant_id} ({mutant_meta['operator_name']})")
        killed = False
        killing_seed = None

        for si in range(n_seeds):
            seed = base_seed + si
            cfg = {
                "mode": "value_stress",
                "problem_file": resolved["problem_file"],
                "kernel_code": resolved["kernel_code"],
                "mutated_code": resolved["mutated_code"],
                "policy_name": "__identity__",
                "seed": seed,
                "atol": ATOL,
                "rtol": RTOL,
                "device": DEVICE,
            }

            data = _run_worker(cfg, timeout=TIMEOUT)
            if data and data.get("original_ok") and not data.get("mutant_ok"):
                killed = True
                killing_seed = seed
                break

        status = "KILLED" if killed else "survived"
        P(f"    -> {status}" + (f" (seed={killing_seed})" if killed else ""))

        if killed:
            total_killed += 1

        result_entry = {
            "mutant_id": mutant_id,
            "operator_name": mutant_meta["operator_name"],
            "operator_category": mutant_meta["operator_category"],
            "kernel_name": kernel_name,
            "killed": killed,
            "killing_seed": killing_seed,
        }
        results.append(result_entry)

        completed.add(mutant_id)
        with open(completed_file, "w") as f:
            json.dump(sorted(completed), f)
        gpu_cleanup()

    summary = {
        "baseline": "random20",
        "n_seeds": n_seeds,
        "total_tested": len(results),
        "total_killed": total_killed,
        "kill_rate": total_killed / max(len(results), 1),
        "results": results,
    }

    out_path = output_dir / "b1_random20.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    P(f"\n  Baseline 1 Results:")
    P(f"    Tested:  {len(results)}")
    P(f"    Killed:  {total_killed}")
    P(f"    Rate:    {total_killed / max(len(results), 1):.4f}")
    P(f"    Saved:   {out_path}")


# ---------------------------------------------------------------------------
# Baseline 2: Random stress (unguided, 3 random policies per mutant)
# ---------------------------------------------------------------------------

def run_baseline_random_stress(survived_list, best_kernels, output_dir: Path):
    """For each mutant, pick 3 random stress policies and test."""
    P(f"\n{'='*70}")
    P(f"  Baseline 2: Random Stress (3 random policies, unguided)")
    P(f"{'='*70}")
    P(f"  Mutants to test: {len(survived_list)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_file = output_dir / "completed_b2.json"
    completed = set()
    if completed_file.exists():
        completed = set(json.loads(completed_file.read_text()))

    all_policies = get_all_policy_names()
    n_random_policies = 3
    rng = random.Random(12345)

    total_killed = 0
    results = []
    per_policy_kills = defaultdict(int)

    for idx, (kernel_name, kernel_meta, mutant_meta) in enumerate(survived_list):
        mutant_id = mutant_meta["id"]
        if mutant_id in completed:
            continue

        resolved = _resolve_mutant(kernel_name, kernel_meta, mutant_meta, best_kernels)
        if resolved is None:
            P(f"  [{idx+1}] {mutant_id} -- could not resolve, skip")
            continue

        chosen = rng.sample(all_policies, min(n_random_policies, len(all_policies)))
        P(f"  [{idx+1}/{len(survived_list)}] {mutant_id} -> policies: {chosen}")

        killed = False
        killing_policy = None
        killing_seed = None

        for pi, policy_name in enumerate(chosen):
            for si in range(N_SEEDS_PER_POLICY):
                seed = 42 + pi * N_SEEDS_PER_POLICY + si
                cfg = {
                    "mode": "value_stress",
                    "problem_file": resolved["problem_file"],
                    "kernel_code": resolved["kernel_code"],
                    "mutated_code": resolved["mutated_code"],
                    "policy_name": policy_name,
                    "seed": seed,
                    "atol": ATOL,
                    "rtol": RTOL,
                    "device": DEVICE,
                }

                data = _run_worker(cfg, timeout=TIMEOUT)
                if data and data.get("ref_ok") and data.get("original_ok") and not data.get("mutant_ok"):
                    killed = True
                    killing_policy = policy_name
                    killing_seed = seed
                    per_policy_kills[policy_name] += 1
                    break
            if killed:
                break

        status = "KILLED" if killed else "survived"
        P(f"    -> {status}" + (f" (policy={killing_policy})" if killed else ""))

        if killed:
            total_killed += 1

        result_entry = {
            "mutant_id": mutant_id,
            "operator_name": mutant_meta["operator_name"],
            "operator_category": mutant_meta["operator_category"],
            "kernel_name": kernel_name,
            "policies_tried": chosen,
            "seeds_per_policy": N_SEEDS_PER_POLICY,
            "killed": killed,
            "killing_policy": killing_policy,
            "killing_seed": killing_seed,
        }
        results.append(result_entry)

        completed.add(mutant_id)
        with open(completed_file, "w") as f:
            json.dump(sorted(completed), f)
        gpu_cleanup()

    summary = {
        "baseline": "random_stress",
        "n_random_policies": n_random_policies,
        "seeds_per_policy": N_SEEDS_PER_POLICY,
        "total_worker_calls_per_mutant": n_random_policies * N_SEEDS_PER_POLICY,
        "total_tested": len(results),
        "total_killed": total_killed,
        "kill_rate": total_killed / max(len(results), 1),
        "per_policy_kills": dict(per_policy_kills),
        "results": results,
    }

    out_path = output_dir / "b2_random_stress.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    P(f"\n  Baseline 2 Results:")
    P(f"    Tested:  {len(results)}")
    P(f"    Killed:  {total_killed}")
    P(f"    Rate:    {total_killed / max(len(results), 1):.4f}")
    P(f"    Saved:   {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline experiments")
    parser.add_argument("--mode", required=True, choices=["random20", "random_stress"],
                        help="Which baseline to run")
    args = parser.parse_args()

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    survived_list = load_all_survived(BLOCK12_RESULT_DIR)
    P(f"  Loaded {len(survived_list)} survived mutants from Block 1-2 results")

    BASELINE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if args.mode == "random20":
        run_baseline_random20(survived_list, best_kernels, BASELINE_RESULT_DIR)
    elif args.mode == "random_stress":
        run_baseline_random_stress(survived_list, best_kernels, BASELINE_RESULT_DIR)

    elapsed = time.time() - t_start
    P(f"\n  Total elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
