#!/usr/bin/env python3
"""Ablation experiments for ICSE evaluation (RQ3).

Two ablation studies:

1. Leave-one-out (--mode leave_one_out):
   For each policy, re-run the experiment WITHOUT that policy.
   Measures the marginal contribution of each policy.

2. Strategy count curve (--mode strategy_curve):
   Run with {1, 3, 5, 7, 12} randomly selected policies.
   Shows how kill rate scales with the number of policies.

Both operate on the stress_enhance_results from the main experiment.

Usage:
  python scripts/run_ablation.py --mode leave_one_out
  python scripts/run_ablation.py --mode strategy_curve
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
from src.mutrepair.enhanced_inputs import STRATEGY_MAP

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

BLOCK12_RESULT_DIR = PROJECT_ROOT / "full_block12_results"
ABLATION_RESULT_DIR = PROJECT_ROOT / "ablation_results"
WORKER_SCRIPT = SCRIPT_DIR / "_stress_worker.py"

TIMEOUT = 120
DEVICE = "cuda"
SEED = 42
ATOL = 1e-2
RTOL = 1e-2

N_SEEDS_PER_POLICY = 3   # Same per-policy seed budget as main experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ablation")


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
    stale = [k for k in sys.modules if k.startswith(("mutant_", "ref_", "stress_", "build_", "abl_"))]
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
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="ablcfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="ablres_")
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


def _test_mutant_with_policies(resolved, policies, mutant_meta):
    """Test one mutant with a given set of policies (N_SEEDS_PER_POLICY seeds each).

    Returns (killed, killing_policy).
    """
    for pi, policy_name in enumerate(policies):
        for si in range(N_SEEDS_PER_POLICY):
            seed = SEED + pi * N_SEEDS_PER_POLICY + si
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
                return True, policy_name
    return False, None


# ---------------------------------------------------------------------------
# Leave-one-out ablation
# ---------------------------------------------------------------------------

def run_leave_one_out(survived_list, best_kernels, output_dir: Path):
    """For each policy, remove it and measure the kill count drop."""
    P(f"\n{'='*70}")
    P(f"  Ablation: Leave-One-Out")
    P(f"{'='*70}")

    all_policies = get_all_policy_names()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to Type 3 candidates (operators with STRATEGY_MAP entries)
    type3_candidates = [
        (kn, km, mm) for kn, km, mm in survived_list
        if mm["operator_name"] in STRATEGY_MAP
    ]
    P(f"  Type 3 candidates (with STRATEGY_MAP): {len(type3_candidates)}")
    P(f"  Policies to ablate: {len(all_policies)}")

    # Full run (all policies)
    P(f"\n  --- Full run (all {len(all_policies)} policies) ---")
    full_kills = 0
    full_results = {}

    for idx, (kn, km, mm) in enumerate(type3_candidates):
        mid = mm["id"]
        resolved = _resolve_mutant(kn, km, mm, best_kernels)
        if not resolved:
            continue

        op = mm["operator_name"]
        mapped = STRATEGY_MAP.get(op, [])
        remaining = [p for p in all_policies if p not in mapped]
        priority_policies = mapped + remaining

        killed, kp = _test_mutant_with_policies(resolved, priority_policies, mm)
        full_results[mid] = {"killed": killed, "killing_policy": kp}
        if killed:
            full_kills += 1

        if (idx + 1) % 20 == 0:
            P(f"    Progress: {idx+1}/{len(type3_candidates)}, kills so far: {full_kills}")
        gpu_cleanup()

    P(f"  Full run: {full_kills} / {len(type3_candidates)} killed")

    # Leave-one-out runs
    loo_results = {}
    for removed_policy in all_policies:
        P(f"\n  --- Without '{removed_policy}' ---")
        reduced_policies = [p for p in all_policies if p != removed_policy]
        kills = 0

        for kn, km, mm in type3_candidates:
            mid = mm["id"]
            resolved = _resolve_mutant(kn, km, mm, best_kernels)
            if not resolved:
                continue

            op = mm["operator_name"]
            mapped = [p for p in STRATEGY_MAP.get(op, []) if p != removed_policy]
            remaining = [p for p in reduced_policies if p not in mapped]
            priority = mapped + remaining

            killed, _ = _test_mutant_with_policies(resolved, priority, mm)
            if killed:
                kills += 1
            gpu_cleanup()

        drop = full_kills - kills
        P(f"    Kills: {kills}, Drop: {drop}")
        loo_results[removed_policy] = {
            "kills_without": kills,
            "drop": drop,
            "contribution": drop / max(full_kills, 1),
        }

    summary = {
        "ablation": "leave_one_out",
        "full_kills": full_kills,
        "total_candidates": len(type3_candidates),
        "per_policy": loo_results,
        "policy_ranking": sorted(
            loo_results.items(), key=lambda x: x[1]["drop"], reverse=True
        ),
    }

    out_path = output_dir / "leave_one_out.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    P(f"\n  Saved to {out_path}")


# ---------------------------------------------------------------------------
# Strategy count curve
# ---------------------------------------------------------------------------

def run_strategy_curve(survived_list, best_kernels, output_dir: Path):
    """Measure kill rate with {1, 3, 5, 7, 12} random policies."""
    P(f"\n{'='*70}")
    P(f"  Ablation: Strategy Count Curve")
    P(f"{'='*70}")

    all_policies = get_all_policy_names()
    output_dir.mkdir(parents=True, exist_ok=True)

    type3_candidates = [
        (kn, km, mm) for kn, km, mm in survived_list
        if mm["operator_name"] in STRATEGY_MAP
    ]
    P(f"  Type 3 candidates: {len(type3_candidates)}")

    counts = [1, 3, 5, 7, len(all_policies)]
    n_repeats = 3
    rng = random.Random(99999)

    curve_results = {}

    for count in counts:
        P(f"\n  --- {count} policies ---")
        repeat_kills = []

        for rep in range(n_repeats):
            if count >= len(all_policies):
                chosen = list(all_policies)
            else:
                chosen = rng.sample(all_policies, count)

            kills = 0
            for kn, km, mm in type3_candidates:
                resolved = _resolve_mutant(kn, km, mm, best_kernels)
                if not resolved:
                    continue
                killed, _ = _test_mutant_with_policies(resolved, chosen, mm)
                if killed:
                    kills += 1
                gpu_cleanup()

            repeat_kills.append(kills)
            P(f"    Repeat {rep+1}: {kills} killed")

        avg_kills = sum(repeat_kills) / len(repeat_kills)
        curve_results[count] = {
            "per_repeat": repeat_kills,
            "avg_kills": avg_kills,
            "avg_rate": avg_kills / max(len(type3_candidates), 1),
        }

    summary = {
        "ablation": "strategy_curve",
        "total_candidates": len(type3_candidates),
        "policy_counts": counts,
        "n_repeats": n_repeats,
        "curve": curve_results,
    }

    out_path = output_dir / "strategy_curve.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    P(f"\n  Saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation experiments")
    parser.add_argument("--mode", required=True,
                        choices=["leave_one_out", "strategy_curve"],
                        help="Which ablation to run")
    args = parser.parse_args()

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    survived_list = load_all_survived(BLOCK12_RESULT_DIR)
    P(f"  Loaded {len(survived_list)} survived mutants")

    ABLATION_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if args.mode == "leave_one_out":
        run_leave_one_out(survived_list, best_kernels, ABLATION_RESULT_DIR)
    elif args.mode == "strategy_curve":
        run_strategy_curve(survived_list, best_kernels, ABLATION_RESULT_DIR)

    elapsed = time.time() - t_start
    P(f"\n  Total elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
