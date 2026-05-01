#!/usr/bin/env python3
"""External CUDA Kernel Differential Testing.

Uses MutaKernel's 21 stress policies to detect numerical discrepancies
between external CUDA/Triton kernels and PyTorch reference implementations.

Reuses the existing _stress_worker.py subprocess protocol — no mutation
is involved. The external kernel is loaded as the "original", and
mutated_code is set to the same source (no mutant).

Results are stored in:
  MutaKernel/第三次实验汇总/results/
  MutaKernel/external_benchmarks/results/  (symlink / copy)
"""
from __future__ import annotations

import gc
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.stress.policy_bank import get_all_policy_names

WORKER_SCRIPT = SCRIPT_DIR / "_stress_worker.py"
RESULT_DIR = PROJECT_ROOT / "第三次实验汇总" / "results"
DETAIL_DIR = RESULT_DIR / "details"

DEVICE = "cuda"
STRESS_TIMEOUT = 180
BASELINE_SEEDS = 50
SEEDS_PER_POLICY = 3
TOLERANCE_LEVELS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("external_diff_test")


def P(msg: str):
    print(msg, flush=True)


def gpu_cleanup():
    stale = [k for k in sys.modules if k.startswith(("mutant_", "ref_", "stress_", "ext_"))]
    for k in stale:
        del sys.modules[k]
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def _run_stress_worker(cfg: dict, timeout: int) -> dict | None:
    """Run _stress_worker.py in a subprocess (copied from run_stress_enhance.py)."""
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="extcfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="extres_")
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
            _, stderr = proc.communicate(timeout=timeout)
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


def _check_package(pkg_name: str) -> bool:
    """Check whether a Python package is importable."""
    try:
        __import__(pkg_name)
        return True
    except ImportError:
        return False


BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
REPEATED_TRIALS = 10
DTYPE_TARGETS = ["float16", "bfloat16"]


def _classify_worker_result(data: dict | None) -> str:
    """Classify a value_stress / training_stress worker result into a status string."""
    if data is None:
        return "timeout"
    if not data.get("ref_ok", True):
        return "ref_fail"
    if data.get("original_ok", False):
        return "pass"
    return "discrepancy"


def _run_policy_sweep(
    problem_file: str,
    kernel_code: str,
    mode: str,
    seed_base: int,
    all_policies: list[str],
    label: str,
) -> tuple[dict, int, int]:
    """Run 21 policies × SEEDS_PER_POLICY for value_stress or training_stress.

    Returns (results_dict, passes, discrepancies).
    """
    mutated_code = kernel_code
    total = len(all_policies) * SEEDS_PER_POLICY
    P(f"\n  {label} ({len(all_policies)} policies x {SEEDS_PER_POLICY} seeds = {total} rounds)")

    results: dict[str, Any] = {}
    passes = discreps = 0

    for pi, policy_name in enumerate(all_policies):
        policy_results = []
        policy_has_discrepancy = False
        for si in range(SEEDS_PER_POLICY):
            seed = seed_base + pi * SEEDS_PER_POLICY + si
            cfg = {
                "mode": mode,
                "problem_file": problem_file,
                "kernel_code": kernel_code,
                "mutated_code": mutated_code,
                "policy_name": policy_name,
                "seed": seed,
                "atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL,
                "device": DEVICE,
                "sync_weights": True,
            }
            data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)
            status = _classify_worker_result(data)
            pr: dict[str, Any] = {"seed": seed, "status": status}
            if status == "discrepancy":
                pr["error"] = data.get("error", "") if data else ""
                discreps += 1
                policy_has_discrepancy = True
            elif status == "pass":
                passes += 1
            policy_results.append(pr)

        call_idx = (pi + 1) * SEEDS_PER_POLICY
        if call_idx % 9 == 0 or policy_has_discrepancy:
            tag = "DISCREPANCY" if policy_has_discrepancy else "pass"
            P(f"    [{call_idx}/{total}] {policy_name}: {tag}")

        results[policy_name] = {
            "has_discrepancy": policy_has_discrepancy,
            "results": policy_results,
        }

    P(f"  {label} result: {passes} pass, {discreps} discrepancies across {total} rounds")
    return results, passes, discreps


def run_single_kernel(kernel_entry: dict) -> dict:
    """Run full six-dimension differential test suite for one external kernel."""
    kid = kernel_entry["id"]
    P(f"\n{'='*60}")
    P(f"  Testing: {kid}  ({kernel_entry['repo']})")
    P(f"{'='*60}")

    problem_file = str(PROJECT_ROOT / kernel_entry["reference_file"])
    kernel_code = kernel_entry["kernel_source"]
    mutated_code = kernel_code  # no mutation — differential test

    result: dict[str, Any] = {
        "id": kid,
        "repo": kernel_entry["repo"],
        "kernel_name": kernel_entry["kernel_name"],
        "reference_file": kernel_entry["reference_file"],
        "baseline": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "results": []},
        "value_stress": {},
        "dtype_stress": {},
        "training_stress": {},
        "repeated_run": {},
        "config_stress": {},
        "multi_tolerance": {},
        "summary": {},
    }

    # ---- Dim 1: Baseline (identity policy, 50 seeds) ----
    P(f"\n  Dim 1: Baseline test ({BASELINE_SEEDS} seeds)")
    baseline = result["baseline"]
    for seed in range(BASELINE_SEEDS):
        cfg = {
            "mode": "value_stress",
            "problem_file": problem_file,
            "kernel_code": kernel_code,
            "mutated_code": mutated_code,
            "policy_name": "__identity__",
            "seed": seed,
            "atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL,
            "device": DEVICE,
            "sync_weights": True,
        }
        data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)
        baseline["total"] += 1
        if data is None:
            baseline["errors"] += 1
            baseline["results"].append({"seed": seed, "status": "timeout"})
            P(f"    [baseline {seed+1}/{BASELINE_SEEDS}] TIMEOUT")
        elif data.get("error"):
            if "orig compile" in data.get("error", "") or "orig crash" in data.get("error", ""):
                baseline["errors"] += 1
                baseline["results"].append({"seed": seed, "status": "error",
                                            "error": data["error"]})
                P(f"    [baseline {seed+1}/{BASELINE_SEEDS}] ERROR: {data['error'][:80]}")
            elif data.get("original_ok", False):
                baseline["passed"] += 1
                baseline["results"].append({"seed": seed, "status": "pass"})
            else:
                baseline["failed"] += 1
                baseline["results"].append({"seed": seed, "status": "discrepancy"})
                P(f"    [baseline {seed+1}/{BASELINE_SEEDS}] DISCREPANCY")
        else:
            if data.get("original_ok", False):
                baseline["passed"] += 1
                baseline["results"].append({"seed": seed, "status": "pass"})
            else:
                baseline["failed"] += 1
                baseline["results"].append({"seed": seed, "status": "discrepancy"})
                P(f"    [baseline {seed+1}/{BASELINE_SEEDS}] DISCREPANCY")

    P(f"  Baseline result: {baseline['passed']}/{baseline['total']} passed, "
      f"{baseline['failed']} discrepancies, {baseline['errors']} errors")

    if baseline["errors"] == baseline["total"]:
        P(f"  SKIP: All baseline tests errored — likely package not installed or incompatible")
        result["summary"]["status"] = "SKIPPED"
        return result

    all_policies = get_all_policy_names()

    # ---- Dim 2: value_stress (21 policies × 3 seeds) ----
    vs_results, vs_pass, vs_disc = _run_policy_sweep(
        problem_file, kernel_code,
        mode="value_stress", seed_base=50000,
        all_policies=all_policies, label="Dim 2: value_stress",
    )
    result["value_stress"] = vs_results

    # ---- Dim 3: dtype_stress (float16 / bfloat16) ----
    P(f"\n  Dim 3: dtype_stress ({len(DTYPE_TARGETS)} dtypes x {SEEDS_PER_POLICY} seeds = {len(DTYPE_TARGETS) * SEEDS_PER_POLICY} rounds)")
    dtype_results: dict[str, Any] = {}
    dtype_disc = 0
    for di, dname in enumerate(DTYPE_TARGETS):
        dtype_round_results = []
        has_disc = False
        for si in range(SEEDS_PER_POLICY):
            seed = 60000 + di * SEEDS_PER_POLICY + si
            cfg = {
                "mode": "dtype_stress",
                "problem_file": problem_file,
                "kernel_code": kernel_code,
                "mutated_code": mutated_code,
                "seed": seed,
                "target_dtypes": [dname],
                "atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL,
                "device": DEVICE,
                "sync_weights": True,
            }
            data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)
            pr: dict[str, Any] = {"seed": seed}
            if data is None:
                pr["status"] = "timeout"
            elif data.get("error"):
                pr["status"] = "error"
                pr["error"] = data["error"][:200]
            else:
                rpd = data.get("results_per_dtype", {}).get(dname, {})
                if rpd.get("error"):
                    is_unsupported = rpd.get("dtype_unsupported", False)
                    pr["status"] = "unsupported" if is_unsupported else "error"
                    pr["error"] = rpd["error"][:200]
                elif rpd.get("orig_ok", True) is False:
                    pr["status"] = "discrepancy"
                    has_disc = True
                    dtype_disc += 1
                else:
                    pr["status"] = "pass"
            dtype_round_results.append(pr)

        tag = "DISCREPANCY" if has_disc else "pass"
        P(f"    {dname}: {tag}")
        dtype_results[dname] = {
            "has_discrepancy": has_disc,
            "results": dtype_round_results,
        }

    result["dtype_stress"] = dtype_results
    P(f"  dtype_stress result: {dtype_disc} discrepancies across {len(DTYPE_TARGETS) * SEEDS_PER_POLICY} rounds")

    # ---- Dim 4: training_stress (21 policies × 3 seeds, .train() mode) ----
    ts_results, ts_pass, ts_disc = _run_policy_sweep(
        problem_file, kernel_code,
        mode="training_stress", seed_base=70000,
        all_policies=all_policies, label="Dim 4: training_stress",
    )
    result["training_stress"] = ts_results

    # ---- Dim 5: repeated_run (3 seeds × 10 trials each) ----
    P(f"\n  Dim 5: repeated_run ({SEEDS_PER_POLICY} seeds x {REPEATED_TRIALS} trials)")
    repeat_results = {"seeds_tested": SEEDS_PER_POLICY, "results": []}
    repeat_disc = 0
    for si in range(SEEDS_PER_POLICY):
        seed = 80000 + si
        cfg = {
            "mode": "repeated_run",
            "problem_file": problem_file,
            "kernel_code": kernel_code,
            "mutated_code": mutated_code,
            "seed": seed,
            "n_trials": REPEATED_TRIALS,
            "atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL,
            "device": DEVICE,
            "sync_weights": True,
        }
        data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT * 2)
        pr: dict[str, Any] = {"seed": seed}
        if data is None:
            pr["status"] = "timeout"
        elif data.get("error"):
            pr["status"] = "error"
            pr["error"] = data["error"][:200]
        else:
            killed = data.get("killed", False)
            self_inc = data.get("self_inconsistent", False)
            if self_inc:
                pr["status"] = "self_inconsistent"
                pr["divergent_trial"] = data.get("divergent_trial")
                repeat_disc += 1
            elif killed:
                pr["status"] = "ref_mismatch"
                pr["divergent_trial"] = data.get("divergent_trial")
                repeat_disc += 1
            else:
                pr["status"] = "pass"
                pr["self_consistent"] = True
                pr["matches_ref"] = True
        repeat_results["results"].append(pr)
        tag = pr["status"].upper()
        P(f"    seed={seed}: {tag}")

    repeat_results["any_discrepancy"] = repeat_disc > 0
    result["repeated_run"] = repeat_results
    P(f"  repeated_run result: {repeat_disc} discrepancies across {SEEDS_PER_POLICY} rounds")

    # ---- Dim 6: config_stress (7 batch_sizes × 3 seeds) ----
    total_cfg = len(BATCH_SIZES) * SEEDS_PER_POLICY
    P(f"\n  Dim 6: config_stress ({len(BATCH_SIZES)} batch_sizes x {SEEDS_PER_POLICY} seeds = {total_cfg} rounds)")
    config_results: dict[str, Any] = {}
    config_disc = 0
    cfg_seeds = [90000 + i for i in range(SEEDS_PER_POLICY)]
    cfg = {
        "mode": "config_stress",
        "problem_file": problem_file,
        "kernel_code": kernel_code,
        "mutated_code": mutated_code,
        "batch_sizes": BATCH_SIZES,
        "seeds": cfg_seeds,
        "device": DEVICE,
        "sync_weights": True,
    }
    data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT * 3)
    if data is None:
        P(f"    config_stress TIMEOUT")
        config_results["status"] = "timeout"
    elif data.get("error"):
        P(f"    config_stress ERROR: {data['error'][:100]}")
        config_results["status"] = "error"
        config_results["error"] = data["error"][:300]
    else:
        rpb = data.get("results_per_batch", {})
        for bs_str, bs_data in rpb.items():
            has_disc = False
            if isinstance(bs_data, dict):
                seeds_tested = bs_data.get("seeds_tested", [])
                for sr in seeds_tested:
                    if isinstance(sr, dict) and sr.get("status") == "orig_diverges_from_ref":
                        has_disc = True
                        break
                if bs_data.get("status") in ("killed_by_crash", "killed_by_divergence"):
                    has_disc = True
            if has_disc:
                config_disc += 1
            config_results[f"batch_{bs_str}"] = {
                "has_discrepancy": has_disc,
                "raw": bs_data,
            }
            tag = "DISCREPANCY" if has_disc else "pass"
            P(f"    batch_size={bs_str}: {tag}")

    result["config_stress"] = config_results
    P(f"  config_stress result: {config_disc} discrepancies across {len(BATCH_SIZES)} batch_sizes")

    # ---- Multi-tolerance re-test for any dimension with discrepancies ----
    discrepant_vs = [p for p, v in result["value_stress"].items()
                     if isinstance(v, dict) and v.get("has_discrepancy")]
    discrepant_ts = [p for p, v in result["training_stress"].items()
                     if isinstance(v, dict) and v.get("has_discrepancy")]

    all_discrepant = []
    for p in discrepant_vs:
        all_discrepant.append(("value_stress", p))
    for p in discrepant_ts:
        all_discrepant.append(("training_stress", p))

    if all_discrepant:
        P(f"\n  Multi-tolerance re-test ({len(all_discrepant)} policies x {len(TOLERANCE_LEVELS)} levels)")
        mt: dict[str, Any] = {}
        for dim, policy_name in all_discrepant:
            mode = dim
            if dim == "value_stress":
                seed_base = 50000
                pi = all_policies.index(policy_name) if policy_name in all_policies else 0
            else:
                seed_base = 70000
                pi = all_policies.index(policy_name) if policy_name in all_policies else 0
            seed = seed_base + pi * SEEDS_PER_POLICY

            tol_results = {}
            for atol in TOLERANCE_LEVELS:
                tol_cfg = {
                    "mode": mode,
                    "problem_file": problem_file,
                    "kernel_code": kernel_code,
                    "mutated_code": mutated_code,
                    "policy_name": policy_name,
                    "seed": seed,
                    "atol": atol, "rtol": atol,
                    "device": DEVICE,
                    "sync_weights": True,
                }
                d = _run_stress_worker(tol_cfg, timeout=STRESS_TIMEOUT)
                tol_results[str(atol)] = d.get("original_ok", False) if d else None
            key = f"{dim}/{policy_name}"
            mt[key] = tol_results
            P(f"    {key}: {tol_results}")
        result["multi_tolerance"] = mt

    # ---- Summary ----
    discrepant_dims = []
    if vs_disc > 0:
        discrepant_dims.append("value_stress")
    if dtype_disc > 0:
        discrepant_dims.append("dtype_stress")
    if ts_disc > 0:
        discrepant_dims.append("training_stress")
    if repeat_disc > 0:
        discrepant_dims.append("repeated_run")
    if config_disc > 0:
        discrepant_dims.append("config_stress")

    total_rounds = (len(all_policies) * SEEDS_PER_POLICY  # value_stress
                    + len(DTYPE_TARGETS) * SEEDS_PER_POLICY  # dtype_stress
                    + len(all_policies) * SEEDS_PER_POLICY  # training_stress
                    + SEEDS_PER_POLICY  # repeated_run
                    + len(BATCH_SIZES))  # config_stress (1 call, 7 bs)

    result["summary"] = {
        "status": "COMPLETED",
        "baseline_pass_rate": f"{baseline['passed']}/{baseline['total']}",
        "value_stress_discrepancies": vs_disc,
        "dtype_stress_discrepancies": dtype_disc,
        "training_stress_discrepancies": ts_disc,
        "repeated_run_discrepancies": repeat_disc,
        "config_stress_discrepancies": config_disc,
        "total_discrepancies": vs_disc + dtype_disc + ts_disc + repeat_disc + config_disc,
        "total_rounds": total_rounds,
        "discrepant_dimensions": discrepant_dims,
        "discrepant_value_policies": discrepant_vs,
        "discrepant_training_policies": discrepant_ts,
    }

    gpu_cleanup()
    return result


def main():
    P("=" * 60)
    P("  MutaKernel External Differential Testing (6 Dimensions)")
    P("  value_stress | dtype_stress | training_stress | repeated_run | config_stress")
    P("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT))
    from external_benchmarks.registry import EXTERNAL_KERNELS

    DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    available_kernels = []
    skipped_kernels = []

    # Pre-check package availability
    P("\nPackage availability check:")
    checked_pkgs: dict[str, bool] = {}
    for entry in EXTERNAL_KERNELS:
        for pkg in entry.get("requires", []):
            if pkg not in checked_pkgs:
                checked_pkgs[pkg] = _check_package(pkg)
                status = "OK" if checked_pkgs[pkg] else "NOT FOUND"
                P(f"  {pkg}: {status}")

    for entry in EXTERNAL_KERNELS:
        missing = [p for p in entry.get("requires", []) if not checked_pkgs.get(p, False)]
        if missing:
            P(f"\n  SKIP {entry['id']}: missing packages {missing}")
            skipped_kernels.append({"id": entry["id"], "reason": f"missing {missing}"})
        else:
            available_kernels.append(entry)

    P(f"\nAvailable: {len(available_kernels)}, Skipped: {len(skipped_kernels)}")

    for entry in available_kernels:
        t0 = time.time()
        try:
            result = run_single_kernel(entry)
        except Exception as e:
            P(f"\n  FATAL ERROR testing {entry['id']}: {e}")
            result = {
                "id": entry["id"],
                "repo": entry["repo"],
                "summary": {"status": "FATAL_ERROR", "error": str(e)[:500]},
            }
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        P(f"\n  Completed {entry['id']} in {elapsed:.1f}s")

        detail_path = DETAIL_DIR / f"{entry['id']}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        P(f"  Saved: {detail_path}")

        all_results.append(result)

    # ---- Write summary ----
    summary = {
        "experiment": "External CUDA Kernel Differential Testing (6 Dimensions)",
        "framework": "MutaKernel stress policies (21 strategies) + 5 test dimensions",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "default_tolerance": {"atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL},
        "baseline_seeds": BASELINE_SEEDS,
        "seeds_per_policy": SEEDS_PER_POLICY,
        "dimensions": ["value_stress", "dtype_stress", "training_stress",
                        "repeated_run", "config_stress"],
        "tested_kernels": len(available_kernels),
        "skipped_kernels": skipped_kernels,
        "results": [],
    }
    for r in all_results:
        s = r.get("summary", {})
        summary["results"].append({
            "id": r["id"],
            "repo": r.get("repo", ""),
            "kernel_name": r.get("kernel_name", ""),
            "status": s.get("status", "UNKNOWN"),
            "baseline_pass_rate": s.get("baseline_pass_rate", "N/A"),
            "value_stress_discrepancies": s.get("value_stress_discrepancies", 0),
            "dtype_stress_discrepancies": s.get("dtype_stress_discrepancies", 0),
            "training_stress_discrepancies": s.get("training_stress_discrepancies", 0),
            "repeated_run_discrepancies": s.get("repeated_run_discrepancies", 0),
            "config_stress_discrepancies": s.get("config_stress_discrepancies", 0),
            "total_discrepancies": s.get("total_discrepancies", 0),
            "discrepant_dimensions": s.get("discrepant_dimensions", []),
            "elapsed_s": r.get("elapsed_s", 0),
        })

    summary_path = RESULT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    P(f"\nSummary saved: {summary_path}")

    # Also copy to external_benchmarks/results/ for convenience
    ext_summary_path = PROJECT_ROOT / "external_benchmarks" / "results" / "summary.json"
    ext_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ext_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    P("\n" + "=" * 60)
    P("  DONE")
    P("=" * 60)


if __name__ == "__main__":
    main()
