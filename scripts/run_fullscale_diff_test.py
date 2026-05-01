#!/usr/bin/env python3
"""Full-scale differential testing across multiple external datasets.

Datasets:
  1. NVIDIA/apex (new kernels only - MLP + optimizers)
  2. CUDA-L1 (250 RL-optimized kernels)
  3. AI CUDA Engineer Archive (250 best-per-task LLM-generated kernels)

Uses MutaKernel's 5 deterministic test dimensions (excludes LLM).
Results stored in 第三次实验汇总/results/{dataset_name}/
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

WORKER_SCRIPT = SCRIPT_DIR / "_stress_worker.py"
RESULT_BASE = PROJECT_ROOT / "第三次实验汇总" / "results"

DEVICE = "cuda"
STRESS_TIMEOUT = 600   # 10 min per worker (compilation can take 2min + run up to 80s)
KERNEL_TIMEOUT = 10800  # 3 hours max per kernel (slowest observed: 6946s ≈ 116min)
BASELINE_SEEDS = 10
SEEDS_PER_POLICY = 2
TOLERANCE_LEVELS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2
BATCH_SIZES = [1, 4, 16, 64]
REPEATED_TRIALS = 5
DTYPE_TARGETS = ["float16", "bfloat16"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("fullscale_diff_test")


def P(msg: str):
    print(msg, flush=True)


def gpu_cleanup():
    stale = [k for k in sys.modules if k.startswith(("mutant_", "ref_", "stress_", "ext_", "build_"))]
    for k in stale:
        del sys.modules[k]
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def _force_gpu_cleanup():
    """Attempt to reset GPU state after a hang/OOM."""
    P("    [GPU-CLEANUP] Attempting GPU recovery...")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
        P("    [GPU-CLEANUP] nvidia-smi responsive - GPU OK")
    except Exception:
        P("    [GPU-CLEANUP] nvidia-smi NOT responsive - GPU may be hung!")


def _run_stress_worker(cfg: dict, timeout: int) -> dict | None:
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="fscfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="fsres_")
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
            P(f"    [TIMEOUT] Worker exceeded {timeout}s, killing...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                P("    [TIMEOUT] proc.wait() also hung (GPU driver stuck?) - abandoning process")
                _force_gpu_cleanup()
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


def _classify(data: dict | None) -> str:
    if data is None:
        return "timeout"
    if not data.get("ref_ok", True):
        return "ref_fail"
    if data.get("original_ok", False):
        return "pass"
    return "discrepancy"


def run_quick_baseline(problem_file: str, kernel_code: str, n_seeds: int = 3) -> tuple[int, int, int]:
    """Run a quick baseline check. Returns (passed, failed, errors)."""
    mutated_code = kernel_code
    passed = failed = errors = 0
    for seed in range(n_seeds):
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
        if data is None:
            errors += 1
        elif data.get("error"):
            if "compile" in data.get("error", "") or "crash" in data.get("error", ""):
                errors += 1
            elif data.get("original_ok", False):
                passed += 1
            else:
                failed += 1
        elif data.get("original_ok", False):
            passed += 1
        else:
            failed += 1
    return passed, failed, errors


def _extract_worker_details(data: dict | None, policy: str = "", seed: int = 0) -> dict:
    """Extract key fields from worker return for detailed logging."""
    if data is None:
        return {"policy": policy, "seed": seed, "status": "timeout"}
    record = {
        "policy": policy,
        "seed": seed,
        "status": _classify(data),
        "diff_summary": data.get("diff_summary", ""),
        "time_ms": data.get("time_ms", 0),
        "original_ok": data.get("original_ok", None),
        "ref_ok": data.get("ref_ok", None),
    }
    if data.get("error"):
        record["error"] = str(data["error"])[:300]
    return record


def run_kernel_5dim(problem_file: str, kernel_code: str, kid: str) -> dict:
    """Run 5-dimension test on a single kernel. Returns result dict."""
    from src.stress.policy_bank import get_all_policy_names
    all_policies = get_all_policy_names()
    mutated_code = kernel_code
    _kernel_start = time.time()

    def _over_budget() -> bool:
        return (time.time() - _kernel_start) > KERNEL_TIMEOUT

    result: dict[str, Any] = {
        "id": kid,
        "value_stress": {"discrepancies": 0, "passes": 0, "details": {}, "test_cases": []},
        "dtype_stress": {"discrepancies": 0, "passes": 0, "details": {}, "test_cases": []},
        "training_stress": {"discrepancies": 0, "passes": 0, "details": {}, "test_cases": []},
        "repeated_run": {"discrepancies": 0, "passes": 0, "test_cases": []},
        "config_stress": {"discrepancies": 0, "passes": 0, "raw_results": {}},
    }

    # Dim 1: value_stress
    for pi, policy_name in enumerate(all_policies):
        if _over_budget():
            P(f"    TIMEOUT: skipping remaining value_stress policies")
            break
        has_disc = False
        for si in range(SEEDS_PER_POLICY):
            seed = 50000 + pi * SEEDS_PER_POLICY + si
            cfg = {
                "mode": "value_stress",
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
            status = _classify(data)
            record = _extract_worker_details(data, policy=policy_name, seed=seed)
            result["value_stress"]["test_cases"].append(record)
            if status == "pass":
                result["value_stress"]["passes"] += 1
            elif status == "discrepancy":
                result["value_stress"]["discrepancies"] += 1
                has_disc = True
        if has_disc:
            result["value_stress"]["details"][policy_name] = "discrepancy"

    # Dim 2: dtype_stress
    if _over_budget():
        P(f"    TIMEOUT: skipping dtype_stress")
    for dname in (DTYPE_TARGETS if not _over_budget() else []):
        for si in range(SEEDS_PER_POLICY):
            seed = 60000 + si
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
            tc_record = {"dtype": dname, "seed": seed, "status": "timeout"}
            if data is None:
                result["dtype_stress"]["test_cases"].append(tc_record)
                continue
            rpd = data.get("results_per_dtype", {}).get(dname, {})
            tc_record["time_ms"] = data.get("time_ms", 0)
            tc_record["raw_dtype_result"] = {
                k: v for k, v in rpd.items()
                if k in ("orig_ok", "error", "dtype_unsupported", "diff_summary", "max_diff")
            }
            if rpd.get("orig_ok", True) is False and not rpd.get("dtype_unsupported", False):
                result["dtype_stress"]["discrepancies"] += 1
                result["dtype_stress"]["details"][dname] = "discrepancy"
                tc_record["status"] = "discrepancy"
            elif not rpd.get("error"):
                result["dtype_stress"]["passes"] += 1
                tc_record["status"] = "pass"
            else:
                tc_record["status"] = "error"
                tc_record["error"] = str(rpd.get("error", ""))[:300]
            result["dtype_stress"]["test_cases"].append(tc_record)

    # Dim 3: training_stress
    for pi, policy_name in enumerate(all_policies):
        if _over_budget():
            P(f"    TIMEOUT: skipping remaining training_stress policies")
            break
        has_disc = False
        for si in range(SEEDS_PER_POLICY):
            seed = 70000 + pi * SEEDS_PER_POLICY + si
            cfg = {
                "mode": "training_stress",
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
            status = _classify(data)
            record = _extract_worker_details(data, policy=policy_name, seed=seed)
            result["training_stress"]["test_cases"].append(record)
            if status == "pass":
                result["training_stress"]["passes"] += 1
            elif status == "discrepancy":
                result["training_stress"]["discrepancies"] += 1
                has_disc = True
        if has_disc:
            result["training_stress"]["details"][policy_name] = "discrepancy"

    # Dim 4: repeated_run
    if _over_budget():
        P(f"    TIMEOUT: skipping repeated_run")
    for si in (range(SEEDS_PER_POLICY) if not _over_budget() else []):
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
        tc_record = {"seed": seed, "status": "timeout"}
        if data:
            tc_record["status"] = "discrepancy" if data.get("killed", False) else "pass"
            tc_record["time_ms"] = data.get("time_ms", 0)
            tc_record["diff_summary"] = data.get("diff_summary", "")
            tc_record["n_trials"] = REPEATED_TRIALS
            if data.get("killed"):
                tc_record["kill_reason"] = data.get("kill_reason", "non_deterministic")
            if not data.get("killed", False):
                result["repeated_run"]["passes"] += 1
            else:
                result["repeated_run"]["discrepancies"] += 1
        result["repeated_run"]["test_cases"].append(tc_record)

    # Dim 5: config_stress
    if _over_budget():
        P(f"    TIMEOUT: skipping config_stress")
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
    data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT * 3) if not _over_budget() else None
    if data:
        rpb = data.get("results_per_batch", {})
        result["config_stress"]["raw_results"] = rpb
        for bs_str, bs_data in rpb.items():
            has_disc = False
            if isinstance(bs_data, dict):
                for sr in bs_data.get("seeds_tested", []):
                    if isinstance(sr, dict) and sr.get("status") == "orig_diverges_from_ref":
                        has_disc = True
                        break
                if bs_data.get("status") in ("killed_by_crash", "killed_by_divergence"):
                    has_disc = True
            if has_disc:
                result["config_stress"]["discrepancies"] += 1
            else:
                result["config_stress"]["passes"] += 1

    total_disc = sum(result[d]["discrepancies"] for d in
                     ["value_stress", "dtype_stress", "training_stress", "repeated_run", "config_stress"])
    result["total_discrepancies"] = total_disc
    discrepant_dims = [d for d in ["value_stress", "dtype_stress", "training_stress", "repeated_run", "config_stress"]
                       if result[d]["discrepancies"] > 0]
    result["discrepant_dimensions"] = discrepant_dims

    return result


def run_dataset(dataset_name: str, registry: list[dict], result_dir: Path,
                checkpoint_path: Path | None = None):
    """Run 5-dimension tests on all kernels in a dataset registry."""
    P(f"\n{'='*60}")
    P(f"  Dataset: {dataset_name} ({len(registry)} kernels)")
    P(f"  Result dir: {result_dir}")
    P(f"{'='*60}")

    result_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = result_dir / "details"
    detail_dir.mkdir(exist_ok=True)

    completed = {}
    checkpoint = checkpoint_path or (result_dir / "checkpoint.json")
    if checkpoint.exists():
        with open(checkpoint, "r", encoding="utf-8") as f:
            completed = json.load(f)
        P(f"  Resuming: {len(completed)} kernels already completed")

    all_results = []
    total = len(registry)
    skip_compile = 0
    skip_baseline = 0

    for idx, entry in enumerate(registry):
        kid = entry["id"]

        if kid in completed:
            all_results.append(completed[kid])
            continue

        P(f"\n  [{idx+1}/{total}] {kid}")

        problem_file = entry["reference_file"]
        if not os.path.isabs(problem_file):
            problem_file = str(PROJECT_ROOT / problem_file)
        elif not os.path.exists(problem_file):
            rel = os.path.join("external_benchmarks", *Path(problem_file).parts[-3:])
            alt = str(PROJECT_ROOT / rel)
            if os.path.exists(alt):
                problem_file = alt
        kernel_code = entry["kernel_source"]

        if not os.path.exists(problem_file):
            P(f"    SKIP: reference file not found: {problem_file}")
            r = {"id": kid, "status": "SKIPPED", "reason": "ref_not_found"}
            all_results.append(r)
            completed[kid] = r
            continue

        passed, failed, errors = run_quick_baseline(problem_file, kernel_code, n_seeds=3)
        P(f"    Baseline: {passed} pass, {failed} fail, {errors} errors (of 3)")

        if errors == 3:
            P(f"    SKIP: All baseline tests errored (compile/crash)")
            r = {"id": kid, "status": "SKIPPED", "reason": "baseline_all_error",
                 "baseline": {"passed": passed, "failed": failed, "errors": errors}}
            all_results.append(r)
            completed[kid] = r
            skip_compile += 1
            gpu_cleanup()
            continue

        if failed >= 2:
            P(f"    NOTE: Baseline mostly failing ({failed}/3) - kernel may have inherent differences")

        t0 = time.time()
        try:
            result = run_kernel_5dim(problem_file, kernel_code, kid)
            result["status"] = "COMPLETED"
            result["baseline"] = {"passed": passed, "failed": failed, "errors": errors}
        except Exception as e:
            P(f"    FATAL ERROR: {e}")
            result = {"id": kid, "status": "FATAL_ERROR", "error": str(e)[:500]}

        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        result["repo"] = entry.get("repo", "")
        result["kernel_name"] = entry.get("kernel_name", "")

        if elapsed > KERNEL_TIMEOUT:
            P(f"    WARNING: kernel took {elapsed:.0f}s (>{KERNEL_TIMEOUT}s), forcing GPU cleanup")
            try:
                subprocess.run(["nvidia-smi", "--gpu-reset"], timeout=10,
                               capture_output=True)
            except Exception:
                pass

        P(f"    Done in {elapsed:.1f}s | disc={result.get('total_discrepancies', '?')} "
          f"dims={result.get('discrepant_dimensions', [])}")

        detail_path = detail_dir / f"{kid}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        all_results.append(result)
        completed[kid] = result

        with open(checkpoint, "w", encoding="utf-8") as f:
            json.dump(completed, f, indent=2, ensure_ascii=False)

        gpu_cleanup()

    summary = {
        "dataset": dataset_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_kernels": total,
        "completed": sum(1 for r in all_results if r.get("status") == "COMPLETED"),
        "skipped_compile": skip_compile,
        "skipped_baseline": skip_baseline,
        "dimensions": ["value_stress", "dtype_stress", "training_stress", "repeated_run", "config_stress"],
        "tolerance": {"atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL},
        "seeds_per_policy": SEEDS_PER_POLICY,
        "baseline_seeds_quick": 3,
        "kernels_with_discrepancies": sum(1 for r in all_results
                                          if r.get("total_discrepancies", 0) > 0),
        "results": [],
    }

    for r in all_results:
        entry = {
            "id": r.get("id", "?"),
            "status": r.get("status", "UNKNOWN"),
            "repo": r.get("repo", ""),
            "kernel_name": r.get("kernel_name", ""),
            "total_discrepancies": r.get("total_discrepancies", 0),
            "discrepant_dimensions": r.get("discrepant_dimensions", []),
            "elapsed_s": r.get("elapsed_s", 0),
        }
        if r.get("status") == "COMPLETED":
            for dim in ["value_stress", "dtype_stress", "training_stress", "repeated_run", "config_stress"]:
                dd = r.get(dim, {})
                entry[f"{dim}_disc"] = dd.get("discrepancies", 0)
        summary["results"].append(entry)

    summary_path = result_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    P(f"\n  Summary saved: {summary_path}")
    P(f"  Completed: {summary['completed']}/{total}")
    P(f"  With discrepancies: {summary['kernels_with_discrepancies']}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["apex", "cuda-l1", "sakana", "tritonbench", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Limit kernels per dataset (0=all)")
    args = parser.parse_args()

    P("=" * 60)
    P("  MutaKernel Full-Scale Differential Testing")
    P("  5 Dimensions: value_stress | dtype_stress | training_stress | repeated_run | config_stress")
    P("=" * 60)

    # --- Apex new kernels ---
    if args.dataset in ("apex", "all"):
        from external_benchmarks.registry import EXTERNAL_KERNELS
        already_tested = {
            "apex__fused_layer_norm", "apex__fused_rms_norm",
            "apex__fused_dense", "apex__fused_dense_gelu_dense",
            "flash_attn__flash_attention_2"
        }
        new_apex = [e for e in EXTERNAL_KERNELS if e["id"] not in already_tested]
        if new_apex:
            apex_registry = []
            for e in new_apex:
                missing = []
                for pkg in e.get("requires", []):
                    try:
                        __import__(pkg)
                    except ImportError:
                        missing.append(pkg)
                if missing:
                    P(f"  SKIP {e['id']}: missing {missing}")
                    continue
                apex_registry.append({
                    "id": e["id"],
                    "repo": e["repo"],
                    "kernel_name": e["kernel_name"],
                    "reference_file": str(PROJECT_ROOT / e["reference_file"]),
                    "kernel_source": e["kernel_source"],
                })
            if apex_registry:
                if args.limit:
                    apex_registry = apex_registry[:args.limit]
                run_dataset("apex_new", apex_registry,
                           RESULT_BASE / "apex_new")
        else:
            P("  No new Apex kernels to test")

    # --- CUDA-L1 ---
    if args.dataset in ("cuda-l1", "all"):
        cuda_l1_reg_path = PROJECT_ROOT / "external_benchmarks" / "cuda_l1" / "registry.json"
        if not cuda_l1_reg_path.exists():
            P("\n  CUDA-L1 registry not found. Run prepare_external_datasets.py --cuda-l1 first.")
        else:
            with open(cuda_l1_reg_path, encoding="utf-8") as f:
                cuda_l1_registry = json.load(f)
            if args.limit:
                cuda_l1_registry = cuda_l1_registry[:args.limit]
            run_dataset("cuda_l1", cuda_l1_registry,
                       RESULT_BASE / "cuda_l1")

    # --- AI CUDA Engineer ---
    if args.dataset in ("sakana", "all"):
        sakana_reg_path = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"
        if not sakana_reg_path.exists():
            P("\n  AI CUDA Engineer registry not found. Run prepare_external_datasets.py --sakana first.")
        else:
            with open(sakana_reg_path, encoding="utf-8") as f:
                sakana_registry = json.load(f)
            if args.limit:
                sakana_registry = sakana_registry[:args.limit]
            run_dataset("ai_cuda_engineer", sakana_registry,
                       RESULT_BASE / "ai_cuda_engineer")

    # --- TritonBench-G ---
    if args.dataset in ("tritonbench", "all"):
        tb_reg_path = PROJECT_ROOT / "external_benchmarks" / "tritonbench_g" / "registry.json"
        if not tb_reg_path.exists():
            P("\n  TritonBench-G registry not found. Run prepare_external_datasets.py --tritonbench first.")
        else:
            with open(tb_reg_path, encoding="utf-8") as f:
                tb_registry = json.load(f)
            # Filter out identity-passthrough kernels: these have no real
            # PyTorch reference implementation, so correctness comparison
            # against the reference is meaningless (always reports discrepancy).
            tb_before = len(tb_registry)
            tb_filtered = []
            for e in tb_registry:
                ref = e.get("reference_file", "")
                ref_path = Path(ref) if os.path.isabs(ref) else PROJECT_ROOT / ref
                if ref_path.exists():
                    try:
                        with open(ref_path, encoding="utf-8") as rf:
                            if "identity passthrough" not in rf.read():
                                tb_filtered.append(e)
                    except Exception:
                        pass
            tb_registry = tb_filtered
            P(f"\n  TritonBench-G: {len(tb_registry)}/{tb_before} kernels with valid PyTorch reference"
              f" (filtered {tb_before - len(tb_registry)} identity-passthrough)")
            if args.limit:
                tb_registry = tb_registry[:args.limit]
            run_dataset("tritonbench_g", tb_registry,
                       RESULT_BASE / "tritonbench_g")

    P("\n" + "=" * 60)
    P("  ALL DATASETS DONE")
    P("=" * 60)


if __name__ == "__main__":
    main()
