#!/usr/bin/env python3
"""Re-run only config_stress (Dim 6) for all kernels with fixed allclose comparison.

Updates existing result JSONs in-place.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORKER_SCRIPT = SCRIPT_DIR / "_stress_worker.py"
RESULT_DIR = PROJECT_ROOT / "第三次实验汇总" / "results" / "details"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SEEDS_PER_POLICY = 3
STRESS_TIMEOUT = 540
DEVICE = "cuda"
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2


def P(msg):
    print(msg, flush=True)


def _run_stress_worker(cfg, timeout):
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="cfgre_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="cfgre_r_")
    os.close(cfg_fd)
    os.close(res_fd)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    try:
        proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT), cfg_path, res_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT), start_new_session=True,
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


def main():
    from external_benchmarks.registry import EXTERNAL_KERNELS

    P("=" * 60)
    P("  Re-running Dim 6: config_stress (allclose fix)")
    P("=" * 60)

    for entry in EXTERNAL_KERNELS:
        kid = entry["id"]
        result_path = RESULT_DIR / f"{kid}.json"
        if not result_path.exists():
            P(f"\n  SKIP {kid}: no existing result file")
            continue

        with open(result_path) as f:
            result = json.load(f)

        problem_file = str(PROJECT_ROOT / entry["reference_file"])
        kernel_code = entry["kernel_source"]
        cfg_seeds = [90000 + i for i in range(SEEDS_PER_POLICY)]

        P(f"\n  {kid}: running config_stress ({len(BATCH_SIZES)} batch_sizes x {SEEDS_PER_POLICY} seeds)")

        cfg = {
            "mode": "config_stress",
            "problem_file": problem_file,
            "kernel_code": kernel_code,
            "mutated_code": kernel_code,
            "batch_sizes": BATCH_SIZES,
            "seeds": cfg_seeds,
            "atol": DEFAULT_ATOL,
            "rtol": DEFAULT_RTOL,
            "device": DEVICE,
            "sync_weights": True,
        }

        t0 = time.time()
        data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)
        elapsed = time.time() - t0

        config_results = {}
        config_disc = 0

        if data is None:
            P(f"    TIMEOUT ({elapsed:.1f}s)")
            config_results["status"] = "timeout"
        elif data.get("error"):
            P(f"    ERROR: {data['error'][:100]}")
            config_results["status"] = "error"
            config_results["error"] = data["error"][:300]
        elif data.get("killed"):
            bs = data.get("killing_batch_size", "?")
            kill_type = data.get("kill_type", "?")
            P(f"    KILLED at batch_size={bs} ({kill_type})")
            config_disc = 1
            config_results["killed"] = True
            config_results["killing_batch_size"] = bs
            config_results["kill_type"] = kill_type
            rpb = data.get("results_per_batch", {})
            for bs_str, bs_data in rpb.items():
                config_results[f"batch_{bs_str}"] = {
                    "has_discrepancy": bs_str == str(bs),
                    "raw": bs_data,
                }
                tag = "DISCREPANCY" if bs_str == str(bs) else "pass"
                P(f"    batch_size={bs_str}: {tag}")
        else:
            rpb = data.get("results_per_batch", {})
            for bs_str, bs_data in rpb.items():
                has_disc = False
                if isinstance(bs_data, dict):
                    if bs_data.get("status") in ("killed_by_crash", "killed_by_divergence"):
                        has_disc = True
                    seeds_tested = bs_data.get("seeds_tested", [])
                    for sr in seeds_tested:
                        if isinstance(sr, dict) and sr.get("status") == "orig_diverges_from_ref":
                            has_disc = True
                            break
                if has_disc:
                    config_disc += 1
                config_results[f"batch_{bs_str}"] = {
                    "has_discrepancy": has_disc,
                    "raw": bs_data,
                }
                tag = "DISCREPANCY" if has_disc else "pass"
                P(f"    batch_size={bs_str}: {tag}")

        P(f"    Result: {config_disc} discrepancies ({elapsed:.1f}s)")

        # Update existing result
        result["config_stress"] = config_results
        old_cs_disc = result.get("summary", {}).get("config_stress_discrepancies", 0)
        result["summary"]["config_stress_discrepancies"] = config_disc

        # Recalculate total and discrepant_dimensions
        dims_disc = []
        total = 0
        for dim_key in ["value_stress", "dtype_stress", "training_stress",
                        "repeated_run", "config_stress"]:
            d_count = result["summary"].get(f"{dim_key}_discrepancies", 0)
            total += d_count
            if d_count > 0:
                dims_disc.append(dim_key)
        result["summary"]["total_discrepancies"] = total
        result["summary"]["discrepant_dimensions"] = dims_disc

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        P(f"    Updated: {result_path}")

    # Update summary.json
    summary_path = RESULT_DIR.parent / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for sr in summary.get("results", []):
            detail_path = RESULT_DIR / f"{sr['id']}.json"
            if detail_path.exists():
                with open(detail_path) as f:
                    d = json.load(f)
                s = d.get("summary", {})
                sr["config_stress_discrepancies"] = s.get("config_stress_discrepancies", 0)
                sr["total_discrepancies"] = s.get("total_discrepancies", 0)
                sr["discrepant_dimensions"] = s.get("discrepant_dimensions", [])
        summary["config_stress_fix"] = "replaced _bitwise_eq with _allclose in run_config_stress"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        P(f"\n  Updated: {summary_path}")

    P("\n" + "=" * 60)
    P("  DONE - config_stress re-run complete")
    P("=" * 60)


if __name__ == "__main__":
    main()
