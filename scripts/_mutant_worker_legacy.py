#!/usr/bin/env python3
"""Isolated subprocess worker running the *legacy* (pre-FSE-rework) runner.

Purpose: the E0 flip-rate study (Table 3) needs a same-environment paired
comparison — legacy substrate vs corrected substrate on identical probes —
so that verdict flips are attributable to the substrate rather than to
GPU/torch drift.  The legacy runner is the frozen pre-rework implementation
(`src/mutengine/mutant_runner_legacy.py`, extracted verbatim from tag
pre-fse-rework-20260719) with its known soundness defects intact:
consecutive model construction without RNG replay or state sync.

Usage: python _mutant_worker_legacy.py <config.json> <result.json>
Config schema: identical to scripts/_mutant_worker.py mode="run".
"""
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _run_mode(cfg):
    import torch  # noqa: F401
    from src.models import KernelInfo, Mutant, MutationSite
    from src.mutengine.mutant_runner_legacy import MutantRunner as LegacyRunner
    from src.bridge.eval_bridge import _load_module_from_path

    kernel = KernelInfo(
        problem_id=cfg["problem_id"],
        level=cfg["level"],
        problem_name=cfg["problem_name"],
        source_path=cfg.get("source_path", ""),
        kernel_code=cfg.get("kernel_code", ""),
        reference_module_path=cfg["problem_file"],
        language=cfg.get("language", "cuda"),
    )
    site = MutationSite(
        line_start=cfg["site"]["line_start"],
        line_end=cfg["site"]["line_end"],
        col_start=cfg["site"].get("col_start", 0),
        col_end=cfg["site"].get("col_end", 0),
        original_code=cfg["site"].get("original_code", ""),
        node_type=cfg["site"].get("node_type", ""),
    )
    mutant = Mutant(
        id=cfg["mutant_id"],
        operator_name=cfg["operator_name"],
        operator_category=cfg["operator_category"],
        site=site,
        original_code=cfg.get("original_code", ""),
        mutated_code=cfg["mutated_code"],
        description=cfg.get("description", ""),
    )
    runner = LegacyRunner(
        atol=cfg["atol"], rtol=cfg["rtol"],
        num_test_inputs=cfg["num_test_inputs"],
        device=cfg["device"], seed=cfg["seed"],
    )
    safe_id = cfg["mutant_id"].replace("-", "_").replace(".", "_")
    ref_mod = _load_module_from_path(cfg["problem_file"], f"lref_w_{safe_id}")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])

    runner.run_mutant(kernel, mutant, ref_mod, get_inputs, get_init_inputs)
    return {
        "status": mutant.status.value,
        "time_ms": mutant.execution_time_ms,
        "error": mutant.error_message or "",
        "kill_seed": mutant.kill_input_seed,
    }


def _apply_gpu_memory_budget():
    import os
    fraction = float(os.environ.get("MK_GPU_MEMORY_FRACTION", "0") or 0)
    if fraction <= 0:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
    except Exception:
        pass


def main():
    cfg_path, res_path = sys.argv[1], sys.argv[2]
    with open(cfg_path) as f:
        cfg = json.load(f)
    _apply_gpu_memory_budget()
    t0 = time.time()
    try:
        result = _run_mode(cfg)
    except Exception as e:  # noqa: BLE001 - worker boundary
        result = {
            "status": "stillborn",
            "error": f"LegacyWorkerCrash: {str(e)[:300]}",
            "time_ms": (time.time() - t0) * 1000,
            "kill_seed": None,
        }
    with open(res_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
