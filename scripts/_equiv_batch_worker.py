#!/usr/bin/env python3
"""Batch equivalence check worker: processes ALL mutants from one kernel
in a single process to avoid redundant compilation of the original kernel.

Usage:
    python _equiv_batch_worker.py <config.json> <result.json>

config.json:
  {
    "problem_file": "...",
    "kernel_code": "...",
    "mutants": [{"mutant_id": "...", "mutated_code": "..."}, ...],
    "device": "cuda",
    "equiv_runs": 100,
    "base_seed": 10000
  }
"""
import hashlib
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

EQUIV_STRESS_POLICIES = [
    "large_magnitude", "near_zero", "structured_ramp",
    "all_negative", "sparse", "boundary_last_element",
]


def _bitwise_identical(a, b):
    """NaN-aware bitwise comparison."""
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point():
            nan_a = torch.isnan(a)
            nan_b = torch.isnan(b)
            if not torch.equal(nan_a, nan_b):
                return False
            finite = ~nan_a
            if finite.any():
                return torch.equal(a[finite], b[finite])
            return True
        return torch.equal(a, b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_bitwise_identical(x, y) for x, y in zip(a, b))
    return a == b


def main():
    import torch
    from src.mutengine.mutant_runner import _load_module_from_source, CompilationError
    from src.bridge.eval_bridge import _load_module_from_path
    from src.stress.policy_bank import STRESS_POLICIES

    cfg_path, res_path = sys.argv[1], sys.argv[2]
    with open(cfg_path) as f:
        cfg = json.load(f)

    device = cfg["device"]
    equiv_runs = cfg.get("equiv_runs", 100)
    base_seed = cfg.get("base_seed", 10000)
    kernel_code = cfg["kernel_code"]
    mutants = cfg["mutants"]

    tmp_dir = tempfile.mkdtemp(prefix="eqbatch_")
    results = []

    ref_mod = _load_module_from_path(cfg["problem_file"], "ref_eqbatch")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
    init_args = get_init_inputs()

    t_compile_orig = time.time()
    orig_hash = hashlib.md5(kernel_code.encode()).hexdigest()[:10]
    try:
        orig_mod = _load_module_from_source(
            kernel_code, f"eqo_{orig_hash}", tmp_dir,
        )
    except Exception as e:
        for m in mutants:
            results.append({
                "mutant_id": m["mutant_id"],
                "is_equivalent": False,
                "error": f"OrigCompile: {str(e)[:200]}",
            })
        with open(res_path, "w") as f:
            json.dump(results, f)
        return

    orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
    orig_model = (orig_cls(*init_args) if isinstance(init_args, (list, tuple))
                  else orig_cls())
    orig_model = orig_model.to(device).eval()
    compile_orig_ms = (time.time() - t_compile_orig) * 1000
    print(f"  Original kernel compiled in {compile_orig_ms:.0f}ms", flush=True)

    for mi, m_cfg in enumerate(mutants):
        mutant_id = m_cfg["mutant_id"]
        mutated_code = m_cfg["mutated_code"]
        t0 = time.time()

        safe_id = mutant_id.replace("-", "_").replace(".", "_")
        try:
            mut_mod = _load_module_from_source(
                mutated_code, f"eqm_{safe_id}", tmp_dir,
            )
        except Exception as e:
            results.append({
                "mutant_id": mutant_id,
                "is_equivalent": False,
                "error": f"MutCompile: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000,
            })
            print(f"  [{mi+1}/{len(mutants)}] {mutant_id}: COMPILE_ERROR", flush=True)
            continue

        mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
        mut_model = (mut_cls(*init_args) if isinstance(init_args, (list, tuple))
                     else mut_cls())
        mut_model = mut_model.to(device).eval()

        is_equiv = True

        for i in range(equiv_runs):
            seed = base_seed + i
            torch.manual_seed(seed)
            inputs = get_inputs()
            moved = [x.to(device) if isinstance(x, torch.Tensor) else x
                     for x in inputs]
            with torch.no_grad():
                orig_out = orig_model(*moved)
                mut_out = mut_model(*moved)
            if not _bitwise_identical(orig_out, mut_out):
                is_equiv = False
                break

        if is_equiv:
            for policy_name in EQUIV_STRESS_POLICIES:
                policy_fn = STRESS_POLICIES.get(policy_name)
                if policy_fn is None:
                    continue
                for si in range(2):
                    seed = base_seed + equiv_runs + si
                    torch.manual_seed(seed)
                    try:
                        template = get_inputs()
                        stress_inputs = policy_fn(template, seed)
                    except Exception:
                        continue
                    moved = [x.to(device) if isinstance(x, torch.Tensor) else x
                             for x in stress_inputs]

                    orig_exc = mut_exc = None
                    orig_out = mut_out = None
                    try:
                        with torch.no_grad():
                            orig_out = orig_model(*moved)
                    except Exception as e:
                        orig_exc = e
                    try:
                        with torch.no_grad():
                            mut_out = mut_model(*moved)
                    except Exception as e:
                        mut_exc = e

                    if orig_exc is not None and mut_exc is not None:
                        if type(orig_exc) is type(mut_exc):
                            continue
                        is_equiv = False
                        break
                    if orig_exc is not None or mut_exc is not None:
                        oom_msg = str(orig_exc or mut_exc).lower()
                        if "out of memory" in oom_msg:
                            continue
                        is_equiv = False
                        break
                    if not _bitwise_identical(orig_out, mut_out):
                        is_equiv = False
                        break
                if not is_equiv:
                    break

        elapsed = (time.time() - t0) * 1000
        tag = "EQUIVALENT" if is_equiv else "SURVIVED"
        print(f"  [{mi+1}/{len(mutants)}] {mutant_id}: {tag} ({elapsed:.0f}ms)",
              flush=True)

        results.append({
            "mutant_id": mutant_id,
            "is_equivalent": is_equiv,
            "time_ms": elapsed,
        })

        del mut_model, mut_mod
        torch.cuda.empty_cache()

    with open(res_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
