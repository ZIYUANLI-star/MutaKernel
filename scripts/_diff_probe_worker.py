#!/usr/bin/env python3
"""r-o differential probe worker — single subprocess for ONE kernel.

For one kernel (best_kernels entry):
  1. Load reference Module from KernelBench problem file
  2. Load original optimized kernel
  3. For each (stress_policy, seed) in (eval mode + train mode):
     - generate stress input
     - run ref → record ref_ok / ref_has_nan_inf
     - run orig → record orig_ok / orig_has_nan_inf
     - if not orig_ok: capture diff_summary
  4. Write per-event JSON

Usage:
    python _diff_probe_worker.py <config.json> <result.json>

Config keys:
    kernel_name      : str (e.g. "L1_P1")
    kernel_code      : str (the optimized kernel .py source)
    problem_file     : path to KernelBench reference .py
    policies         : list[str]  (stress policy names; "__identity__" allowed)
    seeds            : list[int]  (e.g. [42, 1337])
    device           : "cuda" / "cpu"
    atol, rtol       : tolerances (default 1e-2)
    include_training : bool (also test in .train() mode)
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _code_hash(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()[:10]


def _has_nan_inf(out):
    import torch
    if isinstance(out, torch.Tensor):
        return torch.isnan(out).any().item() or torch.isinf(out).any().item()
    if isinstance(out, (tuple, list)):
        return any(_has_nan_inf(x) for x in out)
    return False


def _allclose(a, b, atol, rtol):
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return False
        return torch.allclose(a.float().cpu(), b.float().cpu(),
                              atol=atol, rtol=rtol)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_allclose(x, y, atol, rtol) for x, y in zip(a, b))
    return a == b


def _diff_summary(ref_out, orig_out):
    import torch
    if isinstance(ref_out, torch.Tensor) and isinstance(orig_out, torch.Tensor):
        if ref_out.shape != orig_out.shape:
            return (f"shape_mismatch: ref={list(ref_out.shape)}, "
                    f"orig={list(orig_out.shape)}")
        diff = (ref_out.float().cpu() - orig_out.float().cpu()).abs()
        return (
            f"max_diff={diff.max().item():.6e}, "
            f"mean_diff={diff.mean().item():.6e}, "
            f"ref_range=[{ref_out.float().min().item():.4e},"
            f"{ref_out.float().max().item():.4e}], "
            f"orig_range=[{orig_out.float().min().item():.4e},"
            f"{orig_out.float().max().item():.4e}]"
        )
    return "complex_output_type"


def _sync_weights(src, dst):
    """Try to copy weights src → dst regardless of param naming differences."""
    import torch
    try:
        dst.load_state_dict(src.state_dict())
        return True
    except Exception:
        pass
    try:
        src_vals = list(src.state_dict().values())
        dst_sd = dst.state_dict()
        dst_keys = list(dst_sd.keys())
        if len(src_vals) != len(dst_keys):
            return False
        for i, key in enumerate(dst_keys):
            if src_vals[i].shape == dst_sd[key].shape:
                dst_sd[key] = src_vals[i].clone()
            else:
                return False
        dst.load_state_dict(dst_sd)
        return True
    except Exception:
        return False


def run(cfg):
    import torch
    from src.mutengine.mutant_runner import _load_module_from_source
    from src.bridge.eval_bridge import _load_module_from_path
    from src.stress.policy_bank import STRESS_POLICIES

    t0_total = time.time()
    device = cfg.get("device", "cuda")
    atol = cfg.get("atol", 1e-2)
    rtol = cfg.get("rtol", 1e-2)
    policies = cfg["policies"]
    seeds = cfg["seeds"]
    include_training = cfg.get("include_training", True)

    kernel_name = cfg["kernel_name"]
    kernel_code = cfg["kernel_code"]
    problem_file = cfg["problem_file"]

    # Load reference once
    ref_hash = _code_hash(problem_file)
    ref_mod = _load_module_from_path(problem_file, f"diff_ref_{ref_hash}")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
    init_args = get_init_inputs()
    ref_cls = ref_mod.Model

    # Load optimized kernel once
    tmp_dir = tempfile.mkdtemp(prefix="diff_probe_")
    orig_hash = _code_hash(kernel_code)
    t_compile_start = time.time()
    try:
        orig_mod = _load_module_from_source(
            kernel_code, f"diff_orig_{orig_hash}", tmp_dir,
        )
    except Exception as e:
        return {
            "kernel_name": kernel_name,
            "status": "orig_compile_failed",
            "error": f"{e!r}"[:400],
            "elapsed_sec": time.time() - t0_total,
            "events": [],
        }
    compile_sec = time.time() - t_compile_start

    orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")

    events = []
    bad_count = 0

    def _eval_one(policy_name, seed, train_mode):
        nonlocal bad_count
        # 1. Build stress input deterministically
        torch.manual_seed(seed)
        template_inputs = get_inputs()
        if policy_name == "__identity__":
            stress_inputs = template_inputs
        else:
            fn = STRESS_POLICIES.get(policy_name)
            if fn is None:
                return None
            try:
                stress_inputs = fn(template_inputs, seed)
            except Exception as e:
                return {
                    "policy": policy_name, "seed": seed, "train_mode": train_mode,
                    "stage": "stress_gen", "error": f"{e!r}"[:200],
                    "ref_ok": None, "orig_ok": None,
                }
        stress_on_device = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in stress_inputs
        ]

        # 2. Run reference
        torch.manual_seed(seed)
        ref_model = (ref_cls(*init_args) if isinstance(init_args, (list, tuple))
                     else ref_cls())
        ref_model = ref_model.to(device)
        ref_model.train(train_mode)

        try:
            if train_mode:
                ref_out = ref_model(*stress_on_device)
            else:
                ref_model.eval()
                with torch.no_grad():
                    ref_out = ref_model(*stress_on_device)
        except Exception as e:
            return {
                "policy": policy_name, "seed": seed, "train_mode": train_mode,
                "stage": "ref", "error": f"{e!r}"[:200],
                "ref_ok": False, "orig_ok": None,
            }
        ref_has_nan = _has_nan_inf(ref_out)

        # 3. Run original optimized kernel
        torch.manual_seed(seed)
        orig_model = (orig_cls(*init_args) if isinstance(init_args, (list, tuple))
                      else orig_cls())
        orig_model = orig_model.to(device)
        orig_model.train(train_mode)
        _sync_weights(ref_model, orig_model)

        try:
            if train_mode:
                orig_out = orig_model(*stress_on_device)
            else:
                orig_model.eval()
                with torch.no_grad():
                    orig_out = orig_model(*stress_on_device)
        except Exception as e:
            bad_count += 1
            return {
                "policy": policy_name, "seed": seed, "train_mode": train_mode,
                "stage": "orig_run", "error": f"{e!r}"[:200],
                "ref_ok": not ref_has_nan, "orig_ok": False,
                "ref_has_nan_inf": ref_has_nan,
            }
        orig_has_nan = _has_nan_inf(orig_out)

        # 4. Compare
        if ref_has_nan:
            # ref is NaN/Inf — input is degenerate; we only verify orig also produces NaN/Inf
            orig_ok = not orig_has_nan
            verdict = "ref_nan_orig_nan" if orig_has_nan else "ref_nan_orig_finite"
        else:
            ac = _allclose(ref_out, orig_out, atol, rtol)
            orig_ok = ac and not orig_has_nan
            verdict = "ok" if orig_ok else (
                "orig_nan_inf" if orig_has_nan else "diverged"
            )

        rec = {
            "policy": policy_name, "seed": seed, "train_mode": train_mode,
            "ref_ok": not ref_has_nan, "orig_ok": orig_ok,
            "ref_has_nan_inf": ref_has_nan,
            "orig_has_nan_inf": orig_has_nan,
            "verdict": verdict,
        }
        if not orig_ok and not ref_has_nan:
            try:
                rec["diff_summary"] = _diff_summary(ref_out, orig_out)
            except Exception:
                rec["diff_summary"] = ""
            bad_count += 1
        # Cleanup activations to save GPU memory
        del orig_out, ref_out, stress_on_device, stress_inputs
        torch.cuda.empty_cache()
        return rec

    # Plan: eval mode for all policies, train mode for all policies
    plan = []
    for pol in policies:
        for seed in seeds:
            plan.append((pol, seed, False))   # eval
    if include_training:
        for pol in policies:
            for seed in seeds:
                plan.append((pol, seed, True))   # train

    for pol, seed, tm in plan:
        ev = _eval_one(pol, seed, tm)
        if ev is not None:
            events.append(ev)

    elapsed = time.time() - t0_total
    return {
        "kernel_name": kernel_name,
        "status": "ok",
        "elapsed_sec": round(elapsed, 1),
        "compile_sec": round(compile_sec, 2),
        "policies_tested": len(policies),
        "seeds_per_policy": len(seeds),
        "include_training": include_training,
        "total_events": len(events),
        "bad_events": bad_count,
        "events": events,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: _diff_probe_worker.py <config.json> <result.json>",
              file=sys.stderr)
        sys.exit(2)
    cfg_path, out_path = sys.argv[1], sys.argv[2]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        result = run(cfg)
    except Exception as e:
        result = {
            "kernel_name": cfg.get("kernel_name", "?"),
            "status": "exception",
            "error": f"{e!r}",
            "traceback": traceback.format_exc()[-2000:],
            "events": [],
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[diff_probe_worker] {cfg.get('kernel_name')}: "
          f"status={result.get('status')}, "
          f"events={len(result.get('events',[]))}, "
          f"bad={result.get('bad_events','?')}", flush=True)


if __name__ == "__main__":
    main()
