"""Differential stress-test worker: tests ONE mutant in isolation.

MutaKernel enhanced testing (deterministic dimensions only, NO LLM):
  - value_stress  : 21 value-distribution policies x seeds   (policy_bank)
  - config_stress : batch-size variation only (first/batch dim; all other
                    tensor dimensions stay fixed at the canonical shape)
  - dtype_stress  : re-run under alternative float dtypes
  - repeated_run  : detect nondeterminism (UB / race) vs original

Kill criterion (differential oracle): original kernel runs OK and the mutant
either errors or produces a non-bitwise-identical output -> KILLED.

Reads a job JSON: {refmod, original_code, mutated_code, operator, out}
Writes the result JSON to `out`. Designed to be spawned per mutant so a CUDA
illegal-memory-access only kills this process (orchestrator treats that as a
crash-kill).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from stress.policy_bank import STRESS_POLICIES, get_all_policy_names  # noqa: E402

VALUE_SEEDS = [30000, 30001]
REPEAT_N = 3
# config_stress: vary ONLY the batch (first) dimension; all other dims stay fixed
CONFIG_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
CONFIG_SEEDS = [40000, 40001, 40002]


# ---------------------------------------------------------------------------
def load_refmod(path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_refmod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_kernel_from_file(path: str, name: str):
    """Import a kernel module from a real .py file.

    Must be a real file (not exec'd string) so triton's @triton.jit can use
    inspect.getsource / linecache on the kernel function.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    fn = mod.__dict__.get("kernel_fn")
    if fn is None and "ModelNew" in mod.__dict__:
        inst = mod.__dict__["ModelNew"]()
        fn = lambda *a: inst(*a)  # noqa: E731
    if fn is None:
        raise RuntimeError("no kernel_fn / ModelNew in kernel module")
    return fn


def _to(t, dtype=None, device="cuda"):
    if isinstance(t, torch.Tensor):
        if dtype is not None and t.dtype.is_floating_point:
            t = t.to(dtype)
        return t.to(device)
    return t


def clone_inputs(inputs):
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]


def bitwise_equal(a, b) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(bitwise_equal(x, y) for x, y in zip(a, b))
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return a == b
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a = a.detach().float()
    b = b.detach().float()
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    if not torch.equal(nan_a, nan_b):
        return False
    mask = ~nan_a
    return torch.equal(a[mask], b[mask])


def run_fn(fn, inputs):
    """Returns (ok, output_or_None, err_str). Synchronizes to surface async errs."""
    try:
        out = fn(*clone_inputs(inputs))
        torch.cuda.synchronize()
        return True, out, ""
    except Exception as e:  # noqa: BLE001
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return False, None, f"{type(e).__name__}: {str(e)[:200]}"


def differential(orig_fn, mut_fn, inputs):
    """One differential trial. Returns dict: killed/reason/original_failed."""
    ok_o, out_o, err_o = run_fn(orig_fn, inputs)
    if not ok_o:
        return {"killed": False, "original_failed": True, "orig_err": err_o}
    ok_m, out_m, err_m = run_fn(mut_fn, inputs)
    if not ok_m:
        return {"killed": True, "reason": f"mutant_error: {err_m}"}
    if not bitwise_equal(out_o, out_m):
        return {"killed": True, "reason": "output_differs"}
    return {"killed": False}


# --- config_stress: vary ONLY the batch (first) dimension ---------------------
# Faithful to MutaKernel's Config-Stress contract: "vary the batch dimension
# (the first dimension of input tensors) while keeping all other tensor
# dimensions fixed".  All feature/inner dimensions are taken verbatim from the
# problem's canonical inputs (`template`); only the leading dimension changes.
def config_input_sets(op: str, template, dt: torch.dtype):
    if not template or not isinstance(template[0], torch.Tensor) or template[0].dim() < 1:
        return []
    d = dict(device="cuda", dtype=dt)
    t0 = template[0]
    sets = []
    for bs in CONFIG_BATCH_SIZES:
        for seed in CONFIG_SEEDS:
            g = torch.Generator(device="cuda")
            g.manual_seed(seed)
            label = f"batch_{bs}_s{seed - CONFIG_SEEDS[0]}"
            if op in ("softmax", "reduce"):
                N = t0.shape[-1]
                inp = [torch.randn(bs, N, generator=g, **d)]
            elif op == "rmsnorm":
                N = t0.shape[-1]
                inp = [torch.randn(bs, N, generator=g, **d), template[1]]
            elif op == "layernorm":
                N = t0.shape[-1]
                inp = [torch.randn(bs, N, generator=g, **d), template[1], template[2]]
            elif op == "cross_entropy":
                C = t0.shape[-1]
                inp = [torch.randn(bs, C, generator=g, **d),
                       torch.randint(0, C, (bs,), device="cuda", dtype=torch.long, generator=g)]
            elif op == "matmul":
                K = t0.shape[-1]          # A is [M, K]; vary M (rows), keep B [K, N]
                inp = [torch.randn(bs, K, generator=g, **d), template[1]]
            elif op == "flash_attention":
                H, S, D = t0.shape[1], t0.shape[2], t0.shape[3]
                inp = [torch.randn(bs, H, S, D, generator=g, **d),
                       torch.randn(bs, H, S, D, generator=g, **d),
                       torch.randn(bs, H, S, D, generator=g, **d)]
            elif op == "rotary_embedding":
                D = t0.shape[-1]
                Dc, Ds = template[1].shape[-1], template[2].shape[-1]
                inp = [torch.randn(bs, D, generator=g, **d),
                       torch.randn(bs, Dc, generator=g, **d),
                       torch.randn(bs, Ds, generator=g, **d)]
            else:
                continue
            sets.append((label, inp))
    return sets


def alt_dtypes(base: torch.dtype):
    alld = [torch.float16, torch.bfloat16, torch.float32]
    return [x for x in alld if x != base]


_DT_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32, "float64": torch.float64}


def parse_dtype(kernel_name: str):
    """Authoritative base dtype from the problem name (refmod DT is unreliable)."""
    for tok in kernel_name.split("__"):
        if tok in _DT_MAP:
            return _DT_MAP[tok]
    return None


# ---------------------------------------------------------------------------
def main():
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    out_path = job["out"]
    t0 = time.time()
    result = {
        "mutant_id": job["id"], "operator_name": job["operator_name"],
        "operator_category": job["operator_category"], "kernel_name": job["kernel_name"],
        "final_emd_status": job["final_emd_status"],
        "main_track": {}, "config_track": {},
        "any_killed": False, "first_kill_mode": None, "original_failures": [],
    }
    kill_order = []

    refmod = load_refmod(job["refmod"])
    # problem-name dtype is authoritative; refmod DT is unreliable (scrambled).
    base_dt = parse_dtype(job["kernel_name"]) or getattr(refmod, "DT", torch.float16)
    # write kernel code to real .py files (required for triton inspect.getsource)
    stem = os.path.splitext(out_path)[0]
    orig_py = stem + ".orig_kernel.py"
    mut_py = stem + ".mut_kernel.py"
    with open(orig_py, "w", encoding="utf-8") as f:
        f.write(job["original_code"])
    with open(mut_py, "w", encoding="utf-8") as f:
        f.write(job["mutated_code"])
    orig_fn = load_kernel_from_file(orig_py, "orig_kernel_mod")
    mut_fn = load_kernel_from_file(mut_py, "mut_kernel_mod")

    # sanity: original must run on the canonical input
    torch.manual_seed(12345)
    base_inputs = [_to(x, dtype=base_dt, device="cuda") for x in refmod.get_inputs()]
    ok_o, _, err_o = run_fn(orig_fn, base_inputs)
    if not ok_o:
        result["original_failures"].append(f"base: {err_o}")

    op = job["operator"]

    # ---- value_stress ----
    vs = {"killed": False, "killing_policy": None, "trials": 0, "original_failures": []}
    for pname in get_all_policy_names():
        done = False
        for seed in VALUE_SEEDS:
            try:
                torch.manual_seed(seed)
                template = [_to(x, dtype=base_dt, device="cuda") for x in refmod.get_inputs()]
                inp = STRESS_POLICIES[pname](template, seed)
                inp = [_to(x, device="cuda") for x in inp]
            except Exception as e:  # noqa: BLE001
                continue
            r = differential(orig_fn, mut_fn, inp)
            vs["trials"] += 1
            if r.get("original_failed"):
                continue
            if r.get("killed"):
                vs["killed"] = True
                vs["killing_policy"] = pname
                vs["reason"] = r.get("reason")
                done = True
                break
        if done:
            break
    result["main_track"]["value_stress"] = vs
    if vs["killed"]:
        kill_order.append("value_stress")

    # ---- config_stress (batch-size variation only; other dims fixed) ----
    cs = {"killed": False, "killing_policy": None, "trials": 0}
    for label, inp in config_input_sets(op, base_inputs, base_dt):
        inp = [_to(x, device="cuda") for x in inp]
        r = differential(orig_fn, mut_fn, inp)
        cs["trials"] += 1
        if r.get("original_failed"):
            continue
        if r.get("killed"):
            cs["killed"] = True
            cs["killing_policy"] = label
            cs["reason"] = r.get("reason")
            break
    result["config_track"]["config_stress"] = cs
    if cs["killed"]:
        kill_order.append("config_stress")

    # ---- dtype_stress ----
    ds = {"killed": False, "killing_policy": None, "trials": 0}
    for dt in alt_dtypes(base_dt):
        try:
            torch.manual_seed(777)
            inp = [_to(x, dtype=dt, device="cuda") for x in refmod.get_inputs()]
        except Exception:
            continue
        r = differential(orig_fn, mut_fn, inp)
        ds["trials"] += 1
        if r.get("original_failed"):
            continue
        if r.get("killed"):
            ds["killed"] = True
            ds["killing_policy"] = str(dt).replace("torch.", "")
            ds["reason"] = r.get("reason")
            break
    result["main_track"]["dtype_stress"] = ds
    if ds["killed"]:
        kill_order.append("dtype_stress")

    # ---- repeated_run (nondeterminism / UB) ----
    rr = {"killed": False, "trials": 0}
    torch.manual_seed(999)
    inp = [_to(x, dtype=base_dt, device="cuda") for x in refmod.get_inputs()]
    ok_o, out_o0, _ = run_fn(orig_fn, inp)
    if ok_o:
        for _ in range(REPEAT_N):
            rr["trials"] += 1
            ok_m, out_m, err_m = run_fn(mut_fn, inp)
            if not ok_m:
                rr["killed"] = True
                rr["reason"] = f"mutant_error: {err_m}"
                break
            if not bitwise_equal(out_o0, out_m):
                rr["killed"] = True
                rr["reason"] = "output_differs_or_nondeterministic"
                break
    result["main_track"]["repeated_run"] = rr
    if rr["killed"]:
        kill_order.append("repeated_run")

    result["any_killed"] = bool(kill_order)
    result["first_kill_mode"] = kill_order[0] if kill_order else None
    result["killed_dimensions"] = kill_order
    result["total_time_ms"] = round((time.time() - t0) * 1000, 1)

    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"OK {job['id']} killed={result['any_killed']} mode={result['first_kill_mode']}")


if __name__ == "__main__":
    main()
