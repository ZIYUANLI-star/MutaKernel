#!/usr/bin/env python3
"""Pilot stress test: 5 diverse survived mutants, 3-layer augmentation.

Runs inside WSL with native Linux paths. Self-contained — no project imports needed.
Uses the details JSON (which contains mutated_code) + KernelBench problem files.
"""
import gc
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
# MutaKernel project is on Windows, accessible from WSL via /mnt/d/...
MUTAKERNEL_ROOT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
BLOCK12_DIR = MUTAKERNEL_ROOT / "full_block12_results"
BEST_KERNELS = MUTAKERNEL_ROOT / "best_kernels.json"
RESULT_DIR = MUTAKERNEL_ROOT / "pilot_stress_results"

PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

DEVICE = "cuda"
SEED = 42
ATOL = 1e-2
RTOL = 1e-2
N_SEEDS_PER_POLICY = 3
N_SEEDS_DTYPE = 3
N_SEEDS_REPEATED = 3
N_TRIALS = 10

ALL_POLICIES = [
    "large_magnitude", "near_overflow", "near_zero", "denormals",
    "all_negative", "all_positive", "mixed_extremes", "alternating_sign",
    "sparse", "uniform_constant", "structured_ramp", "boundary_last_element",
]

STRATEGY_MAP = {
    "epsilon_modify":    ["near_zero", "denormals"],
    "scale_modify":      ["uniform_constant", "structured_ramp"],
    "stab_remove":       ["large_magnitude", "near_overflow"],
    "cast_remove":       ["near_overflow", "large_magnitude"],
    "init_modify":       ["all_negative", "sparse"],
    "acc_downgrade":     ["mixed_extremes", "large_magnitude"],
    "reduction_reorder": ["mixed_extremes", "alternating_sign"],
    "broadcast_unsafe":  ["structured_ramp"],
    "layout_assume":     ["structured_ramp"],
    "index_replace":     ["structured_ramp", "large_magnitude"],
    "mask_boundary":     ["boundary_last_element", "sparse"],
    "sync_remove":       ["large_magnitude", "mixed_extremes"],
    "launch_config_mutate": ["structured_ramp", "large_magnitude"],
    "arith_replace":     ["large_magnitude", "mixed_extremes"],
    "relop_replace":     ["boundary_last_element", "structured_ramp"],
    "const_perturb":     ["near_zero", "large_magnitude"],
}


def P(msg):
    print(msg, flush=True)


# ── Policy Bank (inline, no project imports) ──────────────────────────
import torch

def _overflow_threshold(dtype):
    if dtype == torch.float16:   return 60000.0
    if dtype == torch.bfloat16:  return 30000.0
    return 1e30

def _apply(tensor, fn, gen):
    if not tensor.dtype.is_floating_point:
        return tensor.clone()
    return fn(tensor, gen).to(dtype=tensor.dtype, device=tensor.device)

def _make(fn):
    def policy(inputs, seed):
        g = torch.Generator(); g.manual_seed(seed)
        return [_apply(x, fn, g) if isinstance(x, torch.Tensor) else x for x in inputs]
    return policy

_P = {
    "large_magnitude":  _make(lambda t,g: torch.randn(t.shape, dtype=torch.float32, generator=g)*1000),
    "near_overflow":    _make(lambda t,g: torch.randn(t.shape, dtype=torch.float32, generator=g)*_overflow_threshold(t.dtype)),
    "near_zero":        _make(lambda t,g: torch.randn(t.shape, dtype=torch.float32, generator=g)*1e-7),
    "denormals":        _make(lambda t,g: torch.randn(t.shape, dtype=torch.float32, generator=g)*1e-38),
    "all_negative":     _make(lambda t,g: -torch.abs(torch.randn(t.shape, dtype=torch.float32, generator=g))*100),
    "all_positive":     _make(lambda t,g:  torch.abs(torch.randn(t.shape, dtype=torch.float32, generator=g))*100),
    "mixed_extremes":   _make(lambda t,g: (lambda v,m: v.__setitem__(m, v[m]*10000) or v.__setitem__(~m, v[~m]*0.0001) or v)(torch.randn(t.shape,dtype=torch.float32,generator=g), torch.rand(t.shape,generator=g)>0.5)),
    "alternating_sign": _make(lambda t,g: (lambda v,s: v*s)(torch.randn(t.shape,dtype=torch.float32,generator=g).abs()*100, torch.ones(t.shape,dtype=torch.float32).view(-1).__setitem__(slice(1,None,2),-1.0) or torch.ones(t.shape,dtype=torch.float32).tap(lambda s: s.view(-1).__setitem__(slice(1,None,2),-1.0)))),
    "sparse":           _make(lambda t,g: (lambda v,m: v.__setitem__(m, torch.randn(int(m.sum().item()),dtype=torch.float32,generator=g)*100) or v)(torch.zeros(t.shape,dtype=torch.float32), torch.rand(t.shape,generator=g)>0.9)),
    "uniform_constant": _make(lambda t,g: torch.full(t.shape, 88.0, dtype=torch.float32)),
    "structured_ramp":  _make(lambda t,g: torch.arange(max(t.numel(),1), dtype=torch.float32).reshape(t.shape)/max(t.numel(),1)),
}

def _boundary_last_element(inputs, seed):
    g = torch.Generator(); g.manual_seed(seed)
    result = []
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
            v = torch.randn(x.shape, dtype=torch.float32, generator=g)
            v.view(-1)[-1] = 1e4
            result.append(v.to(dtype=x.dtype, device=x.device))
        else:
            result.append(x)
    return result
_P["boundary_last_element"] = _boundary_last_element

# Fix complex lambdas with proper functions
def _mixed_extremes_fn(inputs, seed):
    g = torch.Generator(); g.manual_seed(seed)
    result = []
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
            v = torch.randn(x.shape, dtype=torch.float32, generator=g)
            m = torch.rand(x.shape, generator=g) > 0.5
            v[m] *= 10000; v[~m] *= 0.0001
            result.append(v.to(dtype=x.dtype, device=x.device))
        else:
            result.append(x)
    return result
_P["mixed_extremes"] = _mixed_extremes_fn

def _alternating_sign_fn(inputs, seed):
    g = torch.Generator(); g.manual_seed(seed)
    result = []
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
            v = torch.randn(x.shape, dtype=torch.float32, generator=g).abs() * 100
            s = torch.ones(x.shape, dtype=torch.float32)
            s.view(-1)[1::2] = -1.0
            result.append((v*s).to(dtype=x.dtype, device=x.device))
        else:
            result.append(x)
    return result
_P["alternating_sign"] = _alternating_sign_fn

def _sparse_fn(inputs, seed):
    g = torch.Generator(); g.manual_seed(seed)
    result = []
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
            v = torch.zeros(x.shape, dtype=torch.float32)
            m = torch.rand(x.shape, generator=g) > 0.9
            n = int(m.sum().item())
            if n > 0:
                v[m] = torch.randn(n, dtype=torch.float32, generator=g) * 100
            result.append(v.to(dtype=x.dtype, device=x.device))
        else:
            result.append(x)
    return result
_P["sparse"] = _sparse_fn


# ── Helpers ───────────────────────────────────────────────────────────

def _load_mod(filepath, name):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def _load_source(code, name, tmp_dir):
    p = os.path.join(tmp_dir, f"{name}.py")
    with open(p, "w") as f:
        f.write(code)
    return _load_mod(p, name)

def _allclose(a, b, atol, rtol):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape: return False
        return torch.allclose(a.float().cpu(), b.float().cpu(), atol=atol, rtol=rtol)
    if isinstance(a, (tuple,list)) and isinstance(b, (tuple,list)):
        return len(a)==len(b) and all(_allclose(x,y,atol,rtol) for x,y in zip(a,b))
    return a == b

def _has_nan_inf(out):
    if isinstance(out, torch.Tensor):
        return torch.isnan(out).any().item() or torch.isinf(out).any().item()
    if isinstance(out, (tuple,list)):
        return any(_has_nan_inf(x) for x in out)
    return False

def find_problem_file(problem_dir, pid):
    pid = str(pid)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None

def load_all_survived():
    d = BLOCK12_DIR / "details"
    survived = []
    for jf in sorted(d.glob("*.json")):
        try: data = json.loads(jf.read_text(encoding="utf-8"))
        except: continue
        kn = data["kernel"]["problem_name"]
        for m in data.get("mutants", []):
            if m["status"] == "survived":
                survived.append((kn, data["kernel"], m))
    return survived


def pick_pilots(survived, n=5):
    seen = set()
    picked = []
    for target in ["stab_remove", "cast_remove", "sync_remove", "arith_replace", "epsilon_modify"]:
        if len(picked) >= n: break
        for item in survived:
            if item[2]["operator_name"] == target and target not in seen:
                picked.append(item); seen.add(target); break
    for item in survived:
        if len(picked) >= n: break
        op = item[2]["operator_name"]
        if op not in seen:
            picked.append(item); seen.add(op)
    return picked


# ── In-process stress execution (no subprocess, faster for pilot) ────

def _build_models(problem_file, kernel_code, mutated_code, device, tag):
    tmp = tempfile.mkdtemp(prefix="pilot_")
    ref_mod = _load_mod(str(problem_file), f"ref_{tag}")
    get_inputs = ref_mod.get_inputs
    init_args = getattr(ref_mod, "get_init_inputs", lambda: [])()

    ref_cls = ref_mod.Model
    ref_m = (ref_cls(*init_args) if isinstance(init_args,(list,tuple)) else ref_cls()).to(device).eval()

    orig_mod = _load_source(kernel_code, f"orig_{tag}", tmp)
    orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
    orig_m = (orig_cls(*init_args) if isinstance(init_args,(list,tuple)) else orig_cls()).to(device).eval()

    mut_mod = _load_source(mutated_code, f"mut_{tag}", tmp)
    mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
    mut_m = (mut_cls(*init_args) if isinstance(init_args,(list,tuple)) else mut_cls()).to(device).eval()

    return ref_m, orig_m, mut_m, get_inputs


def run_layer1(problem_file, kernel_code, mutated_code, operator):
    """Layer 1: 12 policies × 3 seeds."""
    mapped = STRATEGY_MAP.get(operator, [])
    priority = mapped + [p for p in ALL_POLICIES if p not in mapped]
    orig_failures = []

    tag = f"l1_{int(time.time()*1000) % 100000}"
    try:
        ref_m, orig_m, mut_m, get_inputs = _build_models(
            problem_file, kernel_code, mutated_code, DEVICE, tag)
    except Exception as e:
        P(f"      BUILD ERROR: {e}")
        return False, None, None, []

    for pi, pol in enumerate(priority):
        for si in range(N_SEEDS_PER_POLICY):
            seed = SEED + pi * N_SEEDS_PER_POLICY + si
            torch.manual_seed(seed)
            template = get_inputs()
            policy_fn = _P.get(pol)
            if policy_fn is None:
                continue
            stress = policy_fn(template, seed)
            stress_dev = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x for x in stress]

            with torch.no_grad():
                try:
                    ref_out = ref_m(*stress_dev)
                except Exception:
                    P(f"      [L1] {pol} s={seed} REF_CRASH"); continue
                if _has_nan_inf(ref_out):
                    P(f"      [L1] {pol} s={seed} REF_NaN"); continue
                try:
                    orig_out = orig_m(*stress_dev)
                    o_ok = _allclose(ref_out, orig_out, ATOL, RTOL) and not _has_nan_inf(orig_out)
                except Exception:
                    o_ok = False
                try:
                    mut_out = mut_m(*stress_dev)
                    m_ok = _allclose(ref_out, mut_out, ATOL, RTOL) and not _has_nan_inf(mut_out)
                except Exception:
                    m_ok = False

            if o_ok and not m_ok:
                P(f"      [L1] {pol} s={seed} -> KILLED!")
                return True, pol, seed, orig_failures
            if not o_ok and not m_ok:
                if pol not in orig_failures: orig_failures.append(pol)
                P(f"      [L1] {pol} s={seed} -> ORIG_ALSO_FAILS")
            elif not o_ok:
                P(f"      [L1] {pol} s={seed} -> ORIG_FAIL_ONLY")
            else:
                P(f"      [L1] {pol} s={seed} -> both OK")

    return False, None, None, orig_failures


def run_layer2(problem_file, kernel_code, mutated_code):
    """Layer 2: 3 seeds × 2 dtypes."""
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}

    tag = f"l2_{int(time.time()*1000) % 100000}"
    try:
        ref_m, orig_m, mut_m, get_inputs = _build_models(
            problem_file, kernel_code, mutated_code, DEVICE, tag)
    except Exception as e:
        P(f"      BUILD ERROR: {e}")
        return False, None, None

    for si in range(N_SEEDS_DTYPE):
        seed = SEED + 100 + si
        torch.manual_seed(seed)
        base = get_inputs()

        for dname, dt in dtype_map.items():
            try:
                cast_in = [
                    x.to(dtype=dt, device=DEVICE) if (isinstance(x, torch.Tensor) and x.dtype.is_floating_point)
                    else (x.to(DEVICE) if isinstance(x, torch.Tensor) else x)
                    for x in base
                ]
                ref_dt = ref_m.to(dt)
                orig_dt = orig_m.to(dt)
                mut_dt = mut_m.to(dt)

                with torch.no_grad():
                    ref_out = ref_dt(*cast_in)
                    if _has_nan_inf(ref_out):
                        P(f"      [L2] {dname} s={seed} REF_NaN"); continue
                    orig_out = orig_dt(*cast_in)
                    o_ok = _allclose(ref_out, orig_out, ATOL, RTOL) and not _has_nan_inf(orig_out)
                    mut_out = mut_dt(*cast_in)
                    m_ok = _allclose(ref_out, mut_out, ATOL, RTOL) and not _has_nan_inf(mut_out)

                if o_ok and not m_ok:
                    P(f"      [L2] {dname} s={seed} -> KILLED!")
                    return True, dname, seed
                P(f"      [L2] {dname} s={seed} -> {'both OK' if m_ok else 'ORIG_FAIL'}")

                ref_m = ref_m.float(); orig_m = orig_m.float(); mut_m = mut_m.float()
            except Exception as e:
                P(f"      [L2] {dname} s={seed} ERROR: {str(e)[:100]}")
                ref_m = ref_m.float(); orig_m = orig_m.float(); mut_m = mut_m.float()
                continue

    return False, None, None


def run_layer3(problem_file, kernel_code, mutated_code):
    """Layer 3: 3 seeds × 10 trials."""
    tag = f"l3_{int(time.time()*1000) % 100000}"
    try:
        ref_m, orig_m, mut_m, get_inputs = _build_models(
            problem_file, kernel_code, mutated_code, DEVICE, tag)
    except Exception as e:
        P(f"      BUILD ERROR: {e}")
        return False, None, False, None

    for si in range(N_SEEDS_REPEATED):
        seed = SEED + 200 + si
        torch.manual_seed(seed)
        base = get_inputs()
        inp = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x for x in base]

        with torch.no_grad():
            try:
                ref_out = ref_m(*inp)
            except Exception:
                P(f"      [L3] s={seed} REF_CRASH"); continue
            if _has_nan_inf(ref_out):
                P(f"      [L3] s={seed} REF_NaN"); continue

        mut_outs = []
        divergent = None
        for t in range(N_TRIALS):
            with torch.no_grad():
                try:
                    mout = mut_m(*inp)
                except Exception:
                    P(f"      [L3] s={seed} t={t} MUT_CRASH -> KILLED")
                    return True, t, False, seed
            if not _allclose(ref_out, mout, ATOL, RTOL) or _has_nan_inf(mout):
                if divergent is None: divergent = t
            if isinstance(mout, torch.Tensor):
                mut_outs.append(mout.float().cpu().clone())

        if divergent is not None:
            P(f"      [L3] s={seed} -> KILLED (diverged at trial {divergent})")
            return True, divergent, False, seed

        if len(mut_outs) >= 2:
            first = mut_outs[0]
            for i, o in enumerate(mut_outs[1:], 1):
                if not _allclose(first, o, 1e-6, 1e-6):
                    P(f"      [L3] s={seed} -> KILLED (self-inconsistent at trial {i})")
                    return True, i, True, seed

        P(f"      [L3] s={seed} -> not killed ({N_TRIALS} trials consistent)")

    return False, None, False, None


def gpu_cleanup():
    stale = [k for k in list(sys.modules) if k.startswith(("ref_","orig_","mut_","l1_","l2_","l3_"))]
    for k in stale: del sys.modules[k]
    gc.collect()
    torch.cuda.empty_cache()


def attribute(killed_layer, orig_failures, operator):
    if killed_layer == 1: return "Type 3 (Input-Distribution-Blind)"
    if killed_layer == 2: return "Type 2 (Configuration-Blind)"
    if killed_layer == 3: return "Type 5 (Non-deterministic)"
    if orig_failures:     return "Type 4 (Code Defect)"
    if operator in ("mask_boundary", "launch_config_mutate"):
        return "Type 1 (Algorithmically Absorbed)"
    return "Unresolved"


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BEST_KERNELS, "r") as f:
        bk = json.load(f)

    P(f"\n{'='*70}")
    P(f"  PILOT Stress Test (5 mutants, in-process, WSL)")
    P(f"  GPU: {torch.cuda.get_device_name(0)}")
    P(f"{'='*70}")

    survived = load_all_survived()
    P(f"  Total survived: {len(survived)}")

    pilots = pick_pilots(survived, 5)
    P(f"  Selected {len(pilots)} pilots:")
    for i, (kn, _, mm) in enumerate(pilots):
        P(f"    {i+1}. {mm['id']} ({mm['operator_name']}, {mm['operator_category']})")
    P("")

    results = []

    for idx, (kn, km, mm) in enumerate(pilots):
        mid = mm["id"]
        op = mm["operator_name"]
        P(f"\n{'_'*70}")
        P(f"  [{idx+1}/5] {mid}  |  {op} ({mm['operator_category']})")
        P(f"{'_'*70}")

        lkey = f"L{km['level']}"
        pdir = PROBLEM_DIRS.get(lkey)
        if not pdir:
            P(f"    SKIP: unknown level"); continue

        bi = bk.get(kn)
        if not bi:
            P(f"    SKIP: not in best_kernels"); continue

        kpath = Path(bi["kernel_path"])
        if not kpath.exists():
            P(f"    SKIP: kernel not found {kpath}"); continue

        pfile = find_problem_file(pdir, km["problem_id"])
        if not pfile:
            P(f"    SKIP: problem file not found"); continue

        kcode = kpath.read_text()
        mcode = mm.get("mutated_code", "")
        if not mcode:
            P(f"    SKIP: no mutated_code"); continue

        P(f"    kernel: {kpath.name}  problem: {pfile.name}  mutant: {len(mcode)} chars")

        # Layer 1
        P(f"\n    === Layer 1: Value Stress (12 pol × 3 seeds = 36) ===")
        t1 = time.time()
        l1_k, l1_pol, l1_seed, ofails = run_layer1(pfile, kcode, mcode, op)
        P(f"    Layer 1 done [{time.time()-t1:.1f}s]")

        l2_k = l2_dt = l2_seed = None
        l3_k = l3_trial = l3_seed = None
        l3_si = False

        if l1_k:
            attr = attribute(1, ofails, op)
        else:
            # Layer 2
            P(f"\n    === Layer 2: Dtype Stress (3 seeds × 2 dtypes = 6) ===")
            t2 = time.time()
            l2_k, l2_dt, l2_seed = run_layer2(pfile, kcode, mcode)
            P(f"    Layer 2 done [{time.time()-t2:.1f}s]")

            if l2_k:
                attr = attribute(2, ofails, op)
            else:
                # Layer 3
                P(f"\n    === Layer 3: Repeated Run (3 seeds × 10 trials = 30) ===")
                t3 = time.time()
                l3_k, l3_trial, l3_si, l3_seed = run_layer3(pfile, kcode, mcode)
                P(f"    Layer 3 done [{time.time()-t3:.1f}s]")

                attr = attribute(3 if l3_k else 0, ofails, op)

        any_killed = l1_k or (l2_k or False) or (l3_k or False)
        P(f"\n    >>> {'KILLED' if any_killed else 'SURVIVED'}  >>>  {attr}")

        results.append({
            "mutant_id": mid, "operator": op, "category": mm["operator_category"],
            "kernel": kn,
            "l1_killed": l1_k, "l1_policy": l1_pol, "l1_seed": l1_seed,
            "l2_killed": l2_k, "l2_dtype": l2_dt, "l2_seed": l2_seed,
            "l3_killed": l3_k, "l3_trial": l3_trial, "l3_self_inconsistent": l3_si, "l3_seed": l3_seed,
            "original_failures": ofails,
            "attribution": attr,
        })
        gpu_cleanup()

    out = RESULT_DIR / "pilot_5_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    P(f"\n{'='*70}")
    P(f"  PILOT COMPLETE  ({elapsed/60:.1f} min)")
    P(f"{'='*70}")
    for r in results:
        k = "KILLED" if any([r["l1_killed"], r.get("l2_killed"), r.get("l3_killed")]) else "SURVIVED"
        P(f"  {r['mutant_id']:45s} {k:8s} {r['attribution']}")
    P(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
