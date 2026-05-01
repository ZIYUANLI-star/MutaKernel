#!/usr/bin/env python3
"""Kill-rate smoke test: sample mutants, compile & run on GPU.

Fixes applied:
  P1 - _patch_load_inline_names: unique CUDA JIT name per mutant
  P2 - Tighter tolerance: atol/rtol = 1e-5
  P3 - Enhanced numerical inputs for survived mutants (second-pass)
"""
import importlib.util
import os
import random
import re
import signal
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.mutengine.operators.arithmetic import ArithReplace, RelOpReplace, ConstPerturb
from src.mutengine.operators.gpu_parallel import (
    IndexReplace, SyncRemove, MaskBoundary, LaunchConfigMutate,
)
from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify, ScaleModify,
    CastRemove, ReductionReorder, InitModify,
)
from src.mutengine.operators.llm_pattern import BroadcastUnsafe, LayoutAssume
from src.mutrepair.enhanced_inputs import EnhancedInputGenerator

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
RUN_DIR = KB_ROOT / "runs" / "iter_full_l1_caesar_paper_v2"
PROBLEM_DIR = KB_ROOT / "KernelBench" / "level1"

ATOL, RTOL = 1e-5, 1e-5
NUM_TRIALS = 3
DEVICE = "cuda"
MUTANT_TIMEOUT = 180
SEED = 42

TARGET_KERNELS = [
    "level_1_problem_5_sample_0_kernel.py",
    "level_1_problem_19_sample_0_kernel.py",
    "level_1_problem_6_sample_0_kernel.py",
]

ALL_OPS = [
    ("A1", ArithReplace()),
    ("A2", RelOpReplace()),
    ("A3", ConstPerturb()),
    ("B1", IndexReplace()),
    ("B2", SyncRemove()),
    ("B3", MaskBoundary()),
    ("B4", LaunchConfigMutate()),
    ("C1", StabRemove()),
    ("C2", AccDowngrade()),
    ("C3", EpsilonModify()),
    ("C4", ScaleModify()),
    ("C5", CastRemove()),
    ("C6", ReductionReorder()),
    ("C7", InitModify()),
    ("D1", BroadcastUnsafe()),
    ("D2", LayoutAssume()),
]

OP_NAME_MAP = {
    "A1": "arith_replace", "A2": "relop_replace", "A3": "const_perturb",
    "B1": "index_replace", "B2": "sync_remove", "B3": "mask_boundary",
    "B4": "launch_config_mutate",
    "C1": "stab_remove", "C2": "acc_downgrade", "C3": "epsilon_modify",
    "C4": "scale_modify", "C5": "cast_remove", "C6": "reduction_reorder",
    "C7": "init_modify",
    "D1": "broadcast_unsafe", "D2": "layout_assume",
}

# --- P1 fix: patch load_inline names ---
_LOAD_INLINE_NAME_RE = re.compile(
    r"""(load_inline\s*\([^)]*?name\s*=\s*)(['"])([^'"]+)\2""",
    re.DOTALL,
)

def _patch_load_inline_names(source: str, unique_suffix: str) -> str:
    def _repl(m: re.Match) -> str:
        prefix, quote, name = m.group(1), m.group(2), m.group(3)
        return f"{prefix}{quote}{name}_{unique_suffix}{quote}"
    return _LOAD_INLINE_NAME_RE.sub(_repl, source)


class MutantTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise MutantTimeout()


def P(msg):
    print(msg, flush=True)


def find_problem_file(problem_id):
    for f in PROBLEM_DIR.iterdir():
        if f.name.startswith(f"{problem_id}_") and f.suffix == ".py":
            return f
    return None


def load_module(filepath, name):
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_module_from_source(source, name, tmp_dir):
    source = _patch_load_inline_names(source, name)
    fp = os.path.join(tmp_dir, f"{name}.py")
    with open(fp, "w") as f:
        f.write(source)
    return load_module(fp, name)


def compare(ref, mut):
    if isinstance(ref, torch.Tensor) and isinstance(mut, torch.Tensor):
        if ref.shape != mut.shape:
            return False
        return torch.allclose(ref.float().cpu(), mut.float().cpu(), atol=ATOL, rtol=RTOL)
    if isinstance(ref, (tuple, list)) and isinstance(mut, (tuple, list)):
        return len(ref) == len(mut) and all(compare(r, m) for r, m in zip(ref, mut))
    return ref == mut


def _run_one_trial(ref_model, mut_model, get_inputs, trial_seed):
    """Run ref and mut models on the same inputs. Returns 'killed' or None."""
    torch.manual_seed(trial_seed)
    inputs = get_inputs()
    ref_d = ref_model.to(DEVICE).eval()
    moved = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x for x in inputs]
    with torch.no_grad():
        ref_out = ref_d(*moved)

    torch.manual_seed(trial_seed)
    inputs2 = get_inputs()
    mut_d = mut_model.to(DEVICE).eval()
    moved2 = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x for x in inputs2]
    with torch.no_grad():
        mut_out = mut_d(*moved2)

    if not compare(ref_out, mut_out):
        return "killed"
    return None


def classify(source, mutated, ref_mod, get_inputs, get_init_inputs, tmp_dir, mut_id,
             op_label=None):
    """Returns: 'killed', 'survived', 'stillborn', 'timeout', 'equivalent',
    or 'killed_enhanced' (killed by enhanced inputs only)."""
    if mutated.strip() == source.strip():
        return "equivalent"

    signal.alarm(MUTANT_TIMEOUT)
    try:
        P(f"        [compile]")
        t0 = time.time()
        mut_mod = load_module_from_source(mutated, f"mut_{mut_id}", tmp_dir)
        P(f"        [compiled in {time.time()-t0:.1f}s]")
    except MutantTimeout:
        P(f"        [compile TIMEOUT {MUTANT_TIMEOUT}s]")
        return "timeout"
    except Exception as e:
        signal.alarm(0)
        P(f"        [stillborn:compile] {str(e)[:100]}")
        return "stillborn"

    try:
        MutCls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
        init_args = get_init_inputs()
        mut_model = MutCls(*init_args) if isinstance(init_args, (list, tuple)) else MutCls()
    except MutantTimeout:
        return "timeout"
    except Exception as e:
        signal.alarm(0)
        P(f"        [stillborn:init] {str(e)[:100]}")
        return "stillborn"

    try:
        RefCls = ref_mod.Model
        init_args = get_init_inputs()
        ref_model = RefCls(*init_args) if isinstance(init_args, (list, tuple)) else RefCls()
    except MutantTimeout:
        return "timeout"
    except Exception as e:
        signal.alarm(0)
        return "stillborn"

    # --- Standard trials ---
    P(f"        [exec {NUM_TRIALS} trials]")
    for trial in range(NUM_TRIALS):
        try:
            r = _run_one_trial(ref_model, mut_model, get_inputs, SEED + trial)
            if r == "killed":
                signal.alarm(0)
                return "killed"
        except MutantTimeout:
            return "timeout"
        except Exception:
            signal.alarm(0)
            return "killed"

    # --- P3: Enhanced inputs for survived mutants ---
    if op_label:
        op_name = OP_NAME_MAP.get(op_label, "")
        if op_name:
            P(f"        [enhanced inputs for {op_name}]")
            try:
                gen = EnhancedInputGenerator(base_seed=30000)
                enhanced_sets = gen.generate_enhanced_inputs(
                    get_inputs, op_name, num_per_strategy=2
                )
                for strategy, enh_inputs in enhanced_sets:
                    try:
                        ref_d = ref_model.to(DEVICE).eval()
                        moved_r = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x
                                   for x in enh_inputs]
                        with torch.no_grad():
                            ref_out = ref_d(*moved_r)

                        mut_d = mut_model.to(DEVICE).eval()
                        enh_inputs2 = gen._apply_strategy(
                            get_inputs(), strategy, 30000 + hash(strategy) % 10000)
                        moved_m = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x
                                   for x in enh_inputs2]
                        with torch.no_grad():
                            mut_out = mut_d(*moved_m)

                        if not compare(ref_out, mut_out):
                            signal.alarm(0)
                            P(f"        [killed by enhanced: {strategy}]")
                            return "killed_enhanced"
                    except MutantTimeout:
                        return "timeout"
                    except Exception:
                        signal.alarm(0)
                        P(f"        [killed by enhanced exception: {strategy}]")
                        return "killed_enhanced"
            except Exception as e:
                P(f"        [enhanced inputs error: {str(e)[:60]}]")

    signal.alarm(0)
    return "survived"


def main():
    random.seed(SEED)
    signal.signal(signal.SIGALRM, _alarm_handler)

    P(f"{'='*60}")
    P(f"  MutaKernel Kill-Rate Smoke Test (v2: fixed)")
    P(f"{'='*60}")
    P(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    P(f"  Kernels: {len(TARGET_KERNELS)}, Operators: {len(ALL_OPS)}")
    P(f"  Timeout: {MUTANT_TIMEOUT}s per mutant")
    P(f"  Tolerance: atol={ATOL}, rtol={RTOL}")
    P(f"  Fixes: P1(load_inline patch) + P2(tight tol) + P3(enhanced inputs)")
    P("")

    tmp_dir = tempfile.mkdtemp(prefix="mutakill_")
    statuses = ["killed", "survived", "stillborn", "equivalent", "timeout", "killed_enhanced"]
    stats = {label: {s: 0 for s in statuses} for label, _ in ALL_OPS}
    cat_stats = {c: {s: 0 for s in statuses} for c in "ABCD"}
    total_time_start = time.time()

    for ki, kname in enumerate(TARGET_KERNELS):
        kf = RUN_DIR / kname
        if not kf.exists():
            P(f"  [{ki+1}] SKIP {kname} (not found)")
            continue

        parts = kname.split("_")
        problem_id = parts[parts.index("problem") + 1]
        problem_file = find_problem_file(problem_id)
        if problem_file is None:
            P(f"  [{ki+1}] SKIP P{problem_id} (no problem file)")
            continue

        gen_source = kf.read_text(encoding="utf-8", errors="replace")

        P(f"\n  ---- [{ki+1}/{len(TARGET_KERNELS)}] P{problem_id} ({len(gen_source)} bytes) ----")

        try:
            ref_mod = load_module(problem_file, f"prob_{problem_id}_{ki}")
            get_inputs = ref_mod.get_inputs
            get_init_inputs = ref_mod.get_init_inputs
        except Exception as e:
            P(f"  SKIP P{problem_id} (ref load: {str(e)[:80]})")
            continue

        P(f"    Pre-warming original kernel...")
        t0 = time.time()
        try:
            signal.alarm(MUTANT_TIMEOUT)
            orig_mod = load_module(kf, f"orig_{problem_id}_{ki}")
            signal.alarm(0)
            P(f"    Original compiled in {time.time()-t0:.1f}s")
        except Exception as e:
            signal.alarm(0)
            P(f"    WARN: original compile failed: {str(e)[:80]}")

        k_stats = {}
        for label, op in ALL_OPS:
            cat = label[0]
            try:
                sites = op.find_sites(gen_source)
            except Exception:
                continue
            if not sites:
                continue

            site = random.choice(sites)
            try:
                mutated = op.apply(gen_source, site)
            except Exception:
                continue

            mut_id = f"k{problem_id}_{label}_{ki}"
            P(f"    {label} @ L{getattr(site, 'line_start', '?')}: "
              f"'{getattr(site, 'original_code', '')[:50]}'")

            t0 = time.time()
            result = classify(
                gen_source, mutated, ref_mod,
                get_inputs, get_init_inputs, tmp_dir, mut_id,
                op_label=label,
            )
            elapsed = time.time() - t0
            P(f"    {label} => {result.upper()} ({elapsed:.1f}s)")

            stats[label][result] += 1
            cat_stats[cat][result] += 1
            k_stats[label] = result

        k_killed = sum(1 for v in k_stats.values() if v in ("killed", "killed_enhanced"))
        k_surv = sum(1 for v in k_stats.values() if v == "survived")
        k_still = sum(1 for v in k_stats.values() if v == "stillborn")
        k_to = sum(1 for v in k_stats.values() if v == "timeout")
        k_nt = k_killed + k_surv
        kr = k_killed / max(1, k_nt)
        P(f"    >> P{problem_id}: killed={k_killed} surv={k_surv} still={k_still} "
          f"timeout={k_to} KR={kr:.0%}")

    total_elapsed = time.time() - total_time_start

    P(f"\n{'='*60}")
    P(f"  KILL RATE SUMMARY (elapsed: {total_elapsed/60:.1f} min)")
    P(f"{'='*60}")

    P(f"\n  Per-Operator:")
    P(f"  {'Op':<6} {'Kill':>5} {'KillE':>6} {'Surv':>5} {'Still':>6} {'TMO':>4} {'Total':>6} {'KR':>7}")
    P(f"  {'-'*48}")
    for label, _ in ALL_OPS:
        s = stats[label]
        total = sum(s.values())
        killed_all = s["killed"] + s["killed_enhanced"]
        nt = killed_all + s["survived"]
        kr = killed_all / max(1, nt)
        if total > 0:
            P(f"  {label:<6} {s['killed']:>5} {s['killed_enhanced']:>6} {s['survived']:>5} "
              f"{s['stillborn']:>6} {s['timeout']:>4} {total:>6} {kr:>6.0%}")

    P(f"\n  Per-Category:")
    P(f"  {'Cat':<6} {'Kill':>5} {'KillE':>6} {'Surv':>5} {'Still':>6} {'TMO':>4} {'KR':>7}")
    P(f"  {'-'*42}")
    grand = {s: 0 for s in statuses}
    for cat in "ABCD":
        s = cat_stats[cat]
        killed_all = s["killed"] + s["killed_enhanced"]
        nt = killed_all + s["survived"]
        kr = killed_all / max(1, nt)
        P(f"  {cat:<6} {s['killed']:>5} {s['killed_enhanced']:>6} {s['survived']:>5} "
          f"{s['stillborn']:>6} {s['timeout']:>4} {kr:>6.0%}")
        for k in grand:
            grand[k] += s[k]

    killed_total = grand["killed"] + grand["killed_enhanced"]
    total_nt = killed_total + grand["survived"]
    overall_kr = killed_total / max(1, total_nt)
    total_all = sum(grand.values())
    P(f"  {'-'*42}")
    P(f"  {'ALL':<6} {grand['killed']:>5} {grand['killed_enhanced']:>6} {grand['survived']:>5} "
      f"{grand['stillborn']:>6} {grand['timeout']:>4} {overall_kr:>6.0%}")

    P(f"\n  Overall Kill Rate: {overall_kr:.1%}")
    P(f"    - Standard kills: {grand['killed']}")
    P(f"    - Enhanced kills: {grand['killed_enhanced']}")
    P(f"    - Survived: {grand['survived']}")
    if total_all > 0:
        P(f"  Stillborn Rate: {grand['stillborn']/total_all:.1%}")
        P(f"  Timeout Rate: {grand['timeout']/total_all:.1%}")
    P(f"  Total mutants tested: {total_all}")
    P(f"  Total time: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
