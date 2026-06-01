"""Benchmark Task B fixed kernels vs original buggy kernels.

For each truly-fixed kernel (where round0 confirmed buggy >= 1):
- Load buggy and fixed kernels
- Run KernelBench default inputs (get_inputs()) for 50 warmup + 200 timed iters
- Report wall-clock mean (ms) and speedup ratio fixed/buggy.
"""
import json, sys, time, importlib.util, gc
from pathlib import Path

PROJECT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
sys.path.insert(0, str(PROJECT))

TASKB = PROJECT / "第二次实验汇总" / "第二次实验汇总_补充" / "task_b_regenerate"
DET_DIR = TASKB / "details"
KERN_DIR = TASKB / "kernels"
BEST = json.loads((PROJECT / "best_kernels.json").read_text())

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
PROBLEM_DIRS = {"L1": KB_ROOT / "KernelBench" / "level1",
                "L2": KB_ROOT / "KernelBench" / "level2"}

WARMUP = 20
TIMED = 50
# Skip kernels we already have or that hang
SKIP_NAMES = {"L1_P1", "L1_P14", "L1_P15", "L1_P16", "L1_P17", "L1_P18", "L1_P2", "L1_P22", "L1_P39",
              "L1_P48"}  # L1_P48 hangs in warmup, skip it


def find_problem_file(level: str, problem_id):
    pid = str(problem_id)
    pdir = PROBLEM_DIRS.get(level)
    for f in pdir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def load_module(path: Path, mod_name: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def benchmark_one(kernel_code: str, problem_file: Path, tag: str,
                  warmup=WARMUP, timed=TIMED):
    """Compile kernel_code as a temp module and time it."""
    import torch
    import tempfile, os
    # Write kernel code to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(kernel_code)
        tmp_path = tf.name
    try:
        kern_mod = load_module(Path(tmp_path), f"benchk_{tag}_{int(time.time()*1e6)}")
        ref_mod = load_module(problem_file, f"benchref_{tag}_{int(time.time()*1e6)}")
        get_inputs = ref_mod.get_inputs
        get_init = getattr(ref_mod, "get_init_inputs", lambda: [])

        torch.manual_seed(42)
        device = "cuda"
        init_args = get_init()
        cls = getattr(kern_mod, "ModelNew", None) or getattr(kern_mod, "Model")
        if isinstance(init_args, (list, tuple)):
            model = cls(*init_args)
        else:
            model = cls()
        model = model.to(device).eval()

        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                  for x in get_inputs()]

        # warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(*inputs)
        torch.cuda.synchronize()

        # timed
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        with torch.no_grad():
            for _ in range(timed):
                start.record()
                _ = model(*inputs)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        times.sort()
        n = len(times)
        # median + 25-75 quantile
        median = times[n//2]
        q25 = times[n//4]
        q75 = times[3*n//4]
        mean = sum(times)/n
        return {"median_ms": median, "q25_ms": q25, "q75_ms": q75,
                "mean_ms": mean, "n": n, "error": None}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}
    finally:
        try: os.unlink(tmp_path)
        except: pass
        gc.collect()
        try: import torch; torch.cuda.empty_cache()
        except: pass


def main():
    # Gather kernels to benchmark (skip pseudo/partial-pseudo)
    targets = []
    for f in sorted(DET_DIR.glob("*.json")):
        d = json.load(open(f))
        name = d.get("kernel_name", f.stem)
        final_status = d.get("final_status", "")
        if not final_status.startswith("fixed"):
            continue
        final_round = d.get("final_round", 0)
        r0 = d.get("round0_stats", {})
        n_conf = r0.get("n_confirmed_buggy", 0)
        n_unexp = r0.get("n_unexpected_pass", 0)
        if name not in BEST or final_round == 0:
            continue
        fixed_p = KERN_DIR / f"{name}_round{final_round}.py"
        if not fixed_p.exists():
            continue
        if name in SKIP_NAMES: continue
        targets.append({
            "name": name, "final_round": final_round,
            "n_confirmed_buggy": n_conf, "n_unexpected_pass": n_unexp,
            "is_pseudo": (n_conf == 0 and n_unexp > 0),
            "is_partial": (n_unexp >= max(n_conf, 1) * 0.5 and n_conf > 0),
            "buggy_path": Path(BEST[name]["kernel_path"]),
            "fixed_path": fixed_p,
            "level": BEST[name]["level"],
            "problem_id": BEST[name]["problem_id"],
            "kb_baseline_speedup": BEST[name]["speedup"],
        })

    print(f"Benchmarking {len(targets)} kernels (warmup={WARMUP}, timed={TIMED})")
    print(f"\n{'Kernel':<8} {'L/P':<7} {'KB_base':>8} {'buggy_ms':>10} {'fixed_ms':>10} "
          f"{'ratio_fb':>9} {'pseudo':<7}")
    print("-" * 90)

    results = []
    for t in targets:
        name = t["name"]
        problem_file = find_problem_file(t["level"], t["problem_id"])
        if problem_file is None:
            print(f"{name:<8} -- no problem_file")
            continue
        buggy_code = t["buggy_path"].read_text(encoding="utf-8")
        fixed_code = t["fixed_path"].read_text(encoding="utf-8")

        b = benchmark_one(buggy_code, problem_file, f"{name}_buggy")
        f_sum = benchmark_one(fixed_code, problem_file, f"{name}_fixed")

        if b.get("error") or f_sum.get("error"):
            print(f"{name:<8} {t['level']}/P{t['problem_id']:<3} -- error "
                  f"buggy={b.get('error','OK')[:30]} fixed={f_sum.get('error','OK')[:30]}")
            results.append({**t, "buggy": b, "fixed": f_sum})
            continue

        ratio = f_sum["median_ms"] / b["median_ms"] if b["median_ms"] > 0 else float("nan")
        pseudo_tag = ("PSEUDO" if t["is_pseudo"]
                      else ("PARTIAL" if t["is_partial"] else ""))
        print(f"{name:<8} {t['level']}/P{t['problem_id']:<3} {t['kb_baseline_speedup']:>8.3f} "
              f"{b['median_ms']:>10.3f} {f_sum['median_ms']:>10.3f} "
              f"{ratio:>9.3f}× {pseudo_tag:<7}")
        results.append({**t, "buggy": b, "fixed": f_sum, "ratio_fb_median": ratio})

    # Persist results
    out = TASKB / "benchmark_speedup.json"
    out.write_text(json.dumps([
        {**{k: v for k, v in r.items() if k not in ("buggy_path", "fixed_path")},
         "buggy_path": str(r["buggy_path"]),
         "fixed_path": str(r["fixed_path"])}
        for r in results
    ], default=str, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
