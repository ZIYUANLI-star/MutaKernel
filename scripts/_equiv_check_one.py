#!/usr/bin/env python3
"""子进程 worker：检测单个变异体是否为等价变异体。

比较 original_kernel vs mutant_kernel (bitwise identical)，
使用 100 轮随机输入 + 6 种压力策略 x 2 种子 = 12 轮。

Usage:  python _equiv_check_one.py <config.json> <result.json>

config.json 格式:
{
  "problem_file": "/path/to/problem.py",
  "kernel_code": "...original kernel source...",
  "mutant_id": "L1_P1__relop_replace__2",
  "mutated_code": "...mutated kernel source...",
  "device": "cuda",
  "equiv_runs": 100,
  "base_seed": 10000
}
"""
import hashlib, json, os, sys, tempfile, time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

STRESS_POLICIES_NAMES = [
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
    cfg = json.load(open(cfg_path))

    device = cfg["device"]
    equiv_runs = cfg.get("equiv_runs", 100)
    base_seed = cfg.get("base_seed", 10000)
    kernel_code = cfg["kernel_code"]
    mutated_code = cfg["mutated_code"]
    mutant_id = cfg["mutant_id"]

    safe_id = mutant_id.replace("-", "_").replace(".", "_")
    tmp_dir = tempfile.mkdtemp(prefix="eq1_")
    t0 = time.time()

    ref_mod = _load_module_from_path(cfg["problem_file"], f"ref_{safe_id}")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
    init_args = get_init_inputs()

    orig_hash = hashlib.md5(kernel_code.encode()).hexdigest()[:10]
    try:
        orig_mod = _load_module_from_source(kernel_code, f"eqo_{orig_hash}", tmp_dir)
    except Exception as e:
        json.dump({"is_equivalent": False, "error": f"OrigCompile: {e}",
                    "time_ms": (time.time()-t0)*1000}, open(res_path, "w"))
        return

    orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
    orig_model = (orig_cls(*init_args) if isinstance(init_args, (list, tuple))
                  else orig_cls())
    orig_model = orig_model.to(device).eval()

    try:
        mut_mod = _load_module_from_source(mutated_code, f"eqm_{safe_id}", tmp_dir)
    except Exception as e:
        json.dump({"is_equivalent": False, "error": f"MutCompile: {e}",
                    "time_ms": (time.time()-t0)*1000}, open(res_path, "w"))
        return

    mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
    mut_model = (mut_cls(*init_args) if isinstance(init_args, (list, tuple))
                 else mut_cls())
    mut_model = mut_model.to(device).eval()

    compile_ms = (time.time() - t0) * 1000

    diverged_at = None
    diverged_policy = ""

    for i in range(equiv_runs):
        torch.manual_seed(base_seed + i)
        inputs = get_inputs()
        moved = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        with torch.no_grad():
            o1 = orig_model(*moved)
            o2 = mut_model(*moved)
        torch.cuda.synchronize()
        if not _bitwise_identical(o1, o2):
            diverged_at = i
            diverged_policy = "random"
            break
        del o1, o2, moved, inputs

    if diverged_at is None:
        for pname in STRESS_POLICIES_NAMES:
            pfn = STRESS_POLICIES.get(pname)
            if pfn is None:
                continue
            for si in range(2):
                seed = base_seed + equiv_runs + si
                torch.manual_seed(seed)
                try:
                    template = get_inputs()
                    sinputs = pfn(template, seed)
                except Exception:
                    continue
                moved = [x.to(device) if isinstance(x, torch.Tensor) else x
                         for x in sinputs]

                orig_exc = mut_exc = None
                o1 = o2 = None
                try:
                    with torch.no_grad():
                        o1 = orig_model(*moved)
                except Exception as e:
                    orig_exc = e
                try:
                    with torch.no_grad():
                        o2 = mut_model(*moved)
                except Exception as e:
                    mut_exc = e
                torch.cuda.synchronize()

                if orig_exc is not None and mut_exc is not None:
                    if type(orig_exc) is type(mut_exc):
                        continue
                    diverged_at = -1
                    diverged_policy = f"{pname}(diff_exc)"
                    break
                if orig_exc is not None or mut_exc is not None:
                    oom_msg = str(orig_exc or mut_exc).lower()
                    if "out of memory" in oom_msg:
                        continue
                    diverged_at = -1
                    diverged_policy = f"{pname}(one_side_exc)"
                    break
                if not _bitwise_identical(o1, o2):
                    diverged_at = -1
                    diverged_policy = pname
                    break
            if diverged_at is not None:
                break

    total_ms = (time.time() - t0) * 1000
    result = {
        "is_equivalent": diverged_at is None,
        "compile_ms": compile_ms,
        "time_ms": total_ms,
    }
    if diverged_at is not None:
        result["diverged_at"] = diverged_at
        result["diverged_policy"] = diverged_policy

    json.dump(result, open(res_path, "w"))


if __name__ == "__main__":
    main()
