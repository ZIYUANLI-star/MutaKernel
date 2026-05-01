"""Debug script: show full mutation process for one KILLED and one SURVIVED case."""
import sys, os, json, random, time, tempfile, importlib, difflib
sys.path.insert(0, "/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
os.chdir("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")

import torch
import re

from pathlib import Path
from src.mutengine.operators.arithmetic import ArithReplace, RelOpReplace
from src.mutengine.operators.gpu_parallel import IndexReplace

SEED = 42
ATOL = 1e-2
RTOL = 1e-2
DEVICE = "cuda"

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
BEST_FILE = Path("best_kernels.json")
PROBLEM_DIR = KB_ROOT / "KernelBench" / "level1"

_LOAD_INLINE_NAME_RE = re.compile(
    r"""(load_inline\s*\([^)]*?name\s*=\s*)(['"])([^'"]+)\2""", re.DOTALL)

def patch_names(source, suffix):
    def _repl(m):
        prefix, quote, name = m.group(1), m.group(2), m.group(3)
        return f"{prefix}{quote}{name}_{suffix}{quote}"
    return _LOAD_INLINE_NAME_RE.sub(_repl, source)

def load_module(filepath, name):
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_from_source(source, name, tmp_dir):
    source = patch_names(source, name)
    fp = os.path.join(tmp_dir, f"{name}.py")
    with open(fp, "w") as f:
        f.write(source)
    return load_module(fp, name)

def find_problem_file(problem_dir, problem_id):
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{problem_id}_") and f.suffix == ".py":
            return f
    return None

def show_diff(original, mutated, label):
    orig_lines = original.splitlines(keepends=True)
    mut_lines = mutated.splitlines(keepends=True)
    diff = list(difflib.unified_diff(orig_lines, mut_lines,
                                      fromfile="original", tofile=f"mutated_{label}",
                                      lineterm=""))
    if not diff:
        print("    [NO DIFFERENCE - equivalent mutant]")
    else:
        for line in diff[:40]:
            print(f"    {line.rstrip()}")
        if len(diff) > 40:
            print(f"    ... ({len(diff)-40} more diff lines)")

def run_one(label, op, gen_source, ref_mod, get_inputs, get_init_inputs, tmp_dir):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  MUTATION: {label} ({op.__class__.__name__})")
    print(f"{sep}")

    print(f"\n[Step 1] find_sites() - 搜索变异位点...")
    t0 = time.time()
    sites = op.find_sites(gen_source)
    print(f"  找到 {len(sites)} 个位点 ({time.time()-t0:.3f}s)")

    if not sites:
        print("  无位点，跳过")
        return

    random.seed(SEED)
    site = random.choice(sites)
    print(f"  选取位点: line={site.line_start}, col={site.col_start}-{site.col_end}")
    print(f"  原始代码片段: `{site.original_code}`")
    print(f"  变异类型: {site.node_type}")

    print(f"\n[Step 2] apply() - 施加变异...")
    t0 = time.time()
    mutated = op.apply(gen_source, site)
    print(f"  变异完成 ({time.time()-t0:.3f}s)")

    if mutated.strip() == gen_source.strip():
        print(f"  结果: EQUIVALENT (变异前后代码一致)")
        return

    print(f"\n[Step 3] Diff - 变异前后差异:")
    show_diff(gen_source, mutated, label)

    print(f"\n[Step 4] 编译变异体 (load_inline JIT)...")
    mut_id = f"debug_{label}"
    t0 = time.time()
    try:
        mut_mod = load_from_source(mutated, f"mut_{mut_id}", tmp_dir)
        print(f"  编译成功 ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  编译失败 ({time.time()-t0:.1f}s)")
        err_msg = str(e)
        print(f"  错误: {err_msg[:300]}")
        print(f"\n  >>> 结果: STILLBORN (编译失败)")
        return

    print(f"\n[Step 5] 实例化变异模型...")
    try:
        MutCls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
        init_args = get_init_inputs()
        mut_model = MutCls(*init_args) if isinstance(init_args, (list, tuple)) else MutCls()
        print(f"  ModelNew 实例化成功")
    except Exception as e:
        print(f"  实例化失败: {e}")
        print(f"\n  >>> 结果: STILLBORN")
        return

    print(f"\n[Step 6] 实例化参考模型...")
    RefCls = ref_mod.Model
    init_args = get_init_inputs()
    ref_model = RefCls(*init_args) if isinstance(init_args, (list, tuple)) else RefCls()
    print(f"  Model 实例化成功")

    print(f"\n[Step 7] 运行对比测试 (3 trials, atol={ATOL}, rtol={RTOL})...")
    for trial in range(3):
        torch.manual_seed(SEED + trial)
        inputs = get_inputs()
        input_shapes = [x.shape if isinstance(x, torch.Tensor) else type(x).__name__ for x in inputs]
        print(f"\n  Trial {trial+1}: input shapes = {input_shapes}")

        ref_d = ref_model.to(DEVICE).eval()
        moved_r = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x for x in inputs]
        with torch.no_grad():
            ref_out = ref_d(*moved_r)

        torch.manual_seed(SEED + trial)
        inputs2 = get_inputs()
        mut_d = mut_model.to(DEVICE).eval()
        moved_m = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x for x in inputs2]

        try:
            with torch.no_grad():
                mut_out = mut_d(*moved_m)
        except Exception as e:
            print(f"  变异体运行异常: {e}")
            print(f"\n  >>> 结果: KILLED (运行时异常)")
            return

        if isinstance(ref_out, torch.Tensor) and isinstance(mut_out, torch.Tensor):
            print(f"  ref output: shape={ref_out.shape}, dtype={ref_out.dtype}")
            print(f"  mut output: shape={mut_out.shape}, dtype={mut_out.dtype}")

            if ref_out.shape != mut_out.shape:
                print(f"  形状不匹配! ref={ref_out.shape} vs mut={mut_out.shape}")
                print(f"\n  >>> 结果: KILLED (shape mismatch)")
                return

            ref_f = ref_out.float().cpu()
            mut_f = mut_out.float().cpu()

            max_abs_diff = (ref_f - mut_f).abs().max().item()
            mean_abs_diff = (ref_f - mut_f).abs().mean().item()
            max_rel_diff = ((ref_f - mut_f).abs() / (ref_f.abs() + 1e-10)).max().item()
            match = torch.allclose(ref_f, mut_f, atol=ATOL, rtol=RTOL)

            print(f"  max_abs_diff  = {max_abs_diff:.6e}")
            print(f"  mean_abs_diff = {mean_abs_diff:.6e}")
            print(f"  max_rel_diff  = {max_rel_diff:.6e}")
            print(f"  allclose(atol={ATOL}, rtol={RTOL}) = {match}")

            if not match:
                n_mismatch = (~torch.isclose(ref_f, mut_f, atol=ATOL, rtol=RTOL)).sum().item()
                total = ref_f.numel()
                print(f"  不匹配元素: {n_mismatch}/{total} ({n_mismatch/total:.1%})")
                print(f"\n  >>> 结果: KILLED (数值差异超过容差)")
                return
        else:
            print(f"  ref type: {type(ref_out)}, mut type: {type(mut_out)}")

    print(f"\n  >>> 结果: SURVIVED (所有 trial 均通过 allclose)")


def main():
    with open(BEST_FILE) as f:
        best = json.load(f)

    info = best.get("L1_P1")
    if not info:
        print("L1_P1 not in best_kernels.json")
        return

    kernel_path = Path(info["kernel_path"])
    problem_id = info["problem_id"]
    print(f"Kernel: L1_P1, turn={info['turn']}, speedup={info['speedup']:.3f}")
    print(f"File: {kernel_path}")

    gen_source = kernel_path.read_text(encoding="utf-8", errors="replace")
    print(f"Source length: {len(gen_source)} bytes, {len(gen_source.splitlines())} lines")

    problem_file = find_problem_file(PROBLEM_DIR, problem_id)
    print(f"Reference: {problem_file}")

    ref_mod = load_module(problem_file, "ref_prob_1")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = ref_mod.get_init_inputs

    tmp_dir = tempfile.mkdtemp(prefix="debug_mut_")
    print(f"Temp dir: {tmp_dir}")

    run_one("A1", ArithReplace(), gen_source, ref_mod, get_inputs, get_init_inputs, tmp_dir)
    run_one("A2", RelOpReplace(), gen_source, ref_mod, get_inputs, get_init_inputs, tmp_dir)

if __name__ == "__main__":
    main()
