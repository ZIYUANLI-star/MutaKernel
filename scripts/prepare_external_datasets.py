#!/usr/bin/env python3
from __future__ import annotations

"""Download and prepare CUDA-L1 and AI CUDA Engineer datasets for MutaKernel testing.

Produces per-kernel directories under external_benchmarks/{dataset}/ with:
  - problem_{task_id}.py   (contains class Model, get_inputs, get_init_inputs)
  - kernel_{task_id}.py    (contains class ModelNew, CUDA code via load_inline)

Also writes a registry JSON file for each dataset.
"""
import json
import os
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _to_relative(abs_path: Path) -> str:
    """Convert an absolute path under PROJECT_ROOT to a relative path string."""
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)

def prepare_cuda_l1():
    """Prepare CUDA-L1 dataset from GitHub JSON files."""
    print("=" * 60)
    print("  Preparing CUDA-L1 dataset")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "external_benchmarks" / "cuda_l1"
    data_dir.mkdir(parents=True, exist_ok=True)
    problems_dir = data_dir / "problems"
    problems_dir.mkdir(exist_ok=True)

    json_path = data_dir / "a100.json"
    if not json_path.exists():
        print(f"  ERROR: {json_path} not found!")
        print(f"  Please download from: https://github.com/deepreinforce-ai/CUDA-L1/raw/main/optimized_cuda_code/a100.json")
        print(f"  Save to: {json_path}")
        return []

    data = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"  Loaded {len(data)} entries from a100.json")

    kernelbench_dir = None
    for candidate in [
        PROJECT_ROOT / "KernelBench",
        PROJECT_ROOT / "KernelBench-0" / "KernelBench",
        Path("/home/kbuser/projects/KernelBench-0/KernelBench"),
        Path.home() / "KernelBench" / "KernelBench",
    ]:
        if candidate.exists():
            kernelbench_dir = candidate
            break

    if kernelbench_dir is None:
        print("  KernelBench not found locally. Attempting to clone...")
        clone_target = PROJECT_ROOT / "KernelBench"
        try:
            import subprocess
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ScalingIntelligence/KernelBench.git",
                 str(clone_target)],
                check=True, capture_output=True, timeout=300,
            )
            kernelbench_dir = clone_target / "KernelBench"
            if kernelbench_dir.exists():
                print(f"  Cloned KernelBench to {clone_target}")
            else:
                kernelbench_dir = clone_target
                print(f"  Cloned KernelBench (flat layout)")
        except Exception as e:
            print(f"  WARNING: Could not clone KernelBench: {e}")
            print(f"  CUDA-L1 will use ref_code from JSON as fallback.")

    registry = []
    skipped = 0

    for entry in data:
        level_id = entry.get("level_id")
        task_id = entry.get("task_id")
        custom_code = entry.get("custom_code", "")
        ref_code = entry.get("ref_code", "")
        score = entry.get("score_default", 0)

        if not custom_code or not ref_code:
            skipped += 1
            continue

        kid = f"cuda_l1__L{level_id}_T{task_id}"
        problem_path = problems_dir / f"problem_L{level_id}_T{task_id}.py"

        kb_problem = None
        if kernelbench_dir:
            kb_path = kernelbench_dir / f"level{level_id}" / f"{task_id}_*.py"
            import glob
            matches = glob.glob(str(kb_path))
            if matches:
                kb_problem = matches[0]

        if kb_problem:
            import shutil
            shutil.copy2(kb_problem, str(problem_path))
        else:
            with open(problem_path, "w", encoding="utf-8") as f:
                f.write(ref_code)

        registry.append({
            "id": kid,
            "repo": "deepreinforce-ai/CUDA-L1",
            "kernel_name": f"CUDA-L1_L{level_id}_T{task_id}",
            "reference_file": _to_relative(problem_path),
            "requires": [],
            "kernel_source": custom_code,
            "score_default": score,
            "level_id": level_id,
            "task_id": task_id,
        })

    registry_path = data_dir / "registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"  Prepared {len(registry)} kernels, skipped {skipped}")
    print(f"  Registry: {registry_path}")
    return registry


def _extract_pybind_fn_names(cuda_code: str) -> list[str]:
    """Extract function names registered in PYBIND11_MODULE."""
    import re
    matches = re.findall(r'm\.def\(\s*"(\w+)"', cuda_code)
    return matches if matches else ["forward"]


def _wrap_cuda_as_modelnew(cuda_code: str, kid: str, problem_path) -> str | None:
    """Wrap raw CUDA C++ code into a Python file with load_inline + class ModelNew.

    Reads the reference problem file to extract Model's __init__ signature and forward args,
    then generates a ModelNew that loads the CUDA extension and calls it.
    """
    fn_names = _extract_pybind_fn_names(cuda_code)
    if not fn_names:
        return None

    ref_content = Path(problem_path).read_text(encoding="utf-8", errors="replace")

    import re
    init_match = re.search(r'def __init__\(self(?:,\s*([^)]*))?\)', ref_content)
    init_params = ""
    init_args = ""
    init_stores = ""
    if init_match and init_match.group(1):
        raw_params = init_match.group(1).strip()
        init_params = ", " + raw_params
        param_names = []
        for p in raw_params.split(","):
            p = p.strip()
            name = p.split(":")[0].split("=")[0].strip()
            if name:
                param_names.append(name)
        init_stores = "\n".join(f"        self.{n} = {n}" for n in param_names)
        init_args = ", ".join(param_names)

    fwd_match = re.search(r'def forward\(self(?:,\s*([^)]*))?\)', ref_content)
    fwd_params = ""
    fwd_args = ""
    if fwd_match and fwd_match.group(1):
        raw_fwd = fwd_match.group(1).strip()
        fwd_params = ", " + raw_fwd
        fwd_names = []
        for p in raw_fwd.split(","):
            p = p.strip()
            name = p.split(":")[0].split("=")[0].strip()
            if name:
                fwd_names.append(name)
        fwd_args = ", ".join(fwd_names)

    escaped_cuda = cuda_code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    primary_fn = fn_names[0]
    fn_list_str = str(fn_names)

    wrapper = f'''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_source = r"""{cuda_code}"""

_ext = load_inline(
    name="sakana_{kid.replace(".", "_")}",
    cpp_sources="",
    cuda_sources=_cuda_source,
    functions={fn_list_str},
    verbose=False,
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    def __init__(self{init_params}):
        super().__init__()
{init_stores if init_stores else "        pass"}

    def forward(self{fwd_params}):
        return _ext.{primary_fn}({fwd_args})
'''
    return wrapper


def prepare_ai_cuda_engineer():
    """Prepare AI CUDA Engineer Archive from HuggingFace dataset."""
    print("=" * 60)
    print("  Preparing AI CUDA Engineer Archive")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer"
    data_dir.mkdir(parents=True, exist_ok=True)
    problems_dir = data_dir / "problems"
    problems_dir.mkdir(exist_ok=True)

    parquet_dir = data_dir / "parquet"

    level_files = {
        "level_1": parquet_dir / "level_1.parquet",
        "level_2": parquet_dir / "level_2.parquet",
        "level_3": parquet_dir / "level_3.parquet",
    }

    all_exist = all(f.exists() for f in level_files.values())

    if not all_exist:
        print("  Parquet files not found. Attempting to download via datasets library...")
        try:
            from datasets import load_dataset
            ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")
            parquet_dir.mkdir(parents=True, exist_ok=True)
            for split_name in ["level_1", "level_2", "level_3"]:
                out_path = parquet_dir / f"{split_name}.parquet"
                ds[split_name].to_parquet(str(out_path))
                print(f"    Saved {split_name} -> {out_path}")
        except ImportError:
            print("  ERROR: 'datasets' library not installed. Install with: pip install datasets")
            print("  Or manually download parquet files from HuggingFace:")
            print("    https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive")
            print(f"  Save to: {parquet_dir}/")
            return []
        except Exception as e:
            print(f"  ERROR downloading dataset: {e}")
            return []

    try:
        import pandas as pd
    except ImportError:
        print("  ERROR: pandas not installed. Install with: pip install pandas pyarrow")
        return []

    registry = []
    seen_tasks = {}

    for level_name, pq_path in level_files.items():
        if not pq_path.exists():
            print(f"  WARNING: {pq_path} not found, skipping")
            continue

        df = pd.read_parquet(pq_path)
        print(f"  {level_name}: {len(df)} rows, {df['Correct'].sum()} correct")

        correct_df = df[df["Correct"] == True].copy()
        if "CUDA_Speedup_Native" in correct_df.columns:
            correct_df = correct_df.sort_values("CUDA_Speedup_Native", ascending=False)

        for _, row in correct_df.iterrows():
            task_key = (row.get("Level_ID"), row.get("Task_ID"))
            if task_key in seen_tasks:
                continue

            level_id = row.get("Level_ID")
            task_id = row.get("Task_ID")
            cuda_code = row.get("CUDA_Code", "")
            pytorch_code = row.get("PyTorch_Code_Module", "")
            speedup = row.get("CUDA_Speedup_Native", 0)
            max_diff = row.get("Max_Diff", 0)
            op_name = row.get("Op_Name", "unknown")

            if not cuda_code or not pytorch_code:
                continue

            kid = f"sakana__L{level_id}_T{task_id}"
            problem_path = problems_dir / f"problem_L{level_id}_T{task_id}.py"

            with open(problem_path, "w", encoding="utf-8") as f:
                f.write(pytorch_code)

            wrapped_code = _wrap_cuda_as_modelnew(cuda_code, kid, problem_path)
            if wrapped_code is None:
                continue

            seen_tasks[task_key] = True
            registry.append({
                "id": kid,
                "repo": "SakanaAI/AI-CUDA-Engineer",
                "kernel_name": f"{op_name}_L{level_id}_T{task_id}",
                "reference_file": _to_relative(problem_path),
                "requires": [],
                "kernel_source": wrapped_code,
                "speedup": float(speedup) if speedup else 0,
                "max_diff_original": float(max_diff) if max_diff else 0,
                "level_id": int(level_id) if level_id else 0,
                "task_id": int(task_id) if task_id else 0,
            })

    registry_path = data_dir / "registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"  Prepared {len(registry)} kernels (best per task)")
    print(f"  Registry: {registry_path}")
    return registry


def _find_wrapper_function(triton_code: str) -> tuple[str | None, list[str]]:
    """Find the last non-decorated top-level function in Triton code (the wrapper).

    Returns (function_name, list_of_param_names_excluding_self_and_defaults).
    """
    import ast
    try:
        tree = ast.parse(triton_code)
    except SyntaxError:
        return None, []

    wrapper = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_triton_jit = any(
                (isinstance(d, ast.Attribute) and d.attr == "jit") or
                (isinstance(d, ast.Name) and d.value if isinstance(d, ast.Call) else False)
                for d in node.decorator_list
            )
            has_autotune = any(
                isinstance(d, ast.Call) and
                (getattr(d.func, "attr", "") == "autotune" or getattr(d.func, "id", "") == "autotune")
                for d in node.decorator_list
            )
            if not is_triton_jit and not has_autotune:
                wrapper = node

        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name in ("forward", "__call__"):
                        wrapper = item

    if wrapper is None:
        return None, []

    params = []
    for arg in wrapper.args.args:
        name = arg.arg
        if name == "self" or name == "ctx":
            continue
        params.append(name)
    return wrapper.name, params


TRITONBENCH_PYTORCH_REFS = {
    "softmax": ("torch.nn.functional.softmax(x, dim=-1)", ["x: torch.randn(4, 1024)"]),
    "SoftMax": ("torch.nn.functional.softmax(x, dim=-1)", ["x: torch.randn(4, 1024)"]),
    "relu": ("torch.relu(x)", ["x: torch.randn(4, 1024)"]),
    "gelu": ("torch.nn.functional.gelu(x)", ["x: torch.randn(4, 1024)"]),
    "silu": ("torch.nn.functional.silu(x)", ["x: torch.randn(4, 1024)"]),
    "sigmoid": ("torch.sigmoid(x)", ["x: torch.randn(4, 1024)"]),
    "tanh": ("torch.tanh(x)", ["x: torch.randn(4, 1024)"]),
    "layernorm": ("torch.nn.functional.layer_norm(x, x.shape[-1:])", ["x: torch.randn(4, 256, 512)"]),
    "layer_norm": ("torch.nn.functional.layer_norm(x, x.shape[-1:])", ["x: torch.randn(4, 256, 512)"]),
    "rmsnorm": ("x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)", ["x: torch.randn(4, 256, 512)"]),
    "rms_norm": ("x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)", ["x: torch.randn(4, 256, 512)"]),
    "dropout": ("torch.nn.functional.dropout(x, p=0.5, training=False)", ["x: torch.randn(4, 1024)"]),
    "add": ("x + y", ["x: torch.randn(4, 1024)", "y: torch.randn(4, 1024)"]),
    "matmul": ("torch.matmul(x, y)", ["x: torch.randn(128, 256)", "y: torch.randn(256, 128)"]),
    "cross_entropy": ("torch.nn.functional.cross_entropy(x, y)", ["x: torch.randn(32, 100)", "y: torch.randint(0, 100, (32,))"]),
    "kldiv": ("torch.nn.functional.kl_div(x.log(), y, reduction='none')", ["x: torch.rand(4, 1024).clamp(min=1e-6)", "y: torch.rand(4, 1024).clamp(min=1e-6)"]),
    "swiglu": ("x * torch.sigmoid(x) * y", ["x: torch.randn(4, 512)", "y: torch.randn(4, 512)"]),
    "geglu": ("x * torch.nn.functional.gelu(y)", ["x: torch.randn(4, 512)", "y: torch.randn(4, 512)"]),
}


def _match_pytorch_ref(filename: str, description: str) -> tuple[str, list[str]] | None:
    """Try to match a TritonBench-G kernel to a known PyTorch equivalent."""
    lower_name = filename.lower().replace(".py", "").replace("_kernel", "").replace("_fwd", "").replace("_bwd", "")
    lower_desc = description.lower()

    for key, (expr, inputs) in TRITONBENCH_PYTORCH_REFS.items():
        if key.lower() in lower_name or key.lower() in lower_desc[:200]:
            return expr, inputs
    return None


def prepare_tritonbench():
    """Prepare TritonBench-G dataset from the cloned repository or JSON download."""
    print("=" * 60)
    print("  Preparing TritonBench-G dataset")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "external_benchmarks" / "tritonbench_g"
    data_dir.mkdir(parents=True, exist_ok=True)
    problems_dir = data_dir / "problems"
    problems_dir.mkdir(exist_ok=True)
    kernels_dir = data_dir / "kernels"
    kernels_dir.mkdir(exist_ok=True)

    json_path = data_dir / "TritonBench_G_v1.json"
    if not json_path.exists():
        repo_json = PROJECT_ROOT / "TritonBench" / "data" / "TritonBench_G_v1.json"
        if repo_json.exists():
            import shutil
            shutil.copy2(str(repo_json), str(json_path))
            print(f"  Copied from cloned repo: {repo_json}")
        else:
            print(f"  ERROR: {json_path} not found!")
            print(f"  Option 1: git clone https://github.com/thunlp/TritonBench.git")
            print(f"            then copy data/TritonBench_G_v1.json to {json_path}")
            print(f"  Option 2: Download directly:")
            print(f"            wget https://raw.githubusercontent.com/thunlp/TritonBench/main/data/TritonBench_G_v1.json")
            print(f"            -O {json_path}")
            return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} kernels from TritonBench_G_v1.json")

    registry = []
    skipped = 0
    skipped_reasons = {}

    for idx, entry in enumerate(data):
        filename = entry.get("file", f"kernel_{idx}.py")
        repo = entry.get("repo", "unknown")
        triton_code = entry.get("output", "")
        description = entry.get("simp_instru", "")
        difficulty = entry.get("difficulty", "?")
        star = entry.get("star", 0)

        if not triton_code or len(triton_code.strip()) < 50:
            skipped += 1
            skipped_reasons[filename] = "empty/short code"
            continue

        wrapper_name, wrapper_params = _find_wrapper_function(triton_code)
        if wrapper_name is None or not wrapper_params:
            skipped += 1
            skipped_reasons[filename] = "no wrapper function found"
            continue

        kid = f"tritonbench__{idx}_{filename.replace('.py', '')}"

        kernel_path = kernels_dir / f"kernel_{idx}.py"
        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(triton_code)

        param_tensors = [f"torch.randn(4, 1024, device='cuda')" for _ in wrapper_params]

        ref_match = _match_pytorch_ref(filename, description)

        problem_code = f'''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, {", ".join(wrapper_params)}):
'''
        if ref_match:
            expr, typed_inputs = ref_match
            problem_code += f'        return {expr}\n'
            input_stmts = []
            for ti in typed_inputs:
                name, init = ti.split(": ", 1)
                input_stmts.append(f"    {init}.cuda()")
            problem_code += f'''

def get_inputs():
    return [
{chr(10).join("        " + s + "," for s in input_stmts)}
    ]

def get_init_inputs():
    return []
'''
        else:
            problem_code += f'        # No known PyTorch equivalent; identity passthrough for structural testing\n'
            problem_code += f'        return {wrapper_params[0]}\n'
            problem_code += f'''

def get_inputs():
    return [{", ".join(param_tensors)}]

def get_init_inputs():
    return []
'''

        problem_path = problems_dir / f"problem_{idx}.py"
        with open(problem_path, "w", encoding="utf-8") as f:
            f.write(problem_code)

        param_call = ", ".join(wrapper_params)
        kernel_source = f'''import torch
import torch.nn as nn
import triton
import triton.language as tl

# --- TritonBench-G kernel: {filename} from {repo} ---
{triton_code}

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, {param_call}):
        return {wrapper_name}({param_call})
'''

        registry.append({
            "id": kid,
            "repo": repo,
            "kernel_name": filename.replace(".py", ""),
            "reference_file": _to_relative(problem_path),
            "requires": ["triton"],
            "kernel_source": kernel_source,
            "difficulty": difficulty,
            "star": star,
            "has_pytorch_ref": ref_match is not None,
            "wrapper_function": wrapper_name,
        })

    registry_path = data_dir / "registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    with_ref = sum(1 for r in registry if r.get("has_pytorch_ref"))
    print(f"  Prepared {len(registry)} kernels, skipped {skipped}")
    print(f"  With PyTorch ref: {with_ref}, without (structural only): {len(registry) - with_ref}")
    if skipped_reasons:
        print(f"  Skip reasons (first 10):")
        for fn, reason in list(skipped_reasons.items())[:10]:
            print(f"    {fn}: {reason}")
    print(f"  Registry: {registry_path}")
    return registry


if __name__ == "__main__":
    if "--cuda-l1" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        prepare_cuda_l1()
    if "--sakana" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        prepare_ai_cuda_engineer()
    if "--tritonbench" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        prepare_tritonbench()
