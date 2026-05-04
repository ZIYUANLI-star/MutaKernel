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


def _replace_module_fn_with_cuda_stub(functional_code: str, pybind_fn: str) -> str | None:
    """Replace the top-level ``def module_fn(...)`` in PyTorch_Code_Functional
    with a stub that delegates to the compiled CUDA extension's ``_ext.<pybind_fn>``.

    The signature is intentionally erased to ``*args, **kwargs`` so the stub
    works for every Sakana entry regardless of how many arguments module_fn
    originally took.

    Returns the rewritten code, or None if module_fn cannot be located.
    """
    import ast
    try:
        tree = ast.parse(functional_code)
    except SyntaxError:
        return None

    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "module_fn":
            target = node
            break
    if target is None:
        return None

    lines = functional_code.split("\n")
    start = target.lineno - 1
    end = target.end_lineno  # 1-indexed inclusive -> use as exclusive slice end

    stub = [
        "def module_fn(*args, **kwargs):",
        f"    return _ext.{pybind_fn}(*args, **kwargs)",
    ]
    return "\n".join(lines[:start] + stub + lines[end:])


def _wrap_cuda_with_functional_module(cuda_code: str, kid: str, functional_code: str) -> str | None:
    """Build a self-contained ``ModelNew`` wrapper for a Sakana CUDA kernel
    using ``PyTorch_Code_Functional`` from the dataset.

    Strategy: Sakana's ``PyTorch_Code_Functional`` field provides a
    ``module_fn(...)`` whose signature exactly matches the CUDA kernel's
    pybind ``forward(...)`` signature, plus a ``Model`` class that calls
    ``fn(<inputs and self.* attributes>)`` via a keyword default
    ``fn=module_fn``. We:

      1. Compile the embedded ``PYBIND11_MODULE`` via ``load_inline``
         (with ``functions=None`` so we don't double-register pybind).
      2. Splice the functional code in but replace ``module_fn``'s body
         with a thin ``_ext.forward(*args, **kwargs)`` stub.
      3. Rename ``class Model`` to ``class ModelNew`` so the MutaKernel
         worker (which expects ``ModelNew``) picks it up.

    Result: ``ref_model`` (loaded from problem_*.py) and ``orig_model``
    (loaded from this wrapper) share an identical ``__init__`` signature
    and ``forward`` interface, and only differ in the implementation of
    ``module_fn`` (PyTorch functional vs. CUDA kernel). Weight sync,
    input shapes, and call conventions are guaranteed to match.
    """
    fn_names = _extract_pybind_fn_names(cuda_code)
    if not fn_names:
        return None
    pybind_fn = "forward" if "forward" in fn_names else fn_names[0]

    rebuilt = _replace_module_fn_with_cuda_stub(functional_code, pybind_fn)
    if rebuilt is None:
        return None

    import re
    rebuilt = re.sub(r'^class\s+Model\b', "class ModelNew", rebuilt, flags=re.MULTILINE)
    # Some Sakana templates use the explicit Python-2-style ``super(Model, self).__init__()``.
    # After we rename Model -> ModelNew, that reference would be a NameError, so we
    # rewrite it to the no-arg form which always works in Python 3.
    rebuilt = re.sub(r'super\(\s*Model\s*,\s*self\s*\)', "super()", rebuilt)

    safe_name = re.sub(r'[^a-zA-Z0-9_]', "_", kid)

    header = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "from torch.utils.cpp_extension import load_inline\n"
        "\n"
        f'_cuda_source = r"""{cuda_code}"""\n'
        "\n"
        "_ext = load_inline(\n"
        f'    name="sakana_{safe_name}",\n'
        '    cpp_sources="",\n'
        "    cuda_sources=_cuda_source,\n"
        "    verbose=False,\n"
        '    extra_cuda_cflags=["-O2"],\n'
        ")\n"
        "\n"
    )
    return header + rebuilt


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
    skipped_no_functional = 0
    skipped_no_module_fn = 0

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
            pytorch_module_code = row.get("PyTorch_Code_Module", "")
            pytorch_functional_code = row.get("PyTorch_Code_Functional", "")
            speedup = row.get("CUDA_Speedup_Native", 0)
            max_diff = row.get("Max_Diff", 0)
            op_name = row.get("Op_Name", "unknown")

            if not cuda_code or not pytorch_functional_code:
                # Functional reference is required for the binding to match
                # the CUDA pybind signature. Module-only entries cannot use
                # the new wrapper.
                skipped_no_functional += 1
                continue

            kid = f"sakana__L{level_id}_T{task_id}"
            problem_path = problems_dir / f"problem_L{level_id}_T{task_id}.py"

            # Save PyTorch_Code_Functional as the reference. ``module_fn`` here
            # uses pure PyTorch, so ``ref_model`` will compute the ground truth.
            with open(problem_path, "w", encoding="utf-8") as f:
                f.write(pytorch_functional_code)

            wrapped_code = _wrap_cuda_with_functional_module(
                cuda_code, kid, pytorch_functional_code,
            )
            if wrapped_code is None:
                skipped_no_module_fn += 1
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
    print(f"  Skipped (no PyTorch_Code_Functional): {skipped_no_functional}")
    print(f"  Skipped (no module_fn or pybind):     {skipped_no_module_fn}")
    print(f"  Registry: {registry_path}")
    return registry


def _find_wrapper_callable(triton_code: str) -> tuple[str | None, list[str], int]:
    """Find the main user-facing callable in a TritonBench-G kernel file.

    Returns (callable_expr, args, n_defaults):
      - callable_expr: How ModelNew should invoke the kernel.
        For ordinary triton wrappers, this is the function name (e.g. "softmax").
        For ``torch.autograd.Function`` subclasses, this is either the
        module-level alias assigned to ``ClassName.apply`` (preferred) or
        ``ClassName.apply`` itself.
      - args:  Positional arg names of the callable (excluding self/ctx).
      - n_defaults: Number of trailing args that have default values.

    Returning ``(None, [], 0)`` means no usable wrapper was found.

    The previous implementation incorrectly returned ``forward`` as the
    wrapper for autograd.Function classes — but at runtime ``forward``
    is not a module-level callable, only ``ClassName.apply`` is, which
    caused 15 TritonBench-G kernels to fail with NameError.
    """
    import ast
    try:
        tree = ast.parse(triton_code)
    except SyntaxError:
        return None, [], 0

    autograd_classes: dict[str, tuple[list[str], int]] = {}
    apply_aliases: list[tuple[str, str]] = []
    top_level_funcs: list[tuple[str, list[str], int]] = []

    def _arg_info(func_node) -> tuple[list[str], int]:
        names = [a.arg for a in func_node.args.args if a.arg not in ("self", "ctx")]
        return names, len(func_node.args.defaults)

    def _is_decorated_kernel(func_node) -> bool:
        for d in func_node.decorator_list:
            if isinstance(d, ast.Attribute) and d.attr in ("jit",):
                return True
            if isinstance(d, ast.Call):
                fn = d.func
                if (getattr(fn, "attr", "") in ("autotune", "jit", "heuristics") or
                        getattr(fn, "id", "") in ("autotune", "jit", "heuristics")):
                    return True
        return False

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_decorated_kernel(node):
                names, n_def = _arg_info(node)
                top_level_funcs.append((node.name, names, n_def))

        elif isinstance(node, ast.ClassDef):
            is_autograd = any(
                (isinstance(b, ast.Attribute) and b.attr == "Function") or
                (isinstance(b, ast.Name) and b.id == "Function")
                for b in node.bases
            )
            if is_autograd:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "forward":
                        names, n_def = _arg_info(item)
                        autograd_classes[node.name] = (names, n_def)

        elif isinstance(node, ast.Assign):
            if (len(node.targets) == 1 and
                    isinstance(node.targets[0], ast.Name) and
                    isinstance(node.value, ast.Attribute) and
                    node.value.attr == "apply" and
                    isinstance(node.value.value, ast.Name)):
                apply_aliases.append((node.targets[0].id, node.value.value.id))

    if apply_aliases:
        for alias_name, class_name in apply_aliases:
            if class_name in autograd_classes:
                args, n_def = autograd_classes[class_name]
                return alias_name, args, n_def

    if autograd_classes:
        cls_name, (args, n_def) = next(iter(autograd_classes.items()))
        return f"{cls_name}.apply", args, n_def

    if top_level_funcs:
        name, args, n_def = top_level_funcs[-1]
        return name, args, n_def

    return None, [], 0


# (key, expr_template, input_specs, n_args)
# - input_specs: list of "name: tensor_init_expr" matching expr_template's variables.
# - The matching is done by lowercased keyword in filename/description.
# Order matters: more specific keys must come first. Each entry is matched
# against the kernel filename and instruction text via substring search.
TRITONBENCH_PYTORCH_REFS = {
    "swiglu":   ("x * torch.sigmoid(x) * y", ["x: torch.randn(4, 512)", "y: torch.randn(4, 512)"]),
    "geglu":    ("x * torch.nn.functional.gelu(y)", ["x: torch.randn(4, 512)", "y: torch.randn(4, 512)"]),
    "rmsnorm":  ("x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)", ["x: torch.randn(4, 256, 512)"]),
    "rms_norm": ("x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)", ["x: torch.randn(4, 256, 512)"]),
    "rms_layernorm": ("x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)", ["x: torch.randn(4, 256, 512)"]),
    "layernorm": ("torch.nn.functional.layer_norm(x, x.shape[-1:])", ["x: torch.randn(4, 256, 512)"]),
    "layer_norm": ("torch.nn.functional.layer_norm(x, x.shape[-1:])", ["x: torch.randn(4, 256, 512)"]),
    "groupnorm": ("torch.nn.functional.group_norm(x, num_groups=4)", ["x: torch.randn(4, 16, 32, 32)"]),
    "group_norm": ("torch.nn.functional.group_norm(x, num_groups=4)", ["x: torch.randn(4, 16, 32, 32)"]),
    "softmax":  ("torch.nn.functional.softmax(x, dim=-1)", ["x: torch.randn(4, 1024)"]),
    "log_softmax": ("torch.nn.functional.log_softmax(x, dim=-1)", ["x: torch.randn(4, 1024)"]),
    "softplus": ("torch.nn.functional.softplus(x)", ["x: torch.randn(4, 1024)"]),
    "kldiv":    ("torch.nn.functional.kl_div(x.log(), y, reduction='none')",
                 ["x: torch.rand(4, 1024).clamp(min=1e-6)", "y: torch.rand(4, 1024).clamp(min=1e-6)"]),
    "kl_div":   ("torch.nn.functional.kl_div(x.log(), y, reduction='none')",
                 ["x: torch.rand(4, 1024).clamp(min=1e-6)", "y: torch.rand(4, 1024).clamp(min=1e-6)"]),
    "cross_entropy": ("torch.nn.functional.cross_entropy(x, y)",
                      ["x: torch.randn(32, 100)", "y: torch.randint(0, 100, (32,))"]),
    "dropout":  ("torch.nn.functional.dropout(x, p=0.5, training=False)", ["x: torch.randn(4, 1024)"]),
    "l2_norm":  ("torch.nn.functional.normalize(x, p=2, dim=-1)", ["x: torch.randn(4, 1024)"]),
    "l2norm":   ("torch.nn.functional.normalize(x, p=2, dim=-1)", ["x: torch.randn(4, 1024)"]),
    "matmul":   ("torch.matmul(x, y)", ["x: torch.randn(128, 256)", "y: torch.randn(256, 128)"]),
    "bmm":      ("torch.bmm(x, y)", ["x: torch.randn(4, 32, 64)", "y: torch.randn(4, 64, 32)"]),
    "vecmat":   ("torch.matmul(x, y)", ["x: torch.randn(64)", "y: torch.randn(64, 128)"]),
    "outer":    ("torch.outer(x, y)", ["x: torch.randn(64)", "y: torch.randn(128)"]),
    "relu":     ("torch.relu(x)", ["x: torch.randn(4, 1024)"]),
    "gelu":     ("torch.nn.functional.gelu(x)", ["x: torch.randn(4, 1024)"]),
    "silu":     ("torch.nn.functional.silu(x)", ["x: torch.randn(4, 1024)"]),
    "sigmoid":  ("torch.sigmoid(x)", ["x: torch.randn(4, 1024)"]),
    "tanh":     ("torch.tanh(x)", ["x: torch.randn(4, 1024)"]),
    "leaky_relu": ("torch.nn.functional.leaky_relu(x, 0.01)", ["x: torch.randn(4, 1024)"]),
    "mish":     ("x * torch.tanh(torch.nn.functional.softplus(x))", ["x: torch.randn(4, 1024)"]),
    "elu":      ("torch.nn.functional.elu(x)", ["x: torch.randn(4, 1024)"]),
    "selu":     ("torch.nn.functional.selu(x)", ["x: torch.randn(4, 1024)"]),
    "add_value":("x + y", ["x: torch.randn(4, 1024)", "y: torch.randn(4, 1024)"]),
    "add":      ("x + y", ["x: torch.randn(4, 1024)", "y: torch.randn(4, 1024)"]),
    "mul":      ("x * y", ["x: torch.randn(4, 1024)", "y: torch.randn(4, 1024)"]),
    "div":      ("x / (y.abs() + 1e-6)", ["x: torch.randn(4, 1024)", "y: torch.randn(4, 1024)"]),
    "sub":      ("x - y", ["x: torch.randn(4, 1024)", "y: torch.randn(4, 1024)"]),
    "cumsum":   ("torch.cumsum(x, dim=-1)", ["x: torch.randn(4, 1024)"]),
    "cumprod":  ("torch.cumprod(x, dim=-1)", ["x: torch.randn(4, 64)"]),
    "rotary":   ("x", ["x: torch.randn(4, 8, 32, 64)"]),
    "rope":     ("x", ["x: torch.randn(4, 8, 32, 64)"]),
    "cosine":   ("torch.nn.functional.cosine_similarity(x, y, dim=-1)",
                 ["x: torch.randn(4, 1024)", "y: torch.randn(4, 1024)"]),
    "lse":      ("torch.logsumexp(x, dim=-1)", ["x: torch.randn(4, 1024)"]),
    "logsumexp": ("torch.logsumexp(x, dim=-1)", ["x: torch.randn(4, 1024)"]),
    "rsqrt":    ("torch.rsqrt(x.clamp(min=1e-6))", ["x: torch.rand(4, 1024).clamp(min=1e-6)"]),
    "abs":      ("torch.abs(x)", ["x: torch.randn(4, 1024)"]),
    "exp":      ("torch.exp(x.clamp(max=20))", ["x: torch.randn(4, 1024)"]),
    "log":      ("torch.log(x.clamp(min=1e-6))", ["x: torch.rand(4, 1024).clamp(min=1e-6)"]),
    "sqrt":     ("torch.sqrt(x.clamp(min=0))", ["x: torch.rand(4, 1024)"]),
    "pow":      ("torch.pow(x, 2)", ["x: torch.randn(4, 1024)"]),
    "softplus": ("torch.nn.functional.softplus(x)", ["x: torch.randn(4, 1024)"]),
}


def _name_token_match(haystack: str, key: str) -> bool:
    """Word-boundary substring match.

    ``mul`` should match ``mul_kernel`` (snake-cased token) but NOT
    ``multiplying``. The original substring matcher conflated these
    and produced false positives like the dequantize kernel matching
    ``mul`` because its description contains the verb "multiplying".
    """
    import re
    pattern = r"(?:^|[^a-z0-9])" + re.escape(key.lower()) + r"(?:[^a-z0-9]|$)"
    return re.search(pattern, haystack.lower()) is not None


def _match_pytorch_ref(filename: str, description: str,
                       n_required: int) -> tuple[str, list[str]] | None:
    """Try to match a TritonBench-G kernel to a known PyTorch equivalent.

    Match logic:
      - Filename token is the primary signal; ``description`` is consulted
        only when no filename match is found, and only for keys >= 5 chars
        (e.g. ``softmax``, ``layernorm``) to avoid false positives like
        ``mul`` matching the verb "multiplying".
      - The match is rejected when the matched ref's input count differs
        from the kernel's required-arg count. Calling the kernel wrapper
        with the wrong number of positional tensors is precisely what
        produced the TypeError/NameError skips that broke downstream
        stress testing in 47/67 TritonBench-G kernels with refs.
    """
    name_stem = (filename.lower()
                 .replace(".py", "")
                 .replace("_kernel", "")
                 .replace("_fwd", "")
                 .replace("_bwd", ""))
    desc_head = description.lower()[:200]

    sorted_keys = sorted(TRITONBENCH_PYTORCH_REFS.keys(), key=len, reverse=True)

    name_match: tuple[str, list[str]] | None = None
    desc_match: tuple[str, list[str]] | None = None

    for key in sorted_keys:
        expr, inputs = TRITONBENCH_PYTORCH_REFS[key]
        if len(inputs) != n_required:
            continue
        if name_match is None and _name_token_match(name_stem, key):
            name_match = (expr, inputs)
            break
        if desc_match is None and len(key) >= 5 and _name_token_match(desc_head, key):
            desc_match = (expr, inputs)

    return name_match or desc_match


def _extract_test_inputs_from_source(test_source_path: Path,
                                     callable_name: str) -> tuple[list[str], list[str]] | None:
    """Mine the upstream TritonBench-G ``test_*`` function for real inputs.

    Each kernel file in ``TritonBench-0/data/TritonBench_G_v1/`` ends with a
    ``def test_<name>():`` block that constructs realistic input tensors and
    invokes the kernel wrapper exactly as the kernel author intended. We walk
    that block until the first call to ``callable_name`` whose args are all
    bare names, then return:

      - ``arg_names``: positional arg names of that call.
      - ``setup_lines``: every assignment statement appearing **before** the
        call, source-unparsed, including scalar/tuple unpacks the tensor
        definitions depend on (e.g. ``b, h, n, d, e = 2, 8, 128, 64, 128``).
        The downstream prep emits these inside ``get_inputs()`` and then
        ``return [arg_names...]``, reproducing the exact shape/dtype the
        kernel was tested with. This fixes the original prep's hard-coded
        ``randn(4, 1024)`` defaults that silently caused 47 TritonBench-G
        kernels to TypeError at runtime.

    ``requires_grad=True`` and ``backward()`` patterns are stripped so the
    inputs are usable in ``with torch.no_grad():`` contexts.

    Returns ``None`` when the file lacks a usable ``test_`` function.
    """
    import ast
    import re

    try:
        src = test_source_path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        return None

    test_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            test_fn = node
            break
    if test_fn is None:
        return None

    def _short_name(call: ast.Call) -> str | None:
        f = call.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return None

    setup_stmts: list[ast.stmt] = []
    target_call: ast.Call | None = None

    for stmt in test_fn.body:
        if isinstance(stmt, ast.Assign):
            for sub in ast.walk(stmt.value):
                if (isinstance(sub, ast.Call) and
                        _short_name(sub) == callable_name and
                        all(isinstance(a, ast.Name) for a in sub.args)):
                    target_call = sub
                    break
            if target_call is not None:
                break
            setup_stmts.append(stmt)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if _short_name(stmt.value) == callable_name:
                if all(isinstance(a, ast.Name) for a in stmt.value.args):
                    target_call = stmt.value
                    break

    if target_call is None:
        return None

    arg_names: list[str] = [a.id for a in target_call.args]

    setup_lines: list[str] = []
    for stmt in setup_stmts:
        try:
            line = ast.unparse(stmt)
        except Exception:
            continue
        line = re.sub(r",?\s*requires_grad\s*=\s*True", "", line)
        setup_lines.append(line)

    return arg_names, setup_lines


def _resolve_test_source(filename: str) -> Path | None:
    """Locate the upstream kernel source file in the cloned TritonBench repo.

    Looks under both common clone locations (``TritonBench/`` and the
    ``TritonBench-0/`` mirror used by this workspace), at the project root
    or at one level above (the workspace root when ``MutaKernel`` is a
    sibling of ``TritonBench-0/``).
    """
    workspace_root = PROJECT_ROOT.parent
    candidates = [
        PROJECT_ROOT / "TritonBench-0" / "data" / "TritonBench_G_v1" / filename,
        PROJECT_ROOT / "TritonBench" / "data" / "TritonBench_G_v1" / filename,
        workspace_root / "TritonBench-0" / "data" / "TritonBench_G_v1" / filename,
        workspace_root / "TritonBench" / "data" / "TritonBench_G_v1" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _build_problem_and_kernel(
    filename: str,
    repo: str,
    triton_code: str,
    callable_expr: str,
    wrapper_args: list[str],
    n_defaults: int,
    ref_match: tuple[str, list[str]] | None,
) -> tuple[str, str, list[str], str]:
    """Build the problem_*.py source, kernel_source string, forward args, and mode tag.

    The forward signature, ``get_inputs``, and ``ModelNew.forward`` are all
    aligned so that ``Model()(*get_inputs())`` and ``ModelNew()(*get_inputs())``
    can both be called without TypeError/NameError. This is the fix for
    the data-prep bugs that caused 47/67 TritonBench-G kernels with PyTorch
    references to be skipped at runtime.

    Three input-source priorities (highest to lowest):
      1. ``ref_match`` - we have a known PyTorch equivalent: use its
         ``typed_inputs`` for the inputs and the matched expression as the ref.
      2. Mined inputs from ``TritonBench-0/data/TritonBench_G_v1/<file>.py`` -
         the kernel's own ``test_*`` block tells us exactly what shapes/dtypes
         the kernel needs. We use those inputs and self-reference (Model and
         ModelNew both call the kernel), so baseline trivially passes and
         stress tests still detect non-determinism (repeated_run).
      3. Generic fallback - randn(4, 1024) per required arg with identity
         passthrough; downstream framework filters these out.

    Returns (problem_code, kernel_source, var_names, mode) where mode is one
    of ``"pytorch_ref"``, ``"self_ref"``, ``"identity"``.
    """
    n_required = max(0, len(wrapper_args) - n_defaults)

    if ref_match:
        expr, typed_inputs = ref_match
        var_names: list[str] = []
        input_stmts: list[str] = []
        for ti in typed_inputs:
            name, init = ti.split(":", 1)
            var_names.append(name.strip())
            input_stmts.append(init.strip() + ".cuda()")
        forward_sig = ", ".join(["self"] + var_names)
        inputs_block = "\n".join(f"        {s}," for s in input_stmts)
        problem_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "\n"
            f"    def forward({forward_sig}):\n"
            f"        return {expr}\n"
            "\n"
            "\n"
            "def get_inputs():\n"
            "    return [\n"
            f"{inputs_block}\n"
            "    ]\n"
            "\n"
            "def get_init_inputs():\n"
            "    return []\n"
        )
        mode = "pytorch_ref"
    else:
        mined = None
        test_path = _resolve_test_source(filename)
        if test_path is not None:
            short_name = callable_expr.split(".")[-1]
            mined = _extract_test_inputs_from_source(test_path, short_name)

        if mined is not None:
            var_names, setup_lines = mined
            forward_sig = ", ".join(["self"] + var_names)
            arg_call = ", ".join(var_names)
            setup_block = "\n".join(f"    {l}" for l in setup_lines)
            return_list = ", ".join(var_names)
            problem_code = (
                "import torch\n"
                "import torch.nn as nn\n"
                "import triton\n"
                "import triton.language as tl\n"
                "\n"
                "# Self-reference Model: defines the same triton kernel as ModelNew\n"
                "# so baseline differential tests (ref vs orig) trivially pass and\n"
                "# stress tests can still detect non-determinism via repeated_run.\n"
                f"# Inputs were mined from the original TritonBench-G test_* block\n"
                f"# in {test_path.name}.\n"
                f"{triton_code}\n"
                "\n"
                "class Model(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "\n"
                f"    def forward({forward_sig}):\n"
                f"        return {callable_expr}({arg_call})\n"
                "\n"
                "\n"
                "def get_inputs():\n"
                f"{setup_block}\n"
                f"    return [{return_list}]\n"
                "\n"
                "def get_init_inputs():\n"
                "    return []\n"
            )
            mode = "self_ref"
        else:
            var_names = list(wrapper_args[:max(1, n_required)]) or ["x"]
            input_stmts = ["torch.randn(4, 1024, device='cuda')"
                           for _ in var_names]
            forward_sig = ", ".join(["self"] + var_names)
            inputs_block = "\n".join(f"        {s}," for s in input_stmts)
            problem_code = (
                "import torch\n"
                "import torch.nn as nn\n"
                "\n"
                "class Model(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "\n"
                f"    def forward({forward_sig}):\n"
                "        # No known PyTorch equivalent; identity passthrough for structural testing\n"
                f"        return {var_names[0]}\n"
                "\n"
                "\n"
                "def get_inputs():\n"
                "    return [\n"
                f"{inputs_block}\n"
                "    ]\n"
                "\n"
                "def get_init_inputs():\n"
                "    return []\n"
            )
            mode = "identity"

    arg_call = ", ".join(var_names)
    kernel_source = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        f"# --- TritonBench-G kernel: {filename} from {repo} ---\n"
        f"{triton_code}\n"
        "\n"
        "class ModelNew(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "\n"
        f"    def forward({forward_sig}):\n"
        f"        return {callable_expr}({arg_call})\n"
    )

    return problem_code, kernel_source, var_names, mode


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
        if not repo_json.exists():
            repo_json = PROJECT_ROOT / "TritonBench-0" / "data" / "TritonBench_G_v1.json"
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

        callable_expr, wrapper_args, n_defaults = _find_wrapper_callable(triton_code)
        if callable_expr is None or not wrapper_args:
            skipped += 1
            skipped_reasons[filename] = "no wrapper callable found"
            continue

        n_required = max(0, len(wrapper_args) - n_defaults)
        ref_match = _match_pytorch_ref(filename, description, n_required)

        kid = f"tritonbench__{idx}_{filename.replace('.py', '')}"

        kernel_path = kernels_dir / f"kernel_{idx}.py"
        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(triton_code)

        problem_code, kernel_source, var_names, mode = _build_problem_and_kernel(
            filename=filename,
            repo=repo,
            triton_code=triton_code,
            callable_expr=callable_expr,
            wrapper_args=wrapper_args,
            n_defaults=n_defaults,
            ref_match=ref_match,
        )

        problem_path = problems_dir / f"problem_{idx}.py"
        with open(problem_path, "w", encoding="utf-8") as f:
            f.write(problem_code)

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
            "wrapper_function": callable_expr,
            "forward_args": var_names,
            "ref_mode": mode,
        })

    registry_path = data_dir / "registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    by_mode: dict[str, int] = {}
    for r in registry:
        by_mode[r.get("ref_mode", "?")] = by_mode.get(r.get("ref_mode", "?"), 0) + 1

    print(f"  Prepared {len(registry)} kernels, skipped {skipped}")
    print(f"  Ref modes:")
    for m in ("pytorch_ref", "self_ref", "identity"):
        if m in by_mode:
            print(f"    {m}: {by_mode[m]}")
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
